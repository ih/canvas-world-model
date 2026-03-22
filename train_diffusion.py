"""Train a Conditional Diffusion ViT on canvas datasets.

Adds noise to the last frame region and trains the model to denoise it,
conditioned on clean context (history frames + action separators) + timestep.

Usage:
    python train_diffusion.py --dataset local/datasets/eval_shoulder_pan_10_minutes
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.canvas_dataset import CanvasDataset
from models.diffusion import ConditionalDiffusionViT, NoiseScheduler
from models.common import (
    set_seed,
    create_plateau_scheduler,
    save_checkpoint,
    load_checkpoint,
    compute_last_frame_patch_mask,
    patchify,
    unpatchify,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Conditional Diffusion ViT on canvas dataset")

    # Data
    p.add_argument("--dataset", type=str, required=True, help="Path to canvas dataset directory")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows)")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop after N epochs without val improvement (0=disabled)")

    # LR scheduler
    p.add_argument("--patience", type=int, default=5, help="Plateau patience (epochs)")
    p.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--lr-schedule", type=str, default="plateau", choices=["plateau", "cosine"])
    p.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs for cosine schedule")
    p.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max norm (0=disabled)")

    # Model
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=8)

    # Diffusion
    p.add_argument("--num-train-timesteps", type=int, default=1000)
    p.add_argument("--beta-schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--prediction-type", type=str, default="epsilon", choices=["epsilon", "sample"])

    # Checkpointing
    p.add_argument("--checkpoint-dir", type=str, default="local/checkpoints/diffusion")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--fine-tune", type=str, default=None,
                   help="Path to checkpoint for fine-tuning (loads weights only, fresh optimizer)")

    # Logging
    p.add_argument("--wandb-project", type=str, default="canvas-world-model")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--log-images-every", type=int, default=10, help="Log sample denoising every N epochs")

    return p.parse_args()


def log_sample_denoising(model, noise_scheduler, val_loader, patch_mask, device,
                         grid_h, grid_w, patch_size, num_inference_steps, epoch, wandb):
    """Run a few DDIM denoising steps and log results to wandb."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        canvas = batch["canvas"][:4].to(device)
        B = canvas.shape[0]
        batch_mask = patch_mask.expand(B, -1)

        target_patches = patchify(canvas, patch_size)

        # Start from pure noise for the last frame patches
        noisy_patches = target_patches.clone()
        noisy_patches[batch_mask] = torch.randn_like(noisy_patches[batch_mask])
        current = unpatchify(noisy_patches, patch_size, grid_h, grid_w)

        # DDIM denoising with evenly spaced timesteps
        step_size = noise_scheduler.num_train_timesteps // num_inference_steps
        timesteps = list(range(noise_scheduler.num_train_timesteps - 1, -1, -step_size))

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            pred = model(current, t_batch)

            # Apply denoising only to last frame patches
            current_patches = patchify(current, patch_size)
            denoised_patches = current_patches.clone()

            pred_last_frame = pred[batch_mask]
            current_last_frame = current_patches[batch_mask]

            stepped = noise_scheduler.step(pred_last_frame, t, current_last_frame)
            denoised_patches[batch_mask] = stepped
            current = unpatchify(denoised_patches, patch_size, grid_h, grid_w)

        images = []
        for i in range(B):
            gt = canvas[i].clamp(-1, 1).cpu() * 0.5 + 0.5  # [-1,1] -> [0,1]
            pred_img = current[i].clamp(-1, 1).cpu() * 0.5 + 0.5
            combined = torch.cat([gt, pred_img], dim=1)
            images.append(wandb.Image(combined, caption=f"Top: GT, Bottom: Denoised (epoch {epoch})"))
        wandb.log({"denoised_predictions": images}, step=epoch)


def train_one_epoch(model, loader, optimizer, noise_scheduler, patch_mask, device,
                    patch_size, num_train_timesteps, prediction_type, epoch, num_epochs,
                    grad_clip=0.0):
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
    for batch in pbar:
        canvas = batch["canvas"].to(device)  # [B, 3, H, W] in [-1, 1]
        B = canvas.shape[0]
        batch_mask = patch_mask.expand(B, -1)  # [B, num_patches]

        target_patches = patchify(canvas, patch_size)  # [B, num_patches, p*p*3]

        # Sample random timesteps
        timesteps = torch.randint(0, num_train_timesteps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(target_patches)

        # Add noise to LAST FRAME patches only; keep context clean
        noisy_patches = noise_scheduler.add_noise(target_patches, noise, timesteps)
        composite_patches = torch.where(
            batch_mask.unsqueeze(-1),
            noisy_patches,
            target_patches,
        )

        # Reconstruct canvas with noisy last frame
        grid_h = model.grid_h
        grid_w = model.grid_w
        noisy_canvas = unpatchify(composite_patches, patch_size, grid_h, grid_w)

        # Forward: predict noise (or clean) given noisy canvas + timestep
        pred_patches = model(noisy_canvas, timesteps)

        # Loss on masked patches only
        if prediction_type == "epsilon":
            loss_target = noise
        else:
            loss_target = target_patches

        loss = F.mse_loss(pred_patches[batch_mask], loss_target[batch_mask])

        optimizer.zero_grad()
        loss.backward()

        clip_val = grad_clip if grad_clip > 0 else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
        total_grad_norm += grad_norm.item()

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_grad_norm = total_grad_norm / max(num_batches, 1)
    return avg_loss, avg_grad_norm


@torch.no_grad()
def validate(model, loader, noise_scheduler, patch_mask, device, patch_size,
             num_train_timesteps, prediction_type, grid_h, grid_w):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        canvas = batch["canvas"].to(device)
        B = canvas.shape[0]
        batch_mask = patch_mask.expand(B, -1)

        target_patches = patchify(canvas, patch_size)
        timesteps = torch.randint(0, num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(target_patches)

        noisy_patches = noise_scheduler.add_noise(target_patches, noise, timesteps)
        composite_patches = torch.where(
            batch_mask.unsqueeze(-1), noisy_patches, target_patches,
        )
        noisy_canvas = unpatchify(composite_patches, patch_size, grid_h, grid_w)

        pred_patches = model(noisy_canvas, timesteps)

        if prediction_type == "epsilon":
            loss_target = noise
        else:
            loss_target = target_patches

        loss = F.mse_loss(pred_patches[batch_mask], loss_target[batch_mask])
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    train_dataset = CanvasDataset(
        args.dataset, split="train", val_ratio=args.val_ratio,
        normalize_mode="neg_one_one", seed=args.seed,
    )
    val_dataset = CanvasDataset(
        args.dataset, split="val", val_ratio=args.val_ratio,
        normalize_mode="neg_one_one", seed=args.seed,
    )
    print(f"Train: {len(train_dataset)} canvases, Val: {len(val_dataset)} canvases")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Canvas geometry ---
    meta = train_dataset.meta
    canvas_h, canvas_w = meta["canvas_size"]
    num_frames = meta["canvas_history_size"]
    sep_width = meta["separator_width"]

    # --- Model ---
    model = ConditionalDiffusionViT(
        img_height=canvas_h, img_width=canvas_w,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        prediction_type=args.prediction_type,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: Conditional Diffusion ViT, {num_params:.1f}M parameters")
    print(f"Canvas: {canvas_h}x{canvas_w}, patches: {model.grid_h}x{model.grid_w} = {model.num_patches}")
    print(f"Diffusion: {args.num_train_timesteps} timesteps, {args.beta_schedule} schedule, {args.prediction_type} prediction")

    # --- Noise scheduler ---
    noise_scheduler = NoiseScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )

    # --- Patch mask ---
    patch_mask = compute_last_frame_patch_mask(
        canvas_h, canvas_w, args.patch_size, num_frames, sep_width, device=device,
    )
    num_masked = patch_mask.sum().item()
    print(f"Target region: {num_masked}/{model.num_patches} patches (last frame)")

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
        use_cosine = True
    else:
        scheduler = create_plateau_scheduler(optimizer, patience=args.patience, factor=args.lr_factor, min_lr=args.min_lr)
        use_cosine = False

    # --- Fine-tune or Resume ---
    start_epoch = 0
    best_val_loss = float("inf")
    if args.fine_tune and args.resume:
        print("Error: --fine-tune and --resume are mutually exclusive")
        sys.exit(1)
    if args.fine_tune:
        ckpt = torch.load(args.fine_tune, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Fine-tuning from: {args.fine_tune} (weights only, fresh optimizer)")
    elif args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    # --- Wandb ---
    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"dit-d{args.depth}-e{args.embed_dim}",
                config=vars(args),
            )
        except ImportError:
            print("wandb not installed, logging to console only")

    # --- Training loop ---
    print(f"\nStarting training for {args.epochs} epochs...")
    prev_lr = optimizer.param_groups[0]["lr"]
    epochs_without_improvement = 0
    training_start = time.time()
    epoch_times = []
    val_loss_history = []
    train_loss_history = []
    best_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss, grad_norm = train_one_epoch(
            model, train_loader, optimizer, noise_scheduler, patch_mask, device,
            args.patch_size, args.num_train_timesteps, args.prediction_type,
            epoch + 1, args.epochs, grad_clip=args.grad_clip,
        )

        # Validate
        val_loss = None
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_loss = validate(
                model, val_loader, noise_scheduler, patch_mask, device,
                args.patch_size, args.num_train_timesteps, args.prediction_type,
                model.grid_h, model.grid_w,
            )
            if use_cosine:
                scheduler.step()
            else:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            lr_changed = current_lr != prev_lr
            if lr_changed:
                print(f"  >> LR reduced: {prev_lr:.2e} -> {current_lr:.2e}")
            prev_lr = current_lr

            # Track history
            val_loss_history.append(val_loss)
            train_loss_history.append(train_loss)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss,
                    Path(args.checkpoint_dir) / "best.pth",
                    extra={"args": vars(args)},
                )
            else:
                epochs_without_improvement += 1

            train_val_gap = train_loss - val_loss

            print(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"lr={current_lr:.2e}, best_val={best_val_loss:.6f}, "
                f"grad_norm={grad_norm:.4f}, tv_gap={train_val_gap:.6f}"
            )

            if wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "train_val_gap": train_val_gap,
                    "epoch": epoch + 1,
                }, step=epoch + 1)

                if (epoch + 1) % args.log_images_every == 0:
                    log_sample_denoising(
                        model, noise_scheduler, val_loader, patch_mask, device,
                        model.grid_h, model.grid_w, args.patch_size,
                        num_inference_steps=50, epoch=epoch + 1, wandb=wandb,
                    )

            # Early stopping
            epoch_times.append(time.time() - epoch_start)
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}: no val improvement for {args.early_stop_patience} epochs")
                break
        else:
            if use_cosine:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, lr={current_lr:.2e}, grad_norm={grad_norm:.4f}")
            if wandb:
                wandb.log({"train_loss": train_loss, "lr": current_lr, "grad_norm": grad_norm, "epoch": epoch + 1}, step=epoch + 1)
            train_loss_history.append(train_loss)
            epoch_times.append(time.time() - epoch_start)

    # Save final
    total_training_time = time.time() - training_start
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, best_val_loss,
        Path(args.checkpoint_dir) / "final.pth",
        extra={"args": vars(args)},
    )

    # Save timing and loss history
    timing = {
        "total_training_seconds": total_training_time,
        "avg_seconds_per_epoch": sum(epoch_times) / len(epoch_times) if epoch_times else 0,
        "best_epoch": best_epoch,
        "time_to_plateau_seconds": sum(epoch_times[:best_epoch]) if epoch_times and best_epoch > 0 else 0,
        "num_epochs_run": len(epoch_times),
        "epoch_times": epoch_times,
        "val_loss_history": val_loss_history,
        "train_loss_history": train_loss_history,
        "best_val_loss": best_val_loss,
    }
    timing_path = Path(args.checkpoint_dir) / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
