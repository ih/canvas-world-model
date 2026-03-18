"""Train a Masked Autoencoder ViT on canvas datasets.

Masks the last frame region and trains the model to reconstruct it
from the visible context (history frames + action separators).

Usage:
    python train_mae.py --dataset local/datasets/eval_shoulder_pan_10_minutes
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.canvas_dataset import CanvasDataset
from models.mae import MaskedAutoencoderViT
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
    p = argparse.ArgumentParser(description="Train MAE ViT on canvas dataset")

    # Data
    p.add_argument("--dataset", type=str, required=True, help="Path to canvas dataset directory")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows)")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop after N epochs without val improvement (0=disabled)")

    # LR scheduler
    p.add_argument("--patience", type=int, default=5, help="Plateau patience (epochs)")
    p.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor")
    p.add_argument("--min-lr", type=float, default=1e-6)

    # Model
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--decoder-embed-dim", type=int, default=128)
    p.add_argument("--decoder-depth", type=int, default=4)
    p.add_argument("--decoder-num-heads", type=int, default=4)

    # Checkpointing
    p.add_argument("--checkpoint-dir", type=str, default="local/checkpoints/mae")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Logging
    p.add_argument("--wandb-project", type=str, default="canvas-world-model")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--log-images-every", type=int, default=10, help="Log sample predictions every N epochs")

    return p.parse_args()


def log_sample_predictions(model, val_loader, patch_mask, device, grid_h, grid_w, patch_size, epoch, wandb):
    """Log a side-by-side comparison of ground truth vs prediction to wandb."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        canvas = batch["canvas"][:4].to(device)  # Take up to 4 samples
        B = canvas.shape[0]
        batch_mask = patch_mask.expand(B, -1)

        pred_patches, _ = model(canvas, batch_mask)
        target_patches = patchify(canvas, patch_size)

        # Reconstruct full prediction image (context + predicted last frame)
        recon_patches = target_patches.clone()
        recon_patches[batch_mask] = pred_patches[batch_mask]
        recon_img = unpatchify(recon_patches, patch_size, grid_h, grid_w)

        images = []
        for i in range(B):
            # Ground truth canvas vs reconstructed canvas
            gt = canvas[i].clamp(0, 1).cpu()
            pred = recon_img[i].clamp(0, 1).cpu()
            # Stack vertically: GT on top, prediction on bottom
            combined = torch.cat([gt, pred], dim=1)  # [3, 2*H, W]
            images.append(wandb.Image(combined, caption=f"Top: GT, Bottom: Pred (epoch {epoch})"))
        wandb.log({"predictions": images}, step=epoch)


def train_one_epoch(model, loader, optimizer, patch_mask, device, patch_size, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
    for batch in pbar:
        canvas = batch["canvas"].to(device)
        B = canvas.shape[0]
        batch_mask = patch_mask.expand(B, -1)

        pred_patches, _ = model(canvas, batch_mask)
        target_patches = patchify(canvas, patch_size)

        # Loss on masked patches only
        loss = F.mse_loss(pred_patches[batch_mask], target_patches[batch_mask])

        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm before optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        total_grad_norm += grad_norm.item()

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_grad_norm = total_grad_norm / max(num_batches, 1)
    return avg_loss, avg_grad_norm


@torch.no_grad()
def validate(model, loader, patch_mask, device, patch_size):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        canvas = batch["canvas"].to(device)
        B = canvas.shape[0]
        batch_mask = patch_mask.expand(B, -1)

        pred_patches, _ = model(canvas, batch_mask)
        target_patches = patchify(canvas, patch_size)

        loss = F.mse_loss(pred_patches[batch_mask], target_patches[batch_mask])
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
        normalize_mode="zero_one", seed=args.seed,
    )
    val_dataset = CanvasDataset(
        args.dataset, split="val", val_ratio=args.val_ratio,
        normalize_mode="zero_one", seed=args.seed,
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

    # --- Canvas geometry from metadata ---
    meta = train_dataset.meta
    canvas_h, canvas_w = meta["canvas_size"]
    num_frames = meta["canvas_history_size"]
    sep_width = meta["separator_width"]

    # --- Model ---
    model = MaskedAutoencoderViT(
        img_height=canvas_h, img_width=canvas_w,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: MAE ViT, {num_params:.1f}M parameters")
    print(f"Canvas: {canvas_h}x{canvas_w}, patches: {model.grid_h}x{model.grid_w} = {model.num_patches}")

    # --- Patch mask (deterministic: last frame region) ---
    patch_mask = compute_last_frame_patch_mask(
        canvas_h, canvas_w, args.patch_size, num_frames, sep_width, device=device,
    )
    num_masked = patch_mask.sum().item()
    print(f"Masking: {num_masked}/{model.num_patches} patches (last frame region)")

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_plateau_scheduler(optimizer, patience=args.patience, factor=args.lr_factor, min_lr=args.min_lr)

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
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
                name=args.wandb_run_name or f"mae-d{args.depth}-e{args.embed_dim}",
                config=vars(args),
            )
        except ImportError:
            print("wandb not installed, logging to console only")

    # --- Training loop ---
    print(f"\nStarting training for {args.epochs} epochs...")
    prev_lr = optimizer.param_groups[0]["lr"]
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, grad_norm = train_one_epoch(
            model, train_loader, optimizer, patch_mask, device,
            args.patch_size, epoch + 1, args.epochs,
        )

        # Validate
        val_loss = None
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_loss = validate(model, val_loader, patch_mask, device, args.patch_size)
            scheduler.step(val_loss)

            # Check for LR change
            current_lr = optimizer.param_groups[0]["lr"]
            lr_changed = current_lr != prev_lr
            if lr_changed:
                print(f"  >> LR reduced: {prev_lr:.2e} -> {current_lr:.2e}")
            prev_lr = current_lr

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss,
                    Path(args.checkpoint_dir) / "best.pth",
                    extra={"args": vars(args)},
                )
            else:
                epochs_without_improvement += 1

            train_val_gap = train_loss - val_loss

            # Console log
            print(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"lr={current_lr:.2e}, best_val={best_val_loss:.6f}, "
                f"grad_norm={grad_norm:.4f}, tv_gap={train_val_gap:.6f}"
            )

            # Wandb log
            if wandb:
                log_dict = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "train_val_gap": train_val_gap,
                    "epoch": epoch + 1,
                }
                wandb.log(log_dict, step=epoch + 1)

                # Log sample predictions
                if (epoch + 1) % args.log_images_every == 0:
                    log_sample_predictions(
                        model, val_loader, patch_mask, device,
                        model.grid_h, model.grid_w, args.patch_size,
                        epoch + 1, wandb,
                    )

            # Early stopping
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}: no val improvement for {args.early_stop_patience} epochs")
                break
        else:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, lr={current_lr:.2e}, grad_norm={grad_norm:.4f}")
            if wandb:
                wandb.log({"train_loss": train_loss, "lr": current_lr, "grad_norm": grad_norm, "epoch": epoch + 1}, step=epoch + 1)

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, best_val_loss,
        Path(args.checkpoint_dir) / "final.pth",
        extra={"args": vars(args)},
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
