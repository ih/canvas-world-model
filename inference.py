"""Run inference with a trained model and save GT vs prediction images.

Usage:
    python inference.py --model-type mae --checkpoint local/checkpoints/mae/best.pth \
        --dataset local/datasets/shoulder_pan_500 --output-dir local/inference/mae
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data.canvas_dataset import CanvasDataset
from models.mae import MaskedAutoencoderViT
from models.gpt import AutoregressiveViT
from models.diffusion import ConditionalDiffusionViT, NoiseScheduler
from models.common import (
    set_seed,
    compute_last_frame_patch_mask,
    patchify,
    unpatchify,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run inference and save prediction images")
    p.add_argument("--model-type", type=str, required=True, choices=["mae", "gpt", "diffusion"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth checkpoint")
    p.add_argument("--dataset", type=str, required=True, help="Path to canvas dataset directory")
    p.add_argument("--output-dir", type=str, default="local/inference", help="Where to save output images")
    p.add_argument("--num-samples", type=int, default=8, help="Number of samples to generate")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model_from_checkpoint(model_type, checkpoint_path, device):
    """Load model and saved args from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]

    canvas_h, canvas_w = None, None
    # Re-derive canvas size from dataset metadata
    dataset_path = saved_args["dataset"]
    meta_path = Path(dataset_path) / "dataset_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    canvas_h, canvas_w = meta["canvas_size"]
    num_frames = meta["canvas_history_size"]
    sep_width = meta["separator_width"]

    patch_size = saved_args["patch_size"]

    if model_type == "mae":
        model = MaskedAutoencoderViT(
            img_height=canvas_h, img_width=canvas_w,
            patch_size=patch_size,
            embed_dim=saved_args["embed_dim"],
            depth=saved_args["depth"],
            num_heads=saved_args["num_heads"],
            decoder_embed_dim=saved_args["decoder_embed_dim"],
            decoder_depth=saved_args["decoder_depth"],
            decoder_num_heads=saved_args["decoder_num_heads"],
        )
    elif model_type == "gpt":
        model = AutoregressiveViT(
            img_height=canvas_h, img_width=canvas_w,
            patch_size=patch_size,
            embed_dim=saved_args["embed_dim"],
            depth=saved_args["depth"],
            num_heads=saved_args["num_heads"],
        )
    elif model_type == "diffusion":
        model = ConditionalDiffusionViT(
            img_height=canvas_h, img_width=canvas_w,
            patch_size=patch_size,
            embed_dim=saved_args["embed_dim"],
            depth=saved_args["depth"],
            num_heads=saved_args["num_heads"],
            prediction_type=saved_args["prediction_type"],
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, saved_args, meta


@torch.no_grad()
def run_mae_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w):
    B = canvas.shape[0]
    batch_mask = patch_mask.expand(B, -1)

    pred_patches, _ = model(canvas, batch_mask)
    target_patches = patchify(canvas, patch_size)

    recon_patches = target_patches.clone()
    recon_patches[batch_mask] = pred_patches[batch_mask]
    recon_img = unpatchify(recon_patches, patch_size, grid_h, grid_w)
    return recon_img.clamp(0, 1)


@torch.no_grad()
def run_gpt_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w):
    B = canvas.shape[0]
    batch_mask = patch_mask.expand(B, -1)

    gen_input_patches = patchify(canvas, patch_size).clone()
    gen_input_patches[batch_mask] = 0.0
    gen_input = unpatchify(gen_input_patches, patch_size, grid_h, grid_w)

    generated = model.generate(gen_input, patch_mask)
    return generated.clamp(0, 1)


@torch.no_grad()
def run_diffusion_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w,
                            noise_scheduler, num_inference_steps=50):
    B = canvas.shape[0]
    batch_mask = patch_mask.expand(B, -1)
    device = canvas.device

    target_patches = patchify(canvas, patch_size)

    # Start from noise for last frame
    noisy_patches = target_patches.clone()
    noisy_patches[batch_mask] = torch.randn_like(noisy_patches[batch_mask])
    current = unpatchify(noisy_patches, patch_size, grid_h, grid_w)

    # DDIM denoising
    step_size = noise_scheduler.num_train_timesteps // num_inference_steps
    timesteps = list(range(noise_scheduler.num_train_timesteps - 1, -1, -step_size))

    for t in timesteps:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        pred = model(current, t_batch)

        current_patches = patchify(current, patch_size)
        denoised_patches = current_patches.clone()

        pred_last_frame = pred[batch_mask]
        current_last_frame = current_patches[batch_mask]

        stepped = noise_scheduler.step(pred_last_frame, t, current_last_frame)
        denoised_patches[batch_mask] = stepped
        current = unpatchify(denoised_patches, patch_size, grid_h, grid_w)

    # Convert [-1,1] -> [0,1]
    return current.clamp(-1, 1) * 0.5 + 0.5


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")

    # Load model
    model, saved_args, meta = load_model_from_checkpoint(args.model_type, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    canvas_h, canvas_w = meta["canvas_size"]
    num_frames = meta["canvas_history_size"]
    sep_width = meta["separator_width"]
    patch_size = saved_args["patch_size"]

    # Load validation data
    normalize_mode = "neg_one_one" if args.model_type == "diffusion" else "zero_one"
    val_dataset = CanvasDataset(
        args.dataset, split="val", val_ratio=0.1,
        normalize_mode=normalize_mode, seed=args.seed,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.num_samples, shuffle=False)
    print(f"Val samples: {len(val_dataset)}")

    # Patch mask
    patch_mask = compute_last_frame_patch_mask(
        canvas_h, canvas_w, patch_size, num_frames, sep_width, device=device,
    )

    # Get a batch
    batch = next(iter(val_loader))
    canvas = batch["canvas"][:args.num_samples].to(device)
    B = canvas.shape[0]

    grid_h = canvas_h // patch_size
    grid_w = canvas_w // patch_size

    # Run inference
    print(f"Running {args.model_type} inference on {B} samples...")
    if args.model_type == "mae":
        pred = run_mae_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w)
        gt = canvas.clamp(0, 1)
    elif args.model_type == "gpt":
        pred = run_gpt_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w)
        gt = canvas.clamp(0, 1)
    elif args.model_type == "diffusion":
        noise_scheduler = NoiseScheduler(
            num_train_timesteps=saved_args["num_train_timesteps"],
            beta_schedule=saved_args["beta_schedule"],
            prediction_type=saved_args["prediction_type"],
        )
        pred = run_diffusion_inference(
            model, canvas, patch_mask, patch_size, grid_h, grid_w,
            noise_scheduler, num_inference_steps=50,
        )
        gt = canvas.clamp(-1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]

    # Save images
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(B):
        # Stack GT on top, prediction on bottom
        combined = torch.cat([gt[i], pred[i]], dim=1)  # [3, 2*H, W]
        save_path = output_dir / f"sample_{i:03d}_gt_vs_pred.png"
        save_image(combined, save_path)

    print(f"\nSaved {B} inference images to: {output_dir}")
    print("Each image shows: GT (top) vs Prediction (bottom)")


if __name__ == "__main__":
    main()
