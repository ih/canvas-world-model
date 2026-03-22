"""Benchmark inference latency for trained models.

Measures per-sample inference time for GPT and diffusion models.

Usage:
    python benchmark_inference.py --model-type gpt --checkpoint local/checkpoints/gpt/best.pth \
        --dataset local/datasets/hold-1500-combined --output timing.json
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median

import torch
from torch.utils.data import DataLoader

from data.canvas_dataset import CanvasDataset
from models.diffusion import NoiseScheduler
from models.common import (
    set_seed,
    compute_last_frame_patch_mask,
    patchify,
    unpatchify,
)
from inference import load_model_from_checkpoint, run_gpt_inference, run_diffusion_inference


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark inference latency")
    p.add_argument("--model-type", type=str, required=True, choices=["gpt", "diffusion"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output", type=str, required=True, help="Path to save timing JSON")
    p.add_argument("--num-samples", type=int, default=50, help="Number of samples to time")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations (not timed)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")

    # Load model
    model, saved_args, meta = load_model_from_checkpoint(args.model_type, args.checkpoint, device)
    print(f"Loaded: {args.checkpoint}")

    canvas_h, canvas_w = meta["canvas_size"]
    num_frames = meta["canvas_history_size"]
    sep_width = meta["separator_width"]
    patch_size = saved_args["patch_size"]
    grid_h = canvas_h // patch_size
    grid_w = canvas_w // patch_size

    # Load data
    normalize_mode = "neg_one_one" if args.model_type == "diffusion" else "zero_one"
    val_dataset = CanvasDataset(
        args.dataset, split="val", val_ratio=0.1,
        normalize_mode=normalize_mode, seed=args.seed,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    patch_mask = compute_last_frame_patch_mask(
        canvas_h, canvas_w, patch_size, num_frames, sep_width, device=device,
    )

    # Setup diffusion scheduler if needed
    noise_scheduler = None
    if args.model_type == "diffusion":
        noise_scheduler = NoiseScheduler(
            num_train_timesteps=saved_args["num_train_timesteps"],
            beta_schedule=saved_args["beta_schedule"],
            prediction_type=saved_args["prediction_type"],
        )

    # Collect samples
    total_needed = args.warmup + args.num_samples
    samples = []
    for i, batch in enumerate(val_loader):
        if i >= total_needed:
            break
        samples.append(batch["canvas"].to(device))
    # Cycle if not enough samples
    while len(samples) < total_needed:
        samples.append(samples[len(samples) % len(val_dataset)])

    print(f"Benchmarking {args.model_type}: {args.warmup} warmup + {args.num_samples} timed iterations")

    # Warmup
    for i in range(args.warmup):
        canvas = samples[i]
        with torch.no_grad():
            if args.model_type == "gpt":
                run_gpt_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w)
            else:
                run_diffusion_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w,
                                        noise_scheduler, num_inference_steps=50)
        if use_cuda:
            torch.cuda.synchronize()

    # Timed runs
    latencies_ms = []
    for i in range(args.num_samples):
        canvas = samples[args.warmup + i]
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if args.model_type == "gpt":
                run_gpt_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w)
            else:
                run_diffusion_inference(model, canvas, patch_mask, patch_size, grid_h, grid_w,
                                        noise_scheduler, num_inference_steps=50)

        if use_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    # Compute statistics
    latencies_ms.sort()
    p95_idx = int(len(latencies_ms) * 0.95)
    results = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_samples": args.num_samples,
        "mean_ms": mean(latencies_ms),
        "median_ms": median(latencies_ms),
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
        "p95_ms": latencies_ms[p95_idx],
        "all_latencies_ms": latencies_ms,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults:")
    print(f"  Mean:   {results['mean_ms']:.1f} ms")
    print(f"  Median: {results['median_ms']:.1f} ms")
    print(f"  P95:    {results['p95_ms']:.1f} ms")
    print(f"  Min:    {results['min_ms']:.1f} ms")
    print(f"  Max:    {results['max_ms']:.1f} ms")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
