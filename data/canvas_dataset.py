"""PyTorch Dataset for loading canvas PNG images."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CanvasDataset(Dataset):
    """Loads canvas PNGs from a dataset directory with train/val splitting.

    Args:
        dataset_dir: Path to directory containing canvas PNGs and dataset_meta.json
        split: "train" or "val"
        val_ratio: Fraction of each episode's canvases used for validation
        normalize_mode: "zero_one" ([0,1]) or "neg_one_one" ([-1,1])
        seed: Random seed for reproducibility (unused currently, split is deterministic)
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
        normalize_mode: str = "zero_one",
        seed: int = 42,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.normalize_mode = normalize_mode

        # Load metadata
        meta_path = self.dataset_dir / "dataset_meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.canvas_h, self.canvas_w = self.meta["canvas_size"]
        self.num_frames = self.meta["canvas_history_size"]
        self.sep_width = self.meta["separator_width"]
        self.frame_h, self.frame_w = self.meta["frame_size"]

        # Build index list with split per episode
        # For single-canvas episodes, use episode index to deterministically assign
        self.indices = []
        rng = np.random.RandomState(seed)
        for ep in self.meta["episodes"]:
            start = ep["canvas_start"]
            end = ep["canvas_end"]
            count = end - start + 1

            if count == 1:
                # Single-canvas episode: assign to train or val based on hash
                is_val = rng.random() < val_ratio
                if split == "val" and is_val:
                    self.indices.append(start)
                elif split == "train" and not is_val:
                    self.indices.append(start)
            else:
                val_count = max(1, int(count * val_ratio))
                train_count = count - val_count
                if split == "train":
                    self.indices.extend(range(start, start + train_count))
                else:
                    self.indices.extend(range(start + train_count, end + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        canvas_idx = self.indices[idx]
        img_path = self.dataset_dir / f"canvas_{canvas_idx:05d}.png"
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]

        if self.normalize_mode == "neg_one_one":
            arr = arr * 2.0 - 1.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        return {"canvas": tensor, "index": canvas_idx}


def extract_last_frame_region(canvas, num_frames, frame_w, sep_width):
    """Split a canvas tensor into context and last frame.

    Args:
        canvas: Tensor [B, 3, H, W] or [3, H, W]
        num_frames: Number of frames in the canvas
        frame_w: Width of each frame in pixels
        sep_width: Width of separator in pixels

    Returns:
        (context, last_frame) where context has the last frame region zeroed out,
        and last_frame is the cropped last frame region.
    """
    squeeze = canvas.dim() == 3
    if squeeze:
        canvas = canvas.unsqueeze(0)

    last_frame_x = (num_frames - 1) * (frame_w + sep_width)
    context = canvas.clone()
    context[:, :, :, last_frame_x:] = 0.0
    last_frame = canvas[:, :, :, last_frame_x:]

    if squeeze:
        context = context.squeeze(0)
        last_frame = last_frame.squeeze(0)

    return context, last_frame
