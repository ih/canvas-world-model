"""Shared utilities for training: scheduling, checkpointing, patching."""

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_plateau_scheduler(optimizer, patience=5, factor=0.5, min_lr=1e-6):
    """Create a ReduceLROnPlateau scheduler.

    Call scheduler.step(val_loss) after each validation epoch.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=factor, min_lr=min_lr,
    )


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path, extra=None):
    """Save a training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Load a training checkpoint. Returns (epoch, val_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


def compute_last_frame_patch_mask(canvas_h, canvas_w, patch_size, num_frames, sep_width, device="cpu"):
    """Compute boolean mask marking patches in the last frame region.

    Returns:
        Tensor [1, num_patches] where True = last frame patch (to be masked/predicted).
    """
    grid_h = canvas_h // patch_size
    grid_w = canvas_w // patch_size
    frame_w_pixels = (canvas_w - (num_frames - 1) * sep_width) // num_frames
    last_frame_x = (num_frames - 1) * (frame_w_pixels + sep_width)
    last_frame_col_start = last_frame_x // patch_size

    mask = torch.zeros(1, grid_h * grid_w, dtype=torch.bool, device=device)
    for row in range(grid_h):
        for col in range(last_frame_col_start, grid_w):
            patch_idx = row * grid_w + col
            mask[0, patch_idx] = True
    return mask


def patchify(imgs, patch_size):
    """Convert images to patches.

    Args:
        imgs: [B, 3, H, W]

    Returns:
        [B, num_patches, patch_size^2 * 3]
    """
    B, C, H, W = imgs.shape
    gh, gw = H // patch_size, W // patch_size
    x = imgs.reshape(B, C, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)  # [B, gh, gw, p, p, C]
    x = x.reshape(B, gh * gw, patch_size * patch_size * C)
    return x


def unpatchify(patches, patch_size, grid_h, grid_w):
    """Convert patches back to images.

    Args:
        patches: [B, num_patches, patch_size^2 * 3]
        patch_size: Size of each patch
        grid_h: Number of patch rows
        grid_w: Number of patch columns

    Returns:
        [B, 3, H, W]
    """
    B = patches.shape[0]
    C = 3
    x = patches.reshape(B, grid_h, grid_w, patch_size, patch_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, gh, p, gw, p]
    x = x.reshape(B, C, grid_h * patch_size, grid_w * patch_size)
    return x


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be divisible by 4)
        grid_h: Number of rows in the grid
        grid_w: Number of columns in the grid

    Returns:
        [grid_h * grid_w, embed_dim] numpy array
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos"
    half_dim = embed_dim // 2

    # Generate 1D embeddings for each axis
    pos_h = np.arange(grid_h, dtype=np.float32)
    pos_w = np.arange(grid_w, dtype=np.float32)

    emb_h = _get_1d_sincos_embed(half_dim, pos_h)  # [grid_h, half_dim]
    emb_w = _get_1d_sincos_embed(half_dim, pos_w)  # [grid_w, half_dim]

    # Combine: for each (h, w) position, concat emb_h[h] and emb_w[w]
    emb_h = np.repeat(emb_h, grid_w, axis=0)  # [grid_h*grid_w, half_dim]
    emb_w = np.tile(emb_w, (grid_h, 1))        # [grid_h*grid_w, half_dim]

    return np.concatenate([emb_h, emb_w], axis=1)  # [grid_h*grid_w, embed_dim]


def _get_1d_sincos_embed(embed_dim, positions):
    """Generate 1D sine-cosine embeddings.

    Args:
        embed_dim: Output dimension (half sin, half cos)
        positions: [N] array of positions

    Returns:
        [N, embed_dim] numpy array
    """
    half = embed_dim // 2
    omega = np.arange(half, dtype=np.float64) / half
    omega = 1.0 / (10000.0 ** omega)

    pos = positions[:, None]  # [N, 1]
    out = pos * omega[None, :]  # [N, half]

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1).astype(np.float32)


# --- Shared building blocks ---

class PatchEmbed(nn.Module):
    """Image to patch embedding using Conv2d."""

    def __init__(self, img_height, img_width, patch_size, in_chans=3, embed_dim=256):
        super().__init__()
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """Simple two-layer MLP with GELU."""

    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHSA -> residual -> LN -> MLP -> residual."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x
