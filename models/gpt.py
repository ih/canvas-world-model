"""GPT-style autoregressive Vision Transformer for canvas next-frame prediction.

Each patch position predicts the pixel values of the NEXT patch in raster order.
Patches are ordered left-to-right, top-to-bottom, so context frames naturally
precede the last frame region in the sequence.
"""

import numpy as np
import torch
import torch.nn as nn

from .common import (
    PatchEmbed, MLP,
    get_2d_sincos_pos_embed, patchify, unpatchify,
)


class CausalTransformerBlock(nn.Module):
    """Pre-norm transformer block with causal attention.

    LN -> MHSA (causal) -> residual -> LN -> MLP -> residual.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class AutoregressiveViT(nn.Module):
    """GPT-style decoder-only ViT for next-patch prediction.

    Processes all patches with causal attention so position i can only attend
    to positions 0..i. The linear head at position i predicts the pixel values
    of patch i+1.

    Args:
        img_height: Canvas height in pixels
        img_width: Canvas width in pixels
        patch_size: Size of square patches
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        drop: Dropout rate
    """

    def __init__(
        self,
        img_height=448,
        img_width=736,
        patch_size=16,
        embed_dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = patch_size ** 2 * 3

        # Patch embedding (reuse shared component)
        self.patch_embed = PatchEmbed(img_height, img_width, patch_size, 3, embed_dim)

        # Fixed 2D sin-cos positional embeddings (no CLS token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        # Causal transformer blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])

        # Final norm + prediction head
        self.norm = nn.LayerNorm(embed_dim)
        self.pred_head = nn.Linear(embed_dim, self.patch_dim)

        # Causal mask: True = blocked (upper triangle)
        causal_mask = torch.triu(
            torch.ones(self.num_patches, self.num_patches, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        self._init_weights()

    def _init_weights(self):
        # Fixed 2D sin-cos positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_h, self.grid_w
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # Zero-init prediction head for stable start
        nn.init.zeros_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

        # Init all linear/conv layers
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear) and m is not self.pred_head:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass with causal attention.

        Args:
            x: [B, 3, H, W] canvas images

        Returns:
            [B, num_patches, patch_dim] where position i predicts patch i+1
        """
        tokens = self.patch_embed(x)            # [B, N, embed_dim]
        tokens = tokens + self.pos_embed        # add positional info

        for blk in self.blocks:
            tokens = blk(tokens, attn_mask=self.causal_mask)

        tokens = self.norm(tokens)
        pred = self.pred_head(tokens)           # [B, N, patch_dim]
        return pred

    def pred_to_image(self, pred_patches):
        """Convert predicted patches to image tensor.

        Args:
            pred_patches: [B, num_patches, patch_size^2 * 3]

        Returns:
            [B, 3, H, W]
        """
        return unpatchify(pred_patches, self.patch_size, self.grid_h, self.grid_w)

    @torch.no_grad()
    def generate(self, x, last_frame_mask):
        """Autoregressive generation of last-frame patches.

        Feeds the full canvas (with last frame zeroed out) and generates
        last-frame patches one at a time in raster order. Each generated
        patch is inserted back into the canvas before the next forward pass.

        Args:
            x: [B, 3, H, W] canvas (last frame region should be zeroed)
            last_frame_mask: [1, num_patches] boolean, True = last frame patch

        Returns:
            [B, 3, H, W] canvas with generated last frame
        """
        self.eval()
        all_patches = patchify(x, self.patch_size)  # [B, N, patch_dim]

        # Indices of last-frame patches in raster order (already sorted ascending)
        target_indices = last_frame_mask[0].nonzero(as_tuple=True)[0]

        for idx in target_indices:
            # Reconstruct image from current patches and run forward
            current_img = unpatchify(all_patches, self.patch_size, self.grid_h, self.grid_w)
            pred = self.forward(current_img)    # [B, N, patch_dim]

            # Position (idx-1) predicts patch idx
            predicted_patch = pred[:, idx - 1, :]
            all_patches[:, idx, :] = predicted_patch

        return unpatchify(all_patches, self.patch_size, self.grid_h, self.grid_w)
