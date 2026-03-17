"""Masked Autoencoder Vision Transformer for canvas next-frame prediction."""

import torch
import torch.nn as nn
import numpy as np

from .common import (
    PatchEmbed, TransformerBlock, MLP,
    get_2d_sincos_pos_embed, patchify, unpatchify,
)


class MaskedAutoencoderViT(nn.Module):
    """MAE ViT that masks the last frame region and reconstructs it.

    The encoder processes only visible (unmasked) patches for efficiency.
    The decoder reconstructs all patches, with loss computed only on masked ones.

    Args:
        img_height: Canvas height in pixels
        img_width: Canvas width in pixels
        patch_size: Size of square patches
        embed_dim: Encoder embedding dimension
        depth: Number of encoder transformer blocks
        num_heads: Number of encoder attention heads
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder transformer blocks
        decoder_num_heads: Number of decoder attention heads
        mlp_ratio: MLP hidden dim ratio
    """

    def __init__(
        self,
        img_height=448,
        img_width=736,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # --- Encoder ---
        self.patch_embed = PatchEmbed(img_height, img_width, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for CLS token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # --- Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3)

        self._init_weights()

    def _init_weights(self):
        # Fixed 2D sin-cos positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_h, self.grid_w
        )
        # Prepend a zero for CLS token position
        cls_pos = np.zeros((1, pos_embed.shape[1]))
        full_pos = np.concatenate([cls_pos, pos_embed], axis=0)
        self.pos_embed.data.copy_(torch.from_numpy(full_pos).float())

        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.grid_h, self.grid_w
        )
        dec_cls_pos = np.zeros((1, dec_pos_embed.shape[1]))
        dec_full_pos = np.concatenate([dec_cls_pos, dec_pos_embed], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_full_pos).float())

        # Token inits
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Zero-init decoder prediction head for stable start
        nn.init.zeros_(self.decoder_pred.weight)
        nn.init.zeros_(self.decoder_pred.bias)

        # Init all linear/conv layers
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear) and m is not self.decoder_pred:
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

    def forward_encoder(self, x, mask):
        """Encode only the visible (unmasked) patches.

        Args:
            x: [B, 3, H, W] canvas images
            mask: [B, num_patches] boolean, True = masked (to predict)

        Returns:
            (latent, ids_restore) where latent is [B, num_visible+1, embed_dim]
            and ids_restore is used to unshuffle in the decoder.
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)  # [B, num_patches, embed_dim]
        tokens = tokens + self.pos_embed[:, 1:, :]  # add patch pos embed

        # Keep only visible (unmasked) patches
        visible_mask = ~mask  # [B, num_patches], True = visible
        # We need ids_restore to put mask tokens back in the right positions
        # Use argsort trick: sort so visible patches come first
        ids_shuffle = torch.argsort(mask.float(), dim=1)  # visible first (0s before 1s)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        num_visible = visible_mask[0].sum().item()
        ids_keep = ids_shuffle[:, :num_visible]  # [B, num_visible]

        # Gather visible tokens
        tokens = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Encoder blocks
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        tokens = self.encoder_norm(tokens)

        return tokens, ids_restore

    def forward_decoder(self, latent, ids_restore):
        """Decode all patches (insert mask tokens for masked positions).

        Args:
            latent: [B, num_visible+1, embed_dim] from encoder
            ids_restore: [B, num_patches] indices to unshuffle

        Returns:
            [B, num_patches, patch_size^2 * 3] predicted pixel patches
        """
        B = latent.shape[0]
        x = self.decoder_embed(latent)  # [B, num_visible+1, decoder_embed_dim]

        # Separate CLS and patch tokens
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # Append mask tokens to fill in masked positions
        num_masked = self.num_patches - patch_tokens.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        full_tokens = torch.cat([patch_tokens, mask_tokens], dim=1)

        # Unshuffle to original patch order
        full_tokens = torch.gather(
            full_tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1])
        )

        # Prepend CLS and add decoder positional embeddings
        full_tokens = torch.cat([cls_token, full_tokens], dim=1)
        full_tokens = full_tokens + self.decoder_pos_embed

        # Decoder blocks
        for blk in self.decoder_blocks:
            full_tokens = blk(full_tokens)
        full_tokens = self.decoder_norm(full_tokens)

        # Predict pixels (skip CLS token)
        pred = self.decoder_pred(full_tokens[:, 1:, :])  # [B, num_patches, p*p*3]
        return pred

    def forward(self, x, mask):
        """Full forward pass.

        Args:
            x: [B, 3, H, W] canvas images
            mask: [B, num_patches] boolean, True = masked

        Returns:
            (pred_patches, latent) where pred_patches is [B, num_patches, patch_size^2*3]
        """
        latent, ids_restore = self.forward_encoder(x, mask)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, latent

    def pred_to_image(self, pred_patches):
        """Convert predicted patches to image tensor.

        Args:
            pred_patches: [B, num_patches, patch_size^2 * 3]

        Returns:
            [B, 3, H, W]
        """
        return unpatchify(pred_patches, self.patch_size, self.grid_h, self.grid_w)
