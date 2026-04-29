"""Conditional Diffusion Vision Transformer for canvas next-frame prediction.

Pixel-space diffusion: noise is added to the last frame region, and the model
denoises conditioned on clean context (history frames + separators) + timestep.
"""

import math

import torch
import torch.nn as nn
import numpy as np

from .common import (
    PatchEmbed, MLP,
    get_2d_sincos_pos_embed, patchify, unpatchify,
)


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding, projected through an MLP."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        """
        Args:
            t: [B] integer timesteps

        Returns:
            [B, dim] timestep embeddings
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    Adaptive layer norm modulates the block based on timestep conditioning.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

        # adaLN-Zero: 6 modulation params (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Zero-init the modulation output for stable training start
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """
        Args:
            x: [B, N, dim] token sequence
            c: [B, dim] conditioning vector (timestep embedding)

        Returns:
            [B, N, dim]
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with adaLN
        h = self.norm1(x) * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + alpha1.unsqueeze(1) * h

        # MLP with adaLN
        h = self.norm2(x) * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.mlp(h)
        x = x + alpha2.unsqueeze(1) * h

        return x


class ConditionalDiffusionViT(nn.Module):
    """Diffusion ViT that denoises the last frame conditioned on context + timestep.

    The full canvas (with noisy last frame) is patchified and fed to the model.
    Timestep conditioning is applied via adaLN-Zero in each transformer block.

    Args:
        img_height: Canvas height in pixels
        img_width: Canvas width in pixels
        patch_size: Size of square patches
        embed_dim: Transformer embedding dimension
        depth: Number of DiT blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        prediction_type: "epsilon" (predict noise) or "sample" (predict clean)
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
        prediction_type="epsilon",
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.prediction_type = prediction_type
        # Gradient checkpointing trades ~30% wall-time for ~3-5x activation
        # memory savings — recompute each block's activations during the
        # backward pass instead of storing them. Required when scaling
        # past ~300M params on a 32GB card.
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # Patch embedding
        self.patch_embed = PatchEmbed(img_height, img_width, patch_size, 3, embed_dim)

        # Positional embedding (fixed sin-cos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(embed_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # Final layer
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )
        self.pred_head = nn.Linear(embed_dim, patch_size ** 2 * 3)

        self._init_weights()

    def _init_weights(self):
        # Fixed 2D sin-cos positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_h, self.grid_w
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # Zero-init final layers
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        nn.init.zeros_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

        # Init other layers
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            # Skip already zero-initialized layers
            if m is self.pred_head or m is self.final_modulation[1]:
                return
            # Skip adaLN modulation layers (already zero-inited in DiTBlock)
            for blk in self.blocks:
                if m is blk.adaLN_modulation[1]:
                    return
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, timesteps):
        """Forward pass.

        Args:
            x: [B, 3, H, W] canvas (context clean + last frame noisy)
            timesteps: [B] integer timesteps

        Returns:
            [B, num_patches, patch_size^2 * 3] predicted noise or clean patches
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)  # [B, num_patches, embed_dim]
        tokens = tokens + self.pos_embed

        # Timestep conditioning
        c = self.time_embed(timesteps)  # [B, embed_dim]

        # DiT blocks
        if self.gradient_checkpointing and self.training:
            import torch.utils.checkpoint as cp
            for blk in self.blocks:
                # use_reentrant=False is the new default and supports
                # all our autograd patterns; explicit to silence warnings.
                tokens = cp.checkpoint(blk, tokens, c, use_reentrant=False)
        else:
            for blk in self.blocks:
                tokens = blk(tokens, c)

        # Final prediction
        gamma, beta = self.final_modulation(c).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        pred = self.pred_head(tokens)  # [B, num_patches, p*p*3]

        return pred

    def pred_to_image(self, pred_patches):
        """Convert predicted patches to image tensor."""
        return unpatchify(pred_patches, self.patch_size, self.grid_h, self.grid_w)


class NoiseScheduler:
    """DDPM noise scheduler with linear or cosine beta schedule.

    Supports forward process (add_noise) and DDIM reverse step.
    """

    def __init__(self, num_train_timesteps=1000, beta_schedule="cosine", prediction_type="epsilon"):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(num_timesteps, s=0.008):
        """Cosine schedule as proposed in Improved DDPM."""
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f_t = torch.cos((steps / num_timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0, 0.999).float()

    def add_noise(self, original, noise, timesteps):
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps.

        Args:
            original: [B, ...] clean data
            noise: [B, ...] sampled noise (same shape)
            timesteps: [B] integer timesteps

        Returns:
            [B, ...] noisy data
        """
        device = original.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps]

        # Reshape for broadcasting
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * original + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def step(self, model_output, timestep, sample, eta=0.0):
        """DDIM reverse step: predict x_{t-1} from x_t and model output.

        Args:
            model_output: [B, ...] predicted noise or clean sample
            timestep: int, current timestep
            sample: [B, ...] current noisy sample x_t
            eta: DDIM stochasticity (0 = deterministic DDIM, 1 = DDPM)

        Returns:
            [B, ...] predicted x_{t-1}
        """
        device = sample.device
        alpha_bar_t = self.alphas_cumprod.to(device)[timestep]

        if timestep > 0:
            alpha_bar_prev = self.alphas_cumprod.to(device)[timestep - 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        if self.prediction_type == "epsilon":
            pred_x0 = (sample - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
        else:
            pred_x0 = model_output

        # DDIM formula
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * model_output if self.prediction_type == "epsilon" else torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * ((sample - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1 - alpha_bar_t))
        prev_sample = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        if eta > 0 and timestep > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + sigma * noise

        return prev_sample
