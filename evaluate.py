"""Post-training evaluation for canvas world models.

Computes comprehensive metrics including loss decomposition, image quality,
action sensitivity, motor state prediction, and model-specific diagnostics.
Generates counterfactual visualizations and a self-contained HTML report.

Usage:
    python evaluate.py --model-type mae --checkpoint local/checkpoints/mae/best.pth \
        --dataset local/datasets/shoulder_pan_500
"""

import argparse
import base64
import io
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.canvas_dataset import CanvasDataset
from data.canvas_builder import _separator_color_for_action
from models.mae import MaskedAutoencoderViT
from models.gpt import AutoregressiveViT
from models.diffusion import ConditionalDiffusionViT, NoiseScheduler
from models.common import (
    set_seed,
    compute_last_frame_patch_mask,
    patchify,
    unpatchify,
)


# ---------------------------------------------------------------------------
# Canvas region extraction helpers
# ---------------------------------------------------------------------------

def get_last_frame_pixel_coords(meta):
    """Get pixel coordinates of the last frame region in the canvas.

    Returns:
        (x_start, y_visual_end, motor_strip_height) where the last frame
        spans canvas[:, :, :y_visual_end, x_start:] for visual pixels and
        canvas[:, :, y_visual_end:, x_start:] for the motor strip.
    """
    frame_h, frame_w = meta["frame_size"]
    # canvas_history_size is the total number of frames in the canvas
    # (e.g., 2 = 1 history frame + 1 prediction target)
    total_frames = meta["canvas_history_size"]
    sep_width = meta["separator_width"]
    motor_strip_height = meta.get("motor_strip_height", 0)

    x_start = (total_frames - 1) * (frame_w + sep_width)

    y_visual_end = frame_h  # visual portion ends at frame_h
    return x_start, y_visual_end, motor_strip_height


def extract_last_frame_visual(canvas, meta):
    """Extract only the visual pixels of the last frame (no motor strip).

    Args:
        canvas: Tensor [B, 3, H, W] or [3, H, W]

    Returns:
        Tensor of last frame visual region
    """
    squeeze = canvas.dim() == 3
    if squeeze:
        canvas = canvas.unsqueeze(0)

    x_start, y_visual_end, _ = get_last_frame_pixel_coords(meta)
    region = canvas[:, :, :y_visual_end, x_start:]

    if squeeze:
        region = region.squeeze(0)
    return region


def extract_last_frame_motor_strip(canvas, meta):
    """Extract only the motor strip of the last frame.

    Args:
        canvas: Tensor [B, 3, H, W] or [3, H, W]

    Returns:
        Tensor of motor strip region, or None if no motor strip
    """
    motor_strip_height = meta.get("motor_strip_height", 0)
    if motor_strip_height == 0:
        return None

    squeeze = canvas.dim() == 3
    if squeeze:
        canvas = canvas.unsqueeze(0)

    x_start, y_visual_end, _ = get_last_frame_pixel_coords(meta)
    region = canvas[:, :, y_visual_end:, x_start:]

    if squeeze:
        region = region.squeeze(0)
    return region


def decode_motor_positions(motor_strip_pixels, meta, patch_size=16):
    """Decode motor position values from grayscale motor strip pixels.

    Args:
        motor_strip_pixels: Tensor [B, 3, strip_h, frame_w] in [0,1]
        meta: Dataset metadata with motor_norm_min/max

    Returns:
        Numpy array [B, num_joints] of decoded positions, or None
    """
    norm_min = meta.get("motor_norm_min")
    norm_max = meta.get("motor_norm_max")
    if norm_min is None or norm_max is None:
        return None

    norm_min = np.array(norm_min)
    norm_max = np.array(norm_max)
    num_joints = len(norm_min)

    B = motor_strip_pixels.shape[0]
    frame_w = motor_strip_pixels.shape[3]

    positions = np.zeros((B, num_joints))
    for j in range(num_joints):
        x_start = j * patch_size
        x_end = x_start + patch_size
        if x_end > frame_w:
            break
        # Average grayscale value of the patch (all 3 channels should be equal)
        patch_val = motor_strip_pixels[:, 0, :, x_start:x_end].mean(dim=(1, 2)).cpu().numpy()
        # Decode: gray_val was norm_pos * 255 mapped to [0,1] -> norm_pos
        pos_range = norm_max[j] - norm_min[j]
        if pos_range < 1e-8:
            pos_range = 1.0
        positions[:, j] = patch_val * pos_range + norm_min[j]

    return positions


def decode_motor_velocities(motor_strip_pixels, meta, patch_size=16):
    """Decode motor velocity values from grayscale motor strip pixels.

    Args:
        motor_strip_pixels: Tensor [B, 3, strip_h, frame_w] in [0,1]
        meta: Dataset metadata with motor_vel_norm_max

    Returns:
        Numpy array [B, num_joints] of decoded velocities, or None
    """
    vel_norm_max = meta.get("motor_vel_norm_max")
    norm_min = meta.get("motor_norm_min")
    if vel_norm_max is None or norm_min is None:
        return None

    vel_norm_max = np.array(vel_norm_max)
    num_joints = len(vel_norm_max)

    B = motor_strip_pixels.shape[0]
    frame_w = motor_strip_pixels.shape[3]

    velocities = np.zeros((B, num_joints))
    for j in range(num_joints):
        x_start = (num_joints + j) * patch_size
        x_end = x_start + patch_size
        if x_end > frame_w:
            break
        patch_val = motor_strip_pixels[:, 0, :, x_start:x_end].mean(dim=(1, 2)).cpu().numpy()
        # Decode: gray_val was (norm_vel * 0.5 + 0.5) mapped to [0,1]
        norm_vel = (patch_val - 0.5) * 2.0
        velocities[:, j] = norm_vel * vel_norm_max[j]

    return velocities


def get_last_separator_coords(meta):
    """Get pixel x-coordinates of the last action separator.

    Returns:
        (x_start, x_end) of the last separator
    """
    frame_h, frame_w = meta["frame_size"]
    num_frames = meta["canvas_history_size"]
    sep_width = meta["separator_width"]

    # Last separator is between frame (num_frames-2) and frame (num_frames-1)
    x_start = (num_frames - 1) * frame_w + (num_frames - 2) * sep_width
    x_end = x_start + sep_width
    return x_start, x_end


def replace_last_separator_action(canvas, meta, action_int, normalize_mode="zero_one"):
    """Replace the last separator color in a canvas to simulate a different action.

    Args:
        canvas: Tensor [B, 3, H, W]
        meta: Dataset metadata
        action_int: Action integer (0=stay, 1=move+, 2=move-)
        normalize_mode: "zero_one" or "neg_one_one"

    Returns:
        Modified canvas tensor (clone)
    """
    modified = canvas.clone()
    x_start, x_end = get_last_separator_coords(meta)
    color = _separator_color_for_action(action_int)

    # Convert RGB [0,255] to tensor values
    color_tensor = torch.tensor(color, dtype=canvas.dtype, device=canvas.device) / 255.0
    if normalize_mode == "neg_one_one":
        color_tensor = color_tensor * 2.0 - 1.0

    # Apply to all rows and the separator column range
    for c in range(3):
        modified[:, c, :, x_start:x_end] = color_tensor[c]

    return modified


# ---------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred, target):
    """Compute PSNR between prediction and target tensors in [0,1].

    Args:
        pred, target: Tensors [B, 3, H, W] in [0, 1]

    Returns:
        Mean PSNR in dB across batch
    """
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))
    return psnr.mean().item()


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM between prediction and target tensors in [0,1].

    Simplified per-channel SSIM averaged across channels and batch.

    Args:
        pred, target: Tensors [B, 3, H, W] in [0, 1]

    Returns:
        Mean SSIM across batch
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = g / g.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(3, 1, window_size, window_size)

    pad = window_size // 2

    mu_pred = F.conv2d(pred, kernel_2d, padding=pad, groups=3)
    mu_target = F.conv2d(target, kernel_2d, padding=pad, groups=3)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, kernel_2d, padding=pad, groups=3) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, kernel_2d, padding=pad, groups=3) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, kernel_2d, padding=pad, groups=3) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Model loading and inference (reused from inference.py patterns)
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(model_type, checkpoint_path, device):
    """Load model and saved args from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", ckpt.get("extra", {}).get("args", {}))

    dataset_path = saved_args["dataset"]
    meta_path = Path(dataset_path) / "dataset_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    canvas_h, canvas_w = meta["canvas_size"]
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
def run_inference(model, model_type, canvas, patch_mask, patch_size, grid_h, grid_w,
                  saved_args=None, noise_scheduler=None):
    """Run inference for any model type. Returns predicted canvas tensor."""
    B = canvas.shape[0]
    batch_mask = patch_mask.expand(B, -1)

    if model_type == "mae":
        pred_patches, _ = model(canvas, batch_mask)
        target_patches = patchify(canvas, patch_size)
        recon_patches = target_patches.clone()
        recon_patches[batch_mask] = pred_patches[batch_mask]
        return unpatchify(recon_patches, patch_size, grid_h, grid_w)

    elif model_type == "gpt":
        gen_input_patches = patchify(canvas, patch_size).clone()
        gen_input_patches[batch_mask] = 0.0
        gen_input = unpatchify(gen_input_patches, patch_size, grid_h, grid_w)
        return model.generate(gen_input, patch_mask)

    elif model_type == "diffusion":
        target_patches = patchify(canvas, patch_size)
        noisy_patches = target_patches.clone()
        noisy_patches[batch_mask] = torch.randn_like(noisy_patches[batch_mask])
        current = unpatchify(noisy_patches, patch_size, grid_h, grid_w)

        num_inference_steps = 50
        step_size = noise_scheduler.num_train_timesteps // num_inference_steps
        timesteps = list(range(noise_scheduler.num_train_timesteps - 1, -1, -step_size))

        for t in timesteps:
            t_batch = torch.full((B,), t, device=canvas.device, dtype=torch.long)
            pred = model(current, t_batch)

            current_patches = patchify(current, patch_size)
            denoised_patches = current_patches.clone()
            pred_last_frame = pred[batch_mask]
            current_last_frame = current_patches[batch_mask]
            stepped = noise_scheduler.step(pred_last_frame, t, current_last_frame)
            denoised_patches[batch_mask] = stepped
            current = unpatchify(denoised_patches, patch_size, grid_h, grid_w)

        return current


def to_display_range(tensor, normalize_mode):
    """Convert tensor to [0,1] for display."""
    if normalize_mode == "neg_one_one":
        return tensor.clamp(-1, 1) * 0.5 + 0.5
    return tensor.clamp(0, 1)


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_all_metrics(model, model_type, val_loader, patch_mask, meta,
                        device, patch_size, grid_h, grid_w, saved_args,
                        noise_scheduler=None):
    """Compute all numeric metrics over the full validation set."""
    normalize_mode = "neg_one_one" if model_type == "diffusion" else "zero_one"

    # Accumulators
    total_mse = 0.0
    total_visual_mse = 0.0
    total_motor_mse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    action_mse = {}  # action_int -> [sum, count]
    static_mse_sum = 0.0
    dynamic_mse_sum = 0.0
    static_count = 0
    dynamic_count = 0
    num_samples = 0

    # Motor metrics
    motor_pos_errors = []  # list of [num_joints] arrays
    motor_vel_errors = []
    motor_pos_errors_by_action = {}  # action_int -> list of errors
    motor_direction_correct = 0
    motor_direction_total = 0
    motor_consistency_errors = []

    has_motor = meta.get("motor_strip_height", 0) > 0
    has_actions = False

    # GPT-specific: per-position loss
    gpt_position_losses = None
    gpt_tf_loss = 0.0
    gpt_fr_loss = 0.0
    gpt_tf_count = 0

    # Diffusion-specific: loss by timestep bucket
    diff_bucket_losses = {0: [], 1: [], 2: [], 3: []}

    B_mask = patch_mask  # [1, num_patches]

    def get_last_action_per_sample(batch_actions, B):
        """Extract the last action int for each sample in the batch.

        DataLoader collates list-of-ints into list-of-tensors:
        [tensor([a0_b0, a0_b1, ...]), tensor([a1_b0, a1_b1, ...])]
        where outer index = separator position, inner = batch sample.
        The last separator's action is what conditioned the prediction.
        """
        if batch_actions is None:
            return None
        if isinstance(batch_actions, list) and len(batch_actions) > 0:
            last_sep = batch_actions[-1]  # tensor of shape [B]
            if isinstance(last_sep, torch.Tensor):
                return [int(last_sep[i].item()) for i in range(B)]
            # Fallback for non-tensor
            return [int(last_sep)] * B
        return None

    for batch in tqdm(val_loader, desc="Computing metrics"):
        canvas = batch["canvas"].to(device)
        B = canvas.shape[0]
        actions_per_sample = get_last_action_per_sample(batch.get("actions", None), B)
        batch_mask = B_mask.expand(B, -1)

        # Run inference
        pred = run_inference(model, model_type, canvas, patch_mask, patch_size,
                             grid_h, grid_w, saved_args, noise_scheduler)

        # Convert to display range for metrics
        gt_disp = to_display_range(canvas, normalize_mode)
        pred_disp = to_display_range(pred, normalize_mode)

        # --- Overall MSE on last frame ---
        gt_last = extract_last_frame_visual(gt_disp, meta)
        pred_last = extract_last_frame_visual(pred_disp, meta)

        # If the last frame region is empty (unlikely), skip
        if gt_last.numel() == 0:
            continue

        mse_val = F.mse_loss(pred_last, gt_last).item()
        total_mse += mse_val * B

        # --- Visual vs motor strip MSE ---
        total_visual_mse += F.mse_loss(pred_last, gt_last).item() * B

        if has_motor:
            gt_motor = extract_last_frame_motor_strip(gt_disp, meta)
            pred_motor = extract_last_frame_motor_strip(pred_disp, meta)
            if gt_motor is not None and gt_motor.numel() > 0:
                total_motor_mse += F.mse_loss(pred_motor, gt_motor).item() * B

        # --- SSIM / PSNR on visual region ---
        if gt_last.shape[2] >= 11 and gt_last.shape[3] >= 11:
            total_ssim += compute_ssim(pred_last, gt_last) * B
            total_psnr += compute_psnr(pred_last, gt_last) * B

        # --- Per-action MSE ---
        if actions_per_sample is not None:
            has_actions = True
            for i in range(B):
                last_action = actions_per_sample[i]
                sample_mse = F.mse_loss(pred_last[i], gt_last[i]).item()
                if last_action not in action_mse:
                    action_mse[last_action] = [0.0, 0]
                action_mse[last_action][0] += sample_mse
                action_mse[last_action][1] += 1

        # --- Static vs dynamic pixel MSE ---
        # Compare second-to-last frame with last frame to find dynamic pixels
        frame_h, frame_w = meta["frame_size"]
        sep_width = meta["separator_width"]
        total_frames = meta["canvas_history_size"]
        prev_frame_x = (total_frames - 2) * (frame_w + sep_width)
        last_frame_x = (total_frames - 1) * (frame_w + sep_width)

        prev_frame = gt_disp[:, :, :frame_h, prev_frame_x:prev_frame_x + frame_w]
        gt_last_frame = gt_disp[:, :, :frame_h, last_frame_x:last_frame_x + frame_w]
        pred_last_frame = pred_disp[:, :, :frame_h, last_frame_x:last_frame_x + frame_w]

        pixel_diff = (gt_last_frame - prev_frame).abs().mean(dim=1)  # [B, H, W]
        dynamic_mask = pixel_diff > 0.02  # threshold for "changed"
        static_mask = ~dynamic_mask

        for i in range(B):
            if dynamic_mask[i].sum() > 0:
                pred_pixels = pred_last_frame[i]  # [3, H, W]
                gt_pixels = gt_last_frame[i]
                err = (pred_pixels - gt_pixels) ** 2

                dyn_m = dynamic_mask[i].unsqueeze(0).expand(3, -1, -1)
                sta_m = static_mask[i].unsqueeze(0).expand(3, -1, -1)

                if dyn_m.sum() > 0:
                    dynamic_mse_sum += err[dyn_m].mean().item()
                    dynamic_count += 1
                if sta_m.sum() > 0:
                    static_mse_sum += err[sta_m].mean().item()
                    static_count += 1

        # --- Motor state metrics ---
        if has_motor:
            gt_motor_strip = extract_last_frame_motor_strip(gt_disp, meta)
            pred_motor_strip = extract_last_frame_motor_strip(pred_disp, meta)

            if gt_motor_strip is not None:
                gt_pos = decode_motor_positions(gt_motor_strip, meta)
                pred_pos = decode_motor_positions(pred_motor_strip, meta)
                gt_vel = decode_motor_velocities(gt_motor_strip, meta)
                pred_vel = decode_motor_velocities(pred_motor_strip, meta)

                if gt_pos is not None and pred_pos is not None:
                    pos_err = np.abs(pred_pos - gt_pos)  # [B, num_joints]
                    motor_pos_errors.append(pos_err)

                    if gt_vel is not None and pred_vel is not None:
                        vel_err = np.abs(pred_vel - gt_vel)
                        motor_vel_errors.append(vel_err)

                    # Per-action motor position error
                    if actions_per_sample is not None:
                        for i in range(B):
                            last_action = actions_per_sample[i]
                            if last_action not in motor_pos_errors_by_action:
                                motor_pos_errors_by_action[last_action] = []
                            motor_pos_errors_by_action[last_action].append(pos_err[i])

                    # Motor direction accuracy (for move actions)
                    # Get history frame's motor positions
                    prev_motor_strip = gt_disp[:, :, frame_h:, prev_frame_x:prev_frame_x + frame_w]
                    if prev_motor_strip.shape[2] > 0:
                        prev_pos = decode_motor_positions(prev_motor_strip, meta)
                        if prev_pos is not None:
                            gt_delta = gt_pos - prev_pos
                            pred_delta = pred_pos - prev_pos

                            if actions_per_sample is not None:
                                for i in range(B):
                                    last_action = actions_per_sample[i]
                                    # Only check direction for move actions
                                    if last_action in (1, 2):
                                        for j in range(gt_delta.shape[1]):
                                            if abs(gt_delta[i, j]) > 1e-6:
                                                motor_direction_total += 1
                                                if np.sign(pred_delta[i, j]) == np.sign(gt_delta[i, j]):
                                                    motor_direction_correct += 1

                            # Motor consistency: pred_vel ≈ pred_pos - prev_pos
                            if pred_vel is not None:
                                consistency_err = np.abs(pred_vel - pred_delta).mean(axis=1)
                                motor_consistency_errors.extend(consistency_err.tolist())

        # --- GPT-specific: per-position loss + teacher-forcing gap ---
        if model_type == "gpt":
            loss_mask = B_mask[:, 1:]  # shifted
            batch_loss_mask = loss_mask.expand(B, -1)

            # Teacher-forcing loss
            pred_tf = model(canvas)
            target_patches = patchify(canvas, patch_size)
            pred_shifted = pred_tf[:, :-1, :]
            target_shifted = target_patches[:, 1:, :]
            tf_loss = F.mse_loss(pred_shifted[batch_loss_mask], target_shifted[batch_loss_mask]).item()
            gpt_tf_loss += tf_loss * B
            gpt_tf_count += B

            # Per-position loss within last frame
            num_patches_total = target_patches.shape[1]
            if gpt_position_losses is None:
                gpt_position_losses = torch.zeros(num_patches_total - 1, device=device)

            for pos in range(num_patches_total - 1):
                if loss_mask[0, pos]:
                    pos_loss = F.mse_loss(pred_shifted[:, pos, :], target_shifted[:, pos, :]).item()
                    gpt_position_losses[pos] += pos_loss * B

            # Free-running loss (already computed via run_inference)
            pred_fr_patches = patchify(pred, patch_size)
            fr_loss = F.mse_loss(pred_fr_patches[batch_mask],
                                 target_patches[batch_mask]).item()
            gpt_fr_loss += fr_loss * B

        # --- Diffusion-specific: loss by timestep bucket ---
        if model_type == "diffusion":
            target_patches = patchify(canvas, patch_size)
            for bucket_idx, (t_low, t_high) in enumerate([(0, 250), (250, 500), (500, 750), (750, 1000)]):
                t_mid = (t_low + t_high) // 2
                timesteps = torch.full((B,), t_mid, device=device, dtype=torch.long)
                noise = torch.randn_like(target_patches)
                noisy = noise_scheduler.add_noise(target_patches, noise, timesteps)
                composite = torch.where(batch_mask.unsqueeze(-1), noisy, target_patches)
                noisy_canvas = unpatchify(composite, patch_size, grid_h, grid_w)
                pred_patches = model(noisy_canvas, timesteps)

                if saved_args.get("prediction_type", "epsilon") == "epsilon":
                    loss_target = noise
                else:
                    loss_target = target_patches

                bucket_loss = F.mse_loss(pred_patches[batch_mask], loss_target[batch_mask]).item()
                diff_bucket_losses[bucket_idx].append(bucket_loss)

        num_samples += B

    # --- Aggregate metrics ---
    results = {}
    results["model_type"] = model_type
    results["num_val_samples"] = num_samples
    results["val_mse"] = total_mse / max(num_samples, 1)
    results["val_mse_visual"] = total_visual_mse / max(num_samples, 1)
    results["ssim"] = total_ssim / max(num_samples, 1)
    results["psnr"] = total_psnr / max(num_samples, 1)

    if has_motor:
        results["val_mse_motor_strip"] = total_motor_mse / max(num_samples, 1)

    if has_actions:
        for action_int, (s, c) in action_mse.items():
            results[f"val_mse_action_{action_int}"] = s / max(c, 1)

    if static_count > 0:
        results["val_mse_static"] = static_mse_sum / static_count
    if dynamic_count > 0:
        results["val_mse_dynamic"] = dynamic_mse_sum / dynamic_count

    # Motor metrics
    if motor_pos_errors:
        all_pos_err = np.concatenate(motor_pos_errors, axis=0)
        results["motor_position_mae_per_joint"] = all_pos_err.mean(axis=0).tolist()
        results["motor_position_mae_mean"] = float(all_pos_err.mean())

    if motor_vel_errors:
        all_vel_err = np.concatenate(motor_vel_errors, axis=0)
        results["motor_velocity_mae_mean"] = float(all_vel_err.mean())

    if motor_pos_errors_by_action:
        results["motor_position_mae_by_action"] = {}
        for a, errs in motor_pos_errors_by_action.items():
            results["motor_position_mae_by_action"][str(a)] = float(np.mean(errs))

    if motor_direction_total > 0:
        results["motor_direction_accuracy"] = motor_direction_correct / motor_direction_total

    if motor_consistency_errors:
        results["motor_consistency_error"] = float(np.mean(motor_consistency_errors))

    # GPT-specific
    if model_type == "gpt" and gpt_tf_count > 0:
        results["gpt_teacher_forcing_loss"] = gpt_tf_loss / gpt_tf_count
        results["gpt_free_running_loss"] = gpt_fr_loss / gpt_tf_count
        results["gpt_tf_fr_gap"] = results["gpt_free_running_loss"] - results["gpt_teacher_forcing_loss"]

        if gpt_position_losses is not None:
            # Normalize and convert to list (only last-frame positions)
            norm_losses = gpt_position_losses / gpt_tf_count
            # Extract only last-frame positions
            loss_mask = B_mask[:, 1:]
            last_frame_positions = []
            for pos in range(loss_mask.shape[1]):
                if loss_mask[0, pos]:
                    last_frame_positions.append(norm_losses[pos].item())
            results["gpt_per_position_loss"] = last_frame_positions

    # Diffusion-specific
    if model_type == "diffusion":
        bucket_names = ["t_0_250", "t_250_500", "t_500_750", "t_750_1000"]
        for idx, name in enumerate(bucket_names):
            if diff_bucket_losses[idx]:
                results[f"diffusion_loss_{name}"] = float(np.mean(diff_bucket_losses[idx]))

    return results


@torch.no_grad()
def compute_action_discrimination(model, model_type, val_loader, patch_mask, meta,
                                  device, patch_size, grid_h, grid_w, saved_args,
                                  noise_scheduler=None, num_samples=16):
    """Compute action and motor discrimination scores."""
    normalize_mode = "neg_one_one" if model_type == "diffusion" else "zero_one"
    has_motor = meta.get("motor_strip_height", 0) > 0

    action_dists = []
    motor_dists = []

    batch = next(iter(val_loader))
    canvas = batch["canvas"][:num_samples].to(device)
    B = canvas.shape[0]

    preds_by_action = {}
    for action_int in range(3):
        modified = replace_last_separator_action(canvas, meta, action_int, normalize_mode)
        pred = run_inference(model, model_type, modified, patch_mask, patch_size,
                             grid_h, grid_w, saved_args, noise_scheduler)
        pred_disp = to_display_range(pred, normalize_mode)
        preds_by_action[action_int] = pred_disp

    # Compute pairwise distances on last frame visual region
    for i in range(3):
        for j in range(i + 1, 3):
            pred_i = extract_last_frame_visual(preds_by_action[i], meta)
            pred_j = extract_last_frame_visual(preds_by_action[j], meta)
            dist = (pred_i - pred_j).pow(2).mean(dim=(1, 2, 3))
            action_dists.extend(dist.cpu().tolist())

            if has_motor:
                motor_i = extract_last_frame_motor_strip(preds_by_action[i], meta)
                motor_j = extract_last_frame_motor_strip(preds_by_action[j], meta)
                if motor_i is not None and motor_j is not None:
                    m_dist = (motor_i - motor_j).pow(2).mean(dim=(1, 2, 3))
                    motor_dists.extend(m_dist.cpu().tolist())

    results = {
        "action_discrimination_score": float(np.mean(action_dists)) if action_dists else 0.0,
    }
    if motor_dists:
        results["motor_discrimination_score"] = float(np.mean(motor_dists))

    return results, preds_by_action


# ---------------------------------------------------------------------------
# Counterfactual visualizations
# ---------------------------------------------------------------------------

def tensor_to_pil(tensor):
    """Convert a [3, H, W] tensor in [0,1] to PIL Image."""
    from PIL import Image
    arr = (tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)  # [H, W, 3]
    return Image.fromarray(arr)


def pil_to_base64(img):
    """Convert PIL Image to base64 string for HTML embedding."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@torch.no_grad()
def generate_counterfactual_images(model, model_type, val_loader, patch_mask, meta,
                                   device, patch_size, grid_h, grid_w, saved_args,
                                   noise_scheduler=None, num_samples=8):
    """Generate counterfactual action grid images.

    Returns:
        List of dicts, each with 'gt_canvas', 'preds' (dict action->canvas),
        'motor_decoded' (dict with GT and per-action decoded values).
    """
    normalize_mode = "neg_one_one" if model_type == "diffusion" else "zero_one"
    has_motor = meta.get("motor_strip_height", 0) > 0

    batch = next(iter(val_loader))
    canvas = batch["canvas"][:num_samples].to(device)
    B = canvas.shape[0]

    gt_disp = to_display_range(canvas, normalize_mode)

    results = []
    action_names = {0: "STAY (red)", 1: "MOVE+ (green)", 2: "MOVE- (blue)"}

    # Run inference for each action
    preds_disp = {}
    for action_int in range(3):
        modified = replace_last_separator_action(canvas, meta, action_int, normalize_mode)
        pred = run_inference(model, model_type, modified, patch_mask, patch_size,
                             grid_h, grid_w, saved_args, noise_scheduler)
        preds_disp[action_int] = to_display_range(pred, normalize_mode)

    for i in range(B):
        sample = {"gt_canvas": gt_disp[i], "preds": {}, "motor_decoded": {}}

        for action_int in range(3):
            sample["preds"][action_int] = preds_disp[action_int][i]

        # Decode motor values
        if has_motor:
            gt_motor_strip = extract_last_frame_motor_strip(gt_disp[i:i+1], meta)
            if gt_motor_strip is not None:
                gt_pos = decode_motor_positions(gt_motor_strip, meta)
                gt_vel = decode_motor_velocities(gt_motor_strip, meta)
                sample["motor_decoded"]["gt_pos"] = gt_pos[0] if gt_pos is not None else None
                sample["motor_decoded"]["gt_vel"] = gt_vel[0] if gt_vel is not None else None

                for action_int in range(3):
                    pred_strip = extract_last_frame_motor_strip(preds_disp[action_int][i:i+1], meta)
                    if pred_strip is not None:
                        pred_pos = decode_motor_positions(pred_strip, meta)
                        pred_vel = decode_motor_velocities(pred_strip, meta)
                        sample["motor_decoded"][f"pred_pos_{action_int}"] = pred_pos[0] if pred_pos is not None else None
                        sample["motor_decoded"][f"pred_vel_{action_int}"] = pred_vel[0] if pred_vel is not None else None

        results.append(sample)

    return results


def save_counterfactual_images(counterfactual_results, output_dir, meta):
    """Save counterfactual grid images and error heatmaps."""
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_names = {0: "STAY", 1: "MOVE+", 2: "MOVE-"}
    saved_images = []

    for idx, sample in enumerate(counterfactual_results):
        gt = sample["gt_canvas"]  # [3, H, W]
        H, W = gt.shape[1], gt.shape[2]

        # Build grid: GT on top, then each action prediction
        rows = [tensor_to_pil(gt)]
        for action_int in range(3):
            rows.append(tensor_to_pil(sample["preds"][action_int]))

        grid_h_px = H * 4
        grid = Image.new("RGB", (W, grid_h_px))
        for r, row_img in enumerate(rows):
            grid.paste(row_img, (0, r * H))

        grid_path = output_dir / f"counterfactual_{idx:03d}.png"
        grid.save(grid_path)

        # Error heatmap (GT vs pred with original action)
        gt_last = extract_last_frame_visual(gt.unsqueeze(0), meta)[0]  # [3, fH, fW]
        # Use action 0 pred as default for error heatmap
        pred_last = extract_last_frame_visual(sample["preds"][0].unsqueeze(0), meta)[0]
        error = (gt_last - pred_last).pow(2).mean(dim=0).cpu().numpy()  # [fH, fW]

        # Normalize error for colormap
        if error.size == 0:
            continue
        error_norm = error / max(error.max(), 1e-8)
        heatmap_rgba = cm.jet(error_norm)[:, :, :3]  # [fH, fW, 3]
        heatmap_uint8 = (heatmap_rgba * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_uint8)

        heatmap_path = output_dir / f"error_heatmap_{idx:03d}.png"
        heatmap_img.save(heatmap_path)

        # Motor decoded table
        motor_decoded = sample.get("motor_decoded", {})
        if motor_decoded.get("gt_pos") is not None:
            lines = ["Joint | GT Pos | STAY Pos | MOVE+ Pos | MOVE- Pos | GT Vel"]
            gt_pos = motor_decoded["gt_pos"]
            gt_vel = motor_decoded.get("gt_vel")
            for j in range(len(gt_pos)):
                vals = [f"J{j}", f"{gt_pos[j]:.4f}"]
                for a in range(3):
                    p = motor_decoded.get(f"pred_pos_{a}")
                    vals.append(f"{p[j]:.4f}" if p is not None else "N/A")
                vals.append(f"{gt_vel[j]:.4f}" if gt_vel is not None else "N/A")
                lines.append(" | ".join(vals))

            table_path = output_dir / f"motor_decoded_{idx:03d}.txt"
            table_path.write_text("\n".join(lines))

        saved_images.append({
            "grid": grid_path,
            "heatmap": heatmap_path,
            "motor_decoded": motor_decoded,
        })

    return saved_images


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html_report(metrics, counterfactual_results, saved_images, output_dir,
                         meta, model_type, checkpoint_path, dataset_path):
    """Generate a self-contained HTML evaluation report."""
    from PIL import Image

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    html_path = docs_dir / f"eval_{model_type}_{timestamp}.html"

    # Color coding thresholds
    def metric_color(name, value):
        if value is None:
            return "#888"
        if "ssim" in name.lower():
            if value > 0.9: return "#2d7d2d"
            if value > 0.7: return "#b8860b"
            return "#c0392b"
        if "psnr" in name.lower():
            if value > 30: return "#2d7d2d"
            if value > 20: return "#b8860b"
            return "#c0392b"
        if "discrimination" in name.lower():
            if value > 0.01: return "#2d7d2d"
            if value > 0.001: return "#b8860b"
            return "#c0392b"
        if "direction_accuracy" in name.lower():
            if value > 0.8: return "#2d7d2d"
            if value > 0.6: return "#b8860b"
            return "#c0392b"
        return "#333"

    # Generate recommendations
    recommendations = []
    if metrics.get("action_discrimination_score", 1) < 0.001:
        recommendations.append("Model may be ignoring actions entirely. Consider increasing separator width, adding learnable action embeddings, or verifying action diversity in training data.")
    if metrics.get("motor_direction_accuracy", 1) < 0.6:
        recommendations.append("Motor direction accuracy is low. The model may not be learning action-to-motor mappings. Try larger separator width or explicit action token embeddings.")
    if metrics.get("motor_consistency_error", 0) > 0.1:
        recommendations.append("Motor position and velocity predictions are inconsistent. Consider adding a consistency loss or simplifying the velocity encoding.")

    visual_mse = metrics.get("val_mse_visual", 0)
    motor_mse = metrics.get("val_mse_motor_strip", 0)
    if motor_mse > visual_mse * 2 and motor_mse > 0:
        recommendations.append("Motor strip MSE is much higher than visual MSE. Consider increasing motor strip height or applying higher loss weight to motor patches.")
    if visual_mse > motor_mse * 2 and visual_mse > 0:
        recommendations.append("Visual MSE is much higher than motor strip MSE. Model capacity may be insufficient for visual prediction.")

    if metrics.get("gpt_tf_fr_gap", 0) > 0.01:
        recommendations.append("Large teacher-forcing vs free-running gap detected (exposure bias). Consider scheduled sampling or a bidirectional refinement pass.")

    diff_low_t = metrics.get("diffusion_loss_t_0_250", 0)
    diff_high_t = metrics.get("diffusion_loss_t_750_1000", 0)
    if diff_low_t > diff_high_t * 2 and diff_low_t > 0:
        recommendations.append("Diffusion model struggles with fine details (low-t loss is high). Consider more decoder capacity or v-prediction parameterization.")

    # Build HTML
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Report: {model_type} - {timestamp}</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.metric-val {{ font-weight: bold; font-family: monospace; font-size: 1.1em; }}
.recommendation {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin: 10px 0; border-radius: 4px; }}
.sample-grid {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 15px 0; }}
.sample-card {{ background: white; padding: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.sample-card img {{ max-width: 100%; height: auto; }}
.motor-table {{ font-size: 0.9em; }}
.motor-table td {{ padding: 4px 8px; font-family: monospace; }}
.header-info {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>Evaluation Report: {model_type.upper()}</h1>
<div class="header-info">
<p><strong>Model Type:</strong> {model_type}</p>
<p><strong>Checkpoint:</strong> {checkpoint_path}</p>
<p><strong>Dataset:</strong> {dataset_path}</p>
<p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<p><strong>Val Samples:</strong> {metrics.get('num_val_samples', 'N/A')}</p>
</div>
""")

    # Metrics table
    html_parts.append("<h2>Metrics</h2>\n<table>\n<tr><th>Metric</th><th>Value</th></tr>\n")
    skip_keys = {"model_type", "num_val_samples", "gpt_per_position_loss",
                 "motor_position_mae_per_joint", "motor_position_mae_by_action"}
    for key, value in metrics.items():
        if key in skip_keys:
            continue
        if isinstance(value, float):
            color = metric_color(key, value)
            html_parts.append(f'<tr><td>{key}</td><td class="metric-val" style="color:{color}">{value:.6f}</td></tr>\n')
        elif isinstance(value, (int, str)):
            html_parts.append(f'<tr><td>{key}</td><td class="metric-val">{value}</td></tr>\n')

    # Per-joint motor position MAE
    per_joint = metrics.get("motor_position_mae_per_joint")
    if per_joint:
        html_parts.append(f'<tr><td>motor_position_mae_per_joint</td><td class="metric-val">[{", ".join(f"{v:.4f}" for v in per_joint)}]</td></tr>\n')

    # Per-action motor MAE
    by_action = metrics.get("motor_position_mae_by_action")
    if by_action:
        for a, v in by_action.items():
            html_parts.append(f'<tr><td>motor_position_mae_action_{a}</td><td class="metric-val">{v:.6f}</td></tr>\n')

    html_parts.append("</table>\n")

    # Recommendations
    if recommendations:
        html_parts.append("<h2>Recommendations</h2>\n")
        for rec in recommendations:
            html_parts.append(f'<div class="recommendation">{rec}</div>\n')

    # Counterfactual images
    if saved_images:
        html_parts.append("<h2>Counterfactual Action Grids</h2>\n")
        html_parts.append('<p>Each grid: Row 1 = GT, Row 2 = STAY (red), Row 3 = MOVE+ (green), Row 4 = MOVE- (blue)</p>\n')
        html_parts.append('<div class="sample-grid">\n')

        for idx, img_info in enumerate(saved_images):
            grid_path = img_info["grid"]
            heatmap_path = img_info["heatmap"]

            # Embed images as base64
            if grid_path.exists():
                grid_b64 = pil_to_base64(Image.open(grid_path))
                html_parts.append(f'<div class="sample-card">')
                html_parts.append(f'<h4>Sample {idx}</h4>')
                html_parts.append(f'<img src="data:image/png;base64,{grid_b64}" alt="Counterfactual grid {idx}">')

                if heatmap_path.exists():
                    hm_b64 = pil_to_base64(Image.open(heatmap_path))
                    html_parts.append(f'<br><img src="data:image/png;base64,{hm_b64}" alt="Error heatmap {idx}">')
                    html_parts.append(f'<p><em>Error heatmap (jet colormap)</em></p>')

                # Motor decoded table
                motor = img_info.get("motor_decoded", {})
                gt_pos = motor.get("gt_pos")
                if gt_pos is not None:
                    html_parts.append('<table class="motor-table">')
                    html_parts.append('<tr><th>Joint</th><th>GT Pos</th><th>STAY</th><th>MOVE+</th><th>MOVE-</th></tr>')
                    for j in range(len(gt_pos)):
                        row = f"<tr><td>J{j}</td><td>{gt_pos[j]:.4f}</td>"
                        for a in range(3):
                            p = motor.get(f"pred_pos_{a}")
                            row += f"<td>{p[j]:.4f}</td>" if p is not None else "<td>N/A</td>"
                        row += "</tr>"
                        html_parts.append(row)
                    html_parts.append("</table>")

                html_parts.append("</div>\n")

        html_parts.append("</div>\n")

    # GPT per-position loss plot (as text table)
    gpt_pos_loss = metrics.get("gpt_per_position_loss")
    if gpt_pos_loss:
        html_parts.append("<h2>GPT Per-Position Loss (last frame, raster order)</h2>\n")
        html_parts.append("<table><tr><th>Position</th><th>MSE</th></tr>\n")
        for i, val in enumerate(gpt_pos_loss):
            html_parts.append(f"<tr><td>{i}</td><td>{val:.6f}</td></tr>\n")
        html_parts.append("</table>\n")

    # Diffusion timestep buckets
    if model_type == "diffusion":
        html_parts.append("<h2>Diffusion Loss by Timestep Bucket</h2>\n")
        html_parts.append("<table><tr><th>Bucket</th><th>Loss</th></tr>\n")
        for name in ["t_0_250", "t_250_500", "t_500_750", "t_750_1000"]:
            val = metrics.get(f"diffusion_loss_{name}", "N/A")
            if isinstance(val, float):
                html_parts.append(f"<tr><td>{name}</td><td>{val:.6f}</td></tr>\n")
        html_parts.append("</table>\n")

    html_parts.append("</body></html>")

    html_content = "".join(html_parts)
    html_path.write_text(html_content, encoding="utf-8")
    print(f"HTML report saved to: {html_path}")

    # Update index.html
    index_path = docs_dir / "index.html"
    if index_path.exists():
        existing = index_path.read_text(encoding="utf-8")
    else:
        existing = """<!DOCTYPE html>
<html><head><title>Evaluation Reports</title>
<style>body{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px;}
a{color:#3498db;}</style></head><body>
<h1>Evaluation Reports</h1>
<ul id="reports">
</ul></body></html>"""

    # Insert new link before </ul>
    link = f'<li><a href="{html_path.name}">{model_type} - {timestamp}</a></li>\n'
    existing = existing.replace("</ul>", link + "</ul>")
    index_path.write_text(existing, encoding="utf-8")

    return html_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained canvas world model")
    p.add_argument("--model-type", type=str, required=True, choices=["mae", "gpt", "diffusion"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output dir for images/report (default: local/eval/{model_type})")
    p.add_argument("--num-counterfactual", type=int, default=8,
                   help="Number of counterfactual samples to generate")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    return p.parse_args()


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
    patch_size = saved_args["patch_size"]
    grid_h = canvas_h // patch_size
    grid_w = canvas_w // patch_size
    num_frames = meta["canvas_history_size"]
    sep_width = meta["separator_width"]

    # Load validation data
    normalize_mode = "neg_one_one" if args.model_type == "diffusion" else "zero_one"
    val_dataset = CanvasDataset(
        args.dataset, split="val", val_ratio=0.1,
        normalize_mode=normalize_mode, seed=args.seed,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"Val samples: {len(val_dataset)}")

    # Patch mask
    patch_mask = compute_last_frame_patch_mask(
        canvas_h, canvas_w, patch_size, num_frames, sep_width, device=device,
    )

    # Noise scheduler for diffusion
    noise_scheduler = None
    if args.model_type == "diffusion":
        noise_scheduler = NoiseScheduler(
            num_train_timesteps=saved_args["num_train_timesteps"],
            beta_schedule=saved_args["beta_schedule"],
            prediction_type=saved_args["prediction_type"],
        )

    # Output directory
    output_dir = args.output_dir or f"local/eval/{args.model_type}"

    # --- Compute all metrics ---
    print("\nComputing metrics...")
    metrics = compute_all_metrics(
        model, args.model_type, val_loader, patch_mask, meta, device,
        patch_size, grid_h, grid_w, saved_args, noise_scheduler,
    )

    # --- Action discrimination ---
    print("Computing action discrimination...")
    disc_results, _ = compute_action_discrimination(
        model, args.model_type, val_loader, patch_mask, meta, device,
        patch_size, grid_h, grid_w, saved_args, noise_scheduler,
    )
    metrics.update(disc_results)

    # --- Save JSON report ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "report.json"

    # Convert any numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    serializable_metrics = make_serializable(metrics)
    with open(report_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"\nMetrics report saved to: {report_path}")

    # Print metrics table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in serializable_metrics.items():
        if isinstance(value, float):
            print(f"  {key:40s} {value:.6f}")
        elif isinstance(value, list) and len(value) < 10:
            print(f"  {key:40s} [{', '.join(f'{v:.4f}' for v in value)}]")
        elif isinstance(value, dict):
            print(f"  {key:40s} {value}")
        else:
            print(f"  {key:40s} {value}")
    print("=" * 60)

    # --- Counterfactual visualizations ---
    print("\nGenerating counterfactual visualizations...")
    counterfactual_results = generate_counterfactual_images(
        model, args.model_type, val_loader, patch_mask, meta, device,
        patch_size, grid_h, grid_w, saved_args, noise_scheduler,
        num_samples=args.num_counterfactual,
    )
    saved_images = save_counterfactual_images(counterfactual_results, output_dir, meta)
    print(f"Saved {len(saved_images)} counterfactual grids + error heatmaps to: {output_dir}")

    # --- HTML report ---
    if not args.no_html:
        print("\nGenerating HTML report...")
        html_path = generate_html_report(
            serializable_metrics, counterfactual_results, saved_images,
            output_dir, meta, args.model_type, args.checkpoint, args.dataset,
        )
        print(f"HTML report: {html_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
