"""Canvas building from interleaved frame and action sequences."""

import numpy as np
from PIL import Image


def _separator_color_for_action(action_dict: dict) -> tuple:
    """Get RGB color for an action separator.

    Args:
        action_dict: Action dictionary with 'action' int key, or plain int.
            0->Red, 1->Green, 2->Blue.

    Returns:
        (R, G, B) tuple with values 0-255
    """
    # Handle plain int action
    if isinstance(action_dict, int):
        action_int = action_dict
    elif isinstance(action_dict, dict):
        action_int = action_dict.get('action', 0)
    else:
        action_int = 0

    # Standard action colors
    if action_int == 0:
        return (255, 0, 0)    # Red: stay
    elif action_int == 1:
        return (0, 255, 0)    # Green: move positive
    elif action_int == 2:
        return (0, 0, 255)    # Blue: move negative
    else:
        return (255, 255, 0)  # Yellow: unknown


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image array to uint8, clipping float values to [0, 1].

    Args:
        img: Numpy array (any dtype)

    Returns:
        uint8 array clipped to [0, 255]
    """
    if img.dtype == np.uint8:
        return img

    # Clip float arrays to [0, 1] and scale
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    # For other dtypes, assume already in [0, 255] range
    return np.clip(img, 0, 255).astype(np.uint8)


def _ensure_hw(img: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image to target (H, W) using PIL LANCZOS.

    Args:
        img: Numpy RGB image array (any shape)
        size: Target (H, W) tuple

    Returns:
        Resized numpy array with shape (H, W, 3), dtype uint8
    """
    img = _to_uint8(img)
    pil_img = Image.fromarray(img)
    target_h, target_w = size
    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil_img, dtype=np.uint8)


def _render_motor_strip(
    motor_state: np.ndarray,
    strip_height: int,
    frame_width: int,
    norm_min: np.ndarray,
    norm_max: np.ndarray,
    patch_size: int = 16,
    motor_velocity: np.ndarray = None,
    vel_norm_max: np.ndarray = None,
) -> np.ndarray:
    """Render motor positions and velocities as a strip of grayscale patches.

    Each joint gets one patch (patch_size x patch_size) for position, then
    one patch for velocity. Positions: black=min, white=max. Velocities:
    mid-gray=zero, black=max negative, white=max positive.

    Layout (for 6 joints, 14 patches wide):
        [pos0][pos1][pos2][pos3][pos4][pos5][vel0][vel1][vel2][vel3][vel4][vel5][--][--]

    Args:
        motor_state: 1D array of joint positions, shape (num_joints,)
        strip_height: Height of the strip in pixels (should equal patch_size)
        frame_width: Width to match the frame above
        norm_min: Per-joint minimum values for position normalization
        norm_max: Per-joint maximum values for position normalization
        patch_size: ViT patch size in pixels (default 16)
        motor_velocity: Optional 1D array of joint velocities (position deltas).
            None on first frame — rendered as mid-gray (zero velocity).
        vel_norm_max: Per-joint max absolute velocity for normalization

    Returns:
        uint8 array of shape (strip_height, frame_width, 3)
    """
    num_joints = len(motor_state)
    strip = np.zeros((strip_height, frame_width, 3), dtype=np.uint8)

    # Normalize positions to [0, 1]
    pos_range = norm_max - norm_min
    pos_range = np.where(pos_range < 1e-8, 1.0, pos_range)
    norm_pos = np.clip((motor_state - norm_min) / pos_range, 0.0, 1.0)

    # Fill position patches
    for j in range(num_joints):
        x_start = j * patch_size
        x_end = x_start + patch_size
        if x_end > frame_width:
            break
        gray_val = int(norm_pos[j] * 255)
        strip[:, x_start:x_end] = gray_val

    # Fill velocity patches
    for j in range(num_joints):
        x_start = (num_joints + j) * patch_size
        x_end = x_start + patch_size
        if x_end > frame_width:
            break

        if motor_velocity is None:
            # First frame: no velocity data, encode as zero (mid-gray)
            gray_val = 128
        else:
            # Normalize velocity to [-1, 1] then map to [0, 255]
            if vel_norm_max is not None and vel_norm_max[j] > 1e-8:
                norm_vel = np.clip(motor_velocity[j] / vel_norm_max[j], -1.0, 1.0)
            else:
                norm_vel = 0.0
            gray_val = int((norm_vel * 0.5 + 0.5) * 255)

        strip[:, x_start:x_end] = gray_val

    return strip


def build_canvas(
    interleaved_history: list,
    frame_size: tuple,
    sep_width: int,
    bg_color: tuple = (0, 0, 0),
    motor_positions: list = None,
    motor_strip_height: int = 0,
    motor_norm_min: np.ndarray = None,
    motor_norm_max: np.ndarray = None,
    motor_vel_norm_max: np.ndarray = None,
    patch_size: int = 16,
) -> np.ndarray:
    """Build a canvas by horizontally concatenating frames with action separators.

    Interleaved history must alternate [frame, action, frame, action, ..., frame].

    Args:
        interleaved_history: List of alternating frames and action dicts
            - Frames are numpy arrays (any dtype/shape)
            - Actions are dicts with action info
            - Must start with a frame, can end with frame or action
        frame_size: Target (H, W) tuple for each frame
        sep_width: Width in pixels of action separator
        bg_color: Background color (R, G, B) for separators, default black
        motor_positions: Optional list of motor state arrays, one per frame.
            Each is a 1D numpy array of joint positions.
        motor_strip_height: Height of motor position strip below each frame (0 to disable)
        motor_norm_min: Per-joint min values for position normalization
        motor_norm_max: Per-joint max values for position normalization
        motor_vel_norm_max: Per-joint max absolute velocity for normalization
        patch_size: ViT patch size in pixels (default 16)

    Returns:
        Canvas as numpy uint8 array with shape (H_total, W_total, 3)
        where H_total = frame_H + motor_strip_height
        and W_total = num_frames * W + num_separators * sep_width
    """
    target_h, target_w = frame_size

    # Separate frames and actions
    frames = []
    actions = []

    for i, item in enumerate(interleaved_history):
        if i % 2 == 0:
            # Even index: frame
            frames.append(item)
        else:
            # Odd index: action
            actions.append(item)

    # Resize all frames to target size
    resized_frames = [_ensure_hw(f, frame_size) for f in frames]

    # Determine if we should render motor strips
    render_motor = (
        motor_positions is not None
        and motor_strip_height > 0
        and motor_norm_min is not None
        and motor_norm_max is not None
    )
    strip_h = motor_strip_height if render_motor else 0
    total_h = target_h + strip_h

    # Compute per-frame velocities (position deltas)
    velocities = []
    if render_motor and motor_positions is not None:
        for i in range(len(motor_positions)):
            if i == 0 or motor_positions[i] is None or motor_positions[i - 1] is None:
                velocities.append(None)  # first frame or missing data
            else:
                velocities.append(motor_positions[i] - motor_positions[i - 1])

    # Calculate total canvas width
    num_frames = len(resized_frames)
    num_seps = num_frames - 1
    total_w = num_frames * target_w + num_seps * sep_width

    # Create canvas
    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    # Place frames, motor strips, and separators
    x_offset = 0

    for frame_idx in range(num_frames):
        # Place frame in top portion
        canvas[:target_h, x_offset:x_offset + target_w] = resized_frames[frame_idx]

        # Place motor strip below frame
        if render_motor and frame_idx < len(motor_positions) and motor_positions[frame_idx] is not None:
            vel = velocities[frame_idx] if frame_idx < len(velocities) else None
            strip = _render_motor_strip(
                motor_positions[frame_idx],
                strip_h,
                target_w,
                motor_norm_min,
                motor_norm_max,
                patch_size=patch_size,
                motor_velocity=vel,
                vel_norm_max=motor_vel_norm_max,
            )
            canvas[target_h:, x_offset:x_offset + target_w] = strip

        x_offset += target_w

        # Place separator (except after last frame) — spans full height
        if frame_idx < num_seps:
            sep_color = _separator_color_for_action(actions[frame_idx])
            canvas[:, x_offset:x_offset + sep_width] = sep_color
            x_offset += sep_width

    return canvas
