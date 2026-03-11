"""Canvas building from interleaved frame and action sequences."""

import numpy as np
from PIL import Image, ImageDraw


def _separator_color_for_action(action_dict: dict) -> tuple:
    """Get RGB color for an action separator.

    Handles multiple action formats:
    - Dict with 'action' int key: 0->Red, 1->Green, 2->Blue
    - Dict with 'motor_right' key: Red to Green linear blend (0.0 to 0.12)
    - Plain int: treated as action int

    Args:
        action_dict: Action dictionary or int

    Returns:
        (R, G, B) tuple with values 0-255
    """
    # Handle plain int action
    if isinstance(action_dict, int):
        action_int = action_dict
    elif isinstance(action_dict, dict):
        # Try 'action' key first
        if 'action' in action_dict:
            action_int = action_dict['action']
        # Try motor-based format
        elif 'motor_right' in action_dict:
            motor_val = action_dict['motor_right']
            # Linear blend: 0.0 -> Red (255,0,0), 0.12 -> Green (0,255,0)
            t = min(1.0, max(0.0, motor_val / 0.12))
            r = int(255 * (1 - t))
            g = int(255 * t)
            return (r, g, 0)
        else:
            action_int = 0  # Default to stay
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


def build_canvas(
    interleaved_history: list,
    frame_size: tuple,
    sep_width: int,
    bg_color: tuple = (0, 0, 0),
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

    Returns:
        Canvas as numpy uint8 array with shape (H, W_total, 3)
        where W_total = num_frames * W + num_separators * sep_width
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

    # Calculate total canvas width
    num_frames = len(resized_frames)
    num_seps = num_frames - 1
    total_w = num_frames * target_w + num_seps * sep_width

    # Create canvas
    canvas = np.full((target_h, total_w, 3), bg_color, dtype=np.uint8)

    # Place frames and separators
    x_offset = 0

    for frame_idx in range(num_frames):
        # Place frame
        canvas[:, x_offset:x_offset + target_w] = resized_frames[frame_idx]
        x_offset += target_w

        # Place separator (except after last frame)
        if frame_idx < num_seps:
            sep_color = _separator_color_for_action(actions[frame_idx])
            canvas[:, x_offset:x_offset + sep_width] = sep_color
            x_offset += sep_width

    return canvas


def build_canvas_for_frame(
    observations: list,
    actions: list,
    frame_idx: int,
    sep_width: int,
    history_size: int,
) -> np.ndarray | None:
    """Convenience wrapper to build a canvas for a single frame.

    Assembles the interleaved frame/action list and calls build_canvas().
    Automatically detects frame size from the first available image.

    Args:
        observations: List of observation dicts from extract_observations()
        actions: List of action dicts from extract_actions()
        frame_idx: Index of the target frame (final frame in the canvas)
        sep_width: Width of action separators (SEPARATOR_WIDTH)
        history_size: Number of frames to include (CANVAS_HISTORY_SIZE)

    Returns:
        Canvas as numpy uint8 array, or None if insufficient history available
    """
    from .session_loader import load_frame_image

    # Check if we have enough history
    start_idx = frame_idx - (history_size - 1)
    if start_idx < 0:
        return None

    # Load frames
    frames = []
    for i in range(start_idx, frame_idx + 1):
        if i >= len(observations):
            return None
        frame_img = load_frame_image(observations[i]['full_path'])
        frames.append(np.array(frame_img))

    # Detect frame size from first frame
    frame_size = (frames[0].shape[0], frames[0].shape[1])

    # Get actions between frames
    frame_actions = []
    for i in range(start_idx, frame_idx):
        if i < len(actions):
            frame_actions.append(actions[i]['action'])
        else:
            # Default to stay action if not available
            frame_actions.append({'action': 0})

    # Build interleaved list
    interleaved = []
    for i in range(len(frames)):
        interleaved.append(frames[i])
        if i < len(frame_actions):
            interleaved.append(frame_actions[i])

    return build_canvas(interleaved, frame_size, sep_width)
