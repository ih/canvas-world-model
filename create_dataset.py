"""Create a canvas dataset from a LeRobot v3.0 dataset.

Usage:
    python create_dataset.py --lerobot-path irvinh/single-action-shoulder-pan-500
    python create_dataset.py --lerobot-path irvinh/single-action-shoulder-pan-500 --output local/datasets/my_dataset
"""

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from data.lerobot_loader import load_dataset
from data.canvas_builder import build_canvas


def create_dataset(
    lerobot_path: str,
    output_dir: str,
    cameras: list,
    stack_mode: str,
    frame_size: tuple,
    episode: int = None,
    state_column: str = None,
) -> None:
    """Build all canvases from a LeRobot dataset and save as PNGs.

    Args:
        lerobot_path: Path to LeRobot v3.0 dataset or HuggingFace repo_id
        output_dir: Path to output directory (will be created)
        cameras: Camera keys to extract
        stack_mode: How to combine cameras
        frame_size: Per-camera (H, W) resize target
        episode: Specific episode index, or None for all episodes
        state_column: Parquet column for motor positions (None to disable motor strips)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load episodes
    episodes = load_dataset(
        lerobot_path=lerobot_path,
        cameras=cameras,
        stack_mode=stack_mode,
        frame_size=frame_size,
        episode=episode,
        state_column=state_column,
    )

    if not episodes:
        print("No episodes loaded.")
        return

    # Detect frame size from first frame (after camera stacking)
    detected_frame_size = (episodes[0].frames[0].shape[0], episodes[0].frames[0].shape[1])
    print(f"\nDetected stacked frame size: {detected_frame_size}")

    # Compute global motor normalization bounds across all episodes
    has_motor = any(ep.motor_positions for ep in episodes)
    motor_norm_min = None
    motor_norm_max = None
    motor_strip_height = 0

    motor_vel_norm_max = None

    if has_motor:
        all_motors = []
        for ep in episodes:
            for m in ep.motor_positions:
                if m is not None:
                    all_motors.append(m)
        if all_motors:
            stacked = np.stack(all_motors)
            motor_norm_min = stacked.min(axis=0)
            motor_norm_max = stacked.max(axis=0)
            motor_strip_height = config.MOTOR_STRIP_HEIGHT

            # Compute max absolute velocity across all consecutive frames
            all_vels = []
            for ep in episodes:
                for i in range(1, len(ep.motor_positions)):
                    if ep.motor_positions[i] is not None and ep.motor_positions[i - 1] is not None:
                        all_vels.append(ep.motor_positions[i] - ep.motor_positions[i - 1])
            if all_vels:
                vel_stacked = np.stack(all_vels)
                motor_vel_norm_max = np.abs(vel_stacked).max(axis=0)
            else:
                motor_vel_norm_max = np.ones(stacked.shape[1])

            print(f"Motor positions: {stacked.shape[1]} joints, strip height: {motor_strip_height}px")

    # Build canvases across all episodes
    print("Building canvases...")
    canvas_count = 0
    history = config.CANVAS_HISTORY_SIZE
    episode_indices = []
    canvas_actions = []  # Per-canvas action labels for evaluation

    for ep in episodes:
        ep_canvas_start = canvas_count
        valid_start = history - 1

        for frame_idx in tqdm(
            range(valid_start, len(ep.frames)),
            desc=f"Episode {ep.episode_index}",
            total=len(ep.frames) - valid_start,
        ):
            start_idx = frame_idx - (history - 1)

            # Collect frames and actions for this window
            window_frames = ep.frames[start_idx : frame_idx + 1]
            window_actions = ep.actions[start_idx : frame_idx]

            # Extract action integer labels for this canvas
            action_ints = []
            for a in window_actions:
                if isinstance(a, int):
                    action_ints.append(a)
                elif isinstance(a, dict):
                    action_ints.append(a.get('action', 0))
                else:
                    action_ints.append(0)
            canvas_actions.append(action_ints)

            # Collect motor positions for this window
            window_motors = None
            if has_motor and ep.motor_positions:
                window_motors = ep.motor_positions[start_idx : frame_idx + 1]

            # Build interleaved list: [frame, action, frame, action, frame]
            interleaved = []
            for i in range(len(window_frames)):
                interleaved.append(window_frames[i])
                if i < len(window_actions):
                    interleaved.append(window_actions[i])

            canvas = build_canvas(
                interleaved,
                frame_size=detected_frame_size,
                sep_width=config.SEPARATOR_WIDTH,
                motor_positions=window_motors,
                motor_strip_height=motor_strip_height,
                motor_norm_min=motor_norm_min,
                motor_norm_max=motor_norm_max,
                motor_vel_norm_max=motor_vel_norm_max,
            )

            canvas_path = Path(output_dir) / f"canvas_{canvas_count:05d}.png"
            Image.fromarray(canvas).save(canvas_path)
            canvas_count += 1

        episode_indices.append({
            "episode_index": ep.episode_index,
            "canvas_start": ep_canvas_start,
            "canvas_end": canvas_count - 1,
            "frame_count": len(ep.frames),
            "action_count": len(ep.actions),
        })

    # Save dataset metadata
    canvas_h = detected_frame_size[0] + motor_strip_height
    canvas_w = detected_frame_size[1] * history + config.SEPARATOR_WIDTH * (history - 1)

    dataset_meta = {
        "source": "lerobot_v3",
        "source_path": lerobot_path,
        "canvas_count": canvas_count,
        "frame_size": detected_frame_size,
        "canvas_size": (canvas_h, canvas_w),
        "separator_width": config.SEPARATOR_WIDTH,
        "canvas_history_size": history,
        "cameras": cameras,
        "stack_mode": stack_mode,
        "motor_strip_height": motor_strip_height,
        "motor_norm_min": motor_norm_min.tolist() if motor_norm_min is not None else None,
        "motor_norm_max": motor_norm_max.tolist() if motor_norm_max is not None else None,
        "motor_vel_norm_max": motor_vel_norm_max.tolist() if motor_vel_norm_max is not None else None,
        "episodes": episode_indices,
        "canvas_actions": canvas_actions,
    }

    meta_path = Path(output_dir) / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"\nDataset created:")
    print(f"  Output: {output_dir}")
    print(f"  Canvases: {canvas_count}")
    print(f"  Canvas size: {dataset_meta['canvas_size']}")
    print(f"  Episodes: {len(episode_indices)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a canvas dataset from a LeRobot v3.0 dataset"
    )
    parser.add_argument(
        "--lerobot-path",
        type=str,
        required=True,
        help="Path to LeRobot v3.0 dataset or HuggingFace repo_id",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="local/datasets",
        help="Path to output dataset directory (default: local/datasets)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=config.DEFAULT_CAMERAS,
        help="Camera keys to use",
    )
    parser.add_argument(
        "--stack-cameras",
        type=str,
        choices=["vertical", "horizontal", "single"],
        default=config.DEFAULT_STACK_MODE,
        help="How to combine multiple cameras",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        default=list(config.DEFAULT_FRAME_SIZE),
        help="Per-camera frame size (height width)",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Episode index (int) or 'all' for all episodes. Default: all.",
    )
    parser.add_argument(
        "--state-column",
        type=str,
        default=config.MOTOR_STATE_KEY,
        help=f"Parquet column for motor positions (default: {config.MOTOR_STATE_KEY}). "
             "Set to 'none' to disable motor strips.",
    )

    args = parser.parse_args()

    # Parse episode argument
    ep = None
    if args.episode is not None and args.episode != "all":
        ep = int(args.episode)

    # Derive output subdirectory from lerobot path if using default output
    output_dir = args.output
    if output_dir == "local/datasets":
        # e.g. "irvinh/single-action-shoulder-pan-500" -> "single-action-shoulder-pan-500"
        dataset_name = args.lerobot_path.rstrip("/").split("/")[-1]
        output_dir = str(Path(output_dir) / dataset_name)

    state_col = args.state_column if args.state_column != "none" else None

    create_dataset(
        lerobot_path=args.lerobot_path,
        output_dir=output_dir,
        cameras=args.cameras,
        stack_mode=args.stack_cameras,
        frame_size=tuple(args.frame_size),
        episode=ep,
        state_column=state_col,
    )
