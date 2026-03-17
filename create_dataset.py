"""Create a canvas dataset from a LeRobot v3.0 dataset.

Usage:
    python create_dataset.py --lerobot-path irvinh/single-action-shoulder-pan-500
    python create_dataset.py --lerobot-path irvinh/single-action-shoulder-pan-500 --output local/datasets/my_dataset
"""

import json
import argparse
from pathlib import Path

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
) -> None:
    """Build all canvases from a LeRobot dataset and save as PNGs.

    Args:
        lerobot_path: Path to LeRobot v3.0 dataset or HuggingFace repo_id
        output_dir: Path to output directory (will be created)
        cameras: Camera keys to extract
        stack_mode: How to combine cameras
        frame_size: Per-camera (H, W) resize target
        episode: Specific episode index, or None for all episodes
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load episodes
    episodes = load_dataset(
        lerobot_path=lerobot_path,
        cameras=cameras,
        stack_mode=stack_mode,
        frame_size=frame_size,
        episode=episode,
    )

    if not episodes:
        print("No episodes loaded.")
        return

    # Detect frame size from first frame (after camera stacking)
    detected_frame_size = (episodes[0].frames[0].shape[0], episodes[0].frames[0].shape[1])
    print(f"\nDetected stacked frame size: {detected_frame_size}")

    # Build canvases across all episodes
    print("Building canvases...")
    canvas_count = 0
    history = config.CANVAS_HISTORY_SIZE
    episode_indices = []

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
    dataset_meta = {
        "source": "lerobot_v3",
        "source_path": lerobot_path,
        "canvas_count": canvas_count,
        "frame_size": detected_frame_size,
        "canvas_size": (
            detected_frame_size[0],
            detected_frame_size[1] * history + config.SEPARATOR_WIDTH * (history - 1),
        ),
        "separator_width": config.SEPARATOR_WIDTH,
        "canvas_history_size": history,
        "cameras": cameras,
        "stack_mode": stack_mode,
        "episodes": episode_indices,
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

    create_dataset(
        lerobot_path=args.lerobot_path,
        output_dir=output_dir,
        cameras=args.cameras,
        stack_mode=args.stack_cameras,
        frame_size=tuple(args.frame_size),
        episode=ep,
    )
