"""Create a dataset by building all canvases from a session and saving as PNGs."""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from data.session_loader import (
    load_session_metadata,
    load_session_events,
    extract_observations,
    extract_actions,
    load_frame_image,
)
from data.canvas_builder import build_canvas


def create_dataset(session_dir: str, output_dir: str) -> None:
    """Build all canvases from a session and save as PNGs.

    Args:
        session_dir: Path to session directory
        output_dir: Path to output directory (will be created)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load session
    print(f"Loading session from {session_dir}...")
    meta = load_session_metadata(session_dir)
    events = load_session_events(session_dir)
    observations = extract_observations(events, session_dir)
    actions = extract_actions(events)

    print(f"  Loaded {len(observations)} observations, {len(actions)} actions")

    # Load all frames to detect frame size
    print("Loading frames and detecting frame size...")
    frame_images = []
    detected_frame_size = None

    for obs in tqdm(observations, desc="Loading frames"):
        img = load_frame_image(obs['full_path'])
        img_array = np.array(img)
        frame_images.append(img_array)

        if detected_frame_size is None:
            detected_frame_size = (img_array.shape[0], img_array.shape[1])

    print(f"  Detected frame size: {detected_frame_size}")

    # Build canvases
    print("Building canvases...")
    canvas_count = 0
    valid_start = config.CANVAS_HISTORY_SIZE - 1

    for frame_idx in tqdm(
        range(valid_start, len(observations)),
        desc="Building canvases",
        total=len(observations) - valid_start,
    ):
        # Get frame indices for history
        start_idx = frame_idx - (config.CANVAS_HISTORY_SIZE - 1)

        # Get frames
        frames = frame_images[start_idx : frame_idx + 1]

        # Get actions
        frame_actions = []
        for i in range(start_idx, frame_idx):
            if i < len(actions):
                frame_actions.append(actions[i]['action'])
            else:
                frame_actions.append({'action': 0})

        # Build interleaved list
        interleaved = []
        for i in range(len(frames)):
            interleaved.append(frames[i])
            if i < len(frame_actions):
                interleaved.append(frame_actions[i])

        # Build canvas
        canvas = build_canvas(
            interleaved,
            frame_size=detected_frame_size,
            sep_width=config.SEPARATOR_WIDTH,
        )

        # Save canvas
        canvas_filename = f"canvas_{canvas_count:05d}.png"
        canvas_path = os.path.join(output_dir, canvas_filename)
        Image.fromarray(canvas).save(canvas_path)

        canvas_count += 1

    # Save dataset metadata
    dataset_meta = {
        'session_name': meta.get('session_name', 'unknown'),
        'frame_count': len(observations),
        'canvas_count': canvas_count,
        'frame_size': detected_frame_size,
        'canvas_size': (
            detected_frame_size[0],
            detected_frame_size[1] * config.CANVAS_HISTORY_SIZE
            + config.SEPARATOR_WIDTH * (config.CANVAS_HISTORY_SIZE - 1),
        ),
        'separator_width': config.SEPARATOR_WIDTH,
        'canvas_history_size': config.CANVAS_HISTORY_SIZE,
        'session_metadata': meta,
    }

    meta_path = os.path.join(output_dir, 'dataset_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"\nDataset created successfully:")
    print(f"  Output directory: {output_dir}")
    print(f"  Total canvases: {canvas_count}")
    print(f"  Canvas size: {dataset_meta['canvas_size']}")
    print(f"  Metadata saved to: {meta_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a dataset by building all canvases from a session'
    )
    parser.add_argument(
        '--session',
        required=True,
        help='Path to session directory',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output dataset directory',
    )

    args = parser.parse_args()
    create_dataset(args.session, args.output)
