"""Test script to verify canvas creation and dataset integrity."""

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np

import config
from data.session_loader import (
    load_session_metadata,
    load_session_events,
    extract_observations,
    extract_actions,
    load_frame_image,
)
from data.canvas_builder import build_canvas


def test_session_loading():
    """Test that sessions load correctly."""
    print("Testing session loading...")

    session_dir = "local/sessions/session_so101_multiheight_part1_1345"

    meta = load_session_metadata(session_dir)
    assert meta is not None, "Failed to load metadata"
    assert meta['session_name'] == 'session_so101_multiheight_part1_1345'
    print(f"  [OK] Metadata loaded: {meta['session_name']}")

    events = load_session_events(session_dir)
    assert len(events) == 2690, f"Expected 2690 events, got {len(events)}"
    print(f"  [OK] Loaded {len(events)} events")

    observations = extract_observations(events, session_dir)
    assert len(observations) == 1345, f"Expected 1345 observations, got {len(observations)}"
    print(f"  [OK] Extracted {len(observations)} observations")

    actions = extract_actions(events)
    assert len(actions) == 1345, f"Expected 1345 actions, got {len(actions)}"
    print(f"  [OK] Extracted {len(actions)} actions")


def test_frame_loading():
    """Test that frames load correctly."""
    print("\nTesting frame loading...")

    session_dir = "local/sessions/session_so101_multiheight_part1_1345"
    events = load_session_events(session_dir)
    observations = extract_observations(events, session_dir)

    # Load first frame
    frame = load_frame_image(observations[0]['full_path'])
    frame_array = np.array(frame)

    assert frame_array.shape == (448, 224, 3), f"Unexpected frame shape: {frame_array.shape}"
    assert frame_array.dtype == np.uint8, f"Unexpected dtype: {frame_array.dtype}"
    print(f"  [OK] Frame shape: {frame_array.shape}")
    print(f"  [OK] Frame dtype: {frame_array.dtype}")


def test_canvas_building():
    """Test that canvases build correctly."""
    print("\nTesting canvas building...")

    session_dir = "local/sessions/session_so101_multiheight_part1_1345"
    events = load_session_events(session_dir)
    observations = extract_observations(events, session_dir)
    actions = extract_actions(events)

    # Load frames for canvas at index 5
    frame_idx = 5
    start_idx = frame_idx - (config.CANVAS_HISTORY_SIZE - 1)

    frames = [np.array(load_frame_image(observations[i]['full_path'])) for i in range(start_idx, frame_idx + 1)]
    actions_list = [actions[i]['action'] for i in range(start_idx, frame_idx)]

    interleaved = []
    for i in range(len(frames)):
        interleaved.append(frames[i])
        if i < len(actions_list):
            interleaved.append(actions_list[i])

    canvas = build_canvas(interleaved, frame_size=(448, 224), sep_width=config.SEPARATOR_WIDTH)

    expected_w = 224 * 3 + 32 * 2  # 3 frames + 2 separators
    expected_h = 448

    assert canvas.shape == (expected_h, expected_w, 3), f"Unexpected canvas shape: {canvas.shape}"
    assert canvas.dtype == np.uint8, f"Unexpected canvas dtype: {canvas.dtype}"
    print(f"  [OK] Canvas shape: {canvas.shape}")
    print(f"  [OK] Canvas dtype: {canvas.dtype}")

    # Verify separators have color (not black)
    sep1_mean = canvas[:, 224:256].mean()
    sep2_mean = canvas[:, 480:512].mean()
    assert sep1_mean > 0, "First separator appears to be all black"
    assert sep2_mean > 0, "Second separator appears to be all black"
    print(f"  [OK] Separators have color (mean: {sep1_mean:.1f}, {sep2_mean:.1f})")


def test_dataset_integrity():
    """Test that created datasets are valid."""
    print("\nTesting dataset integrity...")

    datasets = [
        ("local/datasets/part1", 1343, "session_so101_multiheight_part1_1345"),
        ("local/datasets/part2", 148, "session_so101_multiheight_part2_149"),
    ]

    for dataset_dir, expected_count, expected_session in datasets:
        if not os.path.exists(dataset_dir):
            print(f"  [SKIP] Dataset {dataset_dir} not found")
            continue

        # Count canvases
        canvas_files = list(Path(dataset_dir).glob('canvas_*.png'))
        assert len(canvas_files) == expected_count, f"Expected {expected_count} canvases, got {len(canvas_files)}"
        print(f"  [OK] {dataset_dir}: {len(canvas_files)} canvases")

        # Check metadata
        meta_path = os.path.join(dataset_dir, 'dataset_meta.json')
        assert os.path.exists(meta_path), f"Metadata not found in {dataset_dir}"
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        assert meta['session_name'] == expected_session
        assert meta['canvas_count'] == expected_count
        assert meta['canvas_size'] == [448, 736]
        print(f"      Metadata: {meta['canvas_count']} canvases, size {meta['canvas_size']}")

        # Spot check a few canvases
        for canvas_idx in [0, expected_count // 2, expected_count - 1]:
            canvas_path = os.path.join(dataset_dir, f'canvas_{canvas_idx:05d}.png')
            canvas = Image.open(canvas_path)
            assert canvas.size == (736, 448), f"Canvas {canvas_idx} has wrong size: {canvas.size}"

        print(f"      Canvas files valid (spot checked {min(3, expected_count)} files)")


if __name__ == '__main__':
    print("=" * 60)
    print("Canvas World Model - Test Suite")
    print("=" * 60)

    try:
        test_session_loading()
        test_frame_loading()
        test_canvas_building()
        test_dataset_integrity()

        print("\n" + "=" * 60)
        print("[OK] All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
