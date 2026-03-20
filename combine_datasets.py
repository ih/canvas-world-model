"""Combine multiple canvas datasets into one.

Usage:
    python combine_datasets.py --inputs dir1 dir2 dir3 --output combined_dir
"""

import json
import shutil
import argparse
from pathlib import Path

import numpy as np


def combine_datasets(input_dirs: list, output_dir: str) -> None:
    """Merge multiple canvas datasets by copying canvases and merging metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_metas = []
    for d in input_dirs:
        meta_path = Path(d) / "dataset_meta.json"
        with open(meta_path) as f:
            all_metas.append(json.load(f))

    # Validate compatible canvas sizes
    sizes = set(tuple(m["canvas_size"]) for m in all_metas)
    if len(sizes) > 1:
        raise ValueError(f"Incompatible canvas sizes: {sizes}")

    # Copy canvases with renumbered names and merge metadata
    canvas_count = 0
    all_episodes = []
    all_canvas_actions = []
    source_datasets = []

    # Collect all motor norm values across datasets to compute global bounds
    all_motor_mins = []
    all_motor_maxs = []
    all_vel_maxs = []

    for meta in all_metas:
        if meta.get("motor_norm_min") is not None:
            all_motor_mins.append(np.array(meta["motor_norm_min"]))
            all_motor_maxs.append(np.array(meta["motor_norm_max"]))
        if meta.get("motor_vel_norm_max") is not None:
            all_vel_maxs.append(np.array(meta["motor_vel_norm_max"]))

    for dir_path, meta in zip(input_dirs, all_metas):
        dir_p = Path(dir_path)
        dataset_start = canvas_count

        for i in range(meta["canvas_count"]):
            src = dir_p / f"canvas_{i:05d}.png"
            dst = out / f"canvas_{canvas_count:05d}.png"
            shutil.copy2(src, dst)
            canvas_count += 1

        # Remap episode indices
        for ep in meta["episodes"]:
            all_episodes.append({
                "episode_index": len(all_episodes),
                "canvas_start": ep["canvas_start"] + dataset_start,
                "canvas_end": ep["canvas_end"] + dataset_start,
                "frame_count": ep["frame_count"],
                "action_count": ep["action_count"],
            })

        if meta.get("canvas_actions"):
            all_canvas_actions.extend(meta["canvas_actions"])

        source_datasets.append({
            "source_path": meta.get("source_path", str(dir_path)),
            "canvas_count": meta["canvas_count"],
            "canvas_offset": dataset_start,
        })

    # Compute global motor bounds
    global_motor_min = np.minimum.reduce(all_motor_mins).tolist() if all_motor_mins else None
    global_motor_max = np.maximum.reduce(all_motor_maxs).tolist() if all_motor_maxs else None
    global_vel_max = np.maximum.reduce(all_vel_maxs).tolist() if all_vel_maxs else None

    ref = all_metas[0]
    combined_meta = {
        "source": "combined",
        "source_datasets": source_datasets,
        "canvas_count": canvas_count,
        "frame_size": ref["frame_size"],
        "canvas_size": ref["canvas_size"],
        "separator_width": ref["separator_width"],
        "canvas_history_size": ref["canvas_history_size"],
        "cameras": ref["cameras"],
        "stack_mode": ref["stack_mode"],
        "motor_strip_height": ref.get("motor_strip_height", 0),
        "motor_norm_min": global_motor_min,
        "motor_norm_max": global_motor_max,
        "motor_vel_norm_max": global_vel_max,
        "episodes": all_episodes,
        "canvas_actions": all_canvas_actions if all_canvas_actions else None,
    }

    with open(out / "dataset_meta.json", "w") as f:
        json.dump(combined_meta, f, indent=2)

    print(f"Combined dataset created:")
    print(f"  Output: {output_dir}")
    print(f"  Canvases: {canvas_count}")
    print(f"  Episodes: {len(all_episodes)}")
    print(f"  Source datasets: {len(input_dirs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple canvas datasets")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input dataset directories")
    parser.add_argument("--output", required=True, help="Output combined dataset directory")
    args = parser.parse_args()
    combine_datasets(args.inputs, args.output)
