"""Combine multiple canvas datasets into one.

Usage:
    python combine_datasets.py --inputs dir1 dir2 dir3 --output combined_dir
"""

import json
import shutil
import argparse
from pathlib import Path

import numpy as np


def combine_datasets(
    input_dirs: list,
    output_dir: str,
    motor_bounds_override: dict | None = None,
) -> None:
    """Merge multiple canvas datasets by copying canvases and merging metadata.

    `motor_bounds_override`, if provided, pins the output meta's
    `motor_norm_min` / `motor_norm_max` to fixed values instead of deriving
    them from the per-input global min/max. Keys: `motor_norm_min` (list of
    6 floats), `motor_norm_max` (list of 6 floats), `motor_vel_norm_max`
    (list of 6 floats, optional). Use this to keep the motor-strip
    rendering stable across incremental merges so a checkpoint trained on
    merge N doesn't see a shifted motor-strip distribution in merge N+1.
    """
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
    # Per-canvas acting joint + motor state at decision frame (added in
    # the per-cell MSE breakdown work — sub-phase 1). evaluate.py uses
    # these to bucket prediction error by (joint, position-bin). We must
    # propagate them across merges or merged training datasets lose the
    # signal even when each input dataset has it.
    all_canvas_acting_joints: list = []
    all_canvas_motor_states: list = []
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

        # Forward per-canvas joint + motor state. Pad with None when an
        # individual source dataset lacks the fields (older builds), so
        # the parallel arrays stay aligned with canvas indices.
        n_canvases_in_meta = int(meta["canvas_count"])
        src_joints = meta.get("canvas_acting_joints") or []
        src_states = meta.get("canvas_motor_states_at_decision") or []
        if len(src_joints) == n_canvases_in_meta:
            all_canvas_acting_joints.extend(src_joints)
        else:
            all_canvas_acting_joints.extend([None] * n_canvases_in_meta)
        if len(src_states) == n_canvases_in_meta:
            all_canvas_motor_states.extend(src_states)
        else:
            all_canvas_motor_states.extend([None] * n_canvases_in_meta)

        source_datasets.append({
            "source_path": meta.get("source_path", str(dir_path)),
            "canvas_count": meta["canvas_count"],
            "canvas_offset": dataset_start,
        })

    # Compute global motor bounds — data-driven by default, overridden if
    # the caller pinned them (for cross-merge comparability).
    if motor_bounds_override and motor_bounds_override.get("motor_norm_min") is not None:
        global_motor_min = list(motor_bounds_override["motor_norm_min"])
    else:
        global_motor_min = np.minimum.reduce(all_motor_mins).tolist() if all_motor_mins else None
    if motor_bounds_override and motor_bounds_override.get("motor_norm_max") is not None:
        global_motor_max = list(motor_bounds_override["motor_norm_max"])
    else:
        global_motor_max = np.maximum.reduce(all_motor_maxs).tolist() if all_motor_maxs else None
    if motor_bounds_override and motor_bounds_override.get("motor_vel_norm_max") is not None:
        global_vel_max = list(motor_bounds_override["motor_vel_norm_max"])
    else:
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
        "canvas_acting_joints": (
            all_canvas_acting_joints if any(j is not None for j in all_canvas_acting_joints) else None
        ),
        "canvas_motor_states_at_decision": (
            all_canvas_motor_states if any(s is not None for s in all_canvas_motor_states) else None
        ),
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
    parser.add_argument(
        "--motor-bounds-json",
        default=None,
        help=(
            "Optional JSON dict pinning motor-strip normalization bounds. "
            'Example: \'{"motor_norm_min":[-60,0,0,-90,-90,-45],'
            '"motor_norm_max":[60,180,180,90,90,45]}\'. '
            "If omitted, bounds are derived from the input datasets."
        ),
    )
    args = parser.parse_args()
    override = json.loads(args.motor_bounds_json) if args.motor_bounds_json else None
    combine_datasets(args.inputs, args.output, motor_bounds_override=override)
