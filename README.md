# Canvas World Model

A minimal, vision-only world model framework for robotics learning. Everything (observations and actions) is embedded as images — no separate action tokens or metadata.

## Overview

This repository provides tools to:
1. Load robot session data (recorded trajectories)
2. Build **canvases** — images that concatenate observation frames with colored action separators
3. Browse and visualize datasets

The canvas representation is the foundation for training vision-only world models where action information is purely visual.

## Canvas Format

A canvas concatenates frames horizontally with colored separators between them:

```
[Frame 0] [Separator 0] [Frame 1] [Separator 1] [Frame 2]
```

**Separator colors:**
- Red (255, 0, 0): Action 0 (stay)
- Green (0, 255, 0): Action 1 (move positive)
- Blue (0, 0, 255): Action 2 (move negative)

For SO-101 robot with dual cameras: canvas is 448H × 736W pixels (3 frames × 224W + 2 separators × 32W).

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Create a Dataset

```bash
python create_dataset.py \
    --session local/sessions/session_so101_multiheight_part1_1345 \
    --output local/datasets/part1
```

This will:
- Load all frames and actions from the session
- Build canvases for each valid frame (starting from frame 2 to have 3-frame history)
- Save canvases as `canvas_00000.png`, `canvas_00001.png`, etc.
- Save metadata to `dataset_meta.json`

### View a Dataset

```bash
python view_dataset.py --dataset local/datasets/part1
```

Interactive controls:
- **Left/Right arrow keys** or **P/N**: Navigate canvases
- **Home/End**: Jump to first/last canvas
- **Mouse scroll**: Jump by 5 canvases
- **Q or Escape**: Close viewer

## Directory Structure

```
canvas-world-model/
├── data/
│   ├── session_loader.py       # Load session events and frames
│   ├── canvas_builder.py       # Build canvases from frame sequences
│   └── __init__.py
├── config.py                    # Canvas parameters (separator width, history size)
├── create_dataset.py            # CLI: build canvases from a session
├── view_dataset.py              # CLI: interactive canvas viewer
├── requirements.txt
└── README.md

local/                           # Not in git (local data and artifacts)
├── sessions/                    # Recorded robot sessions
│   ├── session_so101_multiheight_part1_1345/
│   └── session_so101_multiheight_part2_149/
├── datasets/                    # Generated canvas datasets
│   ├── part1/
│   └── part2/
└── checkpoints/                 # (Future) trained model checkpoints
```

## Configuration

Edit `config.py` to customize:
- `SEPARATOR_WIDTH`: Width in pixels of action separators (default 32)
- `CANVAS_HISTORY_SIZE`: Number of frames per canvas (default 3)
- `ACTION_COLORS`: RGB colors for each action type

## Session Format

Sessions are expected to have:
- `session_meta.json`: Metadata (robot type, action space, etc.)
- `events_shard_*.jsonl`: Sharded event logs (observations and actions)
- `frames/frame_*.jpg`: JPEG frame images

See [developmental-robot-movement](https://github.com/your-account/developmental-robot-movement) for example sessions.

## Next Steps

Once you have a canvas dataset, you can:
- Train vision-only world models (in separate modules)
- Analyze action conditioning via counterfactual metrics
- Fine-tune on larger sessions

## License

[Same as developmental-robot-movement]
