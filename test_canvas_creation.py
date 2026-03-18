"""Tests for canvas creation from LeRobot v3.0 datasets.

Unit tests use synthetic data (no external dependencies).
Integration tests require a real LeRobot dataset on disk and are skipped if absent.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import config
from data.canvas_builder import build_canvas
from data.lerobot_loader import (
    DiscreteActionLog,
    EpisodeFrameActions,
    get_decision_frame_indices,
    load_discrete_action_log,
    resize_frame,
    stack_frames,
)


# ---------------------------------------------------------------------------
# Unit tests (always run, no external data needed)
# ---------------------------------------------------------------------------


class TestDiscreteActionLogParsing:
    def test_load_log(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"type": "header", "joint_name": "shoulder_pan.pos",
                                "action_duration": 0.39, "position_delta": 15.0}) + "\n")
            f.write(json.dumps({"type": "action", "discrete_action": 0, "frame_index": 0}) + "\n")
            f.write(json.dumps({"type": "action", "discrete_action": 1, "frame_index": 4}) + "\n")
            f.write(json.dumps({"type": "action", "discrete_action": 2, "frame_index": 8}) + "\n")
            tmp_path = Path(f.name)

        log = load_discrete_action_log(tmp_path)
        tmp_path.unlink()

        assert log is not None
        assert log.joint_name == "shoulder_pan.pos"
        assert log.action_duration == 0.39
        assert len(log.decisions) == 3
        assert log.decisions[0]["discrete_action"] == 0
        assert log.decisions[2]["frame_index"] == 8

    def test_load_missing_file(self):
        assert load_discrete_action_log(Path("/nonexistent.jsonl")) is None

    def test_load_header_only(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"type": "header", "joint_name": "test"}) + "\n")
            tmp_path = Path(f.name)

        log = load_discrete_action_log(tmp_path)
        tmp_path.unlink()

        assert log is not None
        assert len(log.decisions) == 0


class TestDecisionFrameIndices:
    def test_basic_mapping(self):
        log = DiscreteActionLog(
            header={"joint_name": "test"},
            decisions=[
                {"type": "action", "discrete_action": 0, "frame_index": 0},
                {"type": "action", "discrete_action": 1, "frame_index": 4},
                {"type": "action", "discrete_action": 2, "frame_index": 8},
            ],
        )
        result = get_decision_frame_indices(log, total_frames=100)
        assert result == [(0, 0), (4, 1), (8, 2)]

    def test_clamping(self):
        log = DiscreteActionLog(
            header={},
            decisions=[{"type": "action", "discrete_action": 1, "frame_index": 999}],
        )
        result = get_decision_frame_indices(log, total_frames=10)
        assert result == [(9, 1)]

    def test_empty_decisions(self):
        log = DiscreteActionLog(header={}, decisions=[])
        assert get_decision_frame_indices(log, 10) == []

    def test_missing_frame_index_raises(self):
        log = DiscreteActionLog(
            header={},
            decisions=[{"type": "action", "discrete_action": 0}],
        )
        with pytest.raises(ValueError, match="missing 'frame_index'"):
            get_decision_frame_indices(log, 10)


class TestResizeAndStack:
    def test_resize(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized = resize_frame(frame, (224, 224))
        assert resized.shape == (224, 224, 3)

    def test_stack_vertical(self):
        f1 = np.zeros((100, 200, 3), dtype=np.uint8)
        f2 = np.ones((100, 200, 3), dtype=np.uint8) * 255
        stacked = stack_frames([f1, f2], "vertical")
        assert stacked.shape == (200, 200, 3)

    def test_stack_horizontal(self):
        f1 = np.zeros((100, 200, 3), dtype=np.uint8)
        f2 = np.ones((100, 200, 3), dtype=np.uint8) * 255
        stacked = stack_frames([f1, f2], "horizontal")
        assert stacked.shape == (100, 400, 3)

    def test_stack_single(self):
        f1 = np.zeros((100, 200, 3), dtype=np.uint8)
        stacked = stack_frames([f1], "single")
        assert stacked.shape == (100, 200, 3)


class TestCanvasBuilding:
    def _make_frame(self, color, h=448, w=224):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = color
        return frame

    def test_canvas_shape(self):
        frames = [self._make_frame(c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        actions = [0, 1]  # stay, move+
        interleaved = [frames[0], actions[0], frames[1], actions[1], frames[2]]

        canvas = build_canvas(interleaved, frame_size=(448, 224), sep_width=32)

        expected_w = 224 * 3 + 32 * 2
        assert canvas.shape == (448, expected_w, 3)
        assert canvas.dtype == np.uint8

    def test_separator_colors(self):
        frames = [self._make_frame((128, 128, 128)) for _ in range(3)]
        actions = [0, 1]  # stay=red, move+=green
        interleaved = [frames[0], actions[0], frames[1], actions[1], frames[2]]

        canvas = build_canvas(interleaved, frame_size=(448, 224), sep_width=32)

        # First separator (action=0 -> red)
        sep1 = canvas[:, 224:256]
        assert sep1[0, 0, 0] == 255  # R
        assert sep1[0, 0, 1] == 0    # G
        assert sep1[0, 0, 2] == 0    # B

        # Second separator (action=1 -> green)
        sep2 = canvas[:, 480:512]
        assert sep2[0, 0, 0] == 0
        assert sep2[0, 0, 1] == 255
        assert sep2[0, 0, 2] == 0

    def test_canvas_with_action_dicts(self):
        """Canvas builder also accepts action dicts."""
        frames = [self._make_frame((0, 0, 0)) for _ in range(2)]
        interleaved = [frames[0], {"action": 2}, frames[1]]

        canvas = build_canvas(interleaved, frame_size=(448, 224), sep_width=32)
        sep = canvas[:, 224:256]
        assert sep[0, 0, 2] == 255  # Blue for action=2


class TestMotorStrip:
    def _make_frame(self, color, h=448, w=224):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = color
        return frame

    def test_canvas_with_motor_strip(self):
        """Canvas height increases by motor_strip_height when motor positions are provided."""
        frames = [self._make_frame((128, 128, 128)) for _ in range(2)]
        interleaved = [frames[0], 0, frames[1]]

        motor_pos = [
            np.array([0.0, 0.5, 1.0], dtype=np.float32),
            np.array([0.2, 0.7, 0.3], dtype=np.float32),
        ]
        norm_min = np.array([0.0, 0.0, 0.0])
        norm_max = np.array([1.0, 1.0, 1.0])
        vel_max = np.array([1.0, 1.0, 1.0])

        canvas = build_canvas(
            interleaved,
            frame_size=(448, 224),
            sep_width=32,
            motor_positions=motor_pos,
            motor_strip_height=16,
            motor_norm_min=norm_min,
            motor_norm_max=norm_max,
            motor_vel_norm_max=vel_max,
        )

        expected_h = 448 + 16
        expected_w = 224 * 2 + 32
        assert canvas.shape == (expected_h, expected_w, 3)

    def test_motor_strip_position_patches(self):
        """Each joint's position patch should be a uniform grayscale fill."""
        frames = [self._make_frame((0, 0, 0)) for _ in range(2)]
        interleaved = [frames[0], 1, frames[1]]

        # 2 joints: first at max, second at min
        motor_pos = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
        ]
        norm_min = np.array([0.0, 0.0])
        norm_max = np.array([1.0, 1.0])
        vel_max = np.array([1.0, 1.0])

        canvas = build_canvas(
            interleaved,
            frame_size=(448, 224),
            sep_width=32,
            motor_positions=motor_pos,
            motor_strip_height=16,
            motor_norm_min=norm_min,
            motor_norm_max=norm_max,
            motor_vel_norm_max=vel_max,
        )

        # First frame, patch 0 (joint 0 position=1.0) -> white
        assert canvas[448, 0, 0] == 255
        # First frame, patch 1 (joint 1 position=0.0) -> black
        assert canvas[448, 16, 0] == 0

    def test_motor_strip_velocity_patches(self):
        """Velocity patches: first frame mid-gray (no prior), second frame reflects delta."""
        frames = [self._make_frame((0, 0, 0)) for _ in range(2)]
        interleaved = [frames[0], 1, frames[1]]

        motor_pos = [
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),  # delta = +1.0
        ]
        norm_min = np.array([0.0])
        norm_max = np.array([1.0])
        vel_max = np.array([1.0])

        canvas = build_canvas(
            interleaved,
            frame_size=(448, 224),
            sep_width=32,
            motor_positions=motor_pos,
            motor_strip_height=16,
            motor_norm_min=norm_min,
            motor_norm_max=norm_max,
            motor_vel_norm_max=vel_max,
        )

        # First frame velocity patch (patch index 1, x=16:32): mid-gray (no prior frame)
        assert canvas[448, 16, 0] == 128

        # Second frame velocity patch (patch index 1, x=256+16=272): max positive -> white
        assert canvas[448, 272, 0] == 255

    def test_canvas_without_motor_unchanged(self):
        """Canvas without motor positions should have original height."""
        frames = [self._make_frame((128, 128, 128)) for _ in range(2)]
        interleaved = [frames[0], 0, frames[1]]

        canvas = build_canvas(interleaved, frame_size=(448, 224), sep_width=32)
        assert canvas.shape == (448, 480, 3)

    def test_separator_spans_full_height_with_motor(self):
        """Separator should span the full canvas height including motor strip."""
        frames = [self._make_frame((0, 0, 0)) for _ in range(2)]
        interleaved = [frames[0], 1, frames[1]]

        motor_pos = [np.array([0.5]), np.array([0.5])]
        norm_min = np.array([0.0])
        norm_max = np.array([1.0])
        vel_max = np.array([1.0])

        canvas = build_canvas(
            interleaved,
            frame_size=(448, 224),
            sep_width=32,
            motor_positions=motor_pos,
            motor_strip_height=16,
            motor_norm_min=norm_min,
            motor_norm_max=norm_max,
            motor_vel_norm_max=vel_max,
        )

        # Separator at x=224:256 should be green (action=1) across full height
        sep = canvas[:, 224:256]
        assert sep[0, 0, 1] == 255    # top: green
        assert sep[460, 0, 1] == 255  # bottom (in motor strip region): green


class TestEpisodeBoundary:
    def test_no_canvas_crossover(self):
        """Canvases should not span two episodes."""
        history = config.CANVAS_HISTORY_SIZE

        # Two short episodes, each with 4 frames
        ep1 = EpisodeFrameActions(
            frames=[np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)],
            actions=[0, 1, 2],
            episode_index=0,
        )
        ep2 = EpisodeFrameActions(
            frames=[np.ones((10, 10, 3), dtype=np.uint8) * 255 for _ in range(4)],
            actions=[1, 0, 2],
            episode_index=1,
        )

        # Simulate the canvas loop from create_dataset
        canvas_count = 0
        for ep in [ep1, ep2]:
            valid_start = history - 1
            for frame_idx in range(valid_start, len(ep.frames)):
                start_idx = frame_idx - (history - 1)
                window_frames = ep.frames[start_idx : frame_idx + 1]
                window_actions = ep.actions[start_idx : frame_idx]
                assert len(window_frames) == history
                assert len(window_actions) == history - 1
                canvas_count += 1

        # Each episode: 4 frames, history=3, so 2 canvases per episode
        assert canvas_count == 4


# ---------------------------------------------------------------------------
# Integration tests (require real LeRobot dataset on disk)
# ---------------------------------------------------------------------------

DATASET_PATH = Path.home() / ".cache" / "huggingface" / "lerobot" / "irvinh" / "eval_shoulder_pan_10_minutes"
skip_no_data = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="LeRobot dataset not available at expected path",
)


@skip_no_data
class TestLeRobotIntegration:
    def test_reader(self):
        from data.lerobot_loader import LeRobotV3Reader

        reader = LeRobotV3Reader(str(DATASET_PATH))
        assert reader.total_episodes >= 1
        assert reader.fps > 0

        episodes = list(reader.iterate_episodes())
        assert len(episodes) >= 1
        assert episodes[0]["length"] > 0

    def test_load_episode(self):
        from data.lerobot_loader import load_episode, LeRobotV3Reader

        reader = LeRobotV3Reader(str(DATASET_PATH))
        ep = load_episode(
            reader,
            episode_index=0,
            cameras=["base_0_rgb", "left_wrist_0_rgb"],
            stack_mode="vertical",
            frame_size=(224, 224),
        )

        assert len(ep.frames) > 10
        assert len(ep.actions) == len(ep.frames) - 1
        assert ep.frames[0].shape == (448, 224, 3)  # 2 cameras stacked vertically
        assert all(a in (0, 1, 2) for a in ep.actions)

    def test_full_pipeline(self):
        from data.lerobot_loader import load_dataset

        episodes = load_dataset(
            lerobot_path=str(DATASET_PATH),
            cameras=["base_0_rgb"],
            stack_mode="single",
            frame_size=(224, 224),
            episode=0,
        )

        assert len(episodes) == 1
        ep = episodes[0]

        # Build one canvas
        history = config.CANVAS_HISTORY_SIZE
        frames = ep.frames[:history]
        actions = ep.actions[: history - 1]

        interleaved = []
        for i in range(len(frames)):
            interleaved.append(frames[i])
            if i < len(actions):
                interleaved.append(actions[i])

        canvas = build_canvas(
            interleaved,
            frame_size=(frames[0].shape[0], frames[0].shape[1]),
            sep_width=config.SEPARATOR_WIDTH,
        )

        assert canvas.shape[0] == 224
        expected_w = 224 * history + config.SEPARATOR_WIDTH * (history - 1)
        assert canvas.shape[1] == expected_w
