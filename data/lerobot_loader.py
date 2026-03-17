"""LeRobot v3.0 dataset loader for canvas world model.

Reads LeRobot v3.0 datasets (Parquet + MP4) directly, extracting frames
and discrete actions needed for canvas building.

Core classes ported from developmental-robot-movement/convert_lerobot_to_explorer.py.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import av
except ImportError:
    print("Error: PyAV is required for video decoding.")
    print("Install it with: pip install av")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required for reading Parquet files.")
    print("Install it with: pip install pandas pyarrow")
    sys.exit(1)


def resolve_dataset_path(lerobot_path_or_repo: str) -> Path:
    """Resolve dataset path, downloading from Hub if needed.

    Args:
        lerobot_path_or_repo: Either a local path or a HuggingFace repo_id

    Returns:
        Local path to the dataset
    """
    local_path = Path(lerobot_path_or_repo)
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required to download from Hub.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading dataset from Hub: {lerobot_path_or_repo}")
    cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / lerobot_path_or_repo

    snapshot_download(
        repo_id=lerobot_path_or_repo,
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return cache_dir


@dataclass
class DiscreteActionLog:
    """Parsed discrete action log with header and decisions."""

    header: dict
    decisions: list

    @property
    def action_duration(self) -> float:
        return self.header.get("action_duration", 0.5)

    @property
    def position_delta(self) -> float:
        return self.header.get("position_delta", 0.1)

    @property
    def joint_name(self) -> str:
        return self.header.get("joint_name", "shoulder_pan.pos")


def load_discrete_action_log(log_path: Path) -> Optional[DiscreteActionLog]:
    """Load discrete action log from JSONL file.

    First line is header with recording parameters.
    Subsequent lines are action decisions.
    """
    if not log_path.exists():
        return None

    header = None
    decisions = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "header":
                header = entry
            elif entry.get("type") == "action":
                decisions.append(entry)

    if header is None:
        return None

    return DiscreteActionLog(header=header, decisions=decisions)


def load_decision_bearing_logs(log_dir: Path) -> List[DiscreteActionLog]:
    """Load action logs from a directory, filtering out header-only (spurious) files.

    Returns only logs that contain actual action decisions, in file-sorted order.
    Index 0 = first real episode, index 1 = second, etc.
    """
    log_files = sorted(log_dir.glob("episode_*.jsonl"))
    result = []
    for lf in log_files:
        log = load_discrete_action_log(lf)
        if log and log.decisions:
            result.append(log)
    return result


def get_decision_frame_indices(
    log: DiscreteActionLog,
    total_frames: int,
) -> List[Tuple[int, int]]:
    """Map each action decision to a frame index using logged frame indices.

    Args:
        log: The discrete action log with decisions (must have frame_index)
        total_frames: Total number of frames in the episode (for validation)

    Returns:
        List of (frame_index, discrete_action) tuples

    Raises:
        ValueError: If decisions are missing the frame_index field
    """
    if not log.decisions:
        return []

    missing = [i for i, d in enumerate(log.decisions) if "frame_index" not in d]
    if missing:
        raise ValueError(
            f"Discrete action log is missing 'frame_index' field in {len(missing)} "
            f"decision(s) (first at index {missing[0]}). "
            f"Re-record with the updated SimpleJointPolicy that logs frame indices."
        )

    decision_frames = []
    for decision in log.decisions:
        frame_idx = decision["frame_index"]
        if frame_idx < 0 or frame_idx >= total_frames:
            print(f"  Warning: frame_index {frame_idx} out of range [0, {total_frames}), clamping")
            frame_idx = max(0, min(total_frames - 1, frame_idx))
        decision_frames.append((frame_idx, decision["discrete_action"]))

    return decision_frames



class LeRobotV3Reader:
    """Reads LeRobot v3.0 dataset format."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        self.videos_path = self.dataset_path / "videos"

        self.info = self._load_info()
        self.episodes = self._load_episodes()

    def _load_info(self) -> dict:
        info_path = self.meta_path / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        with open(info_path) as f:
            return json.load(f)

    def _load_episodes(self) -> pd.DataFrame:
        episodes_path = self.meta_path / "episodes.parquet"
        if episodes_path.exists():
            return pd.read_parquet(episodes_path)

        episodes_dir = self.meta_path / "episodes"
        if episodes_dir.exists():
            parquet_files = sorted(episodes_dir.glob("**/*.parquet"))
            if parquet_files:
                dfs = [pd.read_parquet(f) for f in parquet_files]
                return pd.concat(dfs, ignore_index=True)

        raise FileNotFoundError(f"Episodes metadata not found in {self.meta_path}")

    def get_data_chunk(self, chunk_idx: int) -> pd.DataFrame:
        chunk_dir = self.data_path / f"chunk-{chunk_idx:03d}"
        if not chunk_dir.exists():
            return pd.DataFrame()

        parquet_files = sorted(chunk_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    def get_video_path(self, camera_key: str, chunk_idx: int, file_idx: int = 0) -> Optional[Path]:
        video_dir = self.videos_path / f"observation.images.{camera_key}" / f"chunk-{chunk_idx:03d}"
        if not video_dir.exists():
            video_dir = self.videos_path / camera_key / f"chunk-{chunk_idx:03d}"

        if not video_dir.exists():
            return None

        mp4_files = sorted(video_dir.glob("*.mp4"))
        if file_idx < len(mp4_files):
            return mp4_files[file_idx]
        return None

    def iterate_episodes(self):
        for idx, row in self.episodes.iterrows():
            yield {
                "episode_index": row.get("episode_index", idx),
                "length": row.get("length", 0),
                "task_index": row.get("task_index", 0),
            }

    @property
    def fps(self) -> float:
        return self.info.get("fps", 30)

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def chunks_size(self) -> int:
        return self.info.get("chunks_size", 1000)


class VideoFrameExtractor:
    """Extracts frames from MP4 video files using PyAV."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.container = av.open(str(self.video_path))
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate) if self.stream.average_rate else 30.0
        self._frames: List[np.ndarray] = []
        self._loaded = False

    def load_all_frames(self):
        if self._loaded:
            return
        self.container.seek(0)
        for frame in self.container.decode(video=0):
            self._frames.append(frame.to_ndarray(format="rgb24"))
        self._loaded = True

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        if not self._loaded:
            self.load_all_frames()
        if 0 <= frame_index < len(self._frames):
            return self._frames[frame_index]
        return None

    @property
    def total_frames(self) -> int:
        if not self._loaded:
            self.load_all_frames()
        return len(self._frames)

    def close(self):
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to target (height, width)."""
    img = Image.fromarray(frame)
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img)


def stack_frames(frames: List[np.ndarray], mode: str = "vertical") -> np.ndarray:
    """Stack multiple camera frames vertically, horizontally, or single."""
    if mode == "single" or len(frames) == 1:
        return frames[0]
    elif mode == "vertical":
        return np.vstack(frames)
    elif mode == "horizontal":
        return np.hstack(frames)
    else:
        raise ValueError(f"Unknown stack mode: {mode}")


@dataclass
class EpisodeFrameActions:
    """Frames and discrete actions for a single episode.

    frames[i] is the i-th observation frame (H, W, 3) uint8.
    actions[i] is the discrete action taken between frames[i] and frames[i+1].
    len(actions) == len(frames) - 1
    """

    frames: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    episode_index: int = 0
    metadata: dict = field(default_factory=dict)


def load_episode(
    reader: LeRobotV3Reader,
    episode_index: int,
    cameras: List[str],
    stack_mode: str,
    frame_size: Tuple[int, int],
    cached_extractors: Optional[dict] = None,
    cached_logs: Optional[List] = None,
    cached_chunk_data: Optional[dict] = None,
) -> EpisodeFrameActions:
    """Load frames and discrete actions for one episode.

    Requires discrete action logs in meta/discrete_action_logs/.

    Args:
        reader: LeRobotV3Reader instance
        episode_index: Which episode to load
        cameras: Camera keys to extract
        stack_mode: How to combine cameras ("vertical", "horizontal", "single")
        frame_size: Per-camera (H, W) resize target
        cached_extractors: Optional pre-loaded {(camera, chunk): VideoFrameExtractor}
        cached_logs: Optional pre-loaded list of DiscreteActionLog
        cached_chunk_data: Optional pre-loaded {chunk_idx: DataFrame}

    Returns:
        EpisodeFrameActions with frames and discrete actions

    Raises:
        ValueError: If no discrete action log is found for the episode
    """
    # Get episode info
    episode_info = None
    for ep in reader.iterate_episodes():
        if ep["episode_index"] == episode_index:
            episode_info = ep
            break

    if episode_info is None:
        raise ValueError(f"Episode {episode_index} not found")

    length = episode_info["length"]
    if length == 0:
        raise ValueError(f"Episode {episode_index} has no frames")

    # Determine chunk
    chunk_idx = episode_index // reader.chunks_size

    # Load parquet data (use cache if available)
    if cached_chunk_data is not None and chunk_idx in cached_chunk_data:
        all_data = cached_chunk_data[chunk_idx]
    else:
        all_data = reader.get_data_chunk(chunk_idx)
        if cached_chunk_data is not None:
            cached_chunk_data[chunk_idx] = all_data

    if all_data.empty:
        raise ValueError(f"No data in chunk {chunk_idx}")

    episode_data = all_data[all_data["episode_index"] == episode_index]
    if len(episode_data) == 0:
        raise ValueError(f"Episode {episode_index} not found in data")

    # Video offset: global index of first frame in this episode
    video_offset = 0
    if "index" in episode_data.columns:
        video_offset = int(episode_data["index"].iloc[0])
    elif "frame_index" in episode_data.columns:
        video_offset = int(episode_data["frame_index"].iloc[0])

    # Load video extractors (use cache if available)
    extractors = {}
    owns_extractors = False
    for camera in cameras:
        cache_key = (camera, chunk_idx)
        if cached_extractors is not None and cache_key in cached_extractors:
            extractors[camera] = cached_extractors[cache_key]
        else:
            video_path = reader.get_video_path(camera, chunk_idx)
            if video_path is None:
                print(f"  Warning: No video for camera {camera}, episode {episode_index}")
                continue
            ext = VideoFrameExtractor(str(video_path))
            ext.load_all_frames()
            extractors[camera] = ext
            if cached_extractors is not None:
                cached_extractors[cache_key] = ext
            else:
                owns_extractors = True

    if not extractors:
        raise ValueError(f"No video data for episode {episode_index}")

    first_camera = cameras[0]
    total_video_frames = extractors[first_camera].total_frames if first_camera in extractors else length

    def get_combined_frame(frame_idx: int) -> Optional[np.ndarray]:
        camera_frames = []
        for camera in cameras:
            if camera not in extractors:
                continue
            frame = extractors[camera].get_frame(frame_idx)
            if frame is not None:
                resized = resize_frame(frame, frame_size)
                camera_frames.append(resized)
        if not camera_frames:
            return None
        return stack_frames(camera_frames, stack_mode)

    # Load discrete action log (use cache if available)
    if cached_logs is not None:
        real_logs = cached_logs
    else:
        log_dir = reader.dataset_path / "meta" / "discrete_action_logs"
        if not log_dir.exists():
            if owns_extractors:
                for ext in extractors.values():
                    ext.close()
            raise ValueError(
                f"No discrete_action_logs directory found at {log_dir}. "
                f"This dataset must have been recorded with discrete action logging."
            )
        real_logs = load_decision_bearing_logs(log_dir)

    if episode_index >= len(real_logs):
        if owns_extractors:
            for ext in extractors.values():
                ext.close()
        raise ValueError(
            f"No discrete action log for episode {episode_index} "
            f"(found {len(real_logs)} logs)"
        )

    action_log = real_logs[episode_index]
    print(f"  Using discrete action log: {len(action_log.decisions)} decisions")

    decision_frames = get_decision_frame_indices(action_log, total_video_frames)

    # Trim trailing no-ops
    original_count = len(decision_frames)
    while decision_frames and decision_frames[-1][1] == 0:
        decision_frames.pop()

    if len(decision_frames) < original_count:
        print(f"  Trimmed {original_count - len(decision_frames)} trailing no-op actions")

    if not decision_frames:
        if owns_extractors:
            for ext in extractors.values():
                ext.close()
        raise ValueError(f"Episode {episode_index}: all actions are no-ops")

    # Calculate action duration in frames
    action_duration_frames = max(1, round(action_log.action_duration * reader.fps))
    max_frame = video_offset + length - 1

    frames = []
    actions = []

    # First frame: at the first decision point
    first_frame = get_combined_frame(decision_frames[0][0] + video_offset)
    if first_frame is not None:
        frames.append(first_frame)

    # For each decision, record the action and the result frame
    # The result frame is action_duration_frames after the decision
    for i, (frame_idx, discrete_action) in enumerate(decision_frames):
        actions.append(discrete_action)

        result_frame_idx = min(frame_idx + action_duration_frames + video_offset, max_frame)

        next_frame = get_combined_frame(result_frame_idx)
        if next_frame is not None:
            frames.append(next_frame)
        elif frames:
            # Use last valid frame as fallback
            frames.append(frames[-1].copy())

    # Only close extractors we own (not cached ones)
    if owns_extractors:
        for ext in extractors.values():
            ext.close()

    # Ensure len(actions) == len(frames) - 1
    if len(actions) >= len(frames):
        actions = actions[: len(frames) - 1]

    metadata = {
        "fps": reader.fps,
        "cameras": cameras,
        "stack_mode": stack_mode,
        "frame_size": frame_size,
        "joint_name": action_log.joint_name,
        "source_path": str(reader.dataset_path),
    }

    return EpisodeFrameActions(
        frames=frames,
        actions=actions,
        episode_index=episode_index,
        metadata=metadata,
    )


def load_dataset(
    lerobot_path: str,
    cameras: List[str],
    stack_mode: str,
    frame_size: Tuple[int, int],
    episode: Optional[int] = None,
) -> List[EpisodeFrameActions]:
    """Load frames and actions from a LeRobot v3.0 dataset.

    Requires discrete action logs in the dataset's meta/discrete_action_logs/.

    Args:
        lerobot_path: Local path or HuggingFace repo_id
        cameras: Camera keys to extract
        stack_mode: How to combine cameras
        frame_size: Per-camera (H, W) resize target
        episode: Specific episode index, or None for all episodes

    Returns:
        List of EpisodeFrameActions, one per episode
    """
    dataset_path = resolve_dataset_path(lerobot_path)
    reader = LeRobotV3Reader(str(dataset_path))

    print(f"LeRobot dataset: {dataset_path}")
    print(f"  {reader.total_episodes} episodes, {reader.fps} FPS")
    print(f"  Cameras: {cameras}, stack: {stack_mode}")

    episodes_to_load = []
    if episode is not None:
        episodes_to_load = [episode]
    else:
        episodes_to_load = [ep["episode_index"] for ep in reader.iterate_episodes()]

    # Pre-load shared resources to avoid redundant work
    log_dir = reader.dataset_path / "meta" / "discrete_action_logs"
    cached_logs = None
    if log_dir.exists():
        print("Loading discrete action logs...")
        cached_logs = load_decision_bearing_logs(log_dir)
        print(f"  Loaded {len(cached_logs)} action logs")

    cached_extractors = {}
    cached_chunk_data = {}

    results = []
    for ep_idx in episodes_to_load:
        print(f"\nLoading episode {ep_idx}...")
        ep_data = load_episode(
            reader, ep_idx, cameras, stack_mode, frame_size,
            cached_extractors=cached_extractors,
            cached_logs=cached_logs,
            cached_chunk_data=cached_chunk_data,
        )
        print(f"  {len(ep_data.frames)} frames, {len(ep_data.actions)} actions")
        results.append(ep_data)

    # Close cached extractors
    for ext in cached_extractors.values():
        ext.close()

    return results
