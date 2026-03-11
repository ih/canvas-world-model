"""Session loading utilities for SO-101 and other robot sessions."""

import os
import json
import glob
from io import BytesIO
from pathlib import Path
from PIL import Image


def load_session_metadata(session_dir: str) -> dict:
    """Load session metadata from session_meta.json.

    Args:
        session_dir: Path to session directory

    Returns:
        Dictionary with session metadata, or {} if file not found
    """
    meta_path = os.path.join(session_dir, "session_meta.json")
    if not os.path.exists(meta_path):
        return {}

    with open(meta_path, 'r') as f:
        return json.load(f)


def load_session_events(session_dir: str) -> list:
    """Load all events from sharded event files.

    Globs for events_shard_*.jsonl files and reads all JSONL lines.
    Returns sorted by 'step' field.

    Args:
        session_dir: Path to session directory

    Returns:
        List of event dictionaries, sorted by step
    """
    events = []

    # Glob for all shard files
    shard_pattern = os.path.join(session_dir, "events_shard_*.jsonl")
    shard_files = sorted(glob.glob(shard_pattern))

    for shard_file in shard_files:
        with open(shard_file, 'r') as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

    # Sort by step
    events.sort(key=lambda e: e.get('step', 0))
    return events


def extract_observations(events: list, session_dir: str) -> list:
    """Extract observation events and add full_path field.

    Args:
        events: List of event dictionaries from load_session_events()
        session_dir: Path to session directory (for resolving frame paths)

    Returns:
        List of observation dictionaries with fields:
        - observation_index: sequential index among observations
        - event_index: index in original events list
        - step, timestamp
        - frame_path: relative path as stored in event
        - full_path: absolute path to frame file
    """
    observations = []
    obs_index = 0

    for event_index, event in enumerate(events):
        if event.get('type') == 'observation' and event.get('data', {}).get('frame_path'):
            obs = {
                'observation_index': obs_index,
                'event_index': event_index,
                'step': event.get('step'),
                'timestamp': event.get('timestamp'),
                'frame_path': event['data']['frame_path'],
            }
            obs['full_path'] = os.path.join(session_dir, event['data']['frame_path'])
            observations.append(obs)
            obs_index += 1

    return observations


def extract_actions(events: list) -> list:
    """Extract action events.

    Args:
        events: List of event dictionaries from load_session_events()

    Returns:
        List of action dictionaries with fields:
        - action_index: sequential index among actions
        - event_index: index in original events list
        - step, timestamp
        - action: the raw action dictionary from event['data']
    """
    actions = []
    action_index = 0

    for event_index, event in enumerate(events):
        if event.get('type') == 'action':
            act = {
                'action_index': action_index,
                'event_index': event_index,
                'step': event.get('step'),
                'timestamp': event.get('timestamp'),
                'action': event.get('data', {}),
            }
            actions.append(act)
            action_index += 1

    return actions


def load_frame_image(full_path: str) -> Image.Image:
    """Load a frame image from disk.

    Args:
        full_path: Absolute path to frame file (JPEG, PNG, etc.)

    Returns:
        PIL Image in RGB mode
    """
    with open(full_path, 'rb') as f:
        img_bytes = f.read()

    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img
