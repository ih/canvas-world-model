"""Interactive canvas viewer for browsing a dataset."""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


class DatasetViewer:
    """Interactive matplotlib viewer for browsing canvas datasets."""

    def __init__(self, dataset_dir: str):
        """Initialize the viewer with a dataset directory.

        Args:
            dataset_dir: Path to dataset directory containing canvas_*.png files
        """
        self.dataset_dir = dataset_dir

        # Load canvas paths
        canvas_pattern = os.path.join(dataset_dir, 'canvas_*.png')
        self.canvas_paths = sorted(Path(dataset_dir).glob('canvas_*.png'))

        if not self.canvas_paths:
            raise ValueError(f"No canvas files found in {dataset_dir}")

        print(f"Loaded {len(self.canvas_paths)} canvases")

        # Load metadata if available
        meta_path = os.path.join(dataset_dir, 'dataset_meta.json')
        self.metadata = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)

        self.current_index = 0

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.suptitle('Canvas Dataset Viewer')

        # Setup keyboard handlers
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    def show(self) -> None:
        """Display the viewer and start interactive mode."""
        self._display_current()
        plt.show()

    def _display_current(self) -> None:
        """Display the canvas at current_index."""
        canvas_path = self.canvas_paths[self.current_index]
        canvas = Image.open(canvas_path)

        self.ax.clear()
        self.ax.imshow(canvas)
        self.ax.axis('off')

        # Build title
        title = f"Canvas {self.current_index:05d} / {len(self.canvas_paths) - 1:05d}"
        title += f" ({canvas_path.name})"
        if self.metadata:
            title += f" | Session: {self.metadata.get('session_name', '?')}"

        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw()

    def _on_key(self, event) -> None:
        """Handle keyboard events."""
        if event.key == 'right' or event.key == 'n':
            self.current_index = min(self.current_index + 1, len(self.canvas_paths) - 1)
            self._display_current()
        elif event.key == 'left' or event.key == 'p':
            self.current_index = max(self.current_index - 1, 0)
            self._display_current()
        elif event.key == 'home':
            self.current_index = 0
            self._display_current()
        elif event.key == 'end':
            self.current_index = len(self.canvas_paths) - 1
            self._display_current()
        elif event.key == 'escape' or event.key == 'q':
            plt.close(self.fig)

    def _on_scroll(self, event) -> None:
        """Handle mouse scroll events."""
        if event.button == 'up':
            self.current_index = min(self.current_index + 5, len(self.canvas_paths) - 1)
        elif event.button == 'down':
            self.current_index = max(self.current_index - 5, 0)
        self._display_current()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactive viewer for canvas datasets'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to dataset directory',
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Starting canvas index',
    )

    args = parser.parse_args()

    viewer = DatasetViewer(args.dataset)
    viewer.current_index = args.index
    viewer.show()
