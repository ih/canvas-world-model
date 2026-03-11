"""Canvas configuration parameters."""

SEPARATOR_WIDTH = 32
CANVAS_HISTORY_SIZE = 3

# Action to RGB color mapping
ACTION_COLORS = {
    0: (255, 0, 0),    # Red: stay
    1: (0, 255, 0),    # Green: move positive
    2: (0, 0, 255),    # Blue: move negative
}
