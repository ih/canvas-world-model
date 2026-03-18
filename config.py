"""Canvas configuration parameters."""

SEPARATOR_WIDTH = 32
CANVAS_HISTORY_SIZE = 2

# Motor position strip
MOTOR_STRIP_HEIGHT = 16  # pixels, aligned to ViT patch_size=16
MOTOR_STATE_KEY = "observation.state"  # parquet column for joint positions

# LeRobot defaults
DEFAULT_CAMERAS = ["base_0_rgb", "left_wrist_0_rgb"]
DEFAULT_STACK_MODE = "vertical"
DEFAULT_FRAME_SIZE = (224, 224)
