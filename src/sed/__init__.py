"""Commentary-based soccer event detection (Goal / Card / Penalty) from ASR transcripts."""

__version__ = "0.1.0"

# Local settings from a git-ignored .env in the working directory (WANDB_API_KEY, HF_HOME, ...). Loaded on import,
# before transformers / huggingface_hub read HF_HOME; variables already set in the environment win.
from .tracking import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()
