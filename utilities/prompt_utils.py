import os
from scripts.logger_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Define the prompts directory relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompts")

def load_prompt(prompt_name: str) -> str:
    """Loads a prompt string from a file in the PROMPT_DIR."""
    prompt_file_path = os.path.join(PROMPT_DIR, prompt_name)
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt file {prompt_file_path}: {e}")
        raise
