import os
from logger_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Define the prompts directory relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompts")

# Preloaded prompts dictionary
_PRELOADED_PROMPTS = {}

def _load_prompt_from_file(prompt_name: str) -> str:
    """Internal function to load a prompt string from a file."""
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

def _preload_all_prompts():
    """Preload all prompt files at startup."""
    global _PRELOADED_PROMPTS
    
    # List of all prompt files to preload
    prompt_files = [
        "chunk_coverage_prompt.txt",
        "compare_prompt.txt", 
        "econ_answer_prompt.txt",
        "econ_classification_prompt.txt",
        "econ_explain_prompt.txt",
        "econ_simplified_prompt.txt",
        "explain_prompt.txt",
        "math_classification_prompt.txt",
        "rephrase_question_prompt.txt",
        "topic_based_answer_prompt.txt",
        "simplified_prompt.txt"
    ]
    
    logger.info("Preloading prompts...")
    for prompt_file in prompt_files:
        try:
            _PRELOADED_PROMPTS[prompt_file] = _load_prompt_from_file(prompt_file)
            logger.debug(f"Preloaded prompt: {prompt_file}")
        except Exception as e:
            logger.error(f"Failed to preload prompt {prompt_file}: {e}")
            # Continue loading other prompts even if one fails
    
    logger.info(f"Successfully preloaded {len(_PRELOADED_PROMPTS)} prompts")

def load_prompt(prompt_name: str) -> str:
    """Loads a prompt string from preloaded prompts or falls back to file loading.
    
    Args:
        prompt_name (str): Name of the prompt file (e.g., "topic_based_answer_prompt.txt")
        
    Returns:
        str: The prompt content
    """
    # Try to get from preloaded prompts first
    if prompt_name in _PRELOADED_PROMPTS:
        return _PRELOADED_PROMPTS[prompt_name]
    
    # Fallback to file loading if not preloaded
    logger.warning(f"Prompt {prompt_name} not preloaded, loading from file")
    return _load_prompt_from_file(prompt_name)

# Preload all prompts when the module is imported
_preload_all_prompts()
