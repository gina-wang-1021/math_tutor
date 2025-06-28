import os
import sys

def load_env_vars():
    """Load environment variables from .env file."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to .env file
    env_path = os.path.join(project_root, '.env')
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')
        return True
    else:
        print(f"Warning: .env file not found at {env_path}")
        return False

if __name__ == "__main__":
    load_env_vars()
