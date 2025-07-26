import os

def load_env_vars():
    """Load environment variables from .env file."""
    # Get the current directory and parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Check current directory for .env file
    env_path = os.path.join(current_dir, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')
        return True
    else:
        # If not found, check parent directory
        env_path = os.path.join(parent_dir, '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value.strip('"\'')
            return True
        else:
            print(f"Warning: .env file not found at {current_dir} or {parent_dir}")
            return False

