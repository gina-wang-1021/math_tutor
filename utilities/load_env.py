import streamlit as st

def load_env_vars(variable_name):
    """Load a specific environment variable from Streamlit secrets.
    
    Args:
        variable_name (str): The name of the secret variable to retrieve
        
    Returns:
        str: The value of the secret variable
        
    Raises:
        KeyError: If the variable is not found in secrets
        FileNotFoundError: If secrets.toml file is not found
    """
    try:
        return st.secrets[variable_name]
    except KeyError:
        raise KeyError(f"Secret '{variable_name}' not found in .streamlit/secrets.toml")
    except FileNotFoundError:
        raise FileNotFoundError("secrets.toml file not found in .streamlit/ directory")

