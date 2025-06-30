import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Dictionary to keep track of configured loggers
configured_loggers = {}

def setup_logger(name, log_dir=os.path.join(os.path.dirname(__file__), 'logs')):
    """Set up a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger (e.g., 'app' or 'engine')
        log_dir (str): Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Check if this logger has already been configured in this session
    if name in configured_loggers:
        return configured_loggers[name]
        
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set logger level to DEBUG
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        configured_loggers[name] = logger
        return logger
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # File handler (rotating to keep log size manageable)
    # Clean up the logger name to avoid directory structure in filenames
    simple_name = name.split('.')[-1]  # Get the last part of the name (e.g., 'qa_utils' from 'utilities.qa_utils')
    log_file = os.path.join(log_dir, f'{simple_name}_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Capture all debug logs in file
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # Keep console at INFO to reduce noise
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Store the configured logger
    configured_loggers[name] = logger
    
    return logger
