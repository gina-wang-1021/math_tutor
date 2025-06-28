import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')):
    """Set up a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger (e.g., 'app' or 'engine')
        log_dir (str): Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # File handler (rotating to keep log size manageable)
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
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
    
    return logger
