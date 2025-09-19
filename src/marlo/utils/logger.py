"""
Unified logging utilities for MARLO.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def get_logger(name: str = "marlo", 
               log_file: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger for the MARLO system.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_experiment_logger(experiment_name: str, 
                           experiment_dir: str = "experiments") -> logging.Logger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        experiment_dir: Base experiments directory
        
    Returns:
        Experiment logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(experiment_dir, experiment_name, "logs")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    return get_logger(f"marlo.{experiment_name}", log_file)
