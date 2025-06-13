"""
Logging utilities for the household segmentation project.

This module provides centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """
    Setup logging configuration for the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs only to console
        log_format: Custom log format string
        include_timestamp: Whether to include timestamp in log messages
    """
    if log_format is None:
        if include_timestamp:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            log_format = '%(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # Get root logger and add console handler
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def create_experiment_logger(
    experiment_name: str,
    log_dir: Union[str, Path] = "logs"
) -> logging.Logger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        
    Returns:
        Logger instance configured for the experiment
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(f"experiment.{experiment_name}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Also log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Experiment logger created: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to other classes.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


# Configure default logging
setup_logging(level=logging.INFO)
