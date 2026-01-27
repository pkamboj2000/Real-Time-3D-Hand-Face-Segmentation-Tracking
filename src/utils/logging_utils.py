"""
Logging utilities for the segmentation pipeline.

Provides consistent logging configuration across all modules with
support for file logging, console output, and structured logging.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import functools
import time


class LoggerFactory:
    """
    Factory class for creating and managing loggers across the application.
    
    Provides centralized logging configuration with support for multiple
    output handlers and consistent formatting.
    """
    
    _instances: dict[str, logging.Logger] = {}
    _initialized: bool = False
    _log_dir: Optional[Path] = None
    _log_level: int = logging.INFO
    
    @classmethod
    def setup(
        cls,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        enable_file_logging: bool = True,
        log_format: Optional[str] = None
    ) -> None:
        """
        Initialize the logging system with the specified configuration.
        
        Args:
            log_dir: Directory for log files (default: project_root/logs)
            level: Logging level (default: INFO)
            enable_file_logging: Whether to write logs to files
            log_format: Custom log format string
        """
        cls._log_level = level
        
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        
        cls._log_dir = log_dir
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_format is None:
            log_format = (
                "%(asctime)s | %(levelname)-8s | %(name)-25s | "
                "%(filename)s:%(lineno)d | %(message)s"
            )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(log_format))
        root_logger.addHandler(console_handler)
        
        # File handler for persistent logging
        if enable_file_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = cls._log_dir / f"segmentation_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name (typically __name__ of the calling module)
            
        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._instances:
            logger = logging.getLogger(name)
            logger.setLevel(cls._log_level)
            cls._instances[name] = logger
        
        return cls._instances[name]


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console logging."""
    
    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance (uses default if not provided)
    
    Example:
        @log_execution_time()
        def process_image(image):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = LoggerFactory.get_logger(func.__module__)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.debug(
                    f"{func.__name__} completed in {elapsed:.4f}s"
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"{func.__name__} failed after {elapsed:.4f}s: {e}"
                )
                raise
        
        return wrapper
    return decorator


def log_method_calls(cls):
    """
    Class decorator to automatically log all method calls.
    
    Useful for debugging and monitoring class behavior.
    
    Args:
        cls: Class to decorate
    """
    logger = LoggerFactory.get_logger(cls.__module__)
    
    for name, method in vars(cls).items():
        if callable(method) and not name.startswith("_"):
            setattr(cls, name, log_execution_time(logger)(method))
    
    return cls


# Convenience function for quick logger access
def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return LoggerFactory.get_logger(name)
