#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################
"""
Centralized logging configuration for TARDIS-em.

This module provides functions to configure logging for the entire TARDIS project
with consistent formatting and output options.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    :param name: Name of the logger. If None, returns root logger.
    :type name: Optional[str]
    :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
    :type level: int
    :param log_file: Path to log file. If None, uses default location.
    :type log_file: Optional[str]
    :param console_output: Whether to output logs to console.
    :type console_output: bool
    :param file_output: Whether to output logs to file.
    :type file_output: bool
    :return: Configured logger instance.
    :rtype: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        if log_file is None:
            # Default log file location
            log_dir = Path.home() / ".tardis_em" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"tardis_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    This is a convenience function that returns a logger configured
    with the module name.

    :param name: Name of the logger (typically __name__).
    :type name: str
    :return: Logger instance.
    :rtype: logging.Logger
    """
    return logging.getLogger(name)


def set_log_level(level: int):
    """
    Set the logging level for all TARDIS loggers.

    :param level: Logging level (e.g., logging.DEBUG, logging.INFO).
    :type level: int
    """
    logging.getLogger("tardis_em").setLevel(level)
    for handler in logging.getLogger("tardis_em").handlers:
        handler.setLevel(level)


def configure_tardis_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
):
    """
    Configure logging for the entire TARDIS project.

    This function should be called once at the start of the application
    to set up logging for all TARDIS modules.

    :param level: Logging level for all loggers.
    :type level: int
    :param log_file: Path to log file. If None, uses default location.
    :type log_file: Optional[str]
    :param console_output: Whether to output logs to console.
    :type console_output: bool
    :param file_output: Whether to output logs to file.
    :type file_output: bool
    """
    # Configure root TARDIS logger
    setup_logger(
        name="tardis_em",
        level=level,
        log_file=log_file,
        console_output=console_output,
        file_output=file_output,
    )

    # Log initial message
    logger = logging.getLogger("tardis_em")
    logger.info("TARDIS-em logging initialized")

