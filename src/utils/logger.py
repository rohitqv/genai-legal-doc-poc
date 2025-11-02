"""
Logging utility for Legal Document Analysis PoC
"""
import sys
from pathlib import Path
from loguru import logger
from .config import LOG_LEVEL, LOG_FILE_PATH

# Create logs directory if it doesn't exist
Path(LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)

# Remove default logger
logger.remove()

# Add console handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
    colorize=True
)

# Add file handler
logger.add(
    LOG_FILE_PATH,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=LOG_LEVEL,
    rotation="10 MB",
    retention="30 days",
    compression="zip"
)

__all__ = ["logger"]

