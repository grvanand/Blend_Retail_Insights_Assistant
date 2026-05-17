# utils/logger.py
# Centralized logger using loguru.
# Import `logger` from here across all modules — never use print().

import sys
from loguru import logger
from config.settings import settings

# Remove default loguru handler
logger.remove()

# Console handler — respects LOG_LEVEL from .env
logger.add(
    sys.stdout,
    level=settings.log_level.upper(),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
           "<level>{message}</level>",
    colorize=True,
)

# File handler — always logs DEBUG and above, rotates at 10MB
logger.add(
    "logs/app.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
)

# Re-export logger — all modules do: from utils.logger import logger
__all__ = ["logger"]
