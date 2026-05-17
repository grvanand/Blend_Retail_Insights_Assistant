# utils/retry.py
# Reusable retry decorator using tenacity.
# Wrap any LLM call or I/O operation that may fail transiently.

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
from config.settings import settings
from utils.logger import logger

# Map loguru logger to standard logging for tenacity compatibility
_std_logger = logging.getLogger("tenacity")


def llm_retry(func):
    """
    Decorator for LLM API calls.
    - Retries up to llm_max_retries times (from settings)
    - Exponential backoff: 2s → 4s → 8s
    - Retries on any Exception (covers rate limits, timeouts, network errors)
    - Logs each retry attempt
    """
    return retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(_std_logger, logging.WARNING),
        reraise=True,                  # re-raise original exception after all retries exhausted
    )(func)


def io_retry(func):
    """
    Decorator for file I/O and vector store operations.
    - Retries up to 2 times with fixed 1s wait
    - Lighter than llm_retry — I/O failures recover faster
    """
    return retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((IOError, OSError, FileNotFoundError)),
        before_sleep=before_sleep_log(_std_logger, logging.WARNING),
        reraise=True,
    )(func)
