"""Safe retries: transient failures get backed-off retries; fatal ones don't.

Retrying a bad API key five times just wastes time -- FatalError lets a
caller mark an error as non-retryable instead of retrying blindly.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable


class FatalError(Exception):
    """Non-retryable -- e.g. bad credentials, malformed request."""


def with_backoff[T](
    fn: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    for attempt in range(max_retries):
        try:
            return fn()
        except FatalError:
            raise
        except Exception:
            if attempt == max_retries - 1:
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random() / 2)
            time.sleep(delay)
    raise RuntimeError("unreachable: max_retries must be >= 1")
