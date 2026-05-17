"""Retry decorator for transient network errors.

Per PRD §6 Phase 4 deliverable 1: retry on RequestException / Qdrant
UnexpectedResponse / ConnectionError / TimeoutError. Do NOT retry on
ValueError or KeyError (programming bugs that retrying cannot fix).
"""
from __future__ import annotations

from typing import Callable, TypeVar

import requests
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")

RETRYABLE = (
    requests.RequestException,
    UnexpectedResponse,
    ConnectionError,
    TimeoutError,
)


def retry_with_backoff(
    max_tries: int = 5, base: float = 1.0, max_delay: float = 60.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a decorator that retries the wrapped function on transient
    network errors with exponential backoff."""
    return retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=base, max=max_delay),
        stop=stop_after_attempt(max_tries),
        reraise=True,
    )
