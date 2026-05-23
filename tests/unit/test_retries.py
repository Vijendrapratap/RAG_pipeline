"""Unit tests for ingestion.utils.retries."""
from __future__ import annotations

import pytest
import requests

from ingestion.utils.retries import retry_with_backoff


def test_retries_on_connection_error_then_succeeds():
    calls = {"n": 0}

    @retry_with_backoff(max_tries=4, base=0.01, max_delay=0.05)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("flapping")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_does_not_retry_value_error():
    """ValueError is a programming bug — must surface immediately."""
    calls = {"n": 0}

    @retry_with_backoff(max_tries=5, base=0.01, max_delay=0.05)
    def boom() -> None:
        calls["n"] += 1
        raise ValueError("never retry me")

    with pytest.raises(ValueError):
        boom()
    assert calls["n"] == 1


def test_does_not_retry_key_error():
    calls = {"n": 0}

    @retry_with_backoff(max_tries=5, base=0.01, max_delay=0.05)
    def boom() -> None:
        calls["n"] += 1
        raise KeyError("missing key")

    with pytest.raises(KeyError):
        boom()
    assert calls["n"] == 1


def test_retries_requests_exception():
    calls = {"n": 0}

    @retry_with_backoff(max_tries=3, base=0.01, max_delay=0.05)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.ConnectTimeout("slow")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2


def test_gives_up_after_max_tries():
    calls = {"n": 0}

    @retry_with_backoff(max_tries=3, base=0.01, max_delay=0.05)
    def always_bad() -> None:
        calls["n"] += 1
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        always_bad()
    assert calls["n"] == 3
