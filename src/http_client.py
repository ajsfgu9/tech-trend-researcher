"""Shared resilient HTTP session with retries and exponential backoff."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "TechTrendResearcher/4.0 (+https://github.com/ajsfgu9/tech-trend-researcher)"


def build_http_session(
    total_retries: int = 3,
    backoff_factor: float = 1.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> requests.Session:
    """Build a requests.Session configured with Retry on transient HTTP failures."""
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session
