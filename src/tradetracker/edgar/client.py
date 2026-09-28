"""HTTP client for SEC endpoints that follows the fair-access policy.

SEC asks for a declared User-Agent with a contact email and at most 10 requests
per second. We stay well under that at 8/s and back off on 429/503.
"""

import threading
import time

import httpx

MAX_PER_SECOND = 8
RETRY_STATUS = {429, 500, 502, 503, 504}


class EdgarClient:
    def __init__(self, user_agent: str, *, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            # The getcurrent feed can take a minute to answer; bulk ZIPs are large.
            timeout=httpx.Timeout(30.0, read=120.0),
            follow_redirects=True,
            transport=transport,
        )
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + 1.0 / MAX_PER_SECOND
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, *, retries: int = 4) -> httpx.Response:
        delay = 2.0
        for attempt in range(retries + 1):
            self._throttle()
            try:
                resp = self._http.get(url)
            except httpx.TimeoutException:
                if attempt == retries:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code not in RETRY_STATUS or attempt == retries:
                resp.raise_for_status()
                return resp
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._http.close()
