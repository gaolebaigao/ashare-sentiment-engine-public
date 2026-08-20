"""Conservative cached HTTP JSON client for free public endpoints."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import requests


class HttpDataError(RuntimeError):
    """Raised when a public endpoint cannot return valid JSON data."""


class CachedJsonClient:
    """GET JSON with a persistent response cache, throttling and backoff.

    Historical requests are safe to reuse indefinitely. ``refresh=True`` is
    available for the current trading day, whose public pool may change intraday.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        timeout: float = 15.0,
        min_interval: float = 0.2,
        retries: int = 3,
        session: requests.Session | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.min_interval = max(0.0, float(min_interval))
        self.retries = max(0, int(retries))
        self.session = session or requests.Session()
        self._last_request_at = 0.0

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any],
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        cache_path = self._cache_path(url, params)
        if cache_path.exists() and not refresh:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except (OSError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                response = self.session.get(
                    url,
                    params=dict(params),
                    timeout=self.timeout,
                    headers={"User-Agent": "ashare-sentiment-engine/0.1"},
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise HttpDataError("Endpoint returned JSON but not an object")
                self._atomic_write(cache_path, payload)
                return payload
            except (requests.RequestException, ValueError, HttpDataError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))
        raise HttpDataError(f"GET failed after {self.retries + 1} attempts: {url}: {last_error}") from last_error

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _cache_path(self, url: str, params: Mapping[str, Any]) -> Path:
        canonical = json.dumps({"url": url, "params": dict(params)}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.cache_root / f"{digest}.json"

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
