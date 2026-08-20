"""Localhost API smoke tests for the browser presentation."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from ashare_sentiment.config import load_config
from ashare_sentiment.webapp.server import create_server


_REQUIRED_ARTIFACTS = (
    Path("data/processed/market_state_daily_v031.parquet"),
    Path("data/processed/market_advisory_daily_v041.parquet"),
)
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in _REQUIRED_ARTIFACTS),
    reason="Requires locally generated research artifacts; run the regime and advisory commands first.",
)


def _get(base: str, path: str):
    with urlopen(base + path, timeout=10) as response:
        return response.status, response.headers.get_content_type(), json.loads(response.read().decode("utf-8"))


def test_localhost_api_serves_cached_advisory_pages(monkeypatch):
    server = create_server(load_config("config/default.yaml"), "config/default.yaml", port=0)
    monkeypatch.setattr(server.application.service, "refresh", lambda *args, **kwargs: "ok")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/", timeout=10) as response:
            assert response.status == 200
            assert response.headers.get_content_type() == "text/html"
            assert "MarketTemperature" in response.read().decode("utf-8")

        status, _, overview = _get(base, "/api/overview?date=2026-07-23")
        assert status == 200
        assert overview["daily"]["advisory_signal"] == "BUY_REFERENCE"
        assert overview["daily"]["date"] == "2026-07-23"
        assert overview["trend"]

        status, _, history = _get(base, "/api/history?date=2026-07-17&range=3M")
        assert status == 200
        assert history["daily"]["advisory_signal"] == "PANIC_WAIT"
        assert history["range"] == "3M"

        status, _, episodes = _get(base, "/api/episodes")
        assert status == 200
        july = next(item for item in episodes["episodes"] if item["start_date"] == "2026-07-13")
        assert july["confirmed_date"] == "2026-07-23"

        status, _, diagnostics = _get(base, "/api/diagnostics")
        assert status == 200
        assert diagnostics["diagnostics"]["latest_valid_date"] == "2026-08-19"

        cached_snapshot = {
            "daily": server.application.service.daily("2026-08-18"),
            "point": {
                "date": "2026-08-19",
                "market_temperature": 34.9,
                "smoothed_temperature": 51.0,
                "advisory_signal": "NEUTRAL",
            },
            "is_intraday": True,
            "market_status": "OPEN",
            "as_of": "2026-08-19T10:30:00+08:00",
            "refresh_seconds": 60,
            "universe_count": 5001,
            "session_fraction": 0.25,
            "source": "tencent-realtime",
        }
        monkeypatch.setattr(server.application.intraday_service, "snapshot", lambda **kwargs: cached_snapshot)
        for _ in range(2):
            status, _, intraday = _get(base, "/api/intraday")
            assert status == 200
            assert intraday["trend"][-1]["date"] == "2026-08-19"
        assert "point" in cached_snapshot

        request = Request(base + "/api/refresh", method="POST")
        with urlopen(request, timeout=10) as response:
            assert response.status == 200
            refresh = json.loads(response.read().decode("utf-8"))
        assert refresh["status"] in {"running", "complete"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
