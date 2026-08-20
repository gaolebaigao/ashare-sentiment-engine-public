"""Dependency-light localhost server for the MarketTemperature browser UI."""

from __future__ import annotations

import json
import html
import math
import mimetypes
import threading
import webbrowser
from dataclasses import asdict
from datetime import date, datetime, time as clock_time, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ..application.service import AdvisoryService
from ..application.intraday import IntradaySnapshotService
from ..config import load_config


STATIC_ROOT = Path(__file__).with_name("static")


def _json_default(value: Any):
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _json_safe(value: Any):
    """Convert pandas/numpy values to strict JSON, including NaN -> null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _frame_records(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    selected = frame[[column for column in columns if column in frame.columns]].copy()
    return json.loads(json.dumps(_json_safe(selected.to_dict(orient="records")), ensure_ascii=False, allow_nan=False))


class LocalWebApplication:
    """API facade over the existing application service."""

    def __init__(self, config: dict[str, Any], config_path: str | Path):
        self.config = config
        self.config_path = str(config_path)
        self.service = AdvisoryService(config)
        self.intraday_service = IntradaySnapshotService(config, self.service.repository)
        self._job_lock = threading.Lock()
        self._job: dict[str, Any] = {"status": "idle", "message": "", "started_at": None, "finished_at": None}
        self._baseline_checked_for: date | None = None

    def overview(self, date_value: str | None = None) -> dict[str, Any]:
        daily = self.service.daily(date_value)
        resolved = daily.date
        return {
            "daily": asdict(daily),
            "trend": _frame_records(self.service.trend("1Y"), ("date", "market_temperature", "smoothed_temperature", "advisory_signal")),
            "previous_date": self.service.resolve_date(resolved, -1),
            "next_date": self.service.resolve_date(resolved, 1),
        }

    def intraday(self, *, force: bool = False) -> dict[str, Any]:
        baseline_end = self._required_baseline_end()
        if baseline_end is not None:
            job = self._start_refresh(end_date=baseline_end, automatic=True)
            if job["status"] == "running":
                raise RuntimeError(f"正在自动补齐 {baseline_end.isoformat()} 收盘基线，请稍后自动重试")
            if job["status"] == "failed":
                raise RuntimeError(f"上一交易日基线自动更新失败：{job['message']}")
        # The intraday service owns and reuses its TTL cache.  Work on a copy:
        # popping `point` from the cached dictionary made every second poll fail
        # with KeyError("point").
        snapshot = dict(self.intraday_service.snapshot(force=force))
        trend = _frame_records(self.service.trend("1Y"), ("date", "market_temperature", "smoothed_temperature", "advisory_signal"))
        point = snapshot.pop("point")
        trend = [item for item in trend if str(item.get("date", ""))[:10] != point["date"]]
        snapshot["daily"] = asdict(snapshot["daily"])
        snapshot["trend"] = trend + [point]
        return snapshot

    def history(self, date_value: str | None = None, range_name: str = "1Y") -> dict[str, Any]:
        daily = self.service.daily(date_value)
        frame = self.service.trend(range_name)
        rows = frame.tail(40)
        return {
            "daily": asdict(daily),
            "range": range_name,
            "trend": _frame_records(frame, ("date", "market_temperature", "smoothed_temperature", "advisory_signal")),
            "rows": _frame_records(rows, ("date", "market_temperature", "smoothed_temperature", "advisory_signal")),
            "previous_date": self.service.resolve_date(daily.date, -1),
            "next_date": self.service.resolve_date(daily.date, 1),
        }

    def episodes(self) -> dict[str, Any]:
        return {"episodes": [asdict(item) for item in self.service.episodes()]}

    def diagnostics(self) -> dict[str, Any]:
        return {"diagnostics": asdict(self.service.diagnostics())}

    def settings(self) -> dict[str, Any]:
        repository = self.service.repository
        return {
            "processed_root": str(repository.processed_root),
            "reports_root": str(repository.reports_root),
            "config_path": self.config_path,
        }

    def start_refresh(self) -> dict[str, Any]:
        return self._start_refresh(end_date=self._safe_download_end(), automatic=False)

    def _start_refresh(self, *, end_date: date, automatic: bool) -> dict[str, Any]:
        with self._job_lock:
            if self._job["status"] == "running":
                return dict(self._job)
            message = "正在自动补齐上一交易日收盘基线…" if automatic else "正在后台刷新本地数据…"
            self._job = {"status": "running", "message": message, "started_at": pd.Timestamp.now().isoformat(), "finished_at": None}
            thread = threading.Thread(target=self._refresh_worker, args=(end_date,), daemon=True)
            thread.start()
            return dict(self._job)

    def refresh_status(self) -> dict[str, Any]:
        with self._job_lock:
            return dict(self._job)

    def _refresh_worker(self, end_date: date) -> None:
        try:
            self.service.refresh(self.config_path, include_download=True, end_date=end_date)
            self.intraday_service = IntradaySnapshotService(self.config, self.service.repository)
            self._baseline_checked_for = datetime.now(self.intraday_service.timezone).date()
            job = {"status": "complete", "message": "刷新完成", "started_at": self._job["started_at"], "finished_at": pd.Timestamp.now().isoformat()}
        except Exception as exc:  # pragma: no cover - exercised by a live refresh
            job = {"status": "failed", "message": str(exc), "started_at": self._job["started_at"], "finished_at": pd.Timestamp.now().isoformat()}
        with self._job_lock:
            self._job = job

    def _required_baseline_end(self) -> date | None:
        today = datetime.now(self.intraday_service.timezone).date()
        if self._baseline_checked_for == today:
            return None
        expected = self._previous_weekday(today)
        metadata = self.service.repository.metadata("full_market_daily")
        try:
            latest = date.fromisoformat(str(metadata.date_end)[:10]) if metadata and metadata.date_end else None
        except ValueError:
            latest = None
        if latest is not None and latest >= expected:
            self._baseline_checked_for = today
            return None
        return expected

    def _safe_download_end(self) -> date:
        now = datetime.now(self.intraday_service.timezone)
        if now.weekday() < 5 and now.time() >= clock_time(15, 30):
            return now.date()
        return self._previous_weekday(now.date())

    @staticmethod
    def _previous_weekday(value: date) -> date:
        result = value - timedelta(days=1)
        while result.weekday() >= 5:
            result -= timedelta(days=1)
        return result


class _RequestHandler(BaseHTTPRequestHandler):
    app: LocalWebApplication

    def do_GET(self):  # noqa: N802 - HTTP API
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._api_get(parsed.path, parse_qs(parsed.query))
            return
        self._static_get(parsed.path)

    def do_POST(self):  # noqa: N802 - HTTP API
        parsed = urlparse(self.path)
        if parsed.path == "/api/refresh":
            self._send_json(self.app.start_refresh())
            return
        self._send_json({"error": "not found"}, status=404)

    def _api_get(self, path: str, query: dict[str, list[str]]):
        try:
            if path == "/api/overview":
                payload = self.app.overview(self._query(query, "date"))
            elif path == "/api/history":
                payload = self.app.history(self._query(query, "date"), self._query(query, "range") or "1Y")
            elif path == "/api/episodes":
                payload = self.app.episodes()
            elif path == "/api/diagnostics":
                payload = self.app.diagnostics()
            elif path == "/api/settings":
                payload = self.app.settings()
            elif path == "/api/refresh/status":
                payload = self.app.refresh_status()
            elif path == "/api/intraday":
                payload = self.app.intraday(force=self._query(query, "force") == "1")
            else:
                self._send_json({"error": "not found"}, status=404)
                return
            self._send_json(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _static_get(self, path: str):
        relative = "index.html" if path in {"", "/"} else path.removeprefix("/static/")
        target = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT not in target.parents and target != STATIC_ROOT:
            self._send_json({"error": "not found"}, status=404)
            return
        if not target.exists() or not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        if relative == "index.html":
            bootstrap_payload = {
                "overview": self.app.overview(),
                "history": self.app.history(),
                "episodes": self.app.episodes(),
                "diagnostics": self.app.diagnostics(),
                "settings": self.app.settings(),
            }
            bootstrap = json.dumps(_json_safe(bootstrap_payload), ensure_ascii=False, allow_nan=False)
            bootstrap_tag = f'<meta id="mt-bootstrap" data-payload="{html.escape(bootstrap, quote=True)}">'
            page_html = body.decode("utf-8").replace("</head>", f"{bootstrap_tag}</head>", 1)
            body = page_html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _query(query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key, [])
        return values[0] if values else None

    def _send_json(self, payload: dict[str, Any], status: int = 200):
        body = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args):  # noqa: A002 - BaseHTTPRequestHandler API
        return


class _ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True


def create_server(config: dict[str, Any], config_path: str | Path, host: str = "127.0.0.1", port: int = 8765):
    app = LocalWebApplication(config, config_path)

    class Handler(_RequestHandler):
        pass

    Handler.app = app
    server = _ReusableServer((host, port), Handler)
    server.application = app
    return server


def run_web(config_path: str = "config/default.yaml", host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    config = load_config(config_path)
    server = create_server(config, config_path, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"MarketTemperature local web app: {url}")
    print("Press Ctrl-C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
