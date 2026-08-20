"""Presentation-layer regression tests for the optional MarketTemperature app."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from ashare_sentiment.application.service import AdvisoryService
from ashare_sentiment.config import load_config


_REQUIRED_ARTIFACTS = (
    Path("data/processed/market_state_daily_v031.parquet"),
    Path("data/processed/market_advisory_daily_v041.parquet"),
)
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in _REQUIRED_ARTIFACTS),
    reason="Requires locally generated research artifacts; run the regime and advisory commands first.",
)


def _service() -> AdvisoryService:
    return AdvisoryService(load_config("config/default.yaml"))


def test_latest_hot_mapping_is_presentation_only():
    daily = _service().daily("2026-08-18")
    assert daily.advisory_signal == "HOT_CAUTION"
    assert daily.state == "HOT"
    assert daily.temperature == "63.1"
    assert daily.smoothed_temperature == "67.6"
    assert daily.advisory_label == "偏热谨慎"
    assert len(daily.warnings) == 5


def test_data_invalid_has_no_buy_or_sell_reference():
    daily = _service().daily("2021-01-04")
    assert daily.data_invalid is True
    assert daily.advisory_signal == "DATA_INVALID"
    assert daily.buy_reference == "—"
    assert daily.sell_reference == "—"
    assert daily.what_to_watch_next == "数据恢复后再查看市场建议。"


def test_history_date_queries_preserve_frozen_semantics():
    service = _service()
    assert service.daily("2026-07-17").advisory_signal == "PANIC_WAIT"
    assert service.daily("2026-07-23").advisory_signal == "BUY_REFERENCE"
    assert service.daily("2026-07-23").horizon_label.startswith("中期")
    assert service.daily("2026-07-24").advisory_signal == "NEUTRAL"
    assert service.resolve_date("2026-07-18") == pd.Timestamp("2026-07-17").date()
    assert service.resolve_date("2026-07-17", direction=1) == pd.Timestamp("2026-07-20").date()


def test_episode_projection_exposes_lifecycle_without_returns():
    episodes = _service().episodes()
    july = next(item for item in episodes if item.episode_type == "Panic" and item.start_date == "2026-07-13")
    assert july.watch_date == "2026-07-22"
    assert july.confirmed_date == "2026-07-23"
    assert july.end_date == "2026-07-23"
    assert not hasattr(july, "cagr")
    assert not hasattr(july, "sharpe")


def test_core_advisory_hash_did_not_drift():
    service = _service()
    frame = service.advisory.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    columns = [
        "date", "market_temperature", "smoothed_temperature", "state", "state_signal",
        "advisory_signal", "buy_reference", "sell_reference", "risk_level", "expected_horizon",
        "signal_confidence", "research_evidence", "reason_codes",
    ]
    selected = frame[frame["date"].isin(pd.to_datetime(["2026-07-17", "2026-07-23", "2026-08-18"]))][columns]
    payload = selected.sort_values("date").to_json(orient="records", date_format="iso", force_ascii=False)
    assert hashlib.sha256(payload.encode()).hexdigest() == "24bed3938d52d8d4cd7a5005c62eab64c880faefc3174ebf2c75ee67be321263"
