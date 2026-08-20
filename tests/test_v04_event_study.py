"""Focused tests for the MarketTemperature v0.4 event-study contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_sentiment.event_study.events import decluster_events
from ashare_sentiment.event_study.forward_returns import outcome_for_date
from ashare_sentiment.event_study.statistics import bootstrap_ci


def _prices(rows: int = 25) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    close = np.arange(100.0, 100.0 + rows)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close + 0.5,
            "close": close,
            "high": close + 2.0,
            "low": close - 2.0,
        }
    )


def test_next_open_forward_return_does_not_use_signal_close() -> None:
    frame = _prices(5)
    frame.loc[0, "close"] = 100.0
    frame.loc[1, "open"] = 110.0
    frame.loc[1, "close"] = 111.0
    result = outcome_for_date(frame, frame.loc[0, "trade_date"], horizons=(1,))
    assert result["return_1d"] == (111.0 / 110.0) - 1.0
    assert result["signal_close_forward_return_1d"] == (111.0 / 100.0) - 1.0
    assert result["return_1d"] != result["signal_close_forward_return_1d"]


def test_mfe_mae_and_future_drawdown_use_future_high_low() -> None:
    frame = _prices(8)
    frame.loc[1, "open"] = 100.0
    frame.loc[1:3, "high"] = [110.0, 108.0, 112.0]
    frame.loc[1:3, "low"] = [95.0, 96.0, 90.0]
    result = outcome_for_date(frame, frame.loc[0, "trade_date"], horizons=(3,))
    assert np.isclose(result["mfe_3d"], 0.12)
    assert np.isclose(result["mae_3d"], -0.10)
    assert np.isclose(result["future_drawdown_3d"], -0.10)


def test_truncated_horizon_is_not_padded() -> None:
    frame = _prices(4)
    result = outcome_for_date(frame, frame.loc[2, "trade_date"], horizons=(1, 2))
    assert result["price_status"] == "TRUNCATED_HORIZON"
    assert np.isfinite(result["return_1d"])
    assert np.isnan(result["return_2d"])


def test_declustering_keeps_first_same_direction_event_after_window() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "event_type": ["ICE_REVERSAL_CONFIRMED"] * 3,
            "direction": ["ICE"] * 3,
            "episode_id": ["1", "2", "3"],
            "date_position": [0, 20, 41],
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
        }
    )
    result = decluster_events(events, mode="declustered", window=20)
    assert result["event_id"].tolist() == ["a", "c"]


def test_episode_level_keeps_first_event_in_episode() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "event_type": ["ICE_REVERSAL_CONFIRMED"] * 3,
            "direction": ["ICE"] * 3,
            "episode_id": ["1", "1", "2"],
            "date_position": [0, 1, 2],
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
        }
    )
    result = decluster_events(events, mode="episode")
    assert result["event_id"].tolist() == ["a", "c"]


def test_bootstrap_ci_is_reproducible() -> None:
    values = [0.01, 0.02, -0.01, 0.04, 0.00]
    left = bootstrap_ci(values, samples=1000, seed=42)
    right = bootstrap_ci(values, samples=1000, seed=42)
    assert left == right
