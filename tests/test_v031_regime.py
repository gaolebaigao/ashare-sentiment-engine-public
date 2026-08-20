import numpy as np
import pandas as pd

from ashare_sentiment.regime import (
    EpisodeAnchorConfig,
    StateMachineConfig,
    add_episode_anchor_flags,
    apply_state_machine,
    build_regime_indicators,
)


def _cfg(**overrides):
    values = {field: getattr(StateMachineConfig(), field) for field in StateMachineConfig.__dataclass_fields__}
    values.update(overrides)
    return StateMachineConfig(**values)


def _rows(values, *, invalid=None):
    invalid = set(invalid or [])
    rows = []
    for index, value in enumerate(values):
        if isinstance(value, dict):
            row = dict(value)
        else:
            row = {"raw_temperature": value, "smoothed_temperature": value, "slope3": 0.0}
        row.setdefault("raw_temperature", 50.0)
        row.setdefault("smoothed_temperature", row["raw_temperature"])
        row.setdefault("slope3", 0.0)
        row.setdefault("improving_modules", 0)
        row.setdefault("deteriorating_modules", 0)
        row.setdefault("breadth_delta3", 0.0)
        row.setdefault("profit_effect_delta3", 0.0)
        row.setdefault("liquidity_delta3", 0.0)
        row.setdefault("stretch_delta3", 0.0)
        row.setdefault("rolling_low_10", row["smoothed_temperature"])
        row.setdefault("rolling_high_10", row["smoothed_temperature"])
        row.setdefault("recovery_from_low", 0.0)
        row.setdefault("drop_from_high", 0.0)
        for module in ("breadth", "profit_effect", "liquidity", "stretch"):
            row.setdefault(f"{module}_score", 50.0)
        if index in invalid:
            row["raw_temperature"] = np.nan
            row["smoothed_temperature"] = np.nan
            row["quality"] = "INVALID"
        else:
            row.setdefault("quality", "B")
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.insert(0, "trade_date", pd.date_range("2026-01-05", periods=len(frame), freq="B"))
    frame["confidence"] = frame["quality"].map({"A": "HIGH", "B": "MEDIUM", "C": "LOW", "INVALID": "NONE"})
    return frame


def _with_anchors(frame):
    frame = frame.copy()
    frame["regime_valid_observation"] = frame["quality"].ne("INVALID")
    return add_episode_anchor_flags(frame, EpisodeAnchorConfig())


def test_raw_panic_anchor_survives_ema_above_threshold():
    frame = _rows([{
        "raw_temperature": 15.0,
        "smoothed_temperature": 50.0,
        "slope3": -5.0,
        "breadth_score": 10.0,
        "profit_effect_score": 12.0,
        "liquidity_score": 50.0,
        "stretch_score": 50.0,
    }])
    flags = _with_anchors(frame)
    assert bool(flags.iloc[0]["raw_panic_anchor"])
    result = apply_state_machine(flags, _cfg())
    assert result.iloc[0]["panic_episode_armed"]
    assert result.iloc[0]["state"] == "PANIC_FALLING"


def test_noisy_one_day_raw_low_does_not_arm_without_modules():
    frame = _rows([{
        "raw_temperature": 15.0,
        "smoothed_temperature": 50.0,
        "slope3": -1.0,
        "breadth_score": 35.0,
        "profit_effect_score": 40.0,
        "liquidity_score": 50.0,
        "stretch_score": 50.0,
    }])
    flags = _with_anchors(frame)
    assert not bool(flags.iloc[0]["raw_panic_anchor"])
    result = apply_state_machine(flags, _cfg())
    assert not bool(result.iloc[0]["panic_episode_armed"])
    assert result.iloc[0]["state"] == "NORMAL"


def test_raw_euphoria_anchor_preserves_euphoria_rising():
    frame = _rows([{
        "raw_temperature": 90.0,
        "smoothed_temperature": 78.0,
        "slope3": 5.0,
        "breadth_score": 90.0,
        "profit_effect_score": 85.0,
        "liquidity_score": 50.0,
        "stretch_score": 50.0,
    }])
    flags = _with_anchors(frame)
    assert bool(flags.iloc[0]["raw_euphoria_anchor"])
    result = apply_state_machine(flags, _cfg())
    assert result.iloc[0]["state"] == "EUPHORIA_RISING"
    assert result.iloc[0]["signal"] == "NONE"


def test_hot_rollover_requires_anchored_euphoria_and_deterioration():
    frame = _rows([
        {"raw_temperature": 90.0, "smoothed_temperature": 90.0, "slope3": 5.0, "breadth_score": 90.0, "profit_effect_score": 85.0},
        {"raw_temperature": 78.0, "smoothed_temperature": 84.0, "slope3": -5.0, "deteriorating_modules": 3, "breadth_delta3": -1.0, "profit_effect_delta3": -1.0},
        {"raw_temperature": 75.0, "smoothed_temperature": 80.0, "slope3": -5.0, "deteriorating_modules": 3, "breadth_delta3": -1.0, "profit_effect_delta3": -1.0},
    ])
    result = apply_state_machine(_with_anchors(frame), _cfg())
    assert list(result["state"]) == ["EUPHORIA_RISING", "HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"]
    assert result.iloc[2]["signal"] == "HOT_ROLLOVER_CONFIRMED"


def test_rearm_uses_a_new_episode_after_confirmation():
    frame = _rows([
        {"raw_temperature": 15.0, "smoothed_temperature": 15.0, "slope3": -5.0, "breadth_score": 10.0, "profit_effect_score": 12.0},
        {"raw_temperature": 25.0, "smoothed_temperature": 22.0, "slope3": 5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
        {"raw_temperature": 30.0, "smoothed_temperature": 24.0, "slope3": 5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
        {"raw_temperature": 15.0, "smoothed_temperature": 15.0, "slope3": -5.0, "breadth_score": 10.0, "profit_effect_score": 12.0},
    ])
    result = apply_state_machine(_with_anchors(frame), _cfg())
    assert result.iloc[2]["state"] == "ICE_REVERSAL"
    assert result.iloc[3]["panic_episode_id"] != result.iloc[2]["panic_episode_id"]
    assert result.iloc[3]["state"] == "PANIC_FALLING"


def test_cold_zone_hysteresis_holds_until_exit_band():
    frame = _with_anchors(_rows([
        {"smoothed_temperature": 28.0},
        {"smoothed_temperature": 32.0},
        {"smoothed_temperature": 34.0},
    ]))
    result = apply_state_machine(frame, _cfg(zone_hysteresis=3))
    assert list(result["state"]) == ["COLD", "COLD", "NORMAL"]


def test_ice_confirmation_uses_episode_memory_and_two_valid_days():
    frame = _rows([
        {"raw_temperature": 15.0, "smoothed_temperature": 15.0, "slope3": -5.0, "breadth_score": 10.0, "profit_effect_score": 12.0},
        {"raw_temperature": 25.0, "smoothed_temperature": 22.0, "slope3": 5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
        {"raw_temperature": 30.0, "smoothed_temperature": 24.0, "slope3": 5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
    ])
    result = apply_state_machine(_with_anchors(frame), _cfg())
    assert list(result["state"]) == ["PANIC_FALLING", "ICE_REVERSAL_WATCH", "ICE_REVERSAL"]
    assert result.iloc[2]["signal"] == "ICE_REVERSAL_CONFIRMED"
    assert result.iloc[2]["panic_episode_closed"]


def test_falling_knife_does_not_trigger_ice_reversal():
    frame = _rows([
        {"raw_temperature": 15.0, "smoothed_temperature": 15.0, "slope3": -5.0, "breadth_score": 10.0, "profit_effect_score": 12.0},
        {"raw_temperature": 12.0, "smoothed_temperature": 12.0, "slope3": -5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
    ])
    result = apply_state_machine(_with_anchors(frame), _cfg())
    assert "ICE_REVERSAL" not in result["state"].tolist()
    assert not result["signal"].str.contains("ICE_REVERSAL_CONFIRMED").any()


def test_invalid_day_resets_episode_and_confirmation():
    frame = _rows([
        {"raw_temperature": 15.0, "smoothed_temperature": 15.0, "slope3": -5.0, "breadth_score": 10.0, "profit_effect_score": 12.0},
        {"raw_temperature": 22.0, "smoothed_temperature": 22.0, "slope3": 5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
        {"raw_temperature": 22.0, "smoothed_temperature": 22.0, "slope3": 5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
        {"raw_temperature": 22.0, "smoothed_temperature": 22.0, "slope3": 5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
    ], invalid={2})
    result = apply_state_machine(_with_anchors(frame), _cfg())
    assert result.iloc[1]["state"] == "ICE_REVERSAL_WATCH"
    assert result.iloc[2]["state"] == "DATA_INVALID"
    assert result.iloc[2]["confirmation_streak"] == 0
    assert result.iloc[3]["signal"] == "NONE"
    assert result.iloc[3]["state"] != "ICE_REVERSAL"


def test_watch_timeout_cancels_unconfirmed_episode():
    frame = _rows([
        {"raw_temperature": 15.0, "smoothed_temperature": 15.0, "slope3": -5.0, "breadth_score": 10.0, "profit_effect_score": 12.0},
        {"raw_temperature": 22.0, "smoothed_temperature": 22.0, "slope3": 5.0, "improving_modules": 3, "breadth_delta3": 1.0, "profit_effect_delta3": 1.0},
        {"raw_temperature": 22.0, "smoothed_temperature": 22.0, "slope3": 5.0, "improving_modules": 1},
        {"raw_temperature": 22.0, "smoothed_temperature": 22.0, "slope3": 5.0, "improving_modules": 1},
    ])
    result = apply_state_machine(_with_anchors(frame), _cfg(watch_timeout=2, confirmation_days=3))
    assert result.iloc[1]["state"] == "ICE_REVERSAL_WATCH"
    assert bool(result.iloc[3]["watch_timeout"])


def test_zone_hysteresis_reduces_hot_normal_churn():
    frame = _with_anchors(_rows([
        {"smoothed_temperature": 72.0},
        {"smoothed_temperature": 68.0},
        {"smoothed_temperature": 66.0},
    ]))
    result = apply_state_machine(frame, _cfg(zone_hysteresis=3))
    assert list(result["state"]) == ["HOT", "HOT", "NORMAL"]


def test_episode_memory_expires_without_new_anchor():
    frame = _with_anchors(_rows([
        {"raw_temperature": 15.0, "smoothed_temperature": 15.0, "slope3": -5.0, "breadth_score": 10.0, "profit_effect_score": 12.0},
        {"smoothed_temperature": 30.0},
        {"smoothed_temperature": 30.0},
    ]))
    mapping = {"state_machine": {field: getattr(_cfg(), field) for field in _cfg().__dataclass_fields__}, "episode_anchor": {"memory_window": 2}}
    result = apply_state_machine(frame, mapping)
    assert bool(result.iloc[1]["panic_episode_armed"])
    assert not bool(result.iloc[2]["panic_episode_armed"])


def test_no_future_data_changes_episode_and_state_history():
    prefix = _rows([50, 55, 60, 65, 60, 55])
    suffix = _rows([10])
    suffix["trade_date"] = pd.Timestamp("2026-01-13")
    full = pd.concat([prefix, suffix], ignore_index=True)
    cfg = _cfg(ema_span=3)
    mapping = {"state_machine": {field: getattr(cfg, field) for field in cfg.__dataclass_fields__}, "episode_anchor": {}}
    left = apply_state_machine(build_regime_indicators(prefix, mapping), mapping)
    right = apply_state_machine(build_regime_indicators(full, mapping), mapping).iloc[: len(left)]
    columns = ["smoothed_temperature", "slope3", "rolling_high_10", "rolling_low_10", "panic_episode_id", "state", "signal"]
    pd.testing.assert_frame_equal(left[columns].reset_index(drop=True), right[columns].reset_index(drop=True), check_dtype=False)
