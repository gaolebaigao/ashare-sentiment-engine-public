import numpy as np
import pandas as pd

from ashare_sentiment.regime import (
    StateMachineConfig,
    apply_state_machine,
    build_regime_indicators,
    causal_valid_ema,
    transition_matrix,
    valid_lag,
    valid_rolling_extreme,
)


def _cfg(**overrides):
    base = StateMachineConfig()
    values = {field: getattr(base, field) for field in base.__dataclass_fields__}
    values.update(overrides)
    return StateMachineConfig(**values)


def _indicators(rows, *, quality=None):
    dates = pd.date_range("2026-01-05", periods=len(rows), freq="B")
    output = pd.DataFrame(rows)
    output.insert(0, "trade_date", dates)
    output["quality"] = quality or "B"
    output["confidence"] = output["quality"].map({"A": "HIGH", "B": "MEDIUM", "C": "LOW", "INVALID": "NONE"})
    for column, default in {
        "raw_temperature": 50.0,
        "smoothed_temperature": 50.0,
        "slope3": 0.0,
        "rolling_low_10": 50.0,
        "rolling_high_10": 50.0,
        "recovery_from_low": 0.0,
        "drop_from_high": 0.0,
        "improving_modules": 0,
        "deteriorating_modules": 0,
        "breadth_delta3": 0.0,
        "profit_effect_delta3": 0.0,
        "liquidity_delta3": 0.0,
        "stretch_delta3": 0.0,
        "breadth_score": 50.0,
        "profit_effect_score": 50.0,
        "liquidity_score": 50.0,
        "stretch_score": 50.0,
    }.items():
        if column not in output:
            output[column] = default
    return output


def _ice_rows(count=2):
    return _indicators([
        {
            "raw_temperature": 18.0,
            "smoothed_temperature": 15.0,
            "slope3": 5.0,
            "rolling_low_10": 10.0,
            "rolling_high_10": 45.0,
            "recovery_from_low": 5.0,
            "drop_from_high": 30.0,
            "improving_modules": 3,
            "breadth_delta3": 1.0,
        }
        for _ in range(count)
    ])


def _hot_rows(count=2):
    return _indicators([
        {
            "raw_temperature": 78.0,
            "smoothed_temperature": 78.0,
            "slope3": -5.0,
            "rolling_low_10": 45.0,
            "rolling_high_10": 88.0,
            "recovery_from_low": 33.0,
            "drop_from_high": 10.0,
            "deteriorating_modules": 3,
            "breadth_delta3": -1.0,
        }
        for _ in range(count)
    ])


def test_ema_causal_and_invalid_rows_are_not_filled():
    values = pd.Series([10.0, 12.0, np.nan, 16.0])
    result = causal_valid_ema(values, span=2)
    assert result.iloc[0] == 10.0
    assert result.iloc[1] == 11.333333333333334
    assert pd.isna(result.iloc[2])
    assert result.iloc[3] == 14.444444444444445


def test_slope3_uses_valid_observations():
    values = pd.Series([10.0, 12.0, np.nan, 16.0, 18.0])
    lag = valid_lag(values, 3)
    assert pd.isna(lag.iloc[0]) and pd.isna(lag.iloc[2])
    assert lag.iloc[4] == 10.0


def test_rolling_high_low_use_valid_observations():
    values = pd.Series([10.0, np.nan, 12.0, 8.0, 14.0])
    high = valid_rolling_extreme(values, 3, maximum=True)
    low = valid_rolling_extreme(values, 3, maximum=False)
    assert pd.isna(high.iloc[1])
    assert high.iloc[4] == 14.0
    assert low.iloc[4] == 8.0


def test_temperature_recovery_from_low_and_drop_from_high():
    source = pd.DataFrame({
        "trade_date": pd.date_range("2026-01-05", periods=4, freq="B"),
        "raw_temperature": [10.0, 20.0, 30.0, 25.0],
        "market_temperature_quality": ["B"] * 4,
        "breadth_score": [50.0] * 4,
        "profit_effect_score": [50.0] * 4,
        "liquidity_score": [50.0] * 4,
        "stretch_score": [50.0] * 4,
    })
    result = build_regime_indicators(source, _cfg(ema_span=1, turning_window=3))
    assert result.iloc[2]["recovery_from_low"] == 20.0
    assert result.iloc[3]["drop_from_high"] == 5.0


def test_panic_falling_is_not_ice_reversal():
    result = apply_state_machine(_indicators([{"smoothed_temperature": 18.0, "slope3": -1.0}]), _cfg())
    assert result.iloc[0]["state"] == "PANIC_FALLING"
    assert result.iloc[0]["signal"] == "NONE"


def test_extreme_panic_requires_no_clear_downward_slope():
    result = apply_state_machine(_indicators([{"smoothed_temperature": 18.0, "slope3": 0.0}]), _cfg())
    assert result.iloc[0]["state"] == "EXTREME_PANIC"


def test_ice_reversal_watch_and_confirmation_require_modules():
    result = apply_state_machine(_ice_rows(), _cfg())
    assert result.iloc[0]["state"] == "ICE_REVERSAL_WATCH"
    assert result.iloc[0]["signal"] == "ICE_REVERSAL_WATCH"
    assert result.iloc[1]["state"] == "ICE_REVERSAL"
    assert result.iloc[1]["signal"] == "ICE_REVERSAL_CONFIRMED"


def test_ice_reversal_is_invalidated_when_slope_turns_down():
    frame = pd.concat([_ice_rows(1), _indicators([{
        "raw_temperature": 17.0,
        "smoothed_temperature": 14.0,
        "slope3": -1.0,
        "rolling_low_10": 10.0,
        "recovery_from_low": 4.0,
    }])], ignore_index=True)
    result = apply_state_machine(frame, _cfg())
    assert result.iloc[0]["state"] == "ICE_REVERSAL_WATCH"
    assert result.iloc[1]["state"] == "PANIC_FALLING"
    assert bool(result.iloc[1]["watch_invalidated"])


def test_euphoria_rising_is_not_hot_rollover():
    result = apply_state_machine(_indicators([{"smoothed_temperature": 84.0, "slope3": 5.0}]), _cfg())
    assert result.iloc[0]["state"] == "EUPHORIA_RISING"
    assert result.iloc[0]["signal"] == "NONE"


def test_hot_rollover_watch_and_confirmation_require_deterioration():
    result = apply_state_machine(_hot_rows(), _cfg())
    assert result.iloc[0]["state"] == "HOT_ROLLOVER_WATCH"
    assert result.iloc[1]["state"] == "HOT_ROLLOVER"
    assert result.iloc[1]["signal"] == "HOT_ROLLOVER_CONFIRMED"


def test_watch_timeout_cancels_soft_watch():
    frame = pd.concat([
        _ice_rows(1),
        _indicators([{
            "raw_temperature": 18.0,
            "smoothed_temperature": 18.0,
            "slope3": 5.0,
            "rolling_low_10": 10.0,
            "recovery_from_low": 1.0,
            "improving_modules": 1,
        }]),
        _indicators([{
            "raw_temperature": 18.0,
            "smoothed_temperature": 18.0,
            "slope3": 5.0,
            "rolling_low_10": 10.0,
            "recovery_from_low": 1.0,
            "improving_modules": 1,
        }]),
    ], ignore_index=True)
    result = apply_state_machine(frame, _cfg(watch_timeout=2, confirmation_days=3))
    assert list(result["state"]) == ["ICE_REVERSAL_WATCH", "ICE_REVERSAL_WATCH", "EXTREME_PANIC"]
    assert bool(result.iloc[2]["watch_timeout"])


def test_invalid_day_is_data_invalid_and_resets_confirmation():
    invalid = _indicators([{"raw_temperature": np.nan, "smoothed_temperature": np.nan}], quality="INVALID")
    frame = pd.concat([_ice_rows(1), invalid, _ice_rows(1)], ignore_index=True)
    result = apply_state_machine(frame, _cfg())
    assert result.iloc[1]["state"] == "DATA_INVALID"
    assert result.iloc[1]["signal"] == "NONE"
    assert result.iloc[1]["confirmation_streak"] == 0
    assert result.iloc[2]["state"] == "ICE_REVERSAL_WATCH"
    assert result.iloc[2]["signal"] != "ICE_REVERSAL_CONFIRMED"


def test_state_priority_prefers_turning_point_watch():
    result = apply_state_machine(_ice_rows(1), _cfg())
    assert result.iloc[0]["state"] == "ICE_REVERSAL_WATCH"


def test_state_duration_and_transition_matrix_are_deterministic():
    frame = pd.concat([
        _indicators([{"smoothed_temperature": 50.0}] * 2),
        _indicators([{"smoothed_temperature": 75.0}] * 2),
    ], ignore_index=True)
    frame["trade_date"] = pd.date_range("2026-01-05", periods=4, freq="B")
    first = apply_state_machine(frame, _cfg())
    second = apply_state_machine(frame, _cfg())
    pd.testing.assert_frame_equal(first, second)
    assert list(first["state_duration"]) == [1, 2, 1, 2]
    matrix = transition_matrix(first)
    assert matrix.loc["NONE", "NORMAL"] == 1
    assert matrix.loc["NORMAL", "HOT"] == 1


def test_no_future_data_changes_historical_indicators_or_state():
    prefix = pd.DataFrame({
        "trade_date": pd.date_range("2026-01-05", periods=6, freq="B"),
        "raw_temperature": [50, 55, 60, 65, 60, 55],
        "market_temperature_quality": ["B"] * 6,
        "breadth_score": [50] * 6,
        "profit_effect_score": [50] * 6,
        "liquidity_score": [50] * 6,
        "stretch_score": [50] * 6,
    })
    suffix = prefix.iloc[[-1]].copy()
    suffix["trade_date"] = pd.Timestamp("2026-01-13")
    suffix["raw_temperature"] = 10
    full = pd.concat([prefix, suffix], ignore_index=True)
    cfg = _cfg(ema_span=3)
    historical = apply_state_machine(build_regime_indicators(prefix, cfg), cfg)
    appended = apply_state_machine(build_regime_indicators(full, cfg), cfg)
    columns = ["smoothed_temperature", "slope1", "slope3", "rolling_high_10", "rolling_low_10", "state", "signal"]
    pd.testing.assert_frame_equal(historical[columns], appended.iloc[: len(prefix)][columns], check_dtype=False)
