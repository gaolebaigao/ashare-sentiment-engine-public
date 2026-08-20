from __future__ import annotations

import pandas as pd

from ashare_sentiment.advisory import build_advisory, build_advisory_frame, no_future_data_stability


def _row(state: str, *, date: str = "2026-07-23", raw: float = 50.0, smooth: float = 45.0, slope: float = 0.0, signal: str = "NONE", quality: str = "A") -> pd.Series:
    row = {
        "trade_date": pd.Timestamp(date),
        "date": pd.Timestamp(date),
        "state": state,
        "signal": signal,
        "raw_temperature": raw,
        "smoothed_temperature": smooth,
        "slope3": slope,
        "quality": quality,
        "breadth_score": 50.0,
        "profit_effect_score": 50.0,
        "liquidity_score": 50.0,
        "stretch_score": 50.0,
        "breadth_delta3": 0.0,
        "profit_effect_delta3": 0.0,
        "liquidity_delta3": 0.0,
        "stretch_delta3": 0.0,
        "state_duration": 1,
    }
    return pd.Series(row)


def test_panic_falling_is_wait_not_buy_reference() -> None:
    row = _row("PANIC_FALLING", raw=15, smooth=18, slope=-5)
    result = build_advisory(row)
    assert result.advisory_signal == "PANIC_WAIT"
    assert result.buy_reference == "LOW"
    assert "仍在恶化" in result.headline


def test_ice_watch_is_buy_watch_with_medium_reference() -> None:
    row = _row("ICE_REVERSAL_WATCH", raw=28, smooth=24, slope=5, signal="ICE_REVERSAL_WATCH")
    row["breadth_delta3"] = 3
    row["profit_effect_delta3"] = 2
    row["liquidity_delta3"] = 1
    result = build_advisory(row)
    assert result.advisory_signal == "BUY_WATCH"
    assert result.buy_reference == "MEDIUM"


def test_ice_confirmed_is_medium_term_and_warns_short_term() -> None:
    row = _row("ICE_REVERSAL", raw=45, smooth=35, slope=6, signal="ICE_REVERSAL_CONFIRMED")
    row["breadth_delta3"] = 3
    row["profit_effect_delta3"] = 3
    result = build_advisory(row, context={"state_days": 1, "ice_days": 1, "panic_was_extreme": True})
    assert result.advisory_signal == "BUY_REFERENCE"
    assert result.expected_horizon == "MEDIUM_TERM"
    assert result.reference_horizon == "20–60 trading observations"
    assert result.short_term_warning is True
    assert result.research_evidence == "MODERATE"
    assert "不是精确底部" in result.why
    assert "未来 1–5 个交易日上涨" in result.why


def test_euphoria_is_caution_not_sell_reference() -> None:
    result = build_advisory(_row("EUPHORIA_RISING", raw=90, smooth=86, slope=5))
    assert result.advisory_signal == "HOT_CAUTION"
    assert "不是卖出确认" in result.headline or "不是卖出确认" in result.why


def test_hot_watch_and_confirmed_are_distinct_sell_references() -> None:
    watch = _row("HOT_ROLLOVER_WATCH", raw=78, smooth=82, slope=-5, signal="HOT_ROLLOVER_WATCH")
    watch["breadth_delta3"] = -3
    watch["profit_effect_delta3"] = -3
    watch_result = build_advisory(watch)
    assert watch_result.advisory_signal == "SELL_WATCH"
    assert watch_result.sell_reference == "MEDIUM"

    confirmed = _row("HOT_ROLLOVER", raw=72, smooth=76, slope=-5, signal="HOT_ROLLOVER_CONFIRMED")
    confirmed["breadth_delta3"] = -3
    confirmed["profit_effect_delta3"] = -3
    confirmed_result = build_advisory(confirmed, context={"state_days": 1, "hot_days": 1, "euphoria_was_extreme": True})
    assert confirmed_result.advisory_signal == "SELL_REFERENCE"
    assert confirmed_result.expected_horizon == "MEDIUM_TERM_RISK"
    assert confirmed_result.reference_horizon == "40–60 trading observations"
    assert confirmed_result.not_exact_top is True
    assert "中期风险" in confirmed_result.why
    assert "精确顶部" in confirmed_result.why


def test_advisory_history_is_stable_when_future_rows_are_appended() -> None:
    prefix = pd.DataFrame([
        _row("PANIC_FALLING", date="2026-07-17", raw=17, smooth=27, slope=-5),
        _row("ICE_REVERSAL_WATCH", date="2026-07-22", raw=31, smooth=32, slope=5, signal="ICE_REVERSAL_WATCH"),
        _row("ICE_REVERSAL", date="2026-07-23", raw=55, smooth=43, slope=8, signal="ICE_REVERSAL_CONFIRMED"),
    ])
    suffix = _row("PANIC_FALLING", date="2026-07-24", raw=15, smooth=20, slope=-8)
    full = pd.concat([prefix, pd.DataFrame([suffix])], ignore_index=True)
    assert no_future_data_stability(full, split=len(prefix))

    output = build_advisory_frame(full)
    assert output.iloc[2]["advisory_signal"] == "BUY_REFERENCE"
    assert "future" not in " ".join(output.columns).lower()


def test_data_invalid_is_a_hard_advisory_gate() -> None:
    row = _row("DATA_INVALID", raw=float("nan"), smooth=float("nan"), quality="INVALID")
    result = build_advisory(row)
    assert result.advisory_signal == "DATA_INVALID"
    assert result.reason_codes == ("DATA_INVALID",)
    assert "无效" in result.headline


def test_one_day_one_top_level_advisory_schema() -> None:
    frame = build_advisory_frame(pd.DataFrame([
        _row("ICE_REVERSAL_WATCH", date="2026-07-22", raw=30, smooth=32, slope=5),
        _row("HOT_ROLLOVER_WATCH", date="2026-07-23", raw=75, smooth=78, slope=-5),
    ]))
    assert "advisory_level" not in frame.columns
    assert set(frame["advisory_signal"]).issubset({
        "PANIC_WAIT", "BUY_WATCH", "BUY_REFERENCE", "NEUTRAL",
        "HOT_CAUTION", "SELL_WATCH", "SELL_REFERENCE", "DATA_INVALID",
    })


def test_nonfatal_warnings_are_concrete_not_generic() -> None:
    row = _row("HOT", raw=63, smooth=68, slope=-1, quality="B")
    row["data_quality_status"] = "WARN"
    row["data_quality_warnings"] = "SURVIVORSHIP_BIAS_WARNING;OPTIONS_UNAVAILABLE;MARGIN_OPTIONAL"
    result = build_advisory(row)
    assert "DATA_QUALITY_WARNING" not in result.reason_codes
    assert "SURVIVORSHIP_BIAS_WARNING" in result.reason_codes
    assert "OPTIONS_UNAVAILABLE" in result.reason_codes
    assert "MARGIN_OPTIONAL" in result.reason_codes
