"""Causal Advisory Layer on top of the frozen MarketTemperature v0.3.1 state."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..regime.models import EpisodeAnchorConfig, StateMachineConfig
from .confidence import calculate_signal_confidence, module_confirmation, research_evidence_for
from .explanations import build_explanation
from .models import MarketAdvisory


def _number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    if pd.isna(value) and column == "raw_temperature":
        value = pd.to_numeric(row.get("market_temperature", row.get("temperature")), errors="coerce")
    if pd.isna(value) and column == "smoothed_temperature":
        value = pd.to_numeric(row.get("ema_temperature"), errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def _as_date(value: Any) -> date | str:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if hasattr(value, "date") and not isinstance(value, date):
        return value.date()
    return value


def _advisory_signal(state: str) -> str:
    return {
        "PANIC_FALLING": "PANIC_WAIT",
        "EXTREME_PANIC": "BUY_WATCH",
        "ICE_REVERSAL_WATCH": "BUY_WATCH",
        "ICE_REVERSAL": "BUY_REFERENCE",
        "COLD": "NEUTRAL",
        "NORMAL": "NEUTRAL",
        "HOT": "HOT_CAUTION",
        "EUPHORIA_RISING": "HOT_CAUTION",
        "HOT_ROLLOVER_WATCH": "SELL_WATCH",
        "HOT_ROLLOVER": "SELL_REFERENCE",
        "DATA_INVALID": "DATA_INVALID",
    }.get(state, "DATA_INVALID")


def _regime(state: str) -> str:
    if state in {"PANIC_FALLING", "EXTREME_PANIC", "COLD"}:
        return "PANIC"
    if state in {"ICE_REVERSAL_WATCH", "ICE_REVERSAL"}:
        return "RECOVERY"
    if state in {"HOT", "EUPHORIA_RISING"}:
        return "EUPHORIA"
    if state in {"HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"}:
        return "RISK_OFF"
    return "NORMAL"


def _references(state: str, *, strong: bool) -> tuple[str, str]:
    if state in {"EXTREME_PANIC", "ICE_REVERSAL_WATCH", "ICE_REVERSAL"}:
        return ("HIGH" if strong else "MEDIUM") if state != "EXTREME_PANIC" else "LOW", "LOW"
    if state in {"HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"}:
        return "LOW", "HIGH" if strong else "MEDIUM"
    return "LOW", "LOW"


def _risk(state: str) -> str:
    return {
        "PANIC_FALLING": "HIGH",
        "EXTREME_PANIC": "EXTREME",
        "ICE_REVERSAL_WATCH": "ELEVATED",
        "ICE_REVERSAL": "ELEVATED",
        "COLD": "ELEVATED",
        "NORMAL": "NORMAL",
        "HOT": "ELEVATED",
        "EUPHORIA_RISING": "HIGH",
        "HOT_ROLLOVER_WATCH": "HIGH",
        "HOT_ROLLOVER": "HIGH",
        "DATA_INVALID": "HIGH",
    }.get(state, "NORMAL")


def _horizon(state: str) -> str:
    if state in {"ICE_REVERSAL_WATCH", "ICE_REVERSAL"}:
        return "MEDIUM_TERM"
    if state in {"HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"}:
        return "MEDIUM_TERM_RISK"
    if state == "PANIC_FALLING":
        return "SHORT_TERM_UNCERTAINTY"
    if state == "EXTREME_PANIC":
        return "SHORT_TERM_UNCERTAINTY"
    return "NOT_APPLICABLE"


def _reference_horizon(state: str) -> str:
    if state in {"ICE_REVERSAL_WATCH", "ICE_REVERSAL"}:
        return "20–60 trading observations"
    if state in {"HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"}:
        return "40–60 trading observations"
    return "N/A"


def _reason_codes(
    row: pd.Series,
    *,
    state: str,
    signal: str,
    anchor_cfg: EpisodeAnchorConfig,
) -> tuple[str, ...]:
    codes: list[str] = []
    raw = _number(row, "raw_temperature")
    smooth = _number(row, "smoothed_temperature")
    slope = _number(row, "slope3")
    if state == "DATA_INVALID":
        return ("DATA_INVALID",)
    if bool(row.get("recent_panic_episode", False)) or state in {"PANIC_FALLING", "EXTREME_PANIC", "ICE_REVERSAL_WATCH", "ICE_REVERSAL"}:
        codes.append("PANIC_EPISODE_ACTIVE" if bool(row.get("recent_panic_episode", False)) else "PANIC_EPISODE_RECENT")
    if bool(row.get("recent_euphoria_episode", False)) or state in {"EUPHORIA_RISING", "HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"}:
        codes.append("EUPHORIA_EPISODE_ACTIVE" if bool(row.get("recent_euphoria_episode", False)) else "EUPHORIA_EPISODE_RECENT")
    if np.isfinite(raw) and raw <= anchor_cfg.raw_panic_threshold:
        codes.append("RAW_TEMPERATURE_EXTREME_LOW")
    if np.isfinite(raw) and raw >= anchor_cfg.raw_euphoria_threshold:
        codes.append("RAW_TEMPERATURE_EXTREME_HIGH")
    if np.isfinite(smooth) and smooth <= anchor_cfg.smoothed_panic_threshold:
        codes.append("EMA_EXTREME_LOW")
    if np.isfinite(smooth) and smooth >= anchor_cfg.smoothed_euphoria_threshold:
        codes.append("EMA_EXTREME_HIGH")
    for name, label in (("breadth", "BREADTH"), ("profit_effect", "PROFIT_EFFECT"), ("liquidity", "LIQUIDITY"), ("stretch", "STRETCH")):
        score = _number(row, f"{name}_score")
        delta = _number(row, f"{name}_delta3")
        if np.isfinite(score) and score <= 20:
            codes.append(f"{label}_EXTREME_LOW")
        elif np.isfinite(score) and score >= 80:
            codes.append(f"{label}_EXTREME_HIGH")
        if np.isfinite(delta) and delta > 0:
            codes.append(f"{label}_RECOVERING")
        elif np.isfinite(delta) and delta < 0:
            codes.append(f"{label}_DETERIORATING")
    if np.isfinite(slope) and slope > 0:
        codes.append("EMA_RECOVERING")
    elif np.isfinite(slope) and slope < 0:
        codes.append("EMA_DETERIORATING")
    if signal == "BUY_WATCH" and state == "ICE_REVERSAL_WATCH":
        codes.append("ICE_WATCH")
    if signal == "BUY_REFERENCE" and state == "ICE_REVERSAL":
        codes.append("ICE_CONFIRMED")
    if signal == "SELL_WATCH":
        codes.append("HOT_WATCH")
    if signal == "SELL_REFERENCE":
        codes.append("HOT_CONFIRMED")
    codes.extend(_specific_warning_codes(row))
    # Stable order, with duplicate prevention, makes audits diff-friendly.
    return tuple(dict.fromkeys(codes))


def _specific_warning_codes(row: pd.Series) -> list[str]:
    """Translate existing pipeline warnings into concrete, auditable codes.

    The cached v0.2.1 pipeline intentionally reports recurring non-fatal
    limitations such as survivorship bias and unavailable optional factors.
    They are not collapsed into a daily generic DATA_QUALITY_WARNING.
    """

    raw_values: list[str] = []
    for column in ("data_quality_warnings", "integrity_warnings", "warnings"):
        value = row.get(column)
        if value is not None and not pd.isna(value):
            raw_values.extend(str(value).replace(",", ";").split(";"))
    known = {
        "SURVIVORSHIP_BIAS_WARNING",
        "OPTIONS_UNAVAILABLE",
        "MARGIN_OPTIONAL",
        "FAILED_LIMIT_RATE_UNAVAILABLE",
        "HISTORICAL_DELIST_DATE_INCOMPLETE",
        "LIMIT_STATUS_APPROXIMATE_BAOSTOCK_BOARD_BANDS",
        "PARTIAL_LIMIT_DATA",
        "LOW_UNIVERSE_COVERAGE_WARNING",
        "STALE_OPTIONAL_SERIES",
        "MISSING_OPTIONAL_FIELD",
    }
    result: list[str] = []
    for value in raw_values:
        code = value.strip().upper()
        if code in known:
            result.append(code)
    missing = str(row.get("market_temperature_missing_factors", row.get("missing_factors", ""))).lower()
    if "options_score" in missing and "OPTIONS_UNAVAILABLE" not in result:
        result.append("OPTIONS_UNAVAILABLE")
    if "margin_buy_ratio" in missing and "MARGIN_OPTIONAL" not in result:
        result.append("MARGIN_OPTIONAL")
    if "failed_limit_rate" in missing and "FAILED_LIMIT_RATE_UNAVAILABLE" not in result:
        result.append("FAILED_LIMIT_RATE_UNAVAILABLE")
    return result


def _strong_confirmation(state: str, row: pd.Series, context: dict[str, Any]) -> tuple[bool, int, tuple[str, ...]]:
    """Apply the deliberately hard-to-trigger extra advisory level."""

    if state == "ICE_REVERSAL":
        modules, names = module_confirmation(row, "ice")
        both_recover = _number(row, "breadth_delta3") > 0 and _number(row, "profit_effect_delta3") > 0
        strong = (
            context.get("ice_days", 0) >= 3
            and context.get("panic_was_extreme", False)
            and both_recover
            and _number(row, "slope3") > 0
            and modules >= 3
        )
        return strong, modules, names
    if state == "HOT_ROLLOVER":
        modules, names = module_confirmation(row, "hot")
        both_deteriorate = _number(row, "breadth_delta3") < 0 and _number(row, "profit_effect_delta3") < 0
        strong = (
            context.get("hot_days", 0) >= 3
            and context.get("euphoria_was_extreme", False)
            and both_deteriorate
            and _number(row, "slope3") < 0
            and modules >= 3
        )
        return strong, modules, names
    direction = "ice" if state in {"PANIC_FALLING", "EXTREME_PANIC", "ICE_REVERSAL_WATCH"} else "hot"
    return False, *module_confirmation(row, direction)


def build_advisory(row: pd.Series, *, context: dict[str, Any] | None = None, config: dict[str, Any] | StateMachineConfig | None = None) -> MarketAdvisory:
    """Build one advisory from the current row and prior-only context."""

    state = str(row.get("state", "DATA_INVALID"))
    state_context = context if context is not None else {}
    anchor_cfg = config if isinstance(config, EpisodeAnchorConfig) else EpisodeAnchorConfig.from_mapping(config if isinstance(config, dict) else {})
    signal = _advisory_signal(state)
    strong, confirming, module_names = _strong_confirmation(state, row, state_context)
    if confirming == 0 and state not in {"NORMAL", "COLD", "HOT", "EUPHORIA_RISING", "DATA_INVALID"}:
        direction = "ice" if "ICE" in state or "PANIC" in state else "hot"
        confirming, module_names = module_confirmation(row, direction)
    buy_reference, sell_reference = _references(state, strong=strong)
    evidence = research_evidence_for(state, signal)
    confidence = calculate_signal_confidence(
        row,
        confirming_modules=confirming,
        state_persistence=state_context.get("state_days", row.get("state_duration", 1)),
    )
    regime = _regime(state)
    short_warning = state in {"ICE_REVERSAL_WATCH", "ICE_REVERSAL"}
    not_exact_top = state in {"HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"}
    codes = _reason_codes(row, state=state, signal=signal, anchor_cfg=anchor_cfg)
    explanation = build_explanation(
        row,
        advisory_signal=signal,
        advisory_regime=regime,
        short_term_warning=short_warning,
        not_exact_top=not_exact_top,
    )
    return MarketAdvisory(
        date=_as_date(row.get("date", row.get("trade_date"))),
        market_temperature=_number(row, "raw_temperature") if np.isfinite(_number(row, "raw_temperature")) else None,
        smoothed_temperature=_number(row, "smoothed_temperature") if np.isfinite(_number(row, "smoothed_temperature")) else None,
        state=state,
        state_signal=str(row.get("signal", "NONE")),
        advisory_signal=signal,
        buy_reference=buy_reference,
        sell_reference=sell_reference,
        risk_level=_risk(state),
        expected_horizon=_horizon(state),
        reference_horizon=_reference_horizon(state),
        signal_confidence=confidence,
        research_evidence=evidence["research_evidence"],
        confirming_modules=confirming,
        reason_codes=codes,
        headline=explanation["headline"],
        summary=explanation["summary"],
        why=explanation["why"],
        what_to_watch_next=explanation["what_to_watch_next"],
        details=explanation["details"],
        advisory_regime=regime,
        short_term_evidence=evidence["short_term_evidence"],
        medium_term_evidence=evidence["medium_term_evidence"],
        confirming_module_names=module_names,
        short_term_warning=short_warning,
        not_exact_top=not_exact_top,
        active_advisory_window=state in {"ICE_REVERSAL", "HOT_ROLLOVER"},
    )


def build_advisory_frame(
    state_frame: pd.DataFrame,
    config: dict[str, Any] | StateMachineConfig | None = None,
) -> pd.DataFrame:
    """Build a complete daily advisory table causally, in date order.

    All derived fields use the current row and an in-memory context of prior
    valid rows. Appending later rows therefore cannot change earlier output.
    """

    if state_frame.empty:
        return pd.DataFrame()
    frame = state_frame.copy()
    if "trade_date" not in frame.columns and "date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    context: dict[str, Any] = {
        "state_days": 0,
        "previous_state": None,
        "ice_days": 0,
        "hot_days": 0,
        "panic_was_extreme": False,
        "euphoria_was_extreme": False,
    }
    for _, row in frame.iterrows():
        state = str(row.get("state", "DATA_INVALID"))
        valid = state != "DATA_INVALID" and np.isfinite(_number(row, "raw_temperature")) and np.isfinite(_number(row, "smoothed_temperature"))
        if not valid:
            context.update({"state_days": 0, "previous_state": state, "ice_days": 0, "hot_days": 0, "panic_was_extreme": False, "euphoria_was_extreme": False})
        else:
            context["state_days"] = context["state_days"] + 1 if context.get("previous_state") == state else 1
            context["ice_days"] = context["ice_days"] + 1 if state == "ICE_REVERSAL" else 0
            context["hot_days"] = context["hot_days"] + 1 if state == "HOT_ROLLOVER" else 0
            context["panic_was_extreme"] = bool(context.get("panic_was_extreme")) or bool(row.get("recent_panic_episode", False)) or _number(row, "raw_temperature") <= 20 or _number(row, "smoothed_temperature") <= 20
            context["euphoria_was_extreme"] = bool(context.get("euphoria_was_extreme")) or bool(row.get("recent_euphoria_episode", False)) or _number(row, "raw_temperature") >= 85 or _number(row, "smoothed_temperature") >= 80
        advisory = build_advisory(row, context=context, config=config)
        records.append(advisory.to_record())
        context["previous_state"] = state
        if state != "ICE_REVERSAL":
            context["panic_was_extreme"] = context.get("panic_was_extreme", False) if state in {"PANIC_FALLING", "EXTREME_PANIC", "ICE_REVERSAL_WATCH"} else False
        if state != "HOT_ROLLOVER":
            context["euphoria_was_extreme"] = context.get("euphoria_was_extreme", False) if state in {"HOT", "EUPHORIA_RISING", "HOT_ROLLOVER_WATCH"} else False
    output = pd.DataFrame(records)
    output["trade_date"] = pd.to_datetime(output["date"], errors="coerce")
    return output


def apply_advisory_layer(state_frame: pd.DataFrame, config: dict[str, Any] | StateMachineConfig | None = None) -> pd.DataFrame:
    """Public alias emphasizing that this layer does not mutate the state machine."""

    return build_advisory_frame(state_frame, config=config)
