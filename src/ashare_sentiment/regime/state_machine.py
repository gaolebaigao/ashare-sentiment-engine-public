"""Deterministic state machine for MarketTemperature v0.3.1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import EpisodeAnchorConfig, STATES, StateMachineConfig


def _number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def _positive(value: float) -> bool:
    return np.isfinite(value) and value > 0


def _negative(value: float) -> bool:
    return np.isfinite(value) and value < 0


def _flag(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    if pd.isna(value):
        return False
    return bool(value)


def _episode_is_available(row: pd.Series, column: str) -> bool:
    """Whether a v0.3.1 episode field is present, not a legacy test row."""

    if column not in row.index:
        return False
    # apply_state_machine adds lifecycle fields to legacy v0.3 helper rows as
    # it processes them. Anchor-event columns distinguish a real v0.3.1 input
    # contract from that backward-compatible synthetic path.
    if column.startswith("recent_panic") and "panic_anchor_event" not in row.index:
        return False
    if column.startswith("recent_euphoria") and "euphoria_anchor_event" not in row.index:
        return False
    value = row.get(column)
    return not pd.isna(value)


def _ice_candidate(row: pd.Series, cfg: StateMachineConfig) -> bool:
    smooth = _number(row, "smoothed_temperature")
    slope = _number(row, "slope3")
    improving = _number(row, "improving_modules")
    breadth = _number(row, "breadth_delta3")
    profit = _number(row, "profit_effect_delta3")
    if _episode_is_available(row, "recent_panic_episode"):
        low = _number(row, "post_panic_low")
        recovery = _number(row, "recovery_from_post_panic_low")
        if not _flag(row, "recent_panic_episode"):
            return False
        anchor_ok = np.isfinite(low) and np.isfinite(recovery)
    else:
        # Backward-compatible path for small v0.3 hand-built indicator frames.
        low = _number(row, "rolling_low_10")
        recovery = _number(row, "recovery_from_low")
        anchor_ok = np.isfinite(low) and low <= cfg.cold_threshold
    return (
        np.isfinite(smooth)
        and anchor_ok
        and recovery >= cfg.turning_delta
        and slope >= cfg.slope3_threshold
        and improving >= cfg.minimum_confirming_modules
        and (_positive(breadth) or _positive(profit))
    )


def _hot_candidate(row: pd.Series, cfg: StateMachineConfig) -> bool:
    smooth = _number(row, "smoothed_temperature")
    slope = _number(row, "slope3")
    deteriorating = _number(row, "deteriorating_modules")
    breadth = _number(row, "breadth_delta3")
    profit = _number(row, "profit_effect_delta3")
    if _episode_is_available(row, "recent_euphoria_episode"):
        high = _number(row, "post_euphoria_high")
        drop = _number(row, "drop_from_post_euphoria_high")
        if not _flag(row, "recent_euphoria_episode"):
            return False
        anchor_ok = np.isfinite(high) and np.isfinite(drop)
    else:
        high = _number(row, "rolling_high_10")
        drop = _number(row, "drop_from_high")
        anchor_ok = np.isfinite(high) and high >= cfg.hot_threshold
    return (
        np.isfinite(smooth)
        and anchor_ok
        and drop >= cfg.turning_delta
        and slope <= -cfg.slope3_threshold
        and deteriorating >= cfg.minimum_confirming_modules
        and (_negative(breadth) or _negative(profit))
    )


def _ice_watch_invalidated(row: pd.Series) -> bool:
    smooth = _number(row, "smoothed_temperature")
    slope = _number(row, "slope3")
    if not np.isfinite(smooth):
        return True
    # The episode low is updated with the current observation, so the causal
    # slope rule is the explicit invalidation rule for an armed v0.3.1 watch.
    if np.isfinite(slope) and slope <= 0:
        return True
    if "recent_panic_episode" not in row.index:
        low = _number(row, "rolling_low_10")
        recovery = _number(row, "recovery_from_low")
        return (np.isfinite(low) and smooth < low - 1e-9) or (np.isfinite(recovery) and recovery < 0)
    return False


def _hot_watch_invalidated(row: pd.Series) -> bool:
    smooth = _number(row, "smoothed_temperature")
    slope = _number(row, "slope3")
    if not np.isfinite(smooth):
        return True
    if np.isfinite(slope) and slope >= 0:
        return True
    if "recent_euphoria_episode" not in row.index:
        high = _number(row, "rolling_high_10")
        drop = _number(row, "drop_from_high")
        return (np.isfinite(high) and smooth > high + 1e-9) or (np.isfinite(drop) and drop < 0)
    return False


def _zone_state(row: pd.Series, cfg: StateMachineConfig, previous_zone_state: str | None) -> str:
    """Classify HOT/NORMAL/COLD using causal entry and exit hysteresis."""

    smooth = _number(row, "smoothed_temperature")
    if not np.isfinite(smooth):
        return "DATA_INVALID"
    band = cfg.zone_hysteresis
    previous = previous_zone_state if previous_zone_state in {"HOT", "NORMAL", "COLD"} else None
    if previous == "HOT":
        if smooth >= cfg.warm_threshold - band:
            return "HOT"
        return "COLD" if smooth <= cfg.cool_threshold else "NORMAL"
    if previous == "COLD":
        if smooth <= cfg.cool_threshold + band:
            return "COLD"
        return "HOT" if smooth >= cfg.warm_threshold else "NORMAL"
    if smooth >= cfg.warm_threshold:
        return "HOT"
    if smooth <= cfg.cool_threshold:
        return "COLD"
    return "NORMAL"


def _base_state(
    row: pd.Series,
    cfg: StateMachineConfig,
    previous_zone_state: str | None = None,
    anchor_cfg: EpisodeAnchorConfig | None = None,
) -> str:
    smooth = _number(row, "smoothed_temperature")
    slope = _number(row, "slope3")
    raw = _number(row, "raw_temperature")
    if not np.isfinite(smooth):
        return "DATA_INVALID"
    if _episode_is_available(row, "recent_panic_episode"):
        recent_panic = _flag(row, "recent_panic_episode")
        raw_panic_threshold = (anchor_cfg or EpisodeAnchorConfig()).raw_panic_threshold
        if recent_panic and (_negative(slope) or (np.isfinite(raw) and raw <= raw_panic_threshold)):
            return "PANIC_FALLING"
        if recent_panic:
            return "EXTREME_PANIC"
        recent_euphoria = _flag(row, "recent_euphoria_episode")
        smooth_euphoria_threshold = (anchor_cfg or EpisodeAnchorConfig()).smoothed_euphoria_threshold
        raw_euphoria_threshold = (anchor_cfg or EpisodeAnchorConfig()).raw_euphoria_threshold
        if recent_euphoria and _positive(slope) and (
            smooth >= smooth_euphoria_threshold
            or (np.isfinite(raw) and raw >= raw_euphoria_threshold)
        ):
            return "EUPHORIA_RISING"
    else:
        # Preserve the v0.3 helper contract for legacy/synthetic frames.
        if smooth <= cfg.cold_threshold and _negative(slope):
            return "PANIC_FALLING"
        if smooth <= cfg.cold_threshold:
            return "EXTREME_PANIC"
        if smooth >= cfg.hot_threshold and _positive(slope):
            return "EUPHORIA_RISING"
    return _zone_state(row, cfg, previous_zone_state)


def _append_warning(existing: Any, item: str) -> str:
    text = "" if existing is None or pd.isna(existing) else str(existing)
    return ";".join([part for part in (text, item) if part])


def _new_tracker() -> dict[str, Any]:
    return {
        "active": False,
        "closed": False,
        "episode_id": None,
        "start_index": None,
        "last_anchor_index": None,
        "extreme": np.nan,
    }


def _advance_tracker(
    tracker: dict[str, Any],
    *,
    prefix: str,
    row: pd.Series,
    valid_index: int,
    anchor_cfg: EpisodeAnchorConfig,
    next_id: int,
) -> int:
    event = _flag(row, f"{prefix}_anchor_event")
    value = _number(row, "smoothed_temperature")
    if tracker["active"] and tracker["last_anchor_index"] is not None:
        if valid_index - int(tracker["last_anchor_index"]) >= anchor_cfg.memory_window:
            tracker.update({
                "active": False,
                "closed": True,
                "episode_id": None,
                "start_index": None,
                "last_anchor_index": None,
                "extreme": np.nan,
            })
    if event:
        if not tracker["active"]:
            next_id += 1
            tracker["episode_id"] = next_id
            tracker["start_index"] = valid_index
            tracker["extreme"] = value
            tracker["closed"] = False
        tracker["active"] = True
        tracker["last_anchor_index"] = valid_index
        if np.isfinite(value):
            if prefix == "panic":
                tracker["extreme"] = value if not np.isfinite(tracker["extreme"]) else min(float(tracker["extreme"]), value)
            else:
                tracker["extreme"] = value if not np.isfinite(tracker["extreme"]) else max(float(tracker["extreme"]), value)
    return next_id


def _episode_fields(tracker: dict[str, Any], *, prefix: str, valid_index: int) -> dict[str, Any]:
    active = bool(tracker["active"])
    episode_id = tracker["episode_id"] if active else pd.NA
    start_index = tracker["start_index"] if active else None
    age = valid_index - int(start_index) + 1 if active and start_index is not None else 0
    extreme = float(tracker["extreme"]) if active and np.isfinite(tracker["extreme"]) else np.nan
    closed = bool(tracker["closed"] and not active)
    if prefix == "panic":
        return {
            "panic_episode_armed": active,
            "recent_panic_episode": active,
            "panic_episode_age": age,
            "panic_episode_id": episode_id,
            "post_panic_low": extreme,
            "recovery_from_post_panic_low": np.nan,
            "panic_episode_closed": closed,
        }
    return {
        "euphoria_episode_armed": active,
        "recent_euphoria_episode": active,
        "euphoria_episode_age": age,
        "euphoria_episode_id": episode_id,
        "post_euphoria_high": extreme,
        "drop_from_post_euphoria_high": np.nan,
        "euphoria_episode_closed": closed,
    }


def _apply_episode_derivatives(row: pd.Series, fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        row[key] = value
    smooth = _number(row, "smoothed_temperature")
    low = _number(row, "post_panic_low")
    high = _number(row, "post_euphoria_high")
    row["recovery_from_post_panic_low"] = smooth - low if np.isfinite(smooth) and np.isfinite(low) else np.nan
    row["drop_from_post_euphoria_high"] = high - smooth if np.isfinite(smooth) and np.isfinite(high) else np.nan


def apply_state_machine(indicators: pd.DataFrame, config: StateMachineConfig | dict) -> pd.DataFrame:
    """Assign states, episode memory, signals and persistence without lookahead."""

    cfg = config if isinstance(config, StateMachineConfig) else StateMachineConfig.from_mapping(config)
    anchor_cfg = config if isinstance(config, EpisodeAnchorConfig) else EpisodeAnchorConfig.from_mapping(config if isinstance(config, dict) else {})
    output = indicators.copy().sort_values("trade_date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    previous_state = "NONE"
    previous_zone_state: str | None = None
    previous_watch_streak = 0
    previous_confirmation_streak = 0
    previous_panic_episode_id: Any = None
    previous_euphoria_episode_id: Any = None
    valid_index = 0
    panic_id = 0
    euphoria_id = 0
    panic_tracker = _new_tracker()
    euphoria_tracker = _new_tracker()

    for _, source_row in output.iterrows():
        row = source_row.copy()
        quality = str(row.get("quality", row.get("market_temperature_quality", "A"))).upper()
        invalid = quality == "INVALID" or pd.isna(row.get("raw_temperature")) or pd.isna(row.get("smoothed_temperature"))
        close_panic = False
        close_euphoria = False
        if invalid:
            # INVALID is not an observation: no anchor, memory, state or
            # confirmation may cross this date.
            panic_tracker = _new_tracker()
            euphoria_tracker = _new_tracker()
            panic_tracker["closed"] = True
            euphoria_tracker["closed"] = True
            previous_watch_streak = 0
            previous_confirmation_streak = 0
            state = "DATA_INVALID"
            signal = "NONE"
            watch_streak = 0
            confirmation_streak = 0
            watch_timeout = False
            watch_invalidated = False
            _apply_episode_derivatives(row, _episode_fields(panic_tracker, prefix="panic", valid_index=valid_index))
            _apply_episode_derivatives(row, _episode_fields(euphoria_tracker, prefix="euphoria", valid_index=valid_index))
            row["panic_episode_id"] = pd.NA
            row["euphoria_episode_id"] = pd.NA
        else:
            valid_index += 1
            panic_id = _advance_tracker(panic_tracker, prefix="panic", row=row, valid_index=valid_index, anchor_cfg=anchor_cfg, next_id=panic_id)
            euphoria_id = _advance_tracker(euphoria_tracker, prefix="euphoria", row=row, valid_index=valid_index, anchor_cfg=anchor_cfg, next_id=euphoria_id)
            _apply_episode_derivatives(row, _episode_fields(panic_tracker, prefix="panic", valid_index=valid_index))
            _apply_episode_derivatives(row, _episode_fields(euphoria_tracker, prefix="euphoria", valid_index=valid_index))
            # Calculate the causal zone before turning-point priority is applied.
            zone_state = _zone_state(row, cfg, previous_zone_state)
            previous_zone_state = zone_state
            ice = _ice_candidate(row, cfg)
            hot = _hot_candidate(row, cfg)
            state = _base_state(row, cfg, previous_zone_state, anchor_cfg)
            signal = "NONE"
            watch_streak = 0
            confirmation_streak = 0
            watch_timeout = False
            watch_invalidated = False

            if previous_state == "ICE_REVERSAL_WATCH":
                if _ice_watch_invalidated(row):
                    watch_invalidated = True
                    previous_watch_streak = 0
                    previous_confirmation_streak = 0
                elif previous_watch_streak >= cfg.watch_timeout:
                    watch_timeout = True
                    previous_watch_streak = 0
                    previous_confirmation_streak = 0
                else:
                    watch_streak = previous_watch_streak + 1
                    confirmation_streak = watch_streak
                    if ice and watch_streak >= cfg.confirmation_days:
                        state = "ICE_REVERSAL"
                        signal = "ICE_REVERSAL_CONFIRMED"
                        close_panic = True
                        previous_watch_streak = 0
                        previous_confirmation_streak = 0
                    else:
                        state = "ICE_REVERSAL_WATCH"
                        signal = "ICE_REVERSAL_WATCH"
                        previous_watch_streak = watch_streak
                        previous_confirmation_streak = confirmation_streak
            elif previous_state == "HOT_ROLLOVER_WATCH":
                if _hot_watch_invalidated(row):
                    watch_invalidated = True
                    previous_watch_streak = 0
                    previous_confirmation_streak = 0
                elif previous_watch_streak >= cfg.watch_timeout:
                    watch_timeout = True
                    previous_watch_streak = 0
                    previous_confirmation_streak = 0
                else:
                    watch_streak = previous_watch_streak + 1
                    confirmation_streak = watch_streak
                    if hot and watch_streak >= cfg.confirmation_days:
                        state = "HOT_ROLLOVER"
                        signal = "HOT_ROLLOVER_CONFIRMED"
                        close_euphoria = True
                        previous_watch_streak = 0
                        previous_confirmation_streak = 0
                    else:
                        state = "HOT_ROLLOVER_WATCH"
                        signal = "HOT_ROLLOVER_WATCH"
                        previous_watch_streak = watch_streak
                        previous_confirmation_streak = confirmation_streak
            elif previous_state == "ICE_REVERSAL":
                new_episode = _flag(row, "panic_anchor_event") and row.get("panic_episode_id") != previous_panic_episode_id
                if ice and not new_episode:
                    state = "ICE_REVERSAL"
                elif ice:
                    state = "ICE_REVERSAL_WATCH"
                    signal = "ICE_REVERSAL_WATCH"
                    watch_streak = 1
                    confirmation_streak = 1
                    previous_watch_streak = 1
                    previous_confirmation_streak = 1
                else:
                    previous_watch_streak = 0
                    previous_confirmation_streak = 0
            elif previous_state == "HOT_ROLLOVER":
                new_episode = _flag(row, "euphoria_anchor_event") and row.get("euphoria_episode_id") != previous_euphoria_episode_id
                if hot and not new_episode:
                    state = "HOT_ROLLOVER"
                elif hot:
                    state = "HOT_ROLLOVER_WATCH"
                    signal = "HOT_ROLLOVER_WATCH"
                    watch_streak = 1
                    confirmation_streak = 1
                    previous_watch_streak = 1
                    previous_confirmation_streak = 1
                else:
                    previous_watch_streak = 0
                    previous_confirmation_streak = 0
            else:
                # Turning candidates have priority over descriptive labels.
                if ice:
                    state = "ICE_REVERSAL_WATCH"
                    signal = "ICE_REVERSAL_WATCH"
                    watch_streak = 1
                    confirmation_streak = 1
                    previous_watch_streak = 1
                    previous_confirmation_streak = 1
                elif hot:
                    state = "HOT_ROLLOVER_WATCH"
                    signal = "HOT_ROLLOVER_WATCH"
                    watch_streak = 1
                    confirmation_streak = 1
                    previous_watch_streak = 1
                    previous_confirmation_streak = 1
                else:
                    previous_watch_streak = 0
                    previous_confirmation_streak = 0

            if close_panic:
                panic_tracker["active"] = False
                panic_tracker["closed"] = True
                row["panic_episode_armed"] = True
                row["recent_panic_episode"] = True
                row["panic_episode_closed"] = True
            if close_euphoria:
                euphoria_tracker["active"] = False
                euphoria_tracker["closed"] = True
                row["euphoria_episode_armed"] = True
                row["recent_euphoria_episode"] = True
                row["euphoria_episode_closed"] = True

        confidence = "NONE" if invalid else str(row.get("confidence", {"A": "HIGH", "B": "MEDIUM", "C": "LOW"}.get(quality, "LOW")))
        if not invalid and confidence not in {"HIGH", "MEDIUM", "LOW", "NONE"}:
            confidence = {"A": "HIGH", "B": "MEDIUM", "C": "LOW", "INVALID": "NONE"}.get(quality, "LOW")
        row_warning = "WATCH_TIMEOUT" if watch_timeout else "WATCH_INVALIDATED" if watch_invalidated else ""
        record = row.to_dict()
        record.update({
            "previous_state": previous_state,
            "state": state,
            "signal": signal,
            "watch_streak": watch_streak,
            "confirmation_streak": confirmation_streak,
            "watch_timeout": watch_timeout,
            "watch_invalidated": watch_invalidated,
            "state_changed": state != previous_state,
            "confidence": confidence,
            "warnings": _append_warning(row.get("warnings", row.get("data_quality_warnings", "")), row_warning),
            "zone_state": "DATA_INVALID" if invalid else _zone_state(row, cfg, previous_zone_state),
        })
        record["state_duration"] = int(rows[-1]["state_duration"]) + 1 if rows and rows[-1]["state"] == state else 1
        rows.append(record)
        previous_state = state
        previous_panic_episode_id = record.get("panic_episode_id")
        previous_euphoria_episode_id = record.get("euphoria_episode_id")

    assigned = pd.DataFrame(rows)
    # Concatenate once instead of inserting hundreds of columns one by one;
    # this also keeps large full-market research runs quiet and predictable.
    output = pd.concat(
        [output.drop(columns=assigned.columns, errors="ignore").reset_index(drop=True), assigned.reset_index(drop=True)],
        axis=1,
    )
    output.loc[output["state"].eq("DATA_INVALID"), "signal"] = "NONE"
    output.loc[output["state"].eq("DATA_INVALID"), "confidence"] = "NONE"
    return output


def transition_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return counts for previous_state -> state, including zero cells."""

    matrix = pd.DataFrame(0, index=list(STATES) + ["NONE"], columns=list(STATES), dtype=int)
    for previous, current in zip(frame.get("previous_state", []), frame.get("state", [])):
        if current in matrix.columns and previous in matrix.index:
            matrix.loc[previous, current] += 1
    return matrix
