"""Event definitions and deterministic de-clustering for v0.4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CONFIRMED_SIGNALS = ("ICE_REVERSAL_CONFIRMED", "HOT_ROLLOVER_CONFIRMED")
WATCH_SIGNALS = ("ICE_REVERSAL_WATCH", "HOT_ROLLOVER_WATCH")
EPISODE_STARTS = ("PANIC_EPISODE_START", "EUPHORIA_EPISODE_START")
EPISODE_DAYS = ("PANIC_EPISODE_DAY", "EUPHORIA_EPISODE_DAY")
EVENT_TYPES = CONFIRMED_SIGNALS + WATCH_SIGNALS + EPISODE_STARTS + EPISODE_DAYS


def normalize_episode_id(value: Any) -> str | pd._libs.missing.NAType:
    """Make parquet float episode identifiers stable for grouping and joins."""

    if pd.isna(value):
        return pd.NA
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _base_event_row(row: pd.Series, *, event_type: str, direction: str, episode_id: Any) -> dict[str, Any]:
    fields = (
        "trade_date", "raw_temperature", "smoothed_temperature", "state", "quality",
        "confidence", "warnings", "breadth_score", "profit_effect_score", "liquidity_score",
        "stretch_score", "panic_episode_id", "euphoria_episode_id", "recent_panic_episode",
        "recent_euphoria_episode", "panic_episode_age", "euphoria_episode_age",
    )
    event = {field: row.get(field) for field in fields if field in row.index}
    event["date"] = pd.Timestamp(row["trade_date"]).normalize()
    event["event_type"] = event_type
    event["signal"] = event_type
    event["direction"] = direction
    event["episode_id"] = normalize_episode_id(episode_id)
    event["event_id"] = f"{event_type}:{event['date'].date()}"
    event["event_source"] = "v0.3.1_frozen_state_table"
    return event


def build_event_catalog(state: pd.DataFrame) -> pd.DataFrame:
    """Build raw, comparator and episode events from the frozen state table.

    Episode starts and episode-day comparators are descriptive cohorts.  They
    are not additional trading signals and never alter the v0.3.1 state table.
    """

    if state.empty:
        return pd.DataFrame()
    frame = state.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    quality = frame.get("quality", frame.get("market_temperature_quality", pd.Series("A", index=frame.index))).astype(str).str.upper()
    valid = frame["trade_date"].notna() & ~quality.eq("INVALID") & pd.to_numeric(frame.get("raw_temperature"), errors="coerce").notna()
    valid_frame = frame.loc[valid].copy()
    valid_dates = valid_frame["trade_date"].tolist()
    date_position = {date: i for i, date in enumerate(valid_dates)}
    rows: list[dict[str, Any]] = []

    signal_values = frame.get("signal", pd.Series("NONE", index=frame.index)).astype(str)
    for idx, row in frame.loc[valid & signal_values.isin((*CONFIRMED_SIGNALS, *WATCH_SIGNALS))].iterrows():
        signal = str(row["signal"])
        direction = "ICE" if signal.startswith("ICE_") else "HOT"
        episode_col = "panic_episode_id" if direction == "ICE" else "euphoria_episode_id"
        event = _base_event_row(row, event_type=signal, direction=direction, episode_id=row.get(episode_col))
        event["original_signal"] = signal
        event["date_position"] = date_position[event["date"]]
        rows.append(event)

    def add_episode_events(prefix: str, direction: str) -> None:
        id_col = f"{prefix}_episode_id"
        active_col = f"recent_{prefix}_episode"
        if id_col not in frame.columns or active_col not in frame.columns:
            return
        active = frame.loc[valid & frame[active_col].fillna(False).astype(bool) & frame[id_col].notna()].copy()
        if active.empty:
            return
        active["_episode_key"] = active[id_col].map(normalize_episode_id)
        for episode_id, group in active.groupby("_episode_key", sort=True):
            group = group.sort_values("trade_date")
            start_row = group.iloc[0]
            start_type = "PANIC_EPISODE_START" if direction == "ICE" else "EUPHORIA_EPISODE_START"
            day_type = "PANIC_EPISODE_DAY" if direction == "ICE" else "EUPHORIA_EPISODE_DAY"
            start_event = _base_event_row(start_row, event_type=start_type, direction=direction, episode_id=episode_id)
            start_event["original_signal"] = str(start_row.get("signal", "NONE"))
            start_event["date_position"] = date_position[start_event["date"]]
            rows.append(start_event)
            for _, day_row in group.iterrows():
                day_event = _base_event_row(day_row, event_type=day_type, direction=direction, episode_id=episode_id)
                day_event["original_signal"] = str(day_row.get("signal", "NONE"))
                day_event["date_position"] = date_position[day_event["date"]]
                rows.append(day_event)

    add_episode_events("panic", "ICE")
    add_episode_events("euphoria", "HOT")
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values(["date", "event_type"]).reset_index(drop=True)
    result["event_id"] = result["event_id"].astype(str) + ":" + result.groupby("event_id").cumcount().astype(str)
    return result


def _filter_direction(events: pd.DataFrame, direction: str | None) -> pd.DataFrame:
    if direction is None:
        return events
    return events[events["direction"].eq(direction)].copy()


def decluster_events(
    events: pd.DataFrame,
    *,
    mode: str = "raw",
    window: int = 20,
    direction: str | None = None,
    event_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return raw, 20-observation de-clustered, or episode-level events.

    For de-clustering, the first same-direction event is retained and later
    events are suppressed until more than ``window`` valid observations have
    elapsed.  Episode-level selection keeps the first event in each episode.
    """

    if mode not in {"raw", "declustered", "episode"}:
        raise ValueError("mode must be raw, declustered or episode")
    if int(window) < 0:
        raise ValueError("window must be non-negative")
    if events.empty:
        return events.copy()
    output = events.copy()
    if event_types is not None:
        output = output[output["event_type"].isin(event_types)].copy()
    output = _filter_direction(output, direction)
    output = output.sort_values(["date_position", "date", "event_type"]).reset_index(drop=True)
    if mode == "raw" or output.empty:
        return output
    if mode == "episode":
        keys = ["direction", "episode_id"]
        output = output[output["episode_id"].notna()].copy()
        if output.empty:
            return output
        return output.groupby(keys, sort=False, dropna=False, as_index=False).head(1).reset_index(drop=True)

    keep: list[int] = []
    last_by_direction: dict[str, int] = {}
    for idx, row in output.iterrows():
        direction_key = str(row.get("direction", ""))
        position = int(row.get("date_position", idx))
        previous = last_by_direction.get(direction_key)
        if previous is None or position - previous > int(window):
            keep.append(idx)
            last_by_direction[direction_key] = position
    return output.loc[keep].reset_index(drop=True)
