"""History loading, audit projection and causal stability checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.cache import ParquetCache
from .engine import build_advisory_frame


HISTORY_COLUMNS = (
    "date",
    "temperature",
    "ema_temperature",
    "market_temperature",
    "smoothed_temperature",
    "state",
    "signal",
    "advisory_regime",
    "advisory_signal",
    "buy_reference",
    "sell_reference",
    "risk_level",
    "expected_horizon",
    "reference_horizon",
    "signal_confidence",
    "research_evidence",
    "confirming_modules",
    "headline",
    "reason_codes",
)


def load_state_frame(processed_root: str | Path) -> pd.DataFrame:
    """Load the frozen v0.3.1 state table, with v0.3 compatibility fallback."""

    cache = ParquetCache(processed_root)
    for dataset in ("market_state_daily_v031", "market_state_daily"):
        if cache.exists(dataset):
            return cache.load(dataset).sort_values("trade_date").reset_index(drop=True)
    raise FileNotFoundError("market_state_daily_v031 or market_state_daily is missing; run `python -m ashare_sentiment regime` first")


def build_advisory_history(state_frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Return the compact historical audit view without any outcome columns."""

    advisory = build_advisory_frame(state_frame, config=config)
    if advisory.empty:
        return advisory
    advisory = advisory.copy()
    source = state_frame.copy()
    source["trade_date"] = pd.to_datetime(source["trade_date"], errors="coerce")
    source = source.sort_values("trade_date").reset_index(drop=True)
    advisory["signal"] = source.get("signal", pd.Series("NONE", index=advisory.index)).astype(str).to_numpy()
    for column in HISTORY_COLUMNS:
        if column not in advisory:
            advisory[column] = pd.NA
    return advisory[[column for column in HISTORY_COLUMNS if column in advisory.columns] + ["short_term_evidence", "medium_term_evidence", "short_term_warning", "not_exact_top"]]


def save_advisory_outputs(
    advisory: pd.DataFrame,
    history: pd.DataFrame,
    processed_root: str | Path,
    reports_root: str | Path,
) -> tuple[Path, Path]:
    """Write the requested v0.4.1 Parquet and CSV artifacts."""

    processed = ParquetCache(processed_root)
    processed_path = processed.save(
        "market_advisory_daily_v041",
        advisory,
        processed.metadata_now(
            source="market-temperature-v0.4.1-advisory",
            frame=advisory.rename(columns={"trade_date": "trade_date"}),
            version="0.4.1",
            notes="Causal advisory layer over frozen v0.3.1 state; no forward returns or event outcomes.",
            date_column="trade_date",
        ),
    )
    reports = Path(reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    history_path = reports / "advisory_history_v041.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    return processed_path, history_path


def no_future_data_stability(state_frame: pd.DataFrame, config: dict | None = None, split: int | None = None) -> bool:
    """Verify that adding later rows cannot change the earlier advisory history."""

    if split is None:
        split = max(1, len(state_frame) - 1)
    left = build_advisory_frame(state_frame.iloc[:split].copy(), config=config)
    right = build_advisory_frame(state_frame.copy(), config=config).iloc[:split].reset_index(drop=True)
    left = left.reset_index(drop=True)
    return left.equals(right)
