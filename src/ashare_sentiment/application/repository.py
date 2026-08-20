"""Read-only repositories for cached MarketTemperature artifacts.

The desktop layer intentionally reads the same Parquet contracts used by the
CLI.  It does not calculate temperatures, states or advisory signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..advisory.history import load_state_frame
from ..data.cache import CacheError, DatasetMetadata, ParquetCache


class AdvisoryRepository:
    """Locate and load the local v0.4.1/v0.3.1 data contracts."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        data_config = config.get("data", {})
        self.processed_root = Path(data_config.get("processed_root", "data/processed"))
        self.reports_root = Path(data_config.get("reports_root", "reports"))
        self.processed = ParquetCache(self.processed_root)

    def load_advisory(self) -> pd.DataFrame | None:
        if not self.processed.exists("market_advisory_daily_v041"):
            return None
        frame = self.processed.load("market_advisory_daily_v041").copy()
        return self._normalise_dates(frame, "date", fallback="trade_date")

    def load_state(self) -> pd.DataFrame:
        return load_state_frame(self.processed_root)

    def load_coverage(self) -> pd.DataFrame | None:
        if not self.processed.exists("market_coverage_daily"):
            return None
        frame = self.processed.load("market_coverage_daily").copy()
        return self._normalise_dates(frame, "trade_date")

    def metadata(self, dataset: str) -> DatasetMetadata | None:
        try:
            return self.processed.load_metadata(dataset)
        except CacheError:
            return None

    def has_pipeline_artifacts(self) -> bool:
        return all(
            self.processed.exists(name)
            for name in ("market_sentiment_daily", "market_state_daily_v031", "market_advisory_daily_v041")
        )

    @staticmethod
    def _normalise_dates(frame: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.DataFrame:
        if primary not in frame and fallback and fallback in frame:
            frame[primary] = frame[fallback]
        if primary in frame:
            frame[primary] = pd.to_datetime(frame[primary], errors="coerce").dt.normalize()
            frame = frame.dropna(subset=[primary]).sort_values(primary).reset_index(drop=True)
        return frame
