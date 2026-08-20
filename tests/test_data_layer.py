import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ashare_sentiment.data.base import format_date, normalize_trade_date
from ashare_sentiment.data.cache import CacheError, DatasetMetadata, ParquetCache
from ashare_sentiment.data.validation import DataValidationError, validate_timeseries, validate_universe_membership


class DataLayerTests(unittest.TestCase):
    def test_format_date_and_normalization(self):
        self.assertEqual(format_date("2026-08-17"), "20260817")
        frame = normalize_trade_date(pd.DataFrame({"trade_date": ["2026-08-18", "2026-08-17"]}))
        self.assertEqual(frame["trade_date"].dt.strftime("%Y%m%d").tolist(), ["20260817", "20260818"])
        numeric = normalize_trade_date(pd.DataFrame({"trade_date": [20260817]}))
        self.assertEqual(numeric.iloc[0]["trade_date"].strftime("%Y%m%d"), "20260817")

    def test_validation_finds_blocking_and_warning_issues(self):
        frame = pd.DataFrame(
            {
                "ts_code": ["A", "A", "B"],
                "trade_date": ["2026-01-02", "2026-01-02", "2026-01-05"],
                "close": [10, 0, 20],
                "volume": [100, -1, 100],
            }
        )
        issues = validate_timeseries(frame, key_columns=["ts_code", "trade_date"])
        codes = {issue.code for issue in issues}
        self.assertIn("DUPLICATE_DATES", codes)
        self.assertIn("IMPOSSIBLE_PRICES", codes)
        self.assertIn("NEGATIVE_VOLUME", codes)
        with self.assertRaises(DataValidationError):
            validate_timeseries(frame, key_columns=["ts_code", "trade_date"], strict=True)

    def test_universe_membership_is_explicit(self):
        frame = pd.DataFrame({"ts_code": ["A"]})
        issues = validate_universe_membership(frame, {"A", "B"})
        self.assertEqual(issues[0].code, "MISSING_UNIVERSE_MEMBERS")
        self.assertEqual(issues[0].severity, "warning")

    def test_metadata_is_provider_and_date_aware(self):
        frame = pd.DataFrame({"trade_date": ["2026-01-02", "2026-01-05"]})
        metadata = ParquetCache.metadata_now("test", frame, symbol="000300.SH")
        self.assertEqual(metadata.source, "test")
        self.assertEqual(metadata.date_start, "2026-01-02")
        self.assertEqual(metadata.date_end, "2026-01-05")

    def test_cache_round_trip_and_incremental_upsert(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ParquetCache(Path(temp_dir))
            initial = pd.DataFrame({"ts_code": ["A"], "trade_date": ["2026-01-02"], "close": [10.0]})
            metadata = DatasetMetadata(source="test", download_time="now")
            if not (importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")):
                with self.assertRaises(CacheError) as context:
                    cache.save("sample", initial, metadata)
                self.assertIn("Parquet support is not installed", str(context.exception))
                return
            cache.save("sample", initial, metadata)
            future = pd.DataFrame(
                {
                    "ts_code": ["A", "A"],
                    "trade_date": ["2026-01-02", "2026-01-05"],
                    "close": [11.0, 12.0],
                }
            )
            cache.upsert("sample", future, metadata, key_columns=["ts_code", "trade_date"])
            result = cache.load("sample")
            self.assertEqual(result["trade_date"].astype(str).tolist(), ["2026-01-02", "2026-01-05"])
            self.assertEqual(result.iloc[0]["close"], 11.0)
            cached_metadata = cache.load_metadata("sample")
            self.assertEqual(cached_metadata.source, "test")
            self.assertEqual(cached_metadata.date_end, "2026-01-05")


if __name__ == "__main__":
    unittest.main()
