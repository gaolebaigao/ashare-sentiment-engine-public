"""Local Parquet cache with metadata and incremental upsert support."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class CacheError(RuntimeError):
    """Raised for cache read/write errors."""


@dataclass(frozen=True)
class DatasetMetadata:
    source: str
    download_time: str
    date_start: str | None = None
    date_end: str | None = None
    symbol: str | None = None
    version: str = "0.1"
    notes: str | None = None
    endpoint: str | None = None
    symbol_count: int | None = None
    row_count: int | None = None


class ParquetCache:
    """Persist a dataset as Parquet and metadata as a sidecar JSON file.

    The cache deliberately fails with an actionable message when neither
    ``pyarrow`` nor ``fastparquet`` is installed; silently falling back to CSV
    would make the storage contract ambiguous.
    """

    _SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset: str) -> Path:
        safe_name = self._SAFE_NAME.sub("_", dataset).strip("._")
        if not safe_name:
            raise CacheError("Dataset name cannot be empty")
        return self.root / f"{safe_name}.parquet"

    def metadata_path_for(self, dataset: str) -> Path:
        return self.path_for(dataset).with_suffix(".metadata.json")

    def exists(self, dataset: str) -> bool:
        return self.path_for(dataset).exists()

    def save(self, dataset: str, frame: pd.DataFrame, metadata: DatasetMetadata) -> Path:
        path = self.path_for(dataset)
        self._require_parquet_engine()
        self._atomic_write_parquet(frame, path)
        metadata_path = self.metadata_path_for(dataset)
        self._atomic_write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), metadata_path)
        return path

    def load(self, dataset: str) -> pd.DataFrame:
        path = self.path_for(dataset)
        if not path.exists():
            raise CacheError(f"Dataset is not cached: {dataset}")
        self._require_parquet_engine()
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - backend-specific exception types
            raise CacheError(f"Could not read cached Parquet {path}: {exc}") from exc

    def load_metadata(self, dataset: str) -> DatasetMetadata:
        path = self.metadata_path_for(dataset)
        if not path.exists():
            raise CacheError(f"Metadata is not cached: {dataset}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return DatasetMetadata(**payload)
        except (OSError, TypeError, ValueError) as exc:
            raise CacheError(f"Could not read metadata {path}: {exc}") from exc

    def upsert(
        self,
        dataset: str,
        frame: pd.DataFrame,
        metadata: DatasetMetadata,
        date_column: str = "trade_date",
        key_columns: list[str] | None = None,
    ) -> Path:
        """Merge new rows with cache, deduplicate, sort and write atomically."""
        combined = frame.copy()
        if self.exists(dataset):
            combined = pd.concat([self.load(dataset), combined], ignore_index=True)
        keys = key_columns or ([date_column] if date_column in combined.columns else list(combined.columns))
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="last")
        if date_column in combined.columns:
            combined = combined.sort_values(date_column).reset_index(drop=True)
        merged_metadata = self._metadata_for_frame(metadata, combined, date_column)
        return self.save(dataset, combined, merged_metadata)

    @staticmethod
    def metadata_now(
        source: str,
        frame: pd.DataFrame,
        symbol: str | None = None,
        version: str = "0.1",
        notes: str | None = None,
        endpoint: str | None = None,
        date_column: str = "trade_date",
    ) -> DatasetMetadata:
        start = end = None
        if date_column in frame.columns and not frame.empty:
            dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
            if not dates.empty:
                start, end = dates.min().date().isoformat(), dates.max().date().isoformat()
        return DatasetMetadata(
            source=source,
            download_time=datetime.now(timezone.utc).isoformat(),
            date_start=start,
            date_end=end,
            symbol=symbol,
            version=version,
            notes=notes,
            endpoint=endpoint,
            symbol_count=int(frame[symbol_column].nunique()) if (symbol_column := next((column for column in ("ts_code", "symbol") if column in frame.columns), None)) else None,
            row_count=int(len(frame)),
        )

    @staticmethod
    def _metadata_for_frame(
        metadata: DatasetMetadata,
        frame: pd.DataFrame,
        date_column: str,
    ) -> DatasetMetadata:
        if date_column not in frame.columns or frame.empty:
            return metadata
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        if dates.empty:
            return metadata
        return DatasetMetadata(
            source=metadata.source,
            download_time=metadata.download_time,
            date_start=dates.min().date().isoformat(),
            date_end=dates.max().date().isoformat(),
            symbol=metadata.symbol,
            version=metadata.version,
            notes=metadata.notes,
            endpoint=metadata.endpoint,
            symbol_count=int(frame[symbol_column].nunique()) if (symbol_column := next((column for column in ("ts_code", "symbol") if column in frame.columns), None)) else metadata.symbol_count,
            row_count=int(len(frame)),
        )

    @staticmethod
    def _require_parquet_engine() -> None:
        try:
            import pyarrow  # noqa: F401
            return
        except ImportError:
            try:
                import fastparquet  # noqa: F401
                return
            except ImportError as exc:
                raise CacheError(
                    "Parquet support is not installed. Install the project data extra: "
                    "pip install -e '.[data]'"
                ) from exc

    @staticmethod
    def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            frame.to_parquet(temp_path, index=False)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_text(content: str, path: Path) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            temp_path.write_text(content, encoding="utf-8")
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
