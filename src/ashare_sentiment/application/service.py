"""Application service that composes cached domain outputs for the GUI."""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..advisory.engine import build_advisory_frame
from .repository import AdvisoryRepository
from .viewmodels import (
    DailyAdvisoryViewModel,
    DiagnosticsViewModel,
    EpisodeViewModel,
    WARNING_MEANINGS,
    as_date,
)


class AdvisoryService:
    """Single application-facing boundary for the desktop views."""

    def __init__(self, config: dict[str, Any], repository: AdvisoryRepository | None = None):
        self.config = config
        self.repository = repository or AdvisoryRepository(config)
        self._advisory: pd.DataFrame | None = None
        self._state: pd.DataFrame | None = None
        self._coverage: pd.DataFrame | None = None

    def reload(self) -> None:
        self._advisory = None
        self._state = None
        self._coverage = None

    @property
    def advisory(self) -> pd.DataFrame:
        if self._advisory is None:
            cached = self.repository.load_advisory()
            if cached is None:
                state = self.state
                cached = build_advisory_frame(state, config=self.config)
                cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.normalize()
            self._advisory = cached.sort_values("date").reset_index(drop=True)
        return self._advisory

    @property
    def state(self) -> pd.DataFrame:
        if self._state is None:
            self._state = self.repository.load_state().copy()
            self._state["trade_date"] = pd.to_datetime(self._state["trade_date"], errors="coerce").dt.normalize()
            self._state = self._state.sort_values("trade_date").reset_index(drop=True)
        return self._state

    @property
    def coverage(self) -> pd.DataFrame | None:
        if self._coverage is None:
            self._coverage = self.repository.load_coverage()
        return self._coverage

    def latest_valid_date(self) -> date | None:
        frame = self.advisory
        valid = frame[frame["advisory_signal"].ne("DATA_INVALID")]
        return as_date(valid["date"].max()) if not valid.empty else None

    def available_dates(self) -> list[date]:
        return [value.date() for value in pd.to_datetime(self.advisory["date"], errors="coerce").dropna()]

    def resolve_date(self, value: Any = None, direction: int = 0) -> date | None:
        dates = self.available_dates()
        if not dates:
            return None
        target = as_date(value) or self.latest_valid_date() or dates[-1]
        if direction < 0:
            earlier = [item for item in dates if item < target]
            return earlier[-1] if earlier else dates[0]
        if direction > 0:
            later = [item for item in dates if item > target]
            return later[0] if later else dates[-1]
        eligible = [item for item in dates if item <= target]
        return eligible[-1] if eligible else dates[0]

    def daily(self, value: Any = None) -> DailyAdvisoryViewModel:
        resolved = self.resolve_date(value)
        if resolved is None:
            return DailyAdvisoryViewModel.from_row({"advisory_signal": "DATA_INVALID", "state": "DATA_INVALID"})
        row = self.advisory[self.advisory["date"].dt.date.eq(resolved)].iloc[-1].to_dict()
        source = self.state[self.state["trade_date"].dt.date.eq(resolved)]
        if not source.empty:
            row.update(source.iloc[-1].to_dict())
            row["date"] = resolved.isoformat()
        return DailyAdvisoryViewModel.from_row(row)

    def trend(self, range_name: str = "1Y") -> pd.DataFrame:
        frame = self.advisory.copy()
        if frame.empty:
            return frame
        days = {"3M": 92, "6M": 183, "1Y": 366, "All": None}.get(range_name, 366)
        if days is not None:
            cutoff = frame["date"].max() - pd.Timedelta(days=days)
            frame = frame[frame["date"] >= cutoff]
        columns = [column for column in ("date", "market_temperature", "smoothed_temperature", "advisory_signal") if column in frame]
        return frame[columns].reset_index(drop=True)

    def episodes(self) -> list[EpisodeViewModel]:
        frame = self.state.copy()
        if frame.empty:
            return []
        advisory = self.advisory[["date", "advisory_signal"]].copy()
        advisory["trade_date"] = advisory["date"]
        frame = frame.merge(advisory[["trade_date", "advisory_signal"]], on="trade_date", how="left")
        result: list[EpisodeViewModel] = []
        for kind, id_column, closed_column, anchor_column, watch_states, confirmed_state, label in (
            ("panic", "panic_episode_id", "panic_episode_closed", "post_panic_low", {"ICE_REVERSAL_WATCH"}, "ICE_REVERSAL", "Panic"),
            ("euphoria", "euphoria_episode_id", "euphoria_episode_closed", "post_euphoria_high", {"HOT_ROLLOVER_WATCH"}, "HOT_ROLLOVER", "Euphoria"),
        ):
            if id_column not in frame:
                continue
            groups = frame[frame[id_column].notna()].groupby(id_column, sort=True)
            for episode_id, group in groups:
                group = group.sort_values("trade_date")
                start = group.iloc[0]
                extreme_column = "smoothed_temperature"
                values = pd.to_numeric(group[extreme_column], errors="coerce")
                extreme_index = values.idxmin() if kind == "panic" else values.idxmax()
                extreme = group.loc[extreme_index]
                watch = group[group["state"].isin(watch_states)]
                confirmed = group[group["state"].eq(confirmed_state)]
                closed = group[group[closed_column].fillna(False).astype(bool)]
                end = closed.iloc[0] if not closed.empty else group.iloc[-1]
                state_sequence = tuple(dict.fromkeys(str(item) for item in group["state"].tolist()))
                status = "已结束" if not closed.empty else "进行中"
                minimum = pd.to_numeric(extreme.get(extreme_column), errors="coerce")
                extreme_text = "—" if pd.isna(minimum) else f"{float(minimum):.1f}"
                result.append(
                    EpisodeViewModel(
                        episode_type=label,
                        episode_id=str(int(float(episode_id))),
                        status=status,
                        start_date=self._date_text(start["trade_date"]),
                        extreme_date=self._date_text(extreme["trade_date"]),
                        watch_date=self._date_text(watch.iloc[0]["trade_date"]) if not watch.empty else "—",
                        confirmed_date=self._date_text(confirmed.iloc[0]["trade_date"]) if not confirmed.empty else "—",
                        end_date=self._date_text(end["trade_date"]) if not closed.empty else "—",
                        minimum_or_maximum=extreme_text,
                        state_sequence=state_sequence,
                    )
                )
        return sorted(result, key=lambda item: item.start_date, reverse=True)

    def diagnostics(self) -> DiagnosticsViewModel:
        state = self.state
        latest = state.iloc[-1] if not state.empty else pd.Series(dtype=object)
        valid = state[state.get("state", pd.Series(index=state.index)).ne("DATA_INVALID")] if not state.empty else state
        latest_valid = valid["trade_date"].max() if not valid.empty else None
        codes = self.daily(latest_valid).warning_codes if latest_valid is not None else ()
        coverage = self.coverage
        coverage_value = latest.get("market_coverage_ratio", latest.get("coverage_ratio"))
        if coverage is not None and not coverage.empty:
            coverage_row = coverage[coverage["trade_date"].eq(latest.get("trade_date"))]
            if not coverage_row.empty:
                coverage_value = coverage_row.iloc[-1].get("coverage_ratio", coverage_value)
        metadata = self.repository.metadata("market_state_daily_v031")
        last_calculated = metadata.download_time if metadata else "—"
        warning_labels = tuple(WARNING_MEANINGS[code][0] for code in codes if code in WARNING_MEANINGS)
        status = str(latest.get("data_quality_status", latest.get("quality", "UNKNOWN")))
        return DiagnosticsViewModel(
            status=status,
            latest_valid_date=self._date_text(latest_valid),
            latest_calculated_date=self._date_text(latest.get("trade_date")),
            row_count=len(state),
            warning_codes=tuple(codes),
            warning_labels=warning_labels,
            universe=self._number_text(latest.get("observed_universe", latest.get("known_stocks")), digits=0),
            coverage=self._percent_text(coverage_value),
            pipeline_status="就绪" if self.repository.has_pipeline_artifacts() else "缓存不完整",
            last_calculated_at=str(last_calculated),
            source=str(metadata.source) if metadata else "—",
        )

    def refresh_command(
        self, config_path: str | Path, include_download: bool = True, end_date: date | str | None = None
    ) -> list[list[str]]:
        commands: list[list[str]] = []
        if include_download:
            update = [sys.executable, "-m", "ashare_sentiment", "--config", str(config_path), "update", "--dataset", "all"]
            if end_date is not None:
                update.extend(["--end-date", str(end_date)[:10]])
            commands.append(update)
        for command in ("score", "regime", "advisory"):
            commands.append([sys.executable, "-m", "ashare_sentiment", "--config", str(config_path), command])
        return commands

    def refresh(
        self, config_path: str | Path, include_download: bool = True, end_date: date | str | None = None
    ) -> str:
        outputs: list[str] = []
        for command in self.refresh_command(config_path, include_download=include_download, end_date=end_date):
            result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
            outputs.append(result.stdout.strip())
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip() or f"command failed: {' '.join(command)}"
                raise RuntimeError(detail)
        self.reload()
        return "\n".join(item for item in outputs if item)

    @staticmethod
    def _date_text(value: Any) -> str:
        timestamp = pd.to_datetime(value, errors="coerce")
        return "—" if pd.isna(timestamp) else timestamp.strftime("%Y-%m-%d")

    @staticmethod
    def _number_text(value: Any, digits: int = 1) -> str:
        number = pd.to_numeric(value, errors="coerce")
        return "—" if pd.isna(number) else f"{float(number):.{digits}f}"

    @staticmethod
    def _percent_text(value: Any) -> str:
        number = pd.to_numeric(value, errors="coerce")
        return "—" if pd.isna(number) else f"{float(number) * 100:.1f}%"
