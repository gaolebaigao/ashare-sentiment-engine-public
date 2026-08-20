"""Presentation-only view models for the desktop UI.

These mappings are deliberately not signal logic.  The source signal and
state remain visible in every view model; this module only adds readable UI
labels and safe display values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


SIGNAL_LABELS = {
    "PANIC_WAIT": "恐慌下行 · 等待确认",
    "BUY_WATCH": "修复观察",
    "BUY_REFERENCE": "修复参考",
    "NEUTRAL": "中性",
    "HOT_CAUTION": "偏热谨慎",
    "SELL_WATCH": "退潮观察",
    "SELL_REFERENCE": "退潮参考",
    "DATA_INVALID": "数据不可用",
}
STATE_LABELS = {
    "PANIC_FALLING": "恐慌下行",
    "EXTREME_PANIC": "极端恐慌",
    "ICE_REVERSAL_WATCH": "冰点修复观察",
    "ICE_REVERSAL": "冰点修复确认",
    "COLD": "偏冷",
    "NORMAL": "中性",
    "HOT": "偏热",
    "EUPHORIA_RISING": "高温上行",
    "HOT_ROLLOVER_WATCH": "高温退潮观察",
    "HOT_ROLLOVER": "高温退潮确认",
    "DATA_INVALID": "数据不可用",
}
RISK_LABELS = {"LOW": "低", "NORMAL": "正常", "ELEVATED": "偏高", "HIGH": "高", "EXTREME": "极高"}
HORIZON_LABELS = {
    "NOT_APPLICABLE": "不适用",
    "SHORT_TERM": "短期",
    "SHORT_TERM_UNCERTAINTY": "短期不确定性",
    "MEDIUM_TERM": "中期（20–60 个交易观测）",
    "MEDIUM_TERM_RISK": "中期风险（40–60 个交易观测）",
}
CONFIDENCE_LABELS = {"LOW": "低", "MEDIUM": "中", "HIGH": "高"}
EVIDENCE_LABELS = {
    "UNPROVEN": "尚未验证",
    "WEAK": "较弱",
    "WEAK_TO_MODERATE": "较弱至中等",
    "MODERATE": "中等",
    "RELATIVELY_STRONG": "相对较强",
}
WARNING_MEANINGS = {
    "SURVIVORSHIP_BIAS_WARNING": ("生存者偏差提示", "当前股票池不能完全还原历史当日股票集合。", False),
    "HISTORICAL_DELIST_DATE_INCOMPLETE": ("历史退市日期不完整", "历史股票集合的退市日期信息不完整，覆盖率解读需保守。", False),
    "MARGIN_OPTIONAL": ("融资融券数据可选", "融资融券模块没有纳入本日核心温度。", False),
    "OPTIONS_UNAVAILABLE": ("期权数据不可用", "期权模块缺少可靠数据，温度已在可用模块之间重新归一化。", False),
    "FAILED_LIMIT_RATE_UNAVAILABLE": ("失败率数据不可用", "涨停后失败率没有可靠数据，Profit Effect 的解读需保守。", False),
    "DATA_INVALID": ("数据不可用", "本日无法通过生产数据门禁，不生成正常市场建议。", True),
}
MODULE_LABELS = {"breadth": "Breadth", "profit_effect": "Profit Effect", "liquidity": "Liquidity", "stretch": "Stretch"}


def _text(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return default
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    return value or default


def _number(value: Any) -> float | None:
    result = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(result) else float(result)


def _number_text(value: Any, digits: int = 1) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _date_text(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    return "—" if pd.isna(timestamp) else timestamp.strftime("%Y-%m-%d")


def _parse_codes(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item).strip())
    return tuple(item.strip() for item in _text(value, "").replace(",", ";").split(";") if item.strip())


def _parse_details(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass(frozen=True)
class MetricViewModel:
    key: str
    label: str
    value: str
    raw_value: float | None = None


@dataclass(frozen=True)
class DailyAdvisoryViewModel:
    date: str
    state: str
    state_label: str
    advisory_signal: str
    advisory_label: str
    temperature: str
    smoothed_temperature: str
    raw_temperature: float | None
    smooth_temperature: float | None
    buy_reference: str
    sell_reference: str
    risk_level: str
    risk_label: str
    expected_horizon: str
    horizon_label: str
    reference_horizon: str
    signal_confidence: str
    confidence_label: str
    research_evidence: str
    evidence_label: str
    confirming_modules: int
    confirming_module_names: tuple[str, ...]
    headline: str
    summary: str
    why: str
    what_to_watch_next: str
    warning_codes: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[tuple[str, str, str, bool], ...] = field(default_factory=tuple)
    internals: tuple[MetricViewModel, ...] = field(default_factory=tuple)
    data_invalid: bool = False
    source_state_signal: str = "NONE"
    not_exact_top: bool = False

    @classmethod
    def from_row(cls, row: pd.Series | dict[str, Any]) -> "DailyAdvisoryViewModel":
        get = row.get
        signal = _text(get("advisory_signal"), "DATA_INVALID")
        state = _text(get("state"), "DATA_INVALID")
        codes = _parse_codes(get("reason_codes"))
        warnings = tuple(
            (code, WARNING_MEANINGS[code][0], WARNING_MEANINGS[code][1], WARNING_MEANINGS[code][2])
            for code in codes
            if code in WARNING_MEANINGS
        )
        details = _parse_details(get("details"))
        why = _text(get("why"), _text(details.get("why"), _text(get("summary"))))
        watch = _text(get("what_to_watch_next"), _text(details.get("what_to_watch_next")))
        modules = tuple(item.strip() for item in _text(get("confirming_module_names"), "").split(";") if item.strip())
        internals = tuple(
            MetricViewModel(key=key, label=label, value=_number_text(get(f"{key}_score")), raw_value=_number(get(f"{key}_score")))
            for key, label in MODULE_LABELS.items()
        )
        invalid = signal == "DATA_INVALID" or state == "DATA_INVALID"
        return cls(
            date=_date_text(get("date", get("trade_date"))),
            state=state,
            state_label=STATE_LABELS.get(state, state.replace("_", " ").title()),
            advisory_signal=signal,
            advisory_label=SIGNAL_LABELS.get(signal, signal.replace("_", " ").title()),
            temperature=_number_text(get("market_temperature")),
            smoothed_temperature=_number_text(get("smoothed_temperature")),
            raw_temperature=_number(get("market_temperature")),
            smooth_temperature=_number(get("smoothed_temperature")),
            buy_reference="—" if invalid else _text(get("buy_reference")),
            sell_reference="—" if invalid else _text(get("sell_reference")),
            risk_level=_text(get("risk_level")),
            risk_label=RISK_LABELS.get(_text(get("risk_level"), ""), _text(get("risk_level"))),
            expected_horizon=_text(get("expected_horizon")),
            horizon_label=HORIZON_LABELS.get(_text(get("expected_horizon"), ""), _text(get("expected_horizon"))),
            reference_horizon="—" if invalid else _text(get("reference_horizon")),
            signal_confidence=_text(get("signal_confidence")),
            confidence_label=CONFIDENCE_LABELS.get(_text(get("signal_confidence"), ""), _text(get("signal_confidence"))),
            research_evidence=_text(get("research_evidence")),
            evidence_label=EVIDENCE_LABELS.get(_text(get("research_evidence"), ""), _text(get("research_evidence"))),
            confirming_modules=int(_number(get("confirming_modules")) or 0),
            confirming_module_names=modules,
            headline=_text(get("headline"), "数据不可用" if invalid else "暂无摘要"),
            summary=_text(get("summary")),
            why=why,
            what_to_watch_next="数据恢复后再查看市场建议。" if invalid else watch,
            warning_codes=codes,
            warnings=warnings,
            internals=internals,
            data_invalid=invalid,
            source_state_signal=_text(get("state_signal", get("signal")), "NONE"),
            not_exact_top=bool(get("not_exact_top", False)),
        )


@dataclass(frozen=True)
class EpisodeViewModel:
    episode_type: str
    episode_id: str
    status: str
    start_date: str
    extreme_date: str
    watch_date: str
    confirmed_date: str
    end_date: str
    minimum_or_maximum: str
    state_sequence: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticsViewModel:
    status: str
    latest_valid_date: str
    latest_calculated_date: str
    row_count: int
    warning_codes: tuple[str, ...]
    warning_labels: tuple[str, ...]
    universe: str
    coverage: str
    pipeline_status: str
    last_calculated_at: str
    source: str


def as_date(value: Any) -> date | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else timestamp.date()
