"""Stable vocabulary and value objects for MarketTemperature v0.4.1.

The advisory layer is deliberately a presentation and decision-support layer.
It consumes the frozen v0.3.1 state contract and never emits an execution
instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any


ADVISORY_SIGNALS = (
    "PANIC_WAIT",
    "BUY_WATCH",
    "BUY_REFERENCE",
    "NEUTRAL",
    "HOT_CAUTION",
    "SELL_WATCH",
    "SELL_REFERENCE",
    "DATA_INVALID",
)

BUY_REFERENCES = ("LOW", "MEDIUM", "HIGH")
SELL_REFERENCES = ("LOW", "MEDIUM", "HIGH")
RISK_LEVELS = ("LOW", "NORMAL", "ELEVATED", "HIGH", "EXTREME")
EXPECTED_HORIZONS = (
    "NOT_APPLICABLE",
    "SHORT_TERM",
    "SHORT_TERM_UNCERTAINTY",
    "MEDIUM_TERM",
    "MEDIUM_TERM_RISK",
)
ADVISORY_REGIMES = ("PANIC", "RECOVERY", "NORMAL", "EUPHORIA", "RISK_OFF")
SIGNAL_CONFIDENCE = ("LOW", "MEDIUM", "HIGH")
RESEARCH_EVIDENCE = ("UNPROVEN", "WEAK", "WEAK_TO_MODERATE", "MODERATE", "RELATIVELY_STRONG")


@dataclass(frozen=True)
class MarketAdvisory:
    """Human-readable daily market-environment reference.

    ``advisory_signal`` is the one human-facing top-level signal. Buy and sell
    references are supporting context, not a second signal vocabulary.
    """

    date: date | str
    market_temperature: float | None
    smoothed_temperature: float | None
    state: str
    state_signal: str
    advisory_signal: str
    buy_reference: str
    sell_reference: str
    risk_level: str
    expected_horizon: str
    reference_horizon: str
    signal_confidence: str
    research_evidence: str
    confirming_modules: int
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    headline: str = ""
    summary: str = ""
    why: str = ""
    what_to_watch_next: str = ""
    details: dict[str, str] = field(default_factory=dict)
    advisory_regime: str = "NORMAL"
    short_term_evidence: str = "UNPROVEN"
    medium_term_evidence: str = "UNPROVEN"
    confirming_module_names: tuple[str, ...] = field(default_factory=tuple)
    short_term_warning: bool = False
    not_exact_top: bool = False
    active_advisory_window: bool = False

    def to_record(self) -> dict[str, Any]:
        """Return a flat, Parquet/CSV-safe representation."""

        value = self.date.isoformat() if hasattr(self.date, "isoformat") else str(self.date)
        return {
            "date": value,
            "market_temperature": self.market_temperature,
            "smoothed_temperature": self.smoothed_temperature,
            # Compact audit aliases retained for the requested historical
            # schema; the canonical daily names remain market_temperature and
            # smoothed_temperature.
            "temperature": self.market_temperature,
            "ema_temperature": self.smoothed_temperature,
            "state": self.state,
            "state_signal": self.state_signal,
            "advisory_signal": self.advisory_signal,
            "buy_reference": self.buy_reference,
            "sell_reference": self.sell_reference,
            "risk_level": self.risk_level,
            "expected_horizon": self.expected_horizon,
            "reference_horizon": self.reference_horizon,
            "signal_confidence": self.signal_confidence,
            "confidence": self.signal_confidence,
            "research_evidence": self.research_evidence,
            "confirming_modules": int(self.confirming_modules),
            "confirming_module_names": ";".join(self.confirming_module_names),
            "reason_codes": ";".join(self.reason_codes),
            "headline": self.headline,
            "summary": self.summary,
            "why": self.why,
            "what_to_watch_next": self.what_to_watch_next,
            "human_explanation": self.summary,
            "details": json.dumps(self.details, ensure_ascii=False, sort_keys=True),
            "advisory_regime": self.advisory_regime,
            "short_term_evidence": self.short_term_evidence,
            "medium_term_evidence": self.medium_term_evidence,
            "short_term_warning": bool(self.short_term_warning),
            "not_exact_top": bool(self.not_exact_top),
            "active_advisory_window": bool(self.active_advisory_window),
        }
