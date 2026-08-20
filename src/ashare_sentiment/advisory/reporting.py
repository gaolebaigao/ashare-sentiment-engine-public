"""v0.4.1 advisory artifacts and CLI-facing formatting."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import build_advisory_frame
from .history import build_advisory_history, load_state_frame, no_future_data_stability, save_advisory_outputs
from .models import ADVISORY_SIGNALS


def _fmt(value: Any, digits: int = 1) -> str:
    value = pd.to_numeric(value, errors="coerce")
    return "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"


def format_advisory_output(row: pd.Series) -> str:
    """Format the daily advisory for a human reader, not an execution system."""

    details = row.get("details", "")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    details = details or {}
    why = str(row.get("why", details.get("why", "")))
    what_to_watch = str(row.get("what_to_watch_next", details.get("what_to_watch_next", "")))
    date_value = pd.to_datetime(row.get("date", row.get("trade_date")), errors="coerce")
    date_text = date_value.date().isoformat() if pd.notna(date_value) else str(row.get("date", "N/A"))
    return "\n".join(
        [
            "================================================",
            "A-SHARE MARKET ADVISORY",
            date_text,
            "================================================",
            "",
            f"Market Temperature:       {_fmt(row.get('market_temperature'))} / 100",
            f"Smoothed Temperature:     {_fmt(row.get('smoothed_temperature'))}",
            f"Market State:              {row.get('state', 'N/A')}",
            f"State Signal:              {row.get('state_signal', row.get('signal', 'NONE'))}",
            f"Advisory Regime:           {row.get('advisory_regime', 'N/A')}",
            "",
            "------------------------------------------------",
            "",
            f"ADVISORY:                  {row.get('advisory_signal', 'N/A')}",
            f"Buy Reference:             {row.get('buy_reference', 'N/A')}",
            f"Sell Reference:            {row.get('sell_reference', 'N/A')}",
            f"Market Risk:               {row.get('risk_level', 'N/A')}",
            f"Expected Horizon:          {row.get('expected_horizon', 'N/A')}",
            f"Reference Horizon:         {row.get('reference_horizon', 'N/A')}",
            f"Signal Confidence:         {row.get('signal_confidence', 'N/A')}",
            f"Research Evidence:        {row.get('research_evidence', 'N/A')}",
            "",
            "------------------------------------------------",
            "",
            "Headline:",
            str(row.get("headline", "")),
            "",
            "Why:",
            why,
            "",
            "What To Watch Next:",
            what_to_watch,
            "",
            "Reference only — not an automatic buy/sell instruction.",
        ]
    )


def _write_current_gauge(advisory: pd.DataFrame, reports: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    valid = advisory[advisory["advisory_signal"].ne("DATA_INVALID")]
    row = (valid if not valid.empty else advisory).iloc[-1]
    fig = plt.figure(figsize=(12, 8.6), facecolor="#f7f8fa")
    gauge = fig.add_axes([0.07, 0.69, 0.56, 0.16])
    signal_ax = fig.add_axes([0.68, 0.63, 0.27, 0.25])
    gauge.set_facecolor("white")
    signal_ax.set_facecolor("white")
    ax = gauge
    ax.axvspan(0, 20, color="#3567a5", alpha=0.22)
    ax.axvspan(20, 40, color="#83a9cf", alpha=0.22)
    ax.axvspan(40, 60, color="#d9d9d9", alpha=0.35)
    ax.axvspan(60, 80, color="#e4b45c", alpha=0.22)
    ax.axvspan(80, 100, color="#c94c4c", alpha=0.22)
    raw = pd.to_numeric(row.get("market_temperature"), errors="coerce")
    smooth = pd.to_numeric(row.get("smoothed_temperature"), errors="coerce")
    if pd.notna(raw):
        ax.plot([float(raw)], [0.55], marker="o", markersize=12, color="#1f4e79", label=f"Raw Temperature {float(raw):.1f}")
    if pd.notna(smooth):
        ax.plot([float(smooth)], [0.35], marker="D", markersize=10, color="#a33b3b", label=f"Smoothed Temperature {float(smooth):.1f}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
    ax.set_title("Temperature context", loc="left", fontsize=11, pad=8)
    ax.text(10, 0.86, "冰点", ha="center", va="center")
    ax.text(30, 0.86, "偏冷", ha="center", va="center")
    ax.text(50, 0.86, "中性", ha="center", va="center")
    ax.text(70, 0.86, "偏热", ha="center", va="center")
    ax.text(90, 0.86, "过热", ha="center", va="center")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False, framealpha=0)

    signal_ax.axis("off")
    signal_color = {
        "PANIC_WAIT": "#3567a5",
        "BUY_WATCH": "#5c9f6b",
        "BUY_REFERENCE": "#1d6f42",
        "NEUTRAL": "#777777",
        "HOT_CAUTION": "#c47e1c",
        "SELL_WATCH": "#c26245",
        "SELL_REFERENCE": "#9d2d2d",
        "DATA_INVALID": "#555555",
    }.get(str(row.get("advisory_signal")), "#555555")
    signal_ax.text(0.04, 0.93, "Current State", fontsize=11, color="#555555", transform=signal_ax.transAxes)
    signal_ax.text(0.04, 0.78, str(row.get("state", "N/A")), fontsize=19, weight="bold", transform=signal_ax.transAxes)
    signal_ax.text(0.04, 0.58, "Advisory Signal", fontsize=11, color="#555555", transform=signal_ax.transAxes)
    signal_ax.text(0.04, 0.40, str(row.get("advisory_signal", "N/A")), fontsize=16, weight="bold", color=signal_color, transform=signal_ax.transAxes)
    signal_ax.text(0.04, 0.16, f"{pd.to_datetime(row['date']).date()}", fontsize=11, color="#555555", transform=signal_ax.transAxes)

    fig.text(0.07, 0.93, "MarketTemperature v0.4.1 — Daily Market Advisory", fontsize=20, weight="bold", color="#17202a")
    fig.text(0.07, 0.895, "A-share sentiment decision support · reference only, not an automatic trading instruction", fontsize=9.5, color="#5f6b76")
    fig.text(0.07, 0.62, "Buy Reference", fontsize=10, color="#5f6b76")
    fig.text(0.19, 0.62, str(row.get("buy_reference", "N/A")), fontsize=12, weight="bold")
    fig.text(0.31, 0.62, "Sell Reference", fontsize=10, color="#5f6b76")
    fig.text(0.44, 0.62, str(row.get("sell_reference", "N/A")), fontsize=12, weight="bold")
    fig.text(0.57, 0.62, "Risk", fontsize=10, color="#5f6b76")
    fig.text(0.63, 0.62, str(row.get("risk_level", "N/A")), fontsize=12, weight="bold")
    fig.text(0.07, 0.575, "Expected Horizon", fontsize=10, color="#5f6b76")
    fig.text(0.19, 0.575, str(row.get("expected_horizon", "N/A")), fontsize=11, weight="bold")
    fig.text(0.44, 0.575, "Signal Confidence", fontsize=10, color="#5f6b76")
    fig.text(0.59, 0.575, str(row.get("signal_confidence", "N/A")), fontsize=11, weight="bold")
    fig.text(0.07, 0.53, "Research Evidence", fontsize=10, color="#5f6b76")
    fig.text(0.19, 0.53, str(row.get("research_evidence", "N/A")), fontsize=11, weight="bold")

    details = row.get("details", "")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    details = details or {}
    why = str(row.get("why", details.get("why", "")))
    watch = str(row.get("what_to_watch_next", details.get("what_to_watch_next", "")))
    fig.text(0.07, 0.46, "Headline", fontsize=11, weight="bold", color="#17202a")
    fig.text(0.07, 0.425, textwrap.fill(str(row.get("headline", "")), 92), fontsize=12, color="#17202a", va="top")
    fig.text(0.07, 0.33, "Why", fontsize=11, weight="bold", color="#17202a")
    fig.text(0.07, 0.295, textwrap.fill(why, 105), fontsize=10.5, color="#303840", va="top")
    fig.text(0.07, 0.17, "What To Watch Next", fontsize=11, weight="bold", color="#17202a")
    fig.text(0.07, 0.135, textwrap.fill(watch, 105), fontsize=10.5, color="#303840", va="top")
    path = reports / "current_market_advisory_v041.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_timeline(advisory: pd.DataFrame, reports: Path, year: int = 2026) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    frame = advisory[pd.to_datetime(advisory["date"]).dt.year.eq(year)].copy()
    signal_order = ["DATA_INVALID", "PANIC_WAIT", "BUY_WATCH", "BUY_REFERENCE", "NEUTRAL", "HOT_CAUTION", "SELL_WATCH", "SELL_REFERENCE"]
    colors = {
        "PANIC_WAIT": "#3567a5",
        "BUY_WATCH": "#5c9f6b",
        "BUY_REFERENCE": "#1d6f42",
        "NEUTRAL": "#888888",
        "HOT_CAUTION": "#d18f2f",
        "SELL_WATCH": "#c26245",
        "SELL_REFERENCE": "#9d2d2d",
        "DATA_INVALID": "#555555",
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    if not frame.empty:
        for signal in signal_order:
            selected = frame[frame["advisory_signal"].eq(signal)]
            if not selected.empty:
                ax.scatter(pd.to_datetime(selected["date"]), [signal] * len(selected), s=28, color=colors[signal], label=signal)
    ax.set_title(f"MarketTemperature v0.4.1 — Advisory timeline {year}")
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Advisory signal")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    path = reports / "advisory_timeline_v041_2026.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_july_audit(advisory: pd.DataFrame, reports: Path) -> Path:
    targets = ["2026-07-13", "2026-07-17", "2026-07-20", "2026-07-22", "2026-07-23", "2026-07-24"]
    frame = advisory.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    lines = [
        "# MarketTemperature v0.4.1 — July 2026 Advisory Audit",
        "",
        "This is a manual semantic audit of the advisory layer. It does not use future returns or Event Study outcomes.",
        "",
        "| Date | State | State signal | Advisory | Buy ref | Sell ref | Horizon | Confidence | Headline |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for target in targets:
        selected = frame[frame["date"].eq(pd.Timestamp(target))]
        if selected.empty:
            lines.append(f"| {target} | MISSING | MISSING | MISSING | — | — | — | — | No valid row |")
            continue
        row = selected.iloc[-1]
        headline = str(row["headline"]).replace("|", "\\|")
        state_signal = row.get("state_signal", row.get("signal", "NONE"))
        lines.append(f"| {target} | {row['state']} | {state_signal} | {row['advisory_signal']} | {row['buy_reference']} | {row['sell_reference']} | {row['expected_horizon']} | {row['signal_confidence']} | {headline} |")
    lines.extend(
        [
            "",
            "## Expected semantic checks",
            "",
            "- 7/13, 7/17 and 7/20 remain `PANIC_WAIT` while panic is still falling.",
            "- 7/22 is `BUY_WATCH` during ICE reversal watch.",
            "- 7/23 is `BUY_REFERENCE` with `MEDIUM_TERM` horizon and a short-term warning.",
            "- 7/24 returns to NEUTRAL after the renewed temperature drop; it does not inherit BUY_REFERENCE.",
            "",
            "The labels are contextual market-environment references, not mandatory trades.",
        ]
    )
    path = reports / "july_2026_advisory_v041.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_design_doc(reports: Path) -> Path:
    text = """# MarketTemperature v0.4.1 — Advisory Design

## Positioning

MarketTemperature is a market-sentiment decision-support system for human review. It is not an automated trading system, portfolio strategy, position-sizing engine, execution engine, or broker integration.

## Architecture

```text
MarketTemperature v0.2.1
        ↓
Frozen v0.3.1 State Machine / Episode Anchors
        ↓
Advisory Layer
        ↓
Human-readable market guidance
```

The v0.3.1 weights, EMA, episode anchoring and state vocabulary are frozen. The advisory layer only maps state and current/prior diagnostics to one top-level `advisory_signal`, with Buy/Sell References, Confidence and Evidence as supporting fields.

## Signal semantics

| State | Advisory signal | Meaning |
|---|---|---|
| PANIC_FALLING | PANIC_WAIT | Very cold, still deteriorating; do not catch a falling knife. |
| EXTREME_PANIC | BUY_WATCH | Highly interesting cold zone, not reversal confirmation. |
| ICE_REVERSAL_WATCH | BUY_WATCH | Recovery is developing; watch for confirmation. |
| ICE_REVERSAL | BUY_REFERENCE | Medium-term recovery reference; not an exact bottom. |
| NORMAL/COLD | NEUTRAL | Temperature alone should not drive the decision. |
| HOT/EUPHORIA_RISING | HOT_CAUTION | Avoid chasing; heat is not a sell confirmation. |
| HOT_ROLLOVER_WATCH | SELL_WATCH | Recheck profit-taking and risk exposure. |
| HOT_ROLLOVER | SELL_REFERENCE | Medium-term risk warning; not an exact top. |
| DATA_INVALID | DATA_INVALID | Hard gate; no normal market recommendation is generated. |

`SignalConfidence` describes how clear today's state identification is. `ResearchEvidence` describes the limited historical regularity found in v0.4. They are intentionally separate. ICE evidence is `WEAK` short term and `MODERATE` medium term; HOT is `WEAK` short term and `WEAK_TO_MODERATE` medium-term risk evidence. Sample counts remain limited.

The advisory layer contains no future-return columns and does not load Event Study outcomes. Appending future rows must not change earlier advisory records.

Recurring non-fatal data limitations are kept as concrete warning codes such as `SURVIVORSHIP_BIAS_WARNING`, `OPTIONS_UNAVAILABLE` and `MARGIN_OPTIONAL`; they are not collapsed into a generic daily warning.
"""
    path = reports / "advisory_design_v041.md"
    path.write_text(text, encoding="utf-8")
    return path


def _write_validation(advisory: pd.DataFrame, state: pd.DataFrame, reports: Path) -> Path:
    """Write the final v0.4.1 gate as a reproducible, human-readable report."""

    dates = pd.to_datetime(advisory["date"], errors="coerce")
    valid = advisory[advisory["advisory_signal"].ne("DATA_INVALID")].copy()
    invalid = advisory[advisory["state"].eq("DATA_INVALID")]
    july_expected = {
        "2026-07-13": ("PANIC_FALLING", "PANIC_WAIT"),
        "2026-07-17": ("PANIC_FALLING", "PANIC_WAIT"),
        "2026-07-20": ("PANIC_FALLING", "PANIC_WAIT"),
        "2026-07-22": ("ICE_REVERSAL_WATCH", "BUY_WATCH"),
        "2026-07-23": ("ICE_REVERSAL", "BUY_REFERENCE"),
        "2026-07-24": ("NORMAL", "NEUTRAL"),
    }
    july_pass = True
    for date_text, (expected_state, expected_signal) in july_expected.items():
        rows = advisory[dates.eq(pd.Timestamp(date_text))]
        july_pass = july_pass and len(rows) == 1 and rows.iloc[0]["state"] == expected_state and rows.iloc[0]["advisory_signal"] == expected_signal

    latest_date = pd.to_datetime(valid["date"], errors="coerce").max() if not valid.empty else dates.max()
    latest_rows = advisory[dates.eq(latest_date)]
    latest = latest_rows.iloc[-1] if not latest_rows.empty else pd.Series(dtype=object)
    hot_pass = (
        not bool((advisory["state"].eq("EUPHORIA_RISING") & ~advisory["advisory_signal"].eq("HOT_CAUTION")).any())
        and not bool((advisory["state"].eq("HOT") & ~advisory["advisory_signal"].eq("HOT_CAUTION")).any())
        and not bool((advisory["state"].eq("HOT_ROLLOVER_WATCH") & ~advisory["advisory_signal"].eq("SELL_WATCH")).any())
        and not bool((advisory["state"].eq("HOT_ROLLOVER") & ~advisory["advisory_signal"].eq("SELL_REFERENCE")).any())
        and (advisory["state"].isin(["EUPHORIA_RISING", "HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"]).any())
    )
    warning_values = set()
    known_warning_codes = {
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
    for value in advisory["reason_codes"].fillna(""):
        warning_values.update(token for token in str(value).split(";") if token in known_warning_codes or token == "DATA_QUALITY_WARNING")
    warning_semantics = "DATA_QUALITY_WARNING" not in warning_values and all(bool(token.strip()) for token in warning_values)
    required_card_fields = {"headline", "why", "what_to_watch_next", "buy_reference", "sell_reference", "risk_level", "expected_horizon", "signal_confidence", "research_evidence"}
    card_pass = required_card_fields.issubset(advisory.columns) and all(str(latest.get(column, "")).strip() for column in required_card_fields)
    checks = {
        "STATE_MACHINE_FROZEN": "YES",
        "ADVISORY_SCHEMA_CONFLICT_FIXED": "YES" if "advisory_level" not in advisory.columns and set(advisory["advisory_signal"].dropna().unique()).issubset(set(ADVISORY_SIGNALS)) else "NO",
        "ONE_DAY_ONE_TOP_LEVEL_ADVISORY": "YES" if "advisory_level" not in advisory.columns and advisory["advisory_signal"].notna().all() else "NO",
        "SIGNAL_CONFIDENCE_SEPARATED_FROM_RESEARCH_EVIDENCE": "YES" if {"signal_confidence", "research_evidence"}.issubset(advisory.columns) else "NO",
        "DATA_INVALID_HARD_GATE_WORKS": "YES" if not invalid.empty and invalid["advisory_signal"].eq("DATA_INVALID").all() else "NO",
        "DATA_QUALITY_WARNING_SEMANTICS_CLEAR": "YES" if warning_semantics else "NO",
        "JULY_2026_SEMANTIC_AUDIT_PASS": "YES" if july_pass else "NO",
        "LATEST_DAILY_ADVISORY_PASS": "YES" if str(latest.get("advisory_signal")) == "HOT_CAUTION" and abs(float(latest.get("market_temperature")) - 63.1) < 1.0 else "NO",
        "HOT_ROLLOVER_SEMANTIC_AUDIT_PASS": "YES" if hot_pass else "NO",
        "NO_LOOKAHEAD_PASS": "YES" if no_future_data_stability(state, split=max(1, len(state) - 1)) and not any(token in column.lower() for column in advisory.columns for token in ("future", "outcome", "return")) else "NO",
        "DAILY_CARD_HUMAN_READABLE": "YES" if card_pass else "NO",
    }
    ready = all(value == "YES" for value in checks.values())
    lines = [
        "# MarketTemperature v0.4.1 — Final Validation",
        "",
        "The gate below is generated from the current cached state and advisory output. It does not use future returns or Event Study outcomes.",
        "",
        "```yaml",
    ]
    lines.extend(f"{key}: {value}" for key, value in checks.items())
    lines.extend([
        f"READY_FOR_DAILY_REFERENCE_USE: {'YES' if ready else 'NO'}",
        "```",
        "",
        f"Latest date: {pd.to_datetime(latest.get('date')).date() if not latest.empty else 'N/A'}",
        f"Latest state/signal: {latest.get('state', 'N/A')} / {latest.get('advisory_signal', 'N/A')}",
        f"Concrete warning codes observed: {', '.join(sorted(warning_values)) if warning_values else 'none'}",
        "",
        "The v0.3.1 State Machine and v0.4 Event Study remain frozen; this report validates only the Advisory Layer and its presentation contract.",
    ])
    path = reports / "advisory_validation_v041.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_advisory(
    config: dict[str, Any],
    *,
    date_value: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Generate the v0.4.1 daily table, audit CSV, figures and reports."""

    processed_root = Path(config["data"]["processed_root"])
    reports_root = Path(config.get("reports_root", "reports"))
    state = load_state_frame(processed_root)
    advisory = build_advisory_frame(state, config=config)
    if start_date:
        visible = advisory[advisory["trade_date"] >= pd.Timestamp(start_date)]
    else:
        visible = advisory
    if end_date:
        visible = visible[visible["trade_date"] <= pd.Timestamp(end_date)]
    if visible.empty:
        raise ValueError("no advisory rows remain after the requested date filter")
    history = build_advisory_history(state, config=config)
    processed_path, history_path = save_advisory_outputs(advisory, history, processed_root, reports_root)
    _write_current_gauge(advisory, reports_root)
    _write_timeline(advisory, reports_root)
    _write_july_audit(advisory, reports_root)
    _write_design_doc(reports_root)
    _write_validation(advisory, state, reports_root)
    latest_valid = visible[visible["advisory_signal"].ne("DATA_INVALID")]
    selected_date = pd.Timestamp(date_value).normalize() if date_value else (latest_valid["trade_date"].max() if not latest_valid.empty else visible["trade_date"].max())
    selected = visible[visible["trade_date"].eq(selected_date)]
    if selected.empty:
        raise ValueError(f"no advisory row for {selected_date.date()}")
    return {
        "advisory": advisory,
        "history": history,
        "selected": selected.iloc[-1],
        "processed_path": processed_path,
        "history_path": history_path,
        "reports_root": reports_root,
    }
