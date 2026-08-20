"""Command-line entry points for the free data path and Temperature v0.1."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigError, load_config
from .data.base import ProviderDataUnavailable
from .data.cache import CacheError, ParquetCache
from .data.quality import ProductionDataQualityError, ProductionDataQualityGate, UnavailableMarketFactor, build_market_coverage_daily
from .data.factory import create_provider
from .data.validation import validate_timeseries
from .regime import StateMachineConfig, apply_state_machine, build_regime_indicators
from .scoring.market_temperature import calculate_market_temperature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ashare-sentiment", description="A-share Sentiment Timing Engine")
    parser.add_argument("--config", default="config/default.yaml", help="YAML configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("update", help="Download raw data and update local Parquet cache")
    update.add_argument("--provider", choices=["free", "eastmoney", "tencent", "baostock", "tushare", "akshare"], help="Override configured provider")
    update.add_argument("--dataset", choices=["all", "stock", "index", "market-breadth", "full-market", "limit-up-down", "margin", "options"], default="all")
    update.add_argument("--symbol", help="ts_code, required for stock and index")
    update.add_argument("--start-date", help="Inclusive date, YYYY-MM-DD or YYYYMMDD")
    update.add_argument("--end-date", help="Inclusive date, YYYY-MM-DD or YYYYMMDD")
    update.add_argument("--max-symbols", type=int, help="Smoke-test cap for free full-panel downloads")
    update.add_argument("--request-min-interval", type=float, help="Override provider request spacing for a controlled download")
    update.add_argument("--request-timeout", type=float, help="Override provider socket/HTTP timeout in seconds")
    update.add_argument("--workers", type=int, help="BaoStock full-panel worker processes")

    score = subparsers.add_parser("score", help="Calculate MarketTemperature v0.2 from cached full-market data")
    score.add_argument("--date", help="Return one date; defaults to the latest available date")
    score.add_argument("--start-date", help="Optional output start date")
    score.add_argument("--end-date", help="Optional output end date")
    score.add_argument(
        "--allow-partial-data",
        action="store_true",
        help="Research-only: allow a partial panel, but mark the result INVALID",
    )

    regime = subparsers.add_parser("regime", help="Calculate MarketTemperature v0.3.1 states and turning points")
    regime.add_argument("--date", help="Return one date; defaults to the latest available date")
    regime.add_argument("--start-date", help="Optional output start date")
    regime.add_argument("--end-date", help="Optional output end date")

    advisory = subparsers.add_parser("advisory", help="Generate the MarketTemperature v0.4.1 human advisory layer")
    advisory.add_argument("--date", help="Return one date; defaults to the latest available date")
    advisory.add_argument("--start-date", help="Optional visible output start date")
    advisory.add_argument("--end-date", help="Optional visible output end date")

    gui = subparsers.add_parser("gui", help="Start the local MarketTemperature browser app")
    gui.add_argument("--theme", choices=["system", "light", "dark"], default="system")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-open", action="store_true", help="Start the server without opening a browser tab")
    gui.add_argument("--snapshot", action="store_true", help="Render review snapshots without starting the server")
    gui.add_argument("--snapshot-dir", help="Override the snapshot output directory")

    web = subparsers.add_parser("web", help="Start the local MarketTemperature browser app")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--no-open", action="store_true", help="Start the server without opening a browser tab")
    web.add_argument("--snapshot", action="store_true", help="Render review snapshots without starting the server")
    web.add_argument("--snapshot-dir", help="Override the snapshot output directory")

    event_study = subparsers.add_parser("event-study", help="Run the MarketTemperature v0.4 event study")
    event_study.add_argument(
        "--signal",
        choices=["ICE_REVERSAL_CONFIRMED", "HOT_ROLLOVER_CONFIRMED"],
        help="Filter the printed summary; all artifacts are still generated",
    )
    event_study.add_argument(
        "--benchmark",
        choices=["hs300", "csi1000", "chinext"],
        help="Filter the printed summary; all artifacts are still generated",
    )

    for command, help_text in (
        ("backtest", "Run the signal backtest (phase 5)"),
        ("report", "Generate the research report (phase 5)"),
    ):
        subparsers.add_parser(command, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - declared runtime dependency
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overrides: dict[str, Any] = {}
        if args.command == "update" and (args.provider or args.max_symbols is not None or args.request_min_interval is not None or args.request_timeout is not None or args.workers is not None):
            overrides["data"] = {}
            if args.provider:
                overrides["data"]["provider"] = args.provider
            if args.max_symbols is not None:
                overrides["data"]["max_symbols"] = args.max_symbols
            if args.request_min_interval is not None:
                overrides["data"]["request_min_interval_seconds"] = args.request_min_interval
            if args.request_timeout is not None:
                overrides["data"]["request_timeout_seconds"] = args.request_timeout
            if args.workers is not None:
                overrides["data"]["download_workers"] = args.workers
        config = load_config(args.config, overrides)
        if args.command == "update":
            return _run_update(args, config)
        if args.command == "score":
            return _run_score(args, config)
        if args.command == "regime":
            return _run_regime(args, config)
        if args.command == "advisory":
            return _run_advisory(args, config)
        if args.command in {"gui", "web"}:
            if args.snapshot:
                from .ui.snapshots import generate_gui_snapshots

                paths = generate_gui_snapshots(config, args.snapshot_dir)
                for path in paths:
                    print(f"Saved GUI snapshot: {path}")
                return 0
            from .webapp.server import run_web

            return run_web(args.config, host=args.host, port=args.port, open_browser=not args.no_open)
        if args.command == "event-study":
            from .event_study.reporting import run_event_study

            run_event_study(args.config, signal=args.signal, benchmark=args.benchmark)
            return 0
        parser.error(f"{args.command} is reserved for a later phase; this round provides update, score, regime, advisory and event-study.")
    except (CacheError, ConfigError, ProviderDataUnavailable, ProductionDataQualityError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


def _run_update(args: argparse.Namespace, config: dict[str, Any]) -> int:
    today = date.today()
    end_date = args.end_date or today.isoformat()
    start_date = args.start_date or config["data"].get("history_start") or (today - timedelta(days=30)).isoformat()
    provider = create_provider(config)

    if args.dataset == "all":
        datasets = ["market-breadth", "limit-up-down"]
        datasets.extend(f"index:{item['ts_code']}" for item in config.get("benchmarks", [])[:3])
        for dataset in datasets:
            dataset_args = argparse.Namespace(**vars(args))
            if dataset.startswith("index:"):
                dataset_args.dataset = "index"
                dataset_args.symbol = dataset.split(":", 1)[1]
            else:
                dataset_args.dataset = dataset
            _run_update_one(dataset_args, config, provider, start_date, end_date)
        print("Updated all MarketTemperature v0.1 input datasets")
        return 0
    return _run_update_one(args, config, provider, start_date, end_date)


def _run_update_one(args: argparse.Namespace, config: dict[str, Any], provider: Any, start_date: str, end_date: str) -> int:
    if args.dataset == "limit-up-down":
        configured_limit_provider = str(config["data"].get("limit_provider", "")).lower()
        if provider.name == "baostock" and configured_limit_provider in {"eastmoney", "eastmoney-free"}:
            limit_config = dict(config)
            limit_config["data"] = dict(config["data"])
            limit_config["data"]["provider"] = configured_limit_provider
            provider = create_provider(limit_config)
            print(f"Limit dataset source switch: {provider.name}")
    requested_dataset = (
        f"stock_{args.symbol}" if args.dataset == "stock" and args.symbol else
        f"index_{args.symbol}" if args.dataset == "index" and args.symbol else
        {"market-breadth": "market_breadth", "full-market": "market_breadth", "limit-up-down": "limit_up_down", "margin": "margin", "options": "options"}.get(args.dataset, args.dataset)
    )
    start_date = _effective_start_date(args, config, requested_dataset, start_date)
    if args.dataset == "stock":
        if not args.symbol:
            raise ValueError("--symbol is required for --dataset stock")
        frame = provider.get_stock_daily(start_date, end_date, args.symbol)
        dataset_name = f"stock_{args.symbol}"
        symbol = args.symbol
    elif args.dataset == "index":
        if not args.symbol:
            raise ValueError("--symbol is required for --dataset index")
        frame = provider.get_index_daily(start_date, end_date, args.symbol)
        dataset_name = f"index_{args.symbol}"
        symbol = args.symbol
    elif args.dataset in {"market-breadth", "full-market"}:
        frame = provider.get_market_breadth(start_date, end_date)
        dataset_name = "market_breadth"
        symbol = None
    elif args.dataset == "limit-up-down":
        frame = provider.get_limit_up_down(start_date, end_date)
        dataset_name = "limit_up_down"
        symbol = None
    elif args.dataset == "margin":
        frame = provider.get_margin_data(start_date, end_date)
        dataset_name = "margin"
        symbol = None
    else:
        frame = provider.get_option_data(start_date, end_date)
        dataset_name = "options"
        symbol = None

    if frame.empty:
        raise ValueError(f"Provider returned no rows for {args.dataset} ({start_date} to {end_date})")
    key_columns = [column for column in ("ts_code", "trade_date") if column in frame.columns]
    issues = validate_timeseries(frame, key_columns=key_columns or None)
    blocking = [issue for issue in issues if issue.severity == "error"]
    if blocking:
        detail = ", ".join(issue.code for issue in blocking)
        raise ValueError(f"data validation failed: {detail}")

    cache_root = Path(config["data"]["cache_root"])
    cache = ParquetCache(cache_root)
    metadata = cache.metadata_now(
        source=provider.name,
        frame=frame,
        symbol=symbol,
        version="0.2" if args.dataset in {"market-breadth", "full-market", "limit-up-down"} else "0.1",
        endpoint="baostock.query_history_k_data_plus" if provider.name == "baostock" else None,
        notes="; ".join(f"{issue.code}: {issue.message}" for issue in issues) or None,
    )
    # A complete, explicitly bounded full-panel/limit refresh replaces the
    # dataset atomically.  Incremental updates still use keyed upserts.
    replace_complete = bool(
        getattr(args, "start_date", None)
        and getattr(args, "end_date", None)
        and args.dataset in {"market-breadth", "full-market", "limit-up-down"}
    )
    path = cache.save(dataset_name, frame, metadata) if replace_complete else cache.upsert(
        dataset_name, frame, metadata, key_columns=key_columns or None
    )
    print(f"Updated {dataset_name}: {len(frame):,} rows -> {path}")
    if issues:
        print("Validation notes:")
        for issue in issues:
            print(f"- [{issue.severity}] {issue.code}: {issue.message} ({issue.rows})")
    if args.dataset in {"market-breadth", "full-market"}:
        processed = ParquetCache(config["data"]["processed_root"])
        processed.upsert("full_market_daily", frame, metadata, key_columns=["ts_code", "trade_date"])
        coverage = build_market_coverage_daily(frame, config=config)
        coverage_metadata = processed.metadata_now(
            source=provider.name,
            frame=coverage,
            version="0.2",
            endpoint=metadata.endpoint,
            notes="Daily full-market panel coverage diagnostics",
        )
        processed.upsert("market_coverage_daily", coverage, coverage_metadata, key_columns=["trade_date"])
    if config["data"].get("survivorship_bias_warning", True) and args.dataset in {"market-breadth", "full-market"}:
        print("WARNING: current-universe filtering is not point-in-time; survivorship bias may remain.")
    return 0


def _effective_start_date(args: argparse.Namespace, config: dict[str, Any], dataset: str, requested: str) -> str:
    """Use a small overlap on cached data unless the user explicitly pins start_date."""
    if getattr(args, "start_date", None):
        return requested
    cache = ParquetCache(config["data"]["cache_root"])
    if not cache.exists(dataset):
        return requested
    try:
        last_date = date.fromisoformat(cache.load_metadata(dataset).date_end or "")
    except (CacheError, ValueError):
        return requested
    return max(date.fromisoformat(str(requested)[:10]), last_date - timedelta(days=5)).isoformat()


def _run_score(args: argparse.Namespace, config: dict[str, Any]) -> int:
    cache = ParquetCache(config["data"]["cache_root"])
    stock_panel = cache.load("market_breadth")
    limit_panel = cache.load("limit_up_down")
    ProductionDataQualityGate(config).validate(
        stock_panel,
        allow_partial_data=bool(args.allow_partial_data),
    )
    if not args.allow_partial_data and not config["data"].get("allow_approximate_limit_rules", False):
        if "limit_method" in limit_panel.columns and limit_panel["limit_method"].astype(str).str.contains("approx", case=False).any():
            raise UnavailableMarketFactor(
                "Limit-up/down data is only a BaoStock board-band approximation. "
                "Run the configured free limit-pool source before a production score, "
                "or use --allow-partial-data for research-only diagnostics."
            )
        try:
            limit_metadata = cache.load_metadata("limit_up_down")
        except CacheError:
            limit_metadata = None
        if limit_metadata is not None and "baostock" in limit_metadata.source.lower():
            raise UnavailableMarketFactor(
                "The cached limit-up/down dataset was produced by BaoStock approximation; "
                "production v0.2 requires a verified limit-pool source."
            )
    index_frames: dict[str, Any] = {}
    aliases = {"沪深300": "hs300", "中证1000": "csi1000", "创业板指": "chinext"}
    for benchmark in config.get("benchmarks", [])[:3]:
        ts_code = benchmark["ts_code"]
        alias = aliases.get(benchmark.get("name"), ts_code.replace(".", "_").lower())
        index_frames[alias] = cache.load(f"index_{ts_code}")
    daily = calculate_market_temperature(
        stock_panel,
        limit_panel,
        index_frames,
        config,
        production=True,
        allow_partial_data=bool(args.allow_partial_data),
    )
    if args.start_date:
        daily = daily[daily["trade_date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        daily = daily[daily["trade_date"] <= pd.Timestamp(args.end_date)]
    if daily.empty:
        raise ValueError("no scored dates remain after the requested date filter")
    processed = ParquetCache(config["data"]["processed_root"])
    metadata = processed.metadata_now(
        source="market-temperature-v0.2",
        frame=daily,
        version="0.2",
        notes=str(daily["data_quality_warnings"].iloc[-1]),
    )
    processed.save("market_sentiment_daily", daily, metadata)
    coverage = build_market_coverage_daily(
        stock_panel,
        breadth_coverage=daily.set_index("trade_date").get("breadth_coverage"),
        profit_effect_coverage=daily.set_index("trade_date").get("profit_effect_coverage"),
        liquidity_coverage=daily.set_index("trade_date").get("liquidity_coverage"),
        config=config,
    )
    coverage_metadata = processed.metadata_now(
        source="market-temperature-v0.2",
        frame=coverage,
        version="0.2",
        notes="Daily coverage after factor calculation",
    )
    processed.save("market_coverage_daily", coverage, coverage_metadata)
    selected_date = pd.Timestamp(args.date) if args.date else daily["trade_date"].max()
    selected = daily[daily["trade_date"] == selected_date]
    if selected.empty:
        raise ValueError(f"no scored row for {selected_date.date()}")
    row = selected.iloc[-1]
    if not args.allow_partial_data and bool(row.get("integrity_invalid", False)):
        raise ProductionDataQualityError(
            f"MARKET TEMPERATURE INVALID on {selected_date.date()}: "
            f"{row.get('integrity_warnings', 'daily integrity gate failed')}"
        )
    print(_format_score_output(row))
    print(f"Saved {len(daily):,} rows to {processed.path_for('market_sentiment_daily')}")
    return 0


def _format_score_output(row: Any) -> str:
    def fmt(value: Any, digits: int = 1) -> str:
        return "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"

    return "\n".join(
        [
            "========================================",
            f"A-SHARE SENTIMENT v0.2 {row['trade_date'].date()}",
            "========================================",
            "",
            f"Market Temperature: {fmt(row['raw_market_temperature'])} / 100",
            f"Data Quality:      {row.get('market_temperature_quality', row.get('data_quality_status', 'N/A'))}",
            f"Confidence:        {row.get('confidence', 'N/A')}",
            "",
            f"Breadth:        {fmt(row['breadth_score'])}",
            f"Profit Effect:  {fmt(row['profit_effect_score'])}",
            f"Liquidity:      {fmt(row['liquidity_score'])}",
            f"Stretch:        {fmt(row['stretch_score'])}",
            "Options:        N/A",
            "",
            f"Effective weights: {row.get('market_temperature_effective_weights', 'N/A')}",
            f"Missing factors:   {row.get('missing_factors', 'N/A')}",
            f"Warnings:          {row.get('data_quality_warnings', 'N/A')}",
        ]
    )


def _run_regime(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Build and persist the v0.3.1 daily state table from the v0.2.1 table."""
    processed = ParquetCache(config["data"]["processed_root"])
    if not processed.exists("market_sentiment_daily"):
        raise ValueError("market_sentiment_daily is missing; run `python -m ashare_sentiment score` first")
    daily = processed.load("market_sentiment_daily")
    if daily.empty:
        raise ValueError("market_sentiment_daily contains no rows")
    indicators = build_regime_indicators(daily, config)
    regime = apply_state_machine(indicators, config)
    regime["date"] = pd.to_datetime(regime["trade_date"], errors="coerce").dt.normalize()
    metadata = processed.metadata_now(
        source="market-temperature-v0.3.1-state-machine",
        frame=regime,
        version="0.3.1",
        notes="Causal EMA, episode anchors and zone hysteresis; INVALID days remain invalid.",
    )
    processed.save("market_state_daily_v031", regime, metadata)
    visible = regime.copy()
    if args.start_date:
        visible = visible[visible["trade_date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        visible = visible[visible["trade_date"] <= pd.Timestamp(args.end_date)]
    selected_date = pd.Timestamp(args.date).normalize() if args.date else visible["trade_date"].max()
    selected = visible[visible["trade_date"].eq(selected_date)]
    if selected.empty:
        raise ValueError(f"no regime row for {selected_date.date()}")
    print(_format_regime_output(selected.iloc[-1]))
    print(f"Saved {len(regime):,} rows to {processed.path_for('market_state_daily_v031')}")
    return 0


def _run_advisory(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Generate and print the latest human-readable v0.4.1 advisory."""

    from .advisory.reporting import format_advisory_output, run_advisory

    result = run_advisory(
        config,
        date_value=args.date,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(format_advisory_output(result["selected"]))
    print(f"Saved {len(result['advisory']):,} rows to {result['processed_path']}")
    print(f"Saved audit history to {result['history_path']}")
    return 0


def _format_regime_output(row: Any) -> str:
    def fmt(value: Any, digits: int = 1) -> str:
        return "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"

    return "\n".join(
        [
            "========================================",
            f"A-SHARE MARKET REGIME v0.3.1 {row['trade_date'].date()}",
            "========================================",
            "",
            f"Raw Temperature:       {fmt(row.get('raw_temperature'))}",
            f"Smoothed Temperature:   {fmt(row.get('smoothed_temperature'))}",
            f"3D Slope:              {fmt(row.get('slope3'))}",
            f"Rolling 10D Low:       {fmt(row.get('rolling_low_10'))}",
            f"Recovery From Low:     {fmt(row.get('recovery_from_low'))}",
            f"Drop From High:        {fmt(row.get('drop_from_high'))}",
            "",
            "----------------------------------------",
            "",
            f"State:                 {row.get('state', 'N/A')}",
            f"Signal:                {row.get('signal', 'NONE')}",
            "",
            "----------------------------------------",
            "",
            f"Breadth:               {fmt(row.get('breadth_score'))}",
            f"Profit Effect:         {fmt(row.get('profit_effect_score'))}",
            f"Liquidity:              {fmt(row.get('liquidity_score'))}",
            f"Stretch:                {fmt(row.get('stretch_score'))}",
            "",
            f"Data Quality:           {row.get('quality', row.get('market_temperature_quality', 'N/A'))}",
            f"Confidence:             {row.get('confidence', 'N/A')}",
            f"Warnings:               {row.get('warnings', '') or 'NONE'}",
        ]
    )
