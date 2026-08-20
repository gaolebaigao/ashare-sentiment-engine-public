"""Human-readable Chinese explanations for daily advisory objects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _fmt(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "N/A" if pd.isna(number) else f"{float(number):.1f}"


def _direction_text(row: pd.Series) -> str:
    improving = []
    deteriorating = []
    for name, label in (
        ("breadth", "Breadth"),
        ("profit_effect", "Profit Effect"),
        ("liquidity", "Liquidity"),
        ("stretch", "Stretch"),
    ):
        value = pd.to_numeric(row.get(f"{name}_delta3"), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            improving.append(label)
        elif pd.notna(value) and float(value) < 0:
            deteriorating.append(label)
    fragments = []
    if improving:
        fragments.append("改善：" + "、".join(improving))
    if deteriorating:
        fragments.append("走弱：" + "、".join(deteriorating))
    slope = pd.to_numeric(row.get("slope3"), errors="coerce")
    if pd.notna(slope):
        fragments.append(f"EMA 斜率 {'转正/上行' if float(slope) > 0 else '转负/下行' if float(slope) < 0 else '接近持平'}")
    return "；".join(fragments) if fragments else "内部模块方向暂不明显。"


def _watch_next(state: str) -> str:
    if state == "PANIC_FALLING":
        return "观察 Slope3 是否转正，以及 Breadth、Profit Effect 和 EMA 是否开始同步修复。"
    if state == "EXTREME_PANIC":
        return "观察恐慌是否停止扩散，以及内部修复是否继续扩散。"
    if state == "ICE_REVERSAL_WATCH":
        return "观察内部修复是否继续扩散，以及 State Machine 是否升级为 ICE_REVERSAL_CONFIRMED。"
    if state == "ICE_REVERSAL":
        return "短期仍可能震荡或继续回撤；关注修复是否保持，以及是否重新跌回 PANIC_FALLING。"
    if state == "COLD":
        return "市场是否从低温区出现持续修复；不要仅因温度偏低就当作买入信号。"
    if state == "HOT":
        return "观察 Smoothed Temperature 是否继续下降，以及 Breadth、Profit Effect、Liquidity、Stretch 是否开始同步恶化；若高温 episode 持续退潮，可能升级为 SELL_WATCH。"
    if state == "EUPHORIA_RISING":
        return "观察 Smoothed Temperature 是否停止上升，以及 Breadth、Profit Effect、Liquidity、Stretch 是否开始同步恶化。"
    if state == "HOT_ROLLOVER_WATCH":
        return "观察内部恶化是否继续并升级为 HOT_ROLLOVER_CONFIRMED；如果高温重新升温，则 SELL_WATCH 应取消。"
    if state == "HOT_ROLLOVER":
        return "关注退潮是否继续扩大；该状态用于中期风险管理参考，不意味着下一交易日必须卖出。"
    if state == "DATA_INVALID":
        return "等待下一有效交易日数据；在数据恢复前不使用市场环境参考。"
    return "观察温度、Breadth、Profit Effect 和 EMA 斜率是否形成新的极端或转向。"


def _why(state: str, signal: str, row: pd.Series, changed: str) -> str:
    """Explain today's signal using only current/causal diagnostics."""

    raw = _fmt(row.get("raw_temperature", row.get("market_temperature")))
    smooth = _fmt(row.get("smoothed_temperature"))
    if state == "PANIC_FALLING":
        return f"Raw Temperature 已进入极低区域（{raw}），Breadth 与 Profit Effect 仍在恶化，{changed}；EMA / Slope 尚未出现足够修复，因此低温暂时不能视为买入参考。"
    if state == "EXTREME_PANIC":
        return f"市场温度已经非常低（Raw {raw} / Smoothed {smooth}），但 State Machine 尚未确认恐慌结束，因此只能进入 BUY_WATCH。"
    if state == "ICE_REVERSAL_WATCH":
        return f"此前已经形成 panic episode；当前温度、Slope 和内部模块出现修复迹象（{changed}），但 State Machine 仍处于观察阶段，因此是 BUY_WATCH。"
    if state == "ICE_REVERSAL":
        return f"此前已经形成 panic episode，Breadth、Profit Effect 与 EMA 出现修复，State Machine 已确认 ICE_REVERSAL，因此升级为 BUY_REFERENCE；这不是精确底部，也不保证未来 1–5 个交易日上涨。"
    if state == "COLD":
        return f"Smoothed Temperature 偏低（{smooth}），但尚未出现足够一致的修复确认，因此保持 NEUTRAL；COLD 不等于 BUY。"
    if state == "HOT":
        return f"市场温度仍处于偏热区域（Raw {raw} / Smoothed {smooth}）。{changed}，但尚未形成足够一致的退潮信号，因此目前仍为 HOT_CAUTION，而不是 SELL_WATCH；高温本身不是卖出确认。"
    if state == "EUPHORIA_RISING":
        return f"市场已进入明显高温区（Raw {raw} / Smoothed {smooth}）且仍在升温，{changed}；当前最大风险是追高，而不是预测马上见顶，因此保持 HOT_CAUTION。"
    if state == "NORMAL":
        return "当前没有明显市场情绪极端，温度和内部模块没有形成需要升级的环境信号，因此保持 NEUTRAL。"
    if state == "HOT_ROLLOVER_WATCH":
        return f"此前存在明显高温，当前部分内部指标开始恶化（{changed}），但退潮尚未完成确认，因此是 SELL_WATCH。"
    if state == "HOT_ROLLOVER":
        return f"此前高温 episode 已出现持续退潮，Breadth、Profit Effect 与 EMA 方向支持风险恶化，State Machine 已确认 HOT_ROLLOVER，因此为 SELL_REFERENCE；这是中期风险参考，不是精确顶部或强制卖出指令。"
    return "当日缺少可用的 MarketTemperature 或质量门禁未通过，因此 Advisory Signal = DATA_INVALID，不生成正常市场建议。"


def build_explanation(
    row: pd.Series,
    *,
    advisory_signal: str,
    advisory_regime: str,
    short_term_warning: bool = False,
    not_exact_top: bool = False,
) -> dict[str, Any]:
    """Build headline, summary and structured explanation fields."""

    state = str(row.get("state", "DATA_INVALID"))
    raw = _fmt(row.get("raw_temperature", row.get("market_temperature")))
    smooth = _fmt(row.get("smoothed_temperature"))
    current = f"MarketTemperature {raw}，SmoothedTemperature {smooth}。"
    changed = _direction_text(row)

    if state == "PANIC_FALLING":
        headline = "市场处于恐慌下跌阶段：已经很冷，但情绪仍在恶化。"
        meaning = "市场已经明显降温，但情绪仍在恶化。当前是潜在机会观察区，不是精确底部确认。"
        summary = "PANIC_WAIT：继续等待市场内部修复，不因跌幅很大而机械抄底。"
    elif state == "EXTREME_PANIC":
        headline = "市场开始出现冰点关注迹象：值得观察，但反转尚未确认。"
        meaning = "市场已经非常冷，开始进入值得研究的区域；仍需等待恐慌结束和内部修复。"
        summary = "BUY_WATCH：可以提高关注度，但不是必须立即买入的信号。"
    elif state == "ICE_REVERSAL_WATCH":
        headline = "市场开始出现冰点修复迹象：可以开始关注，但反转尚未确认。"
        meaning = "市场曾经历明显恐慌，温度、斜率和部分内部结构出现修复迹象。"
        summary = "BUY_WATCH：准备买的阶段，不是必须马上买的信号。"
    elif state == "ICE_REVERSAL":
        headline = "市场经历明显恐慌后，中期修复已经确认。"
        meaning = "中期修复已经确认，更适合作为市场环境改善参考，而不是精确底部或短线买点。"
        summary = "BUY_REFERENCE：可开始研究或关注目标股票，但不能仅凭该指标作出买入决定。"
    elif state == "COLD":
        headline = "市场偏冷但暂无反转确认：保持观察，不把低温直接当作买点。"
        meaning = "市场温度偏低，但当前没有足够的转向证据。"
        summary = "NEUTRAL：低温不等于 BUY，继续观察内部修复。"
    elif state == "HOT":
        headline = "市场处于偏热环境，但尚未确认高温退潮。"
        meaning = "市场偏热，但还没有明确的退潮确认；高温本身不构成卖出参考。"
        summary = "HOT_CAUTION：提高风险意识，避免因情绪高涨而盲目追高。"
    elif state == "EUPHORIA_RISING":
        headline = "市场处于偏热或高温阶段：避免追高，但高温本身不是卖出确认。"
        meaning = "市场已经进入明显高温区且情绪仍在升温，当前最大风险是追高，而不是预测马上见顶。"
        summary = "HOT_CAUTION：已有盈利仓位提高警惕，但不把高温本身视为卖出确认。"
    elif state == "NORMAL":
        headline = "当前市场情绪处于正常区间，暂无明显极端买卖参考信号。"
        meaning = "当前没有明显市场情绪极端，不应主要依靠市场情绪决定买卖，更应该回到个股基本面、估值和趋势。"
        summary = "NEUTRAL：市场情绪没有给出明确的环境参考。"
    elif state == "HOT_ROLLOVER_WATCH":
        headline = "此前高温环境开始出现退潮迹象，值得提高警惕。"
        meaning = "市场此前处于高温/亢奋阶段，部分内部指标已经开始恶化，但退潮尚未完全确认。"
        summary = "SELL_WATCH：新开仓更严格，已有盈利仓位开始重新检查。"
    elif state == "HOT_ROLLOVER":
        headline = "高温退潮已经确认，未来数周市场风险值得明显提高警惕。"
        meaning = "高温风险环境已经确认。这是中期风险收益比恶化的参考，不是预测明天立即下跌或精确顶部。"
        summary = "SELL_REFERENCE：减少追高、提高止盈意识并重新检查高估值高涨幅标的。"
    else:
        headline = "当前市场数据无效：暂不提供市场环境参考。"
        meaning = "当日缺少可用的 MarketTemperature 或质量门禁未通过。"
        summary = "DATA_INVALID：等待下一有效交易日，不使用无效日作判断。"

    if short_term_warning and state == "ICE_REVERSAL":
        meaning += " 短期仍可能继续震荡甚至回撤，短期跟随并不保证。"
    if not_exact_top and state == "HOT_ROLLOVER":
        meaning += " 该信号不是精确顶部确认。"

    why = _why(state, advisory_signal, row, changed)
    what_to_watch_next = _watch_next(state)

    details = {
        "current_condition": current,
        "what_changed": changed,
        "what_it_means": meaning,
        "advisory": advisory_signal,
        "why": why,
        "what_to_watch_next": what_to_watch_next,
        "decision_support_disclaimer": "这是市场环境参考，不是自动交易指令。",
    }
    return {
        "headline": headline,
        "summary": summary,
        "details": details,
        "current_condition": current,
        "what_changed": changed,
        "what_it_means": meaning,
        "why": why,
        "what_to_watch_next": what_to_watch_next,
    }
