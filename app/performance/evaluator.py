"""Performance Evaluator.

Computes the full metric set from closed positions, the equity curve
(portfolio snapshots) and the decision log, returning a :class:`PerformanceResult`
that the reporting layer and the ``performance_reports`` table both consume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

TRADING_DAYS = 252


@dataclass
class PerformanceResult:
    period: str = "weekly"
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    starting_value: float = 0.0
    ending_value: float = 0.0
    total_return: float = 0.0    # cumulative vs initial capital (since inception)
    period_return: float = 0.0   # v1.2: return over THIS period only (start → end)
    invested_return: float = 0.0
    spy_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    expectancy: float = 0.0
    rule_compliance: float = 1.0
    average_dqs: float = 0.0
    rejected_opportunities: int = 0
    wins: int = 0
    losses: int = 0
    best_decision: str = ""
    worst_decision: str = ""
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "starting_value": self.starting_value,
            "ending_value": self.ending_value,
            "total_return": self.total_return,
            "period_return": self.period_return,
            "invested_return": self.invested_return,
            "spy_return": self.spy_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "expectancy": self.expectancy,
            "rule_compliance": self.rule_compliance,
            "average_dqs": self.average_dqs,
            "rejected_opportunities": self.rejected_opportunities,
            "wins": self.wins,
            "losses": self.losses,
            "best_decision": self.best_decision,
            "worst_decision": self.worst_decision,
            "recommendation": self.recommendation,
        }


def sharpe_ratio(daily_returns: Sequence[float], risk_free: float = 0.0) -> float:
    rets = [r for r in daily_returns if r is not None]
    if len(rets) < 2:
        return 0.0
    excess = [r - risk_free / TRADING_DAYS for r in rets]
    mean = sum(excess) / len(excess)
    var = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(TRADING_DAYS)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    peak = -math.inf
    mdd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = value / peak - 1.0
            mdd = min(mdd, dd)
    return mdd


def expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Expectancy = (WinRate * AvgWin) - (LossRate * AvgLoss).

    ``avg_loss`` is passed as a positive magnitude.
    """
    loss_rate = 1.0 - win_rate
    return (win_rate * avg_win) - (loss_rate * abs(avg_loss))


def evaluate(
    *,
    period: str,
    snapshots: Sequence[Any],
    closed_positions: Sequence[Any],
    decisions: Sequence[Any],
    initial_capital: float,
    spy_return: float = 0.0,
    period_start: Optional[datetime] = None,
    period_end: Optional[datetime] = None,
) -> PerformanceResult:
    """Build a PerformanceResult from raw records.

    ``snapshots`` need ``.total_value`` and ``.daily_pnl_pct``; ``closed_positions``
    need ``.realized_pnl`` / ``.realized_pnl_pct`` / ``.ticker``; ``decisions`` need
    ``.action`` / ``.dqs_score`` / ``.rule_violated`` / ``.rejected_opportunity``.
    """
    res = PerformanceResult(period=period, period_start=period_start, period_end=period_end)

    equity = [float(s.total_value) for s in snapshots if s.total_value is not None]
    res.starting_value = equity[0] if equity else initial_capital
    res.ending_value = equity[-1] if equity else initial_capital
    res.total_return = (res.ending_value / initial_capital - 1.0) if initial_capital else 0.0
    # v1.2: return over this period only (e.g. the week), not since inception.
    res.period_return = (
        (res.ending_value / res.starting_value - 1.0) if res.starting_value else 0.0
    )

    daily_returns = [float(s.daily_pnl_pct) for s in snapshots if s.daily_pnl_pct is not None]
    res.sharpe_ratio = round(sharpe_ratio(daily_returns), 4)
    res.max_drawdown = round(max_drawdown(equity) if equity else 0.0, 4)

    # Trade stats from closed positions.
    wins = [p for p in closed_positions if (p.realized_pnl or 0) > 0]
    losses = [p for p in closed_positions if (p.realized_pnl or 0) <= 0]
    res.wins = len(wins)
    res.losses = len(losses)
    n_trades = len(closed_positions)
    res.win_rate = (len(wins) / n_trades) if n_trades else 0.0
    res.average_win = (
        sum(float(p.realized_pnl_pct or 0) for p in wins) / len(wins) if wins else 0.0
    )
    res.average_loss = (
        sum(float(p.realized_pnl_pct or 0) for p in losses) / len(losses) if losses else 0.0
    )
    res.expectancy = round(expectancy(res.win_rate, res.average_win, res.average_loss), 4)

    # Invested return: realised P/L over total cost basis of closed trades.
    cost_basis = sum(
        float(p.entry_price) * float(p.quantity) for p in closed_positions if p.entry_price
    )
    realized = sum(float(p.realized_pnl or 0) for p in closed_positions)
    res.invested_return = (realized / cost_basis) if cost_basis else 0.0

    res.spy_return = spy_return

    # Decision-quality / compliance stats.
    executed = [d for d in decisions if d.action == "buy"]
    dqs_vals = [d.dqs_score for d in executed if d.dqs_score]
    res.average_dqs = round(sum(dqs_vals) / len(dqs_vals), 2) if dqs_vals else 0.0
    res.rejected_opportunities = sum(1 for d in decisions if getattr(d, "rejected_opportunity", False))
    if decisions:
        compliant = sum(1 for d in decisions if not getattr(d, "rule_violated", False))
        res.rule_compliance = compliant / len(decisions)

    # Best / worst by realised P/L.
    if closed_positions:
        best = max(closed_positions, key=lambda p: float(p.realized_pnl or 0))
        worst = min(closed_positions, key=lambda p: float(p.realized_pnl or 0))
        res.best_decision = f"{best.ticker} {float(best.realized_pnl_pct or 0) * 100:+.2f}%"
        res.worst_decision = f"{worst.ticker} {float(worst.realized_pnl_pct or 0) * 100:+.2f}%"

    res.recommendation = _recommendation(res)
    return res


def _recommendation(res: PerformanceResult) -> str:
    """A conservative, education-only recommendation. Never financial advice."""
    if res.average_dqs and res.average_dqs < 75:
        return "استمر في التداول الورقي. متوسط DQS دون الهدف — لا ترقية بعد."
    if res.max_drawdown <= -0.15:
        return "راجع المخاطر. تجاوز التراجع الحد — ابقَ في الوضع الورقي."
    return "استمر في التداول الورقي. لا ترقية بعد."
