"""Learning from the decisions the system *declined*.

ICS records ~97% of its decisions as rejections (10,592 of 10,923 in a five-year
backtest) and learns from none of them. Yet each rejection stores the full
feature and DQS context at that moment — so the market can answer the question
the system never asks: **did that filter protect capital, or cost opportunity?**

Measured potential: 6,958 of those rejections carry a direct learning signal,
against 164 closed trades the loop currently uses. That is ~42x the sample, and
it is the difference between the feedback loop becoming meaningful in weeks
rather than in nine months.

**What this is not.** ``forward_return`` is a counterfactual. Taking a rejected
trade would have consumed a slot and changed every subsequent decision, so these
numbers must never be read as foregone profit. They are for *calibration*: is a
filter's hit rate better than chance, and is the DQS threshold set in the right
place? Slot-exhaustion rejections are tracked separately precisely because they
say nothing about signal quality — they measure the cost of the position cap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd
from sqlalchemy import select

from app.db.models import Decision, RejectedOutcome
from app.logging_config import get_logger

log = get_logger(__name__)

# Matches the measured median holding period (~18 calendar days) so a rejected
# setup is judged over the horizon a taken trade would actually have run.
DEFAULT_HORIZON_DAYS = 10  # trading days

# Rejection categories.
CAT_STRATEGY_FILTER = "strategy_filter"   # a filter said no -> learnable
CAT_DQS_BELOW = "dqs_below_threshold"     # score too low -> learnable
CAT_SLOTS_FULL = "slots_full"             # capacity, not signal quality
CAT_ALREADY_HOLDING = "already_holding"   # capacity, not signal quality
CAT_RISK_LIMIT = "risk_limit"             # loss limits / kill switch
CAT_OTHER = "other"

# Only these two say anything about how well the system *judges* a setup.
LEARNABLE = (CAT_STRATEGY_FILTER, CAT_DQS_BELOW)


def categorize_rejection(reason: Optional[str], violation_details: Optional[str] = None) -> str:
    """Classify why the system declined, from the recorded reason text."""
    text = f"{reason or ''} {violation_details or ''}"
    if "MAX_OPEN_POSITIONS" in text or "الحد الأقصى للمراكز" in text:
        return CAT_SLOTS_FULL
    if "ALREADY_HOLDING" in text or "مركز مفتوح" in text:
        return CAT_ALREADY_HOLDING
    if any(k in text for k in ("LOSS_LIMIT", "DRAWDOWN", "KILL_SWITCH",
                               "حد الخسارة", "التراجع", "مفتاح الإيقاف")):
        return CAT_RISK_LIMIT
    if "DQS" in text:
        return CAT_DQS_BELOW
    if not (reason or "").strip():
        return CAT_OTHER
    return CAT_STRATEGY_FILTER


BASELINE_RETURN_KEY = "market_baseline_return"
BASELINE_HIT_KEY = "market_baseline_hit_rate"
BASELINE_AT_KEY = "market_baseline_measured_at"


def save_baseline(session, mean_return: float, hit_rate: Optional[float] = None) -> None:
    """Persist the market's own forward return so later reads can judge against it.

    Computing this needs the whole price history, which /learning must not do on
    every invocation. Without a stored value the verdicts degrade to "no baseline
    to compare against" — numbers shown but deliberately not judged.
    """
    from datetime import datetime as _dt

    from app.db.repositories import SystemConfigRepository

    cfg = SystemConfigRepository(session)
    cfg.set(BASELINE_RETURN_KEY, repr(float(mean_return)))
    if hit_rate is not None:
        cfg.set(BASELINE_HIT_KEY, repr(float(hit_rate)))
    cfg.set(BASELINE_AT_KEY, _dt.utcnow().strftime("%Y-%m-%d %H:%M UTC"))


def load_baseline(session):
    """(mean_return, hit_rate) as last measured, or (None, None)."""
    from app.db.repositories import SystemConfigRepository

    cfg = SystemConfigRepository(session)

    def _f(key):
        raw = cfg.get(key)
        try:
            return float(raw) if raw else None
        except (TypeError, ValueError):
            return None

    return _f(BASELINE_RETURN_KEY), _f(BASELINE_HIT_KEY)


# A filter has to beat the market by at least this much to count as skill
# rather than noise (mean forward return, in fraction terms).
EDGE_TOLERANCE = 0.0025


@dataclass
class FilterStat:
    """Calibration of one rejection reason against what the market then did.

    Judged **against the unconditional base rate**, never against zero. In a
    period where the market rose 85%, the average stock on the average day has a
    positive forward return, so "the price went up after we said no" is not
    evidence of a bad filter — it is evidence of a bull market. Comparing to
    zero produced exactly that false alarm before this was fixed.
    """

    category: str
    n: int = 0
    helped: int = 0                 # rejections that avoided a loss
    mean_forward_return: float = 0.0
    median_forward_return: float = 0.0
    # Unconditional forward return of the same universe over the same horizon.
    baseline_return: Optional[float] = None
    baseline_hit_rate: Optional[float] = None

    @property
    def hit_rate(self) -> float:
        """Share of rejections that turned out to be the right call."""
        return (self.helped / self.n) if self.n else 0.0

    @property
    def edge(self) -> Optional[float]:
        """How much worse the rejected setups did than the market.

        Positive = the filter declined setups that beat the market (it costs).
        Negative = it declined setups that lagged the market (it protects).
        """
        if self.baseline_return is None:
            return None
        return self.mean_forward_return - self.baseline_return

    @property
    def verdict(self) -> str:
        if self.n < 30:
            return "عيّنة غير كافية"
        edge = self.edge
        if edge is None:
            return "◐ لا معدّل أساس للمقارنة"
        if abs(edge) <= EDGE_TOLERANCE:
            return "◐ بلا قوة تنبؤية (يطابق السوق)"
        if edge > 0:
            return "⚠️ يرفض فرصًا أفضل من السوق"
        return "✅ يرفض فرصًا أسوأ من السوق"


@dataclass
class CalibrationReport:
    horizon_days: int = DEFAULT_HORIZON_DAYS
    total: int = 0
    baseline_return: Optional[float] = None
    baseline_hit_rate: Optional[float] = None
    by_category: Dict[str, FilterStat] = field(default_factory=dict)
    # Opportunity cost of the position cap, kept apart from signal quality.
    slots_full_n: int = 0
    slots_full_mean_return: float = 0.0

    def learnable_n(self) -> int:
        return sum(s.n for c, s in self.by_category.items() if c in LEARNABLE)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
PriceProvider = Callable[[str], Optional[pd.DataFrame]]


def _default_price_provider(ticker: str) -> Optional[pd.DataFrame]:
    from app.data.market_data import fetch_history

    try:
        return fetch_history(ticker, period="2y")
    except Exception:
        log.warning("No price history for %s; skipping its counterfactuals.", ticker)
        return None


def _forward_metrics(
    frame: pd.DataFrame, at: datetime, horizon: int
) -> Optional[tuple[float, float, float]]:
    """(forward_return, mfe, mae) from the bar after ``at``, over ``horizon`` bars."""
    idx = frame.index
    pos = int(idx.searchsorted(pd.Timestamp(at), "right"))
    if pos >= len(idx):
        return None  # rejection is newer than the data
    end = min(pos + horizon, len(idx) - 1)
    if end <= pos:
        return None  # not enough forward bars yet

    entry = float(frame["open"].iloc[pos])
    if entry <= 0:
        return None
    window = frame.iloc[pos : end + 1]
    fwd = float(window["close"].iloc[-1]) / entry - 1.0
    mfe = float(window["high"].max()) / entry - 1.0
    mae = float(window["low"].min()) / entry - 1.0
    return fwd, mfe, mae


def record_rejected_outcomes(
    session,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    price_provider: Optional[PriceProvider] = None,
    limit: Optional[int] = None,
) -> List[RejectedOutcome]:
    """Score every rejection that is old enough to have a forward outcome.

    Prices are fetched once per ticker, not once per rejection — a five-year
    backtest produces thousands of rejections across ~20 symbols.
    """
    provider = price_provider or _default_price_provider

    known = set(session.scalars(select(RejectedOutcome.decision_id)))
    cutoff = datetime.utcnow() - timedelta(days=horizon_days * 2)  # calendar slack
    stmt = select(Decision).where(Decision.rejected_opportunity.is_(True))
    pending = [
        d for d in session.scalars(stmt)
        if d.id not in known and d.created_at is not None and d.created_at <= cutoff
    ]
    if limit:
        pending = pending[:limit]
    if not pending:
        return []

    by_ticker: Dict[str, List[Decision]] = {}
    for d in pending:
        by_ticker.setdefault(d.ticker, []).append(d)

    created: List[RejectedOutcome] = []
    for ticker, decisions in by_ticker.items():
        frame = provider(ticker)
        if frame is None or frame.empty:
            continue
        for d in decisions:
            metrics = _forward_metrics(frame, d.created_at, horizon_days)
            if metrics is None:
                continue
            fwd, mfe, mae = metrics

            components = None
            if d.audit_log is not None:
                ctx = d.audit_log.raw_context_json or {}
                dqs_ctx = ctx.get("dqs") if isinstance(ctx, dict) else None
                if isinstance(dqs_ctx, dict):
                    components = dqs_ctx.get("components")

            created.append(
                RejectedOutcome(
                    decision_id=d.id,
                    ticker=ticker,
                    strategy=d.strategy,
                    rejected_at=d.created_at,
                    category=categorize_rejection(d.rejection_reason or d.reason,
                                                  d.violation_details),
                    rejection_reason=(d.rejection_reason or d.reason),
                    dqs_score=d.dqs_score,
                    dqs_components_json=components,
                    horizon_days=horizon_days,
                    forward_return=fwd,
                    forward_mfe=mfe,
                    forward_mae=mae,
                    rejection_helped=bool(fwd <= 0.0),
                )
            )

    for row in created:
        session.add(row)
    if created:
        session.flush()
        log.info("Scored %d rejected decisions against the market.", len(created))
    return created


# --------------------------------------------------------------------------- #
# Calibration analysis
# --------------------------------------------------------------------------- #
def compute_baseline(frames, horizon: int = DEFAULT_HORIZON_DAYS, warmup: int = 200):
    """Unconditional forward return of the universe — the bar a filter must beat.

    Returns ``(mean_return, hit_rate)`` over every symbol and every day, using
    the same next-open entry convention as a real fill.
    """
    rets: List[float] = []
    for frame in frames.values():
        if frame is None or len(frame) <= warmup + horizon + 1:
            continue
        d = frame.iloc[warmup:]
        opens = d["open"].tolist()
        closes = d["close"].tolist()
        for i in range(len(d) - horizon - 1):
            entry = opens[i + 1]
            if entry > 0:
                rets.append(closes[min(i + 1 + horizon, len(closes) - 1)] / entry - 1.0)
    if not rets:
        return None, None
    return sum(rets) / len(rets), sum(1 for r in rets if r > 0) / len(rets)


def analyze_calibration(
    session,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    baseline_return: Optional[float] = None,
    baseline_hit_rate: Optional[float] = None,
) -> CalibrationReport:
    """How well each rejection reason actually predicted the market."""
    if baseline_return is None:
        baseline_return, stored_hit = load_baseline(session)
        if baseline_hit_rate is None:
            baseline_hit_rate = stored_hit

    rows = [
        r for r in session.scalars(select(RejectedOutcome))
        if r.forward_return is not None
    ]
    report = CalibrationReport(
        horizon_days=horizon_days, total=len(rows),
        baseline_return=baseline_return, baseline_hit_rate=baseline_hit_rate,
    )

    grouped: Dict[str, List[RejectedOutcome]] = {}
    for r in rows:
        grouped.setdefault(r.category, []).append(r)

    for category, items in grouped.items():
        returns = sorted(float(r.forward_return) for r in items)
        n = len(returns)
        mean = sum(returns) / n
        median = returns[n // 2]
        if category == CAT_SLOTS_FULL:
            report.slots_full_n = n
            report.slots_full_mean_return = mean
        report.by_category[category] = FilterStat(
            category=category,
            n=n,
            helped=sum(1 for r in items if r.rejection_helped),
            mean_forward_return=mean,
            median_forward_return=median,
            baseline_return=baseline_return,
            baseline_hit_rate=baseline_hit_rate,
        )
    return report


def threshold_calibration(session, thresholds=(60, 65, 70, 75, 80)) -> List[dict]:
    """Would a different DQS cut-off have judged these setups better?

    For each candidate threshold, looks at the rejections that scored *above* it
    — those the system would have accepted — and reports how they actually did.
    """
    rows = [
        r for r in session.scalars(select(RejectedOutcome))
        if r.forward_return is not None
        and r.dqs_score is not None
        and r.category in LEARNABLE
    ]
    out: List[dict] = []
    for t in thresholds:
        above = [r for r in rows if (r.dqs_score or 0) >= t]
        if not above:
            out.append({"threshold": t, "n": 0, "mean_return": 0.0, "win_rate": 0.0})
            continue
        rets = [float(r.forward_return) for r in above]
        out.append({
            "threshold": t,
            "n": len(above),
            "mean_return": sum(rets) / len(rets),
            "win_rate": sum(1 for x in rets if x > 0) / len(rets),
        })
    return out
