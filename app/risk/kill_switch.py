"""Kill Switch — three automatic levels plus a manual STOP.

Pure evaluation (``evaluate_kill_switch``) is separated from persistence
(``KillSwitchManager``) so the trigger logic is trivially testable.

Levels (highest active wins):
* L1 Warning   — weekly loss <= -5% OR 3 consecutive losing trades.
                 Stop new entries, report, 48h cooldown.
* L2 Freeze    — monthly loss <= -8% OR severe event flag.
                 Close 50% of open positions, stop new entries, review report.
* L3 Full Stop — monthly loss <= -12% OR max drawdown > 15%.
                 Close all positions, freeze, manual review.
* Manual STOP  — immediate freeze, cancel new entries. No real positions exist
                 to close (paper only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional

from app.domain import KillSwitchLevel, utcnow

# Thresholds (fractions, losses are negative).
L1_WEEKLY = -0.05
L1_CONSECUTIVE_LOSSES = 3
L2_MONTHLY = -0.08
L3_MONTHLY = -0.12
L3_DRAWDOWN = -0.15
L1_COOLDOWN_HOURS = 48


@dataclass
class KillSwitchEvaluation:
    level: KillSwitchLevel
    triggers: List[str] = field(default_factory=list)
    action: str = ""
    blocks_new_entries: bool = False
    close_fraction: float = 0.0  # 0.0, 0.5 or 1.0
    cooldown_hours: int = 0

    @property
    def active(self) -> bool:
        return self.level != KillSwitchLevel.NONE

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "triggers": self.triggers,
            "action": self.action,
            "blocks_new_entries": self.blocks_new_entries,
            "close_fraction": self.close_fraction,
            "cooldown_hours": self.cooldown_hours,
        }


def evaluate_kill_switch(
    weekly_return_pct: float = 0.0,
    monthly_return_pct: float = 0.0,
    drawdown_pct: float = 0.0,
    consecutive_losses: int = 0,
    severe_event: bool = False,
) -> KillSwitchEvaluation:
    """Evaluate automatic kill-switch level from portfolio metrics."""
    # Level 3 — Full Stop (checked first; highest severity wins).
    l3 = []
    if monthly_return_pct <= L3_MONTHLY:
        l3.append(f"خسارة شهرية {monthly_return_pct*100:.1f}% ≤ -12%")
    if drawdown_pct <= L3_DRAWDOWN:
        l3.append(f"أقصى تراجع {drawdown_pct*100:.1f}% تجاوز -15%")
    if l3:
        return KillSwitchEvaluation(
            level=KillSwitchLevel.LEVEL_3,
            triggers=l3,
            action="إغلاق جميع المراكز الورقية، تجميد النظام، مراجعة يدوية مطلوبة.",
            blocks_new_entries=True,
            close_fraction=1.0,
        )

    # Level 2 — Freeze.
    l2 = []
    if monthly_return_pct <= L2_MONTHLY:
        l2.append(f"خسارة شهرية {monthly_return_pct*100:.1f}% ≤ -8%")
    if severe_event:
        l2.append("إشارة حدث اقتصادي/إخباري حاد")
    if l2:
        return KillSwitchEvaluation(
            level=KillSwitchLevel.LEVEL_2,
            triggers=l2,
            action="إغلاق 50% من المراكز الورقية المفتوحة، إيقاف الدخول الجديد، إرسال تقرير مراجعة.",
            blocks_new_entries=True,
            close_fraction=0.5,
        )

    # Level 1 — Warning.
    l1 = []
    if weekly_return_pct <= L1_WEEKLY:
        l1.append(f"خسارة أسبوعية {weekly_return_pct*100:.1f}% ≤ -5%")
    if consecutive_losses >= L1_CONSECUTIVE_LOSSES:
        l1.append(f"{consecutive_losses} صفقات خاسرة متتالية")
    if l1:
        return KillSwitchEvaluation(
            level=KillSwitchLevel.LEVEL_1,
            triggers=l1,
            action="إيقاف الدخول الجديد، إرسال تقرير، تهدئة 48 ساعة.",
            blocks_new_entries=True,
            close_fraction=0.0,
            cooldown_hours=L1_COOLDOWN_HOURS,
        )

    return KillSwitchEvaluation(level=KillSwitchLevel.NONE, action="غير مفعّل.")


def manual_stop_evaluation() -> KillSwitchEvaluation:
    """The evaluation produced by a manual STOP / /stop command."""
    return KillSwitchEvaluation(
        level=KillSwitchLevel.MANUAL,
        triggers=["أمر إيقاف يدوي (STOP)"],
        action="تجميد فوري. إلغاء أي دخول جديد. لا مراكز حقيقية لإغلاقها (تداول ورقي فقط).",
        blocks_new_entries=True,
        close_fraction=0.0,
    )


class KillSwitchManager:
    """Stateful façade over the kill-switch persisted in the database."""

    def __init__(self, session):
        # Imported here to avoid a hard DB import in the pure-logic path.
        from app.db.models import KillSwitchEvent
        from app.db.repositories import KillSwitchRepository, SystemConfigRepository

        self._KillSwitchEvent = KillSwitchEvent
        self.events = KillSwitchRepository(session)
        self.cfg = SystemConfigRepository(session)

    def is_frozen(self) -> bool:
        return self.cfg.get_bool("system_frozen", False)

    def active_level(self) -> KillSwitchLevel:
        event = self.events.active_event()
        if not event:
            return KillSwitchLevel.NONE
        # Respect L1 cooldown expiry.
        if event.cooldown_until and event.cooldown_until < utcnow().replace(tzinfo=None):
            if event.level == KillSwitchLevel.LEVEL_1.value:
                event.active = False
                return KillSwitchLevel.NONE
        return KillSwitchLevel(event.level)

    def is_active(self) -> bool:
        return self.active_level() != KillSwitchLevel.NONE or self.is_frozen()

    def trigger(self, evaluation: KillSwitchEvaluation) -> None:
        """Persist a kill-switch activation."""
        cooldown_until = None
        if evaluation.cooldown_hours:
            cooldown_until = utcnow().replace(tzinfo=None) + timedelta(
                hours=evaluation.cooldown_hours
            )
        self.events.add_event(
            self._KillSwitchEvent(
                level=evaluation.level.value,
                active=True,
                trigger="; ".join(evaluation.triggers),
                action_taken=evaluation.action,
                cooldown_until=cooldown_until,
            )
        )
        if evaluation.level in (KillSwitchLevel.LEVEL_3, KillSwitchLevel.MANUAL):
            self.cfg.set("system_frozen", "true")

    def manual_stop(self) -> KillSwitchEvaluation:
        ev = manual_stop_evaluation()
        self.trigger(ev)
        self.cfg.set("system_frozen", "true")
        return ev

    def resume(self) -> int:
        """Clear all active kill-switch events and unfreeze the system."""
        count = self.events.deactivate_all()
        self.cfg.set("system_frozen", "false")
        return count
