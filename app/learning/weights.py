"""Persisted DQS component weights (ICS-DOC-004 Phase 0).

Weights live in ``system_config`` under a single JSON key so no schema change is
needed and a live database picks them up automatically. Reads always fall back
to :data:`app.decision.dqs.DEFAULT_WEIGHTS`, so a database that has never run a
feedback cycle behaves exactly as before Phase 0.
"""
from __future__ import annotations

import json
from typing import Dict, Mapping

from app.decision.dqs import COMPONENT_NAMES, DEFAULT_WEIGHTS, normalize_weights
from app.db.repositories import SystemConfigRepository
from app.logging_config import get_logger

log = get_logger(__name__)

WEIGHTS_KEY = "dqs_weights"


def load_weights(session) -> Dict[str, float]:
    """Current DQS weights, or the defaults when none were ever stored."""
    raw = SystemConfigRepository(session).get(WEIGHTS_KEY)
    if not raw:
        return dict(DEFAULT_WEIGHTS)
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        log.warning("Stored DQS weights are not valid JSON; falling back to defaults.")
        return dict(DEFAULT_WEIGHTS)
    if not isinstance(data, dict) or not all(n in data for n in COMPONENT_NAMES):
        log.warning("Stored DQS weights are incomplete; falling back to defaults.")
        return dict(DEFAULT_WEIGHTS)
    return normalize_weights(data)


def save_weights(session, weights: Mapping[str, float]) -> Dict[str, float]:
    """Normalise to a sum of 100 and persist. Returns what was stored."""
    normalised = normalize_weights(weights)
    SystemConfigRepository(session).set(
        WEIGHTS_KEY, json.dumps({k: round(v, 6) for k, v in normalised.items()})
    )
    return normalised


def reset_weights(session) -> Dict[str, float]:
    """Restore the documented defaults (manual escape hatch)."""
    return save_weights(session, DEFAULT_WEIGHTS)
