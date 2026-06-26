"""Money / quantity helpers.

Monetary values in the paper broker are handled as ``Decimal`` to avoid float
drift across many small paper trades. These helpers centralise rounding.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Union

Number = Union[int, float, str, Decimal]

CENTS = Decimal("0.01")
SHARE_PRECISION = Decimal("0.000001")  # fractional shares allowed


def to_decimal(value: Number) -> Decimal:
    """Coerce any number-like value to Decimal safely (via str for floats)."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def round_money(value: Number) -> Decimal:
    """Round to cents."""
    return to_decimal(value).quantize(CENTS, rounding=ROUND_HALF_UP)


def round_shares(value: Number) -> Decimal:
    """Round share quantity to 6 dp (fractional shares supported)."""
    return to_decimal(value).quantize(SHARE_PRECISION, rounding=ROUND_HALF_UP)


def money_str(value: Number) -> str:
    """Format as a signed-friendly USD string, e.g. ``$268.40``."""
    d = round_money(value)
    return f"${d:,.2f}"


def signed_money_str(value: Number) -> str:
    """Format with an explicit sign, e.g. ``+$2.40`` / ``-$1.10``."""
    d = round_money(value)
    sign = "+" if d >= 0 else "-"
    return f"{sign}${abs(d):,.2f}"


def pct_str(value: Number, decimals: int = 2) -> str:
    """Format a fraction (0.009) as a signed percent string (``+0.90%``)."""
    pct = float(value) * 100.0
    sign = "+" if pct >= 0 else "-"
    return f"{sign}{abs(pct):.{decimals}f}%"
