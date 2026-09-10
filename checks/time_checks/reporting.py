"""Shared formatting helpers for time-check findings."""

from __future__ import annotations

import cftime
import numpy as np


def _numeric_text(value, decimals: int) -> str:
    if np.ma.is_masked(value):
        return "<missing>"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return repr(value)

    text = f"{numeric:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{text}.0" if "." not in text else text


def _decoded_text(value, *, units: str, calendar: str) -> str:
    if np.ma.is_masked(value):
        return "unavailable"

    try:
        decoded = cftime.num2date(
            float(value),
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=True,
        )
    except Exception:
        return "unavailable"

    base = (
        f"{decoded.year:04d}-{decoded.month:02d}-{decoded.day:02d} "
        f"{decoded.hour:02d}:{decoded.minute:02d}"
    )
    if getattr(decoded, "microsecond", 0):
        return f"{base}:{decoded.second:02d}.{decoded.microsecond:06d}"
    if getattr(decoded, "second", 0):
        return f"{base}:{decoded.second:02d}"
    return base


def format_time_value(value, *, units: str, calendar: str, decimals: int = 6) -> str:
    """Format one stored numeric time value together with its decoded date."""
    numeric = _numeric_text(value, decimals)
    decoded = _decoded_text(value, units=units, calendar=calendar)

    return f"{numeric} (decoded: {decoded})"


def format_time_interval(
    values, *, units: str, calendar: str, decimals: int = 6
) -> str:
    """Format a numeric interval followed by its decoded representation."""
    numeric = ", ".join(_numeric_text(value, decimals) for value in values)
    decoded = ", ".join(
        _decoded_text(value, units=units, calendar=calendar) for value in values
    )
    return f"[{numeric}] (decoded: [{decoded}])"


def count_phrase(count: int, singular: str, plural: str | None = None) -> str:
    """Return a naturally pluralized issue-count phrase."""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"
