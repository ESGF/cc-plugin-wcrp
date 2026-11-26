"""Narrow spelling normalization for units compared against the vocabulary."""

import re


_SCIENTIFIC_NUMBER = re.compile(
    r"(?<![\w.])([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))[eE]([+-]?(?:0[0-9]|[1-9][0-9]*|0))(?![\w.])"
)


def normalize_unit_exponents(units: str) -> str:
    """Normalize E/e, single-digit exponent padding, and a redundant plus sign.

    Preserve the mantissa and every other character: unit symbols are case
    sensitive, and convertible units are not necessarily the prescribed units.
    """
    return _SCIENTIFIC_NUMBER.sub(
        lambda match: f"{match[1]}e{int(match[2])}", units
    )


def exponent_spelling_severity(policy):
    """Return the exception severity, or None when strict spelling is required."""
    if policy is not None and not policy.enabled:
        return None
    return 2 if policy is not None and policy.severity == "M" else 1


def exponent_spelling_message(actual, expected):
    return (
        f"Units {actual!r} differ from the vocabulary spelling {expected!r}, "
        "but represent the same value after normalizing scientific-notation exponents."
    )
