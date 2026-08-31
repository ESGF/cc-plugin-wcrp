from __future__ import annotations

import re

from compliance_checker.cf import util as cfutil


def neutral_dtype(var) -> str:
    """File-side dtype helper function."""
    kind = getattr(getattr(var, "dtype", None), "kind", "")
    if kind in ("S", "U") or getattr(var, "dtype", None) is str:
        return "character"
    if kind in ("i", "u"):
        return "integer"
    if kind == "f":
        return "double"
    return ""

def _compare_units(candidate_units: str, entry_units: str) -> tuple[str, str]:
    """Tiered units comparison backed by udunits.

    Returns (level, message): "ok", "warn" (convertible but not identical,
    e.g. hPa vs Pa) or "fail" (not convertible or invalid). CMOR time
    templates like "days since ?" accept any reference date.
    """
    if not entry_units:
        return "ok", ""
    if not candidate_units:
        return "warn", f"units missing, table expects {entry_units!r}"
    if candidate_units == entry_units:
        return "ok", ""
    if "?" in entry_units:
        pattern = re.escape(entry_units).replace(r"\?", ".+")
        if re.fullmatch(pattern, candidate_units):
            return "ok", ""
        # Compare the base units of "<unit> since <?>" (hours vs days).
        if " since " in entry_units and " since " in candidate_units:
            entry_base = entry_units.split(" since ")[0]
            cand_base = candidate_units.split(" since ")[0]
            if cfutil.units_convertible(cand_base, entry_base):
                return "warn", (f"units {candidate_units!r} use base unit "
                                f"{cand_base!r}, table expects {entry_base!r}")
            return "fail", f"units {candidate_units!r} not convertible to {entry_units!r}"
        # A template can never be parsed by udunits, so do not fall through.
        return "fail", (f"units {candidate_units!r} do not match the table "
                        f"template {entry_units!r}")
    if cfutil.units_convertible(candidate_units, entry_units):
        return "warn", (f"units {candidate_units!r} convertible to, but not "
                        f"identical to, table units {entry_units!r}")
    return "fail", f"units {candidate_units!r} not convertible to {entry_units!r}"
