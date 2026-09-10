"""Match a file-side candidate to its standard entry and diff the two."""
from __future__ import annotations

import re

from compliance_checker.cf import util as cfutil

from plugins.coordinate_standard.model import (
    Coordinate,
    Representation,
    Standard,
    VariableKind,
)

GOOD_OUTCOMES = ("MATCH", "PINNED")
WARN_OUTCOMES = ("AMBIGUOUS", "AMBIGUOUS_UNPINNABLE", "MATCH_WITH_WARNINGS")


def _to_floats(values):
    out = []
    for x in values:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return None
    return out


def _values_match(file_values, ref_values, tolerance) -> bool:
    """Compare value lists. An explicit table tolerance (including 0) is
    used as-is; an absent tolerance falls back to a small relative one."""
    try:
        tol = float(tolerance) if str(tolerance).strip() else None
    except (TypeError, ValueError):
        tol = None
    if len(file_values) != len(ref_values):
        return False
    return all(
        abs(a - b) <= (tol if tol is not None else 1e-6 * max(1.0, abs(b)))
        for a, b in zip(file_values, ref_values))


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
    if "?" in entry_units:      # CMOR template, e.g. "days since ?"
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


def diff(candidate: Coordinate, file_values, entry: Coordinate) -> tuple[list[str], list[str]]:
    """Diff a candidate against its matched standard entry.

    Returns (failures, warnings). Units are compared with udunits semantics
    (see _compare_units), not raw string equality.
    """
    failures: list[str] = []
    warnings: list[str] = []

    # Unit-string validity is udunits' call. "since" templates are handled
    # by _compare_units.
    if (candidate.units and " since " not in candidate.units
            and not cfutil.units_known(candidate.units)):
        failures.append(f"units {candidate.units!r} not recognized by udunits")

    level, msg = _compare_units(candidate.units, entry.units)
    if level == "fail":
        failures.append(msg)
    elif level == "warn":
        warnings.append(msg)

    if entry.positive:
        if not candidate.positive:
            failures.append(f"missing positive (table expects {entry.positive!r})")
        elif entry.positive != candidate.positive:
            failures.append(f"positive {candidate.positive!r}!={entry.positive!r}")
    if entry.must_have_bounds and not candidate.must_have_bounds:
        failures.append("missing bounds")
    if (entry.rank >= 0 and candidate.rank >= 0
            and entry.rank != candidate.rank):
        msg = f"rank {candidate.rank} != table rank {entry.rank}"
        # Unstructured/HEALPix files legitimately store aux lat/lon as 1-D
        # (lat(cells)), while the grids table only knows the 2-D curvilinear
        # form. Rank deviations on auxiliaries are therefore advisory.
        if candidate.representation is Representation.AUXILIARY:
            warnings.append(msg)
        else:
            failures.append(msg)
    # stored_direction is not diffed here: the atomic check [VAR005]
    # (checks/variable_checks/check_coordinate_monotonicity.py) owns
    # direction checking; the glue feeds it the pinned entry's direction.
    fv = _to_floats(file_values) if file_values is not None else None
    if fv:
        if entry.valid_min and min(fv) < float(entry.valid_min):
            failures.append(f"min {min(fv)} < valid_min {entry.valid_min}")
        if entry.valid_max and max(fv) > float(entry.valid_max):
            failures.append(f"max {max(fv)} > valid_max {entry.valid_max}")
    # Pinned values: when the entry prescribes exact values (requested list
    # or scalar value), the file must match them even when the entry was
    # resolved without value-pinning (a pool of one). Label (character)
    # axes are skipped: their requested lists are not numeric.
    if entry.requested or entry.value:
        ref = (_to_floats(entry.requested) if entry.requested
               else _to_floats([entry.value]))
        cand = fv
        if cand is None and candidate.value:
            cand = _to_floats([candidate.value])
        if (ref is not None and cand is not None
                and not _values_match(cand, ref, entry.tolerance)):
            failures.append(
                f"values do not match the table's "
                f"{'requested list' if entry.requested else 'value'}")
    return failures, warnings


def missing_coordinates(ds, kinds: dict, std: Standard,
                        ct_dimensions=None) -> dict[str, str]:
    """Report expected coordinate variables that do not exist in the file.

    Deliberately plain: if a dimension used by a data variable matches a
    standard entry's out_name but no variable of that name exists, the
    coordinate variable does not exist. No inference. Returns
    {expected_variable_name: finding}.

    ct_dimensions, when given, is the token list from a CMOR variable table
    entry's "dimensions" string (e.g. ["longitude", "latitude", "time",
    "height2m"]). Each token names a coordinate-table entry and is satisfied
    if a variable matching any of its entries' out_names exists ("latitude"
    accepts the 1-D lat axis or the 2-D latitude grids auxiliary). This
    catches missing scalar coordinates like height2m, which are never netCDF
    dimensions. Unknown tokens are skipped: a table problem is not the
    file's fault.
    """
    out_names: dict[str, list[str]] = {}
    for c in std.coordinates:
        out_names.setdefault(c.out_name or c.name, []).append(c.name)

    data_dims: set[str] = set()
    for name, kind in kinds.items():
        if kind is VariableKind.DATA:
            data_dims.update(ds.variables[name].dimensions)

    findings: dict[str, str] = {}
    for dim in sorted(data_dims):
        if dim in out_names and dim not in ds.variables:
            entries = ", ".join(sorted(set(out_names[dim])))
            findings[dim] = (f"coordinate variable {dim!r} does not exist "
                             f"(dimension present; table entries: {entries})")

    if ct_dimensions:
        by_entry_name: dict[str, list[Coordinate]] = {}
        for c in std.coordinates:
            by_entry_name.setdefault(c.name, []).append(c)
        for token in ct_dimensions:
            entries = by_entry_name.get(str(token))
            if not entries:
                continue                    # unknown token: not the file's fault
            accepted = {c.out_name or c.name for c in entries}
            if accepted & set(ds.variables):
                continue
            if accepted & set(findings):    # already reported by the pass above
                continue
            key = sorted(accepted)[0]
            names = " / ".join(sorted(accepted))
            findings[key] = (
                f"coordinate variable {names!r} does not exist "
                f"(required by the variable table: dimensions token {token!r})")
    return findings


def match(candidate: Coordinate, file_values, std: Standard) -> tuple[str, object]:
    """Match a candidate to its standard entry and diff. Returns (outcome, detail).

    Outcomes: MATCH / PINNED (good); MATCH_WITH_WARNINGS (matched, soft
    issues); MISMATCH:<why> (attributes or values disagree); NO_VALUE_MATCH /
    NO_STANDARD_ENTRY (no standard fit); AMBIGUOUS / AMBIGUOUS_UNPINNABLE
    (cannot resolve which entry, e.g. time/lat/lon).
    """
    pool = [c for c in std.coordinates
            if c.role == candidate.role
            and c.representation == candidate.representation
            and (not candidate.standard_name
                 or c.standard_name == candidate.standard_name)]
    if not pool:
        return "NO_STANDARD_ENTRY", None
    if len(pool) == 1:
        failures, warnings = diff(candidate, file_values, pool[0])
        if failures:
            return "MISMATCH:" + "; ".join(failures), pool[0].name
        if warnings:
            return "MATCH_WITH_WARNINGS:" + "; ".join(warnings), pool[0].name
        return "MATCH", pool[0].name

    fv = _to_floats(file_values) if file_values is not None else None
    if candidate.value:
        fv = _to_floats([candidate.value])
    pinnable = [c for c in pool if c.requested or c.value]
    if not pinnable:
        return "AMBIGUOUS_UNPINNABLE", [c.name for c in pool]

    pinned = []
    for c in pinnable:
        ref = _to_floats(c.requested) if c.requested else _to_floats([c.value])
        if fv is not None and ref is not None and _values_match(fv, ref, c.tolerance):
            pinned.append(c.name)
    if len(pinned) == 1:
        return "PINNED", pinned[0]
    if pinned:
        return "AMBIGUOUS", pinned
    return "NO_VALUE_MATCH", [c.name for c in pool]
