"""Pure helpers shared by coordinate validators."""

from __future__ import annotations

import re

import numpy as np
from compliance_checker.cf import util as cfutil

from checks.coordinate_checks.model import as_dict, reference_id, reference_ids


def ncattr(obj, name: str, default=""):
    try:
        return obj.getncattr(name)
    except (AttributeError, KeyError):
        return default


def formatted(value) -> str:
    if isinstance(value, str):
        return repr(" ".join(value.split()))
    return repr(value)


def neutral_dtype(var) -> str:
    dtype = getattr(var, "dtype", None)
    kind = getattr(dtype, "kind", "")
    if kind in {"S", "U"} or dtype is str:
        return "character"
    if kind in {"i", "u"}:
        return "integer"
    if kind == "f":
        return "real" if getattr(dtype, "itemsize", 0) == 4 else "double"
    return ""


def compare_units(actual: str, expected: str, *, qualifier: str) -> tuple[bool, str]:
    if not expected:
        return True, ""
    if not actual:
        return (
            False,
            f"units are missing; the {qualifier} units are {expected!r}",
        )
    if actual == expected:
        return True, ""
    if "?" in expected:
        pattern = re.escape(expected).replace(r"\?", ".+")
        if re.fullmatch(pattern, actual):
            return True, ""
        return False, f"units {actual!r} do not match required template {expected!r}"
    if cfutil.units_convertible(actual, expected):
        return False, (
            f"units {actual!r} are convertible to, but not identical with, "
            f"the required units {expected!r}"
        )
    return False, (
        f"units {actual!r} are not convertible to the required units {expected!r}"
    )


def coordinate_type(entry: dict) -> str:
    return reference_id(entry.get("coordinate_type"))


def values(entry: dict) -> list:
    raw = entry.get("coordinate_values")
    if raw is None:
        return []
    return raw if isinstance(raw, list) else [raw]


def bounds_pairs(entry: dict) -> list[tuple[float, float]]:
    edges = entry.get("coordinate_bounds") or []
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def requested_tolerance(index: int, requested: list[float], factor: float) -> float:
    tolerance = 0.001 * factor * abs(requested[index])
    if index == 0 and len(requested) > 1:
        tolerance = min(tolerance, factor * abs(requested[1] - requested[0]))
    elif index > 0:
        tolerance = min(
            tolerance, factor * abs(requested[index] - requested[index - 1])
        )
    return tolerance


def bound_tolerances(pair: tuple[float, float], factor: float) -> tuple[float, float]:
    low, high = pair
    cell = factor * abs(high - low)
    return (
        min(cell, 0.001 * factor * abs(low)),
        min(cell, 0.001 * factor * abs(high)),
    )


def decode_character(var) -> list[str]:
    raw = np.asarray(var[:])
    if raw.dtype.kind == "U":
        if raw.ndim == 1:
            return ["".join(raw.tolist()).rstrip("\x00").strip()]
        return ["".join(row.tolist()).rstrip("\x00").strip() for row in raw]
    if raw.ndim == 1:
        return [raw.tobytes().decode(errors="replace").rstrip("\x00").strip()]
    return [
        raw[index].tobytes().decode(errors="replace").rstrip("\x00").strip()
        for index in range(raw.shape[0])
    ]


def parse_formula_terms(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"(\w+)\s*:\s*([^\s]+)", text or "")
    }


def record_formula_symbol(identifier: str, entry: dict) -> str:
    source = str(entry.get("out_name") or identifier)
    return re.sub(r"_(?:bnds|half)$", "", source)


def _is_bounds_formula_factor(identifier: str, entry: dict) -> bool:
    source = str(entry.get("out_name") or identifier)
    return bool(re.search(r"_(?:bnds|half)$", source))


def _resolve_formula_factor(
    raw_factor,
    formula_terms: dict,
    generic_id: str,
    *,
    bounds: bool,
) -> tuple[str, dict]:
    """Resolve one relationship reference to its complete, context-correct record."""
    factor_id = reference_id(raw_factor)
    embedded = as_dict(raw_factor) if not isinstance(raw_factor, str) else {}
    factor = formula_terms.get(factor_id, embedded)
    if not factor:
        return factor_id, embedded

    # Some resolved ESGVoc relationships expose the bounds factors through both
    # z_factors fields. Only a model-level factor has a distinct bounds variant;
    # scalar and horizontal factors such as ptop and ps retain their referenced ID.
    dimensions = reference_ids(factor.get("dimensions"))
    if generic_id and generic_id in dimensions:
        factor_is_bounds = _is_bounds_formula_factor(factor_id, factor)
        if factor_is_bounds != bounds:
            symbol = record_formula_symbol(factor_id, factor)
            alternatives = [
                (candidate_id, candidate)
                for candidate_id, candidate in formula_terms.items()
                if generic_id in reference_ids(candidate.get("dimensions"))
                and record_formula_symbol(candidate_id, candidate) == symbol
                and _is_bounds_formula_factor(candidate_id, candidate) == bounds
            ]
            if alternatives:
                factor_id, factor = min(alternatives, key=lambda item: item[0])
    return factor_id, factor


def expected_formula_terms(
    level: dict,
    formula_terms: dict,
    *,
    generic_id="",
    bounds=False,
    resolved=None,
) -> dict:
    factor_field = "z_bounds_factors" if bounds else "z_factors"
    expected = {}
    for raw_factor in level.get(factor_field) or []:
        factor_id, factor = _resolve_formula_factor(
            raw_factor,
            formula_terms,
            generic_id,
            bounds=bounds,
        )
        symbol = record_formula_symbol(factor_id, factor)
        expected[symbol] = str(factor.get("out_name") or factor_id)
        if resolved is not None:
            resolved[symbol] = factor

    formula = str(level.get("formula") or "")
    out_name = str(level.get("out_name") or "lev")
    if formula and "=" in formula:
        rhs = " ".join(re.findall(r"=\s*([^;]+)", formula))
        excluded = set(expected) | {
            out_name,
            "exp",
            "log",
            "min",
            "max",
            "where",
            "i",
            "j",
            "k",
            "n",
        }
        for symbol in re.findall(r"\b[A-Za-z_]\w*\b", rhs):
            if symbol not in excluded:
                expected[symbol] = f"{out_name}_bnds" if bounds else out_name
                excluded.add(symbol)
    return expected
