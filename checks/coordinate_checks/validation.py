"""Shared attribute, value, direction, and bounds validation."""

from __future__ import annotations

import numpy as np
from compliance_checker.base import TestCtx

from checks.coordinate_checks.utils import (
    bound_tolerances,
    bounds_pairs,
    compare_units,
    formatted,
    ncattr,
    neutral_dtype,
    requested_tolerance,
    values,
)
from checks.utils import severity_word as configured_severity_word


class Findings:
    """Collect atomic findings and emit one configured Result per family."""

    def __init__(
        self,
        severities: dict[str, int],
        *,
        bounds_dimension_name: str = "bnds",
        vertices_dimension_name: str = "vertices",
        climatology_bounds_name: str = "climatology_bnds",
        time_bounds_delegated: bool = False,
    ):
        self.severities = severities
        self.messages: dict[str, list[str]] = {key: [] for key in severities}
        self.bounds_dimension_name = bounds_dimension_name
        self.vertices_dimension_name = vertices_dimension_name
        self.climatology_bounds_name = climatology_bounds_name
        self.time_bounds_delegated = time_bounds_delegated

    def add(self, family: str, message: str):
        if family in self.messages and message not in self.messages[family]:
            self.messages[family].append(message)

    def severity_word(self, family: str, *, noun: bool = False) -> str:
        """Return wording matching the configured severity of one family."""
        return configured_severity_word(self.severities.get(family), noun=noun)

    def results(self):
        labels = {
            "identity": "[COORD001] Coordinate identity and structure",
            "dimension_order": "[COORD002] Data-variable coordinate order",
            "attributes": "[COORD003] Coordinate metadata",
            "recommendations": (
                "[COORD004] Coordinate metadata "
                f"{self.severity_word('recommendations', noun=True)}s"
            ),
            "direction": "[COORD005] Coordinate monotonicity and stored direction",
            "valid_range": "[COORD006] Coordinate valid range",
            "requested_values": "[COORD007] Requested coordinate values",
            "bounds": "[COORD008] Coordinate bounds",
            "bounds_name": "[COORD009] Coordinate bounds naming",
            "associations": "[COORD010] Auxiliary-coordinate associations",
            "grid": "[COORD011] Horizontal grid coordinates",
            "formula": "[COORD012] Parametric vertical coordinate",
        }
        results = []
        for family, severity in self.severities.items():
            ctx = TestCtx(severity, labels.get(family, f"Coordinate {family}"))
            messages = self.messages.get(family, [])
            if messages:
                for message in messages:
                    ctx.add_failure(message)
            else:
                ctx.add_pass()
            results.append(ctx.to_result())
        return results


def check_dtype(findings: Findings, var, name: str, entry: dict, family="identity"):
    expected = str(entry.get("data_type") or "")
    actual = neutral_dtype(var)
    if expected and actual != expected:
        findings.add(
            family,
            f"'{name}' has storage type {actual or str(var.dtype)!r}; the "
            f"{findings.severity_word(family)} storage type is {expected!r}.",
        )


def check_attributes(
    findings: Findings,
    var,
    name: str,
    entry: dict,
    *,
    fallback_long_name: bool = True,
    missing_ok: bool = False,
):
    expected_standard_name = str(entry.get("cf_standard_name") or "")
    actual_standard_name = ncattr(var, "standard_name")
    if expected_standard_name:
        if actual_standard_name != expected_standard_name and not (
            missing_ok and not actual_standard_name
        ):
            findings.add(
                "attributes",
                f"'{name}' standard_name={formatted(actual_standard_name)}; the "
                f"{findings.severity_word('attributes')} value is "
                f"{expected_standard_name!r}.",
            )
    elif actual_standard_name:
        findings.add(
            "recommendations",
            f"'{name}' has standard_name={formatted(actual_standard_name)}, while "
            "the coordinate definition leaves it model-dependent; it is "
            f"{findings.severity_word('recommendations')} to verify that the extra "
            "metadata is appropriate.",
        )

    expected_long_name = str(entry.get("long_name") or "")
    actual_long_name = ncattr(var, "long_name")
    if not expected_standard_name and fallback_long_name and not missing_ok:
        if not actual_long_name:
            findings.add(
                "attributes",
                f"It is {findings.severity_word('attributes')} for '{name}' to "
                "define long_name because its coordinate definition does not "
                "define a standard_name.",
            )
        elif expected_long_name and actual_long_name != expected_long_name:
            findings.add(
                "recommendations",
                f"'{name}' long_name={formatted(actual_long_name)}; the "
                f"{findings.severity_word('recommendations')} value is "
                f"{expected_long_name!r}.",
            )
    elif (
        expected_long_name
        and actual_long_name != expected_long_name
        and not (missing_ok and not actual_long_name)
    ):
        findings.add(
            "recommendations",
            f"'{name}' long_name={formatted(actual_long_name)}; the "
            f"{findings.severity_word('recommendations')} value is "
            f"{expected_long_name!r}.",
        )

    for file_name, field in (
        ("axis", "axis"),
        ("positive", "positive"),
    ):
        expected = str(entry.get(field) or "")
        actual = ncattr(var, file_name)
        if expected and actual != expected and not (missing_ok and not actual):
            findings.add(
                "attributes",
                f"'{name}' {file_name}={formatted(actual)}; the "
                f"{findings.severity_word('attributes')} value is {expected!r}.",
            )
        elif not expected and actual:
            findings.add(
                "recommendations",
                f"'{name}' has {file_name}={formatted(actual)}, while its coordinate "
                f"definition does not prescribe {file_name}; it is "
                f"{findings.severity_word('recommendations')} to verify the extra "
                "attribute.",
            )

    expected_units = str(entry.get("units") or "")
    actual_units = ncattr(var, "units")
    if expected_units and not (missing_ok and not actual_units):
        matches, detail = compare_units(
            actual_units,
            expected_units,
            qualifier=findings.severity_word("attributes"),
        )
        if not matches:
            findings.add("attributes", f"'{name}' {detail}.")
    elif actual_units:
        findings.add(
            "recommendations",
            f"'{name}' has units={formatted(actual_units)}, while its coordinate "
            "definition does not prescribe units; it is "
            f"{findings.severity_word('recommendations')} to verify the extra "
            "attribute.",
        )


def numeric_values(var) -> np.ndarray | None:
    try:
        array = np.ma.asarray(var[...], dtype="float64")
        # Preserve the index of every coordinate for comparison with its bounds.
        return np.asarray(array.filled(np.nan), dtype="float64").reshape(-1)
    except (TypeError, ValueError, IndexError):
        return None


def check_valid_range(findings: Findings, var, name: str, entry: dict):
    lower = entry.get("valid_min")
    upper = entry.get("valid_max")
    if lower is None and upper is None:
        return
    array = numeric_values(var)
    if array is None or not array.size or not np.all(np.isfinite(array)):
        findings.add("valid_range", f"'{name}' has no finite numeric values to check.")
        return
    if lower is not None:
        slack = 1.0e-6 * abs(float(lower))
        offending = array[array < float(lower) - slack]
        if offending.size:
            findings.add(
                "valid_range",
                f"'{name}' contains {offending.min()} below required valid_min={lower}.",
            )
    if upper is not None:
        slack = 1.0e-6 * abs(float(upper))
        offending = array[array > float(upper) + slack]
        if offending.size:
            findings.add(
                "valid_range",
                f"'{name}' contains {offending.max()} above required valid_max={upper}.",
            )


def _strict_direction(array: np.ndarray) -> str | None:
    """Return the strict direction of a numeric profile, if it has one."""
    differences = np.diff(array)
    if np.all(differences > 0):
        return "increasing"
    if np.all(differences < 0):
        return "decreasing"
    return None


def check_direction(findings: Findings, var, name: str, entry: dict):
    """Check prescribed direction, or strict monotonicity in either direction."""
    if neutral_dtype(var) == "character" or var.ndim == 0:
        return None
    expected = str(entry.get("stored_direction") or "")
    if expected and expected not in {"increasing", "decreasing"}:
        findings.add(
            "direction",
            f"The coordinate definition has unsupported "
            f"stored_direction={expected!r} for '{name}'.",
        )
        return None
    if var.ndim != 1:
        requirement = (
            f"stored_direction={expected!r}" if expected else "strict monotonicity"
        )
        findings.add(
            "direction",
            f"It is {findings.severity_word('direction')} for '{name}' to be "
            f"one-dimensional to verify {requirement}; "
            f"found dimensions {list(var.dimensions)}.",
        )
        return None
    array = numeric_values(var)
    if array is None or array.size < 2 or not np.all(np.isfinite(array)):
        findings.add(
            "direction",
            f"'{name}' has insufficient finite values to verify strict monotonicity.",
        )
        return None
    actual = _strict_direction(array)
    if actual is None:
        requirement = (
            f"strictly {expected} according to its prescribed stored_direction"
            if expected
            else "strictly monotonic (increasing or decreasing)"
        )
        findings.add(
            "direction",
            f"'{name}' is not {requirement}.",
        )
        return None
    if expected and actual != expected:
        findings.add(
            "direction",
            f"'{name}' is strictly {actual}; the "
            f"{findings.severity_word('direction')} stored_direction is "
            f"strictly {expected}.",
        )
    return actual


def check_bounds_direction(
    findings: Findings,
    coord_var,
    coord_name: str,
    bounds_name: str,
    pairs: np.ndarray,
    entry: dict,
    *,
    scalar: bool = False,
):
    """Require bounds to follow the prescribed or observed coordinate direction."""
    direction = str(entry.get("stored_direction") or "")
    source = "the prescribed stored_direction"
    if direction not in {"increasing", "decreasing"}:
        direction = ""
    if not direction and not scalar and coord_var.ndim == 1:
        coordinates = numeric_values(coord_var)
        if (
            coordinates is not None
            and coordinates.size >= 2
            and np.all(np.isfinite(coordinates))
        ):
            direction = _strict_direction(coordinates) or ""
            source = f"the strictly monotonic parent coordinate '{coord_name}'"
    if not direction:
        return

    internal_differences = pairs[:, 1] - pairs[:, 0]
    internal_valid = (
        np.all(internal_differences > 0)
        if direction == "increasing"
        else np.all(internal_differences < 0)
    )
    if not internal_valid:
        findings.add(
            "bounds",
            f"Bounds variable '{bounds_name}' contains bounds pairs that are not "
            f"strictly {direction}, consistently with {source}.",
        )

    if not scalar and pairs.shape[0] > 1:
        overall_differences = np.diff(pairs, axis=0)
        overall_valid = (
            np.all(overall_differences > 0)
            if direction == "increasing"
            else np.all(overall_differences < 0)
        )
        if not overall_valid:
            findings.add(
                "bounds",
                f"Bounds variable '{bounds_name}' is not strictly {direction} "
                f"between successive cells, consistently with {source}.",
            )


def check_trailing_dimension(
    findings: Findings,
    var,
    name: str,
    recommended: str,
    *,
    description: str,
):
    """Check a bounds-like variable's trailing dimension name."""
    if not recommended or not var.ndim:
        return
    actual = var.dimensions[-1]
    if actual != recommended:
        findings.add(
            "bounds_name",
            f"{description} '{name}' uses trailing dimension {actual!r}; the "
            f"configured {findings.severity_word('bounds_name')} name is "
            f"{recommended!r}.",
        )


def check_requested_numeric(findings: Findings, var, name: str, entry: dict):
    requested = values(entry)
    if not requested or var.ndim != 1 or neutral_dtype(var) == "character":
        return
    array = numeric_values(var)
    if array is None:
        findings.add(
            "requested_values", f"Could not read numeric values from '{name}'."
        )
        return
    expected = [float(value) for value in requested]
    factor = entry.get("tolerance")
    missing = []
    for index, expected_value in enumerate(expected):
        tolerance = (
            requested_tolerance(index, expected, float(factor))
            if factor is not None
            else 0.0
        )
        if not np.any(np.abs(array - expected_value) <= tolerance):
            missing.append((expected_value, tolerance))
    if missing:
        findings.add(
            "requested_values",
            f"'{name}' does not contain requested value(s) within their permitted "
            f"absolute differences: {missing}.",
        )


def check_bounds(
    findings: Findings,
    ds,
    coord_var,
    coord_name: str,
    entry: dict,
    *,
    scalar=False,
    recommended_name: str | None = None,
    allowed_attributes=(),
):
    """Follow the coordinate's bounds attribute and validate that variable."""
    time_coverage_owned_by_time002 = (
        findings.time_bounds_delegated
        and entry.get("axis") == "T"
        and not entry.get("is_climatology")
    )
    required = entry.get("bounds_required") is True
    declared = ncattr(coord_var, "bounds")
    if not declared:
        if required:
            findings.add(
                "bounds",
                f"It is {findings.severity_word('bounds')} for '{coord_name}' to "
                "have a 'bounds' attribute according to its coordinate definition.",
            )
        return None
    recommended_name = recommended_name or f"{coord_name}_bnds"
    if declared != recommended_name:
        findings.add(
            "bounds_name",
            f"'{coord_name}' bounds={formatted(declared)}; the "
            f"{findings.severity_word('bounds_name')} name is "
            f"{recommended_name!r}.",
        )
    if declared not in ds.variables:
        findings.add(
            "bounds",
            f"'{coord_name}' bounds attribute names {declared!r}, but that variable "
            "does not exist in the file.",
        )
        return None
    bounds_var = ds.variables[declared]
    check_dtype(findings, bounds_var, declared, entry, family="bounds")
    unexpected = sorted(set(bounds_var.ncattrs()) - set(allowed_attributes))
    if unexpected:
        findings.add(
            "bounds",
            f"Bounds variable '{declared}' has unexpected attributes {unexpected}; "
            f"the {findings.severity_word('bounds')} allowed attributes are "
            f"{list(allowed_attributes)}.",
        )

    check_trailing_dimension(
        findings,
        bounds_var,
        declared,
        findings.bounds_dimension_name,
        description="Bounds variable",
    )

    if scalar:
        valid_shape = bounds_var.ndim == 1 and bounds_var.shape == (2,)
        parent_dimension_valid = True
        expected_shape = "one size-2 dimension"
    else:
        size_shape_valid = (
            coord_var.ndim == 1
            and bounds_var.ndim == 2
            and bounds_var.shape == (coord_var.shape[0], 2)
        )
        parent_dimension_valid = (
            coord_var.ndim == 1
            and bounds_var.ndim == 2
            and bounds_var.dimensions[0] == coord_var.dimensions[0]
        )
        valid_shape = size_shape_valid and (
            parent_dimension_valid or time_coverage_owned_by_time002
        )
        expected_shape = (
            f"dimensions ('{coord_var.dimensions[0]}', <size-2>)"
            if coord_var.ndim == 1
            else "a coordinate dimension followed by a size-2 dimension"
        )
    if not valid_shape:
        if not time_coverage_owned_by_time002:
            findings.add(
                "bounds",
                f"Bounds variable '{declared}' has dimensions "
                f"{list(bounds_var.dimensions)} and shape {bounds_var.shape}; "
                f"expected {expected_shape}.",
            )
        return bounds_var

    if time_coverage_owned_by_time002 and not parent_dimension_valid:
        findings.add(
            "bounds",
            f"Bounds variable '{declared}' has first dimension "
            f"{bounds_var.dimensions[0]!r}; expected the parent time coordinate "
            f"dimension {coord_var.dimensions[0]!r}.",
        )

    try:
        pairs = (
            np.ma.asarray(bounds_var[:], dtype="float64").filled(np.nan).reshape(-1, 2)
        )
    except (TypeError, ValueError):
        return bounds_var
    if not np.all(np.isfinite(pairs)):
        findings.add(
            "bounds",
            f"Bounds variable '{declared}' contains missing or non-finite values.",
        )
        return bounds_var
    check_bounds_direction(
        findings,
        coord_var,
        coord_name,
        declared,
        pairs,
        entry,
        scalar=scalar,
    )

    if not scalar and pairs.shape[0] == coord_var.size:
        coordinates = numeric_values(coord_var)
        lower = np.minimum(pairs[:, 0], pairs[:, 1])
        upper = np.maximum(pairs[:, 0], pairs[:, 1])
        if (
            not time_coverage_owned_by_time002
            and coordinates is not None
            and np.any(
                ~np.isfinite(coordinates)
                | (coordinates < lower)
                | (coordinates > upper)
            )
        ):
            findings.add(
                "bounds",
                f"At least one '{coord_name}' value is missing, non-finite, or lies outside its bounds.",
            )
        if pairs.shape[0] > 1:
            interval_low = lower
            interval_high = upper
            overlap = np.maximum(interval_low[:-1], interval_low[1:]) < np.minimum(
                interval_high[:-1], interval_high[1:]
            )
            if np.any(overlap):
                findings.add(
                    "bounds", f"Adjacent cells in '{declared}' have overlapping bounds."
                )

    expected_pairs = bounds_pairs(entry)
    if expected_pairs:
        if scalar:
            expected_pair = np.asarray(expected_pairs[0])
            if not np.array_equal(pairs[0], expected_pair):
                findings.add(
                    "requested_values",
                    f"Scalar bounds '{declared}'={pairs[0].tolist()}; the "
                    f"{findings.severity_word('requested_values')} pair is "
                    f"{expected_pair.tolist()} (no tolerance applies to scalars).",
                )
        else:
            factor = entry.get("tolerance")
            for expected_pair in expected_pairs:
                tolerances = (
                    bound_tolerances(expected_pair, float(factor))
                    if factor is not None
                    else (0.0, 0.0)
                )
                matches = [
                    pair
                    for pair in pairs
                    if all(
                        abs(float(pair[edge]) - expected_pair[edge]) <= tolerances[edge]
                        for edge in (0, 1)
                    )
                ]
                if not matches:
                    findings.add(
                        "requested_values",
                        f"'{declared}' does not contain requested bound pair "
                        f"{list(expected_pair)} within {list(tolerances)}.",
                    )
                    break
    return bounds_var
