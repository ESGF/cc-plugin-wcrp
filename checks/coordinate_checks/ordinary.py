"""Ordinary, scalar, text, site, and time coordinate validation."""

from __future__ import annotations

import numpy as np

from checks.coordinate_checks.utils import (
    coordinate_type,
    decode_character,
    ncattr,
    neutral_dtype,
    values,
)
from checks.coordinate_checks.validation import (
    Findings,
    check_attributes,
    check_bounds,
    check_bounds_direction,
    check_direction,
    check_dtype,
    check_trailing_dimension,
    check_requested_numeric,
    check_valid_range,
)


def listed_coordinates(data_var) -> list[str]:
    value = ncattr(data_var, "coordinates")
    return value.split() if isinstance(value, str) else []


def validate_standard_1d(findings: Findings, ds, entry_id: str, entry: dict):
    name = str(entry.get("out_name") or entry_id)
    if name not in ds.variables:
        findings.add(
            "identity",
            f"The {findings.severity_word('identity')} one-dimensional coordinate "
            f"'{name}' (coordinate ID "
            f"{entry_id!r}) is absent.",
        )
        return None
    var = ds.variables[name]
    if name not in ds.dimensions or var.dimensions != (name,):
        findings.add(
            "identity",
            f"It is {findings.severity_word('identity')} for coordinate '{name}' "
            f"to be the coordinate variable '{name}({name})'; "
            f"found dimensions {list(var.dimensions)}.",
        )
    check_dtype(findings, var, name, entry)
    check_attributes(findings, var, name, entry)
    check_valid_range(findings, var, name, entry)
    # Time monotonicity remains owned by the existing project-configured check.
    if entry.get("axis") != "T":
        check_direction(findings, var, name, entry)
    check_requested_numeric(findings, var, name, entry)
    validate_time_semantics(findings, ds, var, name, entry)
    if not entry.get("is_climatology"):
        check_bounds(findings, ds, var, name, entry)
    return var


def validate_time_semantics(findings, ds, var, name, entry):
    if entry.get("axis") != "T":
        return
    if not ncattr(var, "calendar"):
        findings.add("attributes", f"Time coordinate '{name}' is missing calendar.")
    if not entry.get("is_climatology"):
        return
    climatology = ncattr(var, "climatology")
    if not climatology:
        findings.add(
            "bounds",
            f"It is {findings.severity_word('bounds')} for climatological time "
            f"coordinate '{name}' to have a climatology "
            "attribute naming its climatology bounds variable.",
        )
    else:
        recommended = findings.climatology_bounds_name
        if recommended and climatology != recommended:
            findings.add(
                "bounds_name",
                f"'{name}' climatology={climatology!r}; the configured "
                f"{findings.severity_word('bounds_name')} climatology bounds name "
                f"is {recommended!r}.",
            )
    if climatology and climatology not in ds.variables:
        findings.add(
            "bounds",
            f"'{name}' climatology attribute names {climatology!r}, but that variable "
            "does not exist.",
        )
    elif climatology:
        bounds = ds.variables[climatology]
        if not (
            var.ndim == 1
            and bounds.ndim == 2
            and bounds.dimensions[0] == var.dimensions[0]
            and bounds.shape == (var.shape[0], 2)
        ):
            findings.add(
                "bounds",
                f"It is {findings.severity_word('bounds')} for climatology bounds "
                f"'{climatology}' to have the time dimension "
                f"followed by a size-2 dimension; found {list(bounds.dimensions)} "
                f"with shape {bounds.shape}.",
            )
        check_dtype(findings, bounds, climatology, entry, family="bounds")
        check_trailing_dimension(
            findings,
            bounds,
            climatology,
            findings.bounds_dimension_name,
            description="Climatology bounds variable",
        )
        if bounds.ndim == 2 and bounds.shape[1] == 2:
            try:
                pairs = np.asarray(bounds[:], dtype="float64").reshape(-1, 2)
            except (TypeError, ValueError):
                pairs = None
            if pairs is not None:
                check_bounds_direction(
                    findings,
                    var,
                    name,
                    climatology,
                    pairs,
                    entry,
                )
    regular = ncattr(var, "bounds")
    if regular:
        findings.add(
            "bounds",
            f"It is {findings.severity_word('bounds')} that climatological time "
            f"coordinate '{name}' does not define a bounds "
            f"attribute; found {regular!r}.",
        )
    regular_name = f"{name}_bnds"
    if regular_name in ds.variables and climatology != regular_name:
        findings.add(
            "bounds",
            f"It is {findings.severity_word('bounds')} that climatological time "
            f"coordinate '{name}' does not define regular "
            f"bounds variable '{regular_name}' unless climatology names it.",
        )


def validate_scalar(
    findings: Findings,
    ds,
    data_var,
    entry_id: str,
    entry: dict,
):
    name = str(entry.get("out_name") or entry_id)
    if name not in ds.variables:
        findings.add(
            "identity",
            f"The {findings.severity_word('identity')} scalar coordinate '{name}' "
            "is absent.",
        )
        return None
    var = ds.variables[name]
    listed = listed_coordinates(data_var) if data_var is not None else []
    if data_var is not None and name not in listed:
        findings.add(
            "associations",
            f"It is {findings.severity_word('associations')} for "
            f"'{data_var.name}' coordinates attribute to include scalar coordinate "
            f"'{name}' (coordinate ID {entry_id!r}).",
        )
    character = neutral_dtype(var) == "character"
    if character:
        if var.ndim != 1:
            findings.add(
                "identity",
                f"It is {findings.severity_word('identity')} for character scalar "
                f"coordinate '{name}' to have one string-length "
                f"dimension; found {list(var.dimensions)}.",
            )
        elif var.dimensions[0] != "strlen":
            findings.add(
                "recommendations",
                f"The {findings.severity_word('recommendations')} string-length "
                f"dimension for character scalar coordinate '{name}' is 'strlen'; "
                f"found {var.dimensions[0]!r}.",
            )
    elif var.ndim != 0:
        findings.add(
            "identity",
            f"It is {findings.severity_word('identity')} for numeric scalar "
            f"coordinate '{name}' to be dimensionless; found "
            f"{list(var.dimensions)}.",
        )
    check_dtype(findings, var, name, entry)
    check_attributes(findings, var, name, entry)
    if not character:
        check_valid_range(findings, var, name, entry)

    requested = values(entry)
    if requested:
        expected = requested[0]
        if character:
            actual = decode_character(var)[0]
            if actual != str(expected):
                findings.add(
                    "requested_values",
                    f"Scalar coordinate '{name}'={actual!r}; the "
                    f"{findings.severity_word('requested_values')} value is "
                    f"{str(expected)!r}.",
                )
        else:
            try:
                actual = float(np.asarray(var[...]).reshape(-1)[0])
                expected_numeric = float(expected)
                # The specification deliberately does not apply coordinate
                # tolerance to scalar values.
                if actual != expected_numeric:
                    lower, upper = entry.get("valid_min"), entry.get("valid_max")
                    in_range = (lower is None or actual >= float(lower)) and (
                        upper is None or actual <= float(upper)
                    )
                    family = (
                        "recommendations"
                        if in_range and (lower is not None or upper is not None)
                        else "requested_values"
                    )
                    findings.add(
                        family,
                        f"Scalar coordinate '{name}'={actual}; the "
                        f"{findings.severity_word(family)} value is "
                        f"{expected_numeric}."
                        + (
                            " The value remains inside its valid range."
                            if in_range
                            else ""
                        ),
                    )
            except (TypeError, ValueError, IndexError):
                findings.add(
                    "requested_values", f"Could not read scalar value from '{name}'."
                )
    if not character:
        check_bounds(findings, ds, var, name, entry, scalar=True)
    return var


def validate_text_auxiliary(findings, ds, data_var, entry_id, entry):
    dimension_name = str(entry.get("out_name") or entry_id)
    name = "sector"
    if name not in ds.variables:
        findings.add(
            "identity",
            f"The {findings.severity_word('identity')} text auxiliary coordinate "
            f"'{name}' for coordinate ID "
            f"{entry_id!r} is absent.",
        )
        return None
    var = ds.variables[name]
    if data_var is not None and name not in listed_coordinates(data_var):
        findings.add(
            "associations",
            f"It is {findings.severity_word('associations')} for "
            f"'{data_var.name}' coordinates attribute to include text auxiliary "
            "coordinate 'sector'.",
        )
    if var.ndim != 2 or not var.dimensions or var.dimensions[0] != dimension_name:
        findings.add(
            "identity",
            f"It is {findings.severity_word('identity')} for 'sector' to be a "
            f"character array whose first dimension is "
            f"'{dimension_name}'; found {list(var.dimensions)}.",
        )
    elif var.dimensions[1] != "strlen":
        findings.add(
            "recommendations",
            f"The {findings.severity_word('recommendations')} string-length "
            "dimension for 'sector' is 'strlen'; "
            f"found {var.dimensions[1]!r}.",
        )
    check_dtype(findings, var, name, entry)
    check_attributes(findings, var, name, entry)
    requested = [str(value) for value in values(entry)]
    if requested:
        actual = decode_character(var)
        missing = [value for value in requested if value not in actual]
        if missing:
            findings.add(
                "requested_values",
                f"'sector' is missing requested label(s) {missing}; found {actual}. "
                "Additional labels and any label order are allowed.",
            )
    return var


def validate_site(findings, ds, data_var, entry_id, entry):
    dimension_name = str(entry.get("out_name") or entry_id)
    if dimension_name not in ds.dimensions:
        findings.add(
            "identity",
            f"The {findings.severity_word('identity')} site dimension "
            f"'{dimension_name}' is absent.",
        )
    if data_var is None:
        return
    listed = listed_coordinates(data_var)
    for standard_name, units in (
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
    ):
        candidates = [
            name
            for name in listed
            if name in ds.variables
            and ncattr(ds.variables[name], "standard_name") == standard_name
        ]
        if not candidates:
            findings.add(
                "associations",
                f"It is {findings.severity_word('associations')} for "
                f"'{data_var.name}' coordinates to include a site auxiliary "
                f"coordinate with standard_name={standard_name!r}.",
            )
            continue
        variable = ds.variables[candidates[0]]
        if variable.dimensions != (dimension_name,):
            findings.add(
                "associations",
                f"It is {findings.severity_word('associations')} for site "
                f"{standard_name} coordinate '{variable.name}' to have "
                f"dimension '{dimension_name}'; found {list(variable.dimensions)}.",
            )
        if ncattr(variable, "units") != units:
            findings.add(
                "attributes",
                f"It is {findings.severity_word('attributes')} for site "
                f"{standard_name} coordinate '{variable.name}' to have "
                f"units={units!r}; found {ncattr(variable, 'units')!r}.",
            )


def validate_ordinary(findings, ds, catalog, data_var):
    resolved = {}
    for entry_id in catalog.coordinate_ids:
        entry = catalog.data_coordinates[entry_id]
        kind = coordinate_type(entry)
        if kind == "standard_1d":
            resolved[entry_id] = validate_standard_1d(findings, ds, entry_id, entry)
        elif kind == "scalar":
            resolved[entry_id] = validate_scalar(
                findings, ds, data_var, entry_id, entry
            )
        elif kind == "auxiliary":
            resolved[entry_id] = validate_text_auxiliary(
                findings, ds, data_var, entry_id, entry
            )
        elif kind == "site":
            validate_site(findings, ds, data_var, entry_id, entry)
    return resolved
