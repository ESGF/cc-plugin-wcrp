"""Adapters exposing legacy CMOR tables through the shared check models."""

from __future__ import annotations

import re

from checks.coordinate_checks.model import Catalog
from checks.variable_checks.known_branded_variable import ExpectedVariableMetadata


def _words(value):
    return str(value or "").split()


def _number(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _references(value):
    if not value:
        return []
    return re.findall(r"(?:^|\s)\w+\s*:\s*([^\s]+)", str(value))


def expected_variable_from_cmor(identifier: str, entry: dict):
    """Normalize one CMOR variable entry for the common attribute checker."""
    return ExpectedVariableMetadata(
        id=identifier,
        cf_standard_name=entry.get("standard_name") or None,
        units=entry.get("units") or None,
        dimensions=_words(entry.get("dimensions")),
        cell_methods=entry.get("cell_methods") or None,
        cell_measures=entry.get("cell_measures") or None,
        description=entry.get("comment") or None,
        long_name=entry.get("long_name") or None,
        out_name=entry.get("out_name") or identifier,
        variable_root_name=identifier,
    )


def _coordinate(identifier: str, source: dict):
    generic = source.get("generic_level_name") or ""
    value = source.get("value")
    if identifier in {"latitude", "longitude"}:
        kind = "generic_horizontal"
    elif generic:
        kind = "generic_vertical"
    elif value not in (None, ""):
        kind = "scalar"
    else:
        kind = "standard_1d"
    requested = _words(source.get("requested"))
    bounds = _words(source.get("requested_bounds") or source.get("bounds_values"))
    return {
        "id": identifier,
        "coordinate_type": kind,
        "axis": source.get("axis") or "",
        "data_type": source.get("type") or "",
        "long_name": source.get("long_name") or "",
        "cf_standard_name": source.get("standard_name") or "",
        "out_name": source.get("out_name") or identifier,
        "units": source.get("units") or "",
        "positive": source.get("positive") or "",
        "stored_direction": source.get("stored_direction") or "",
        "coordinate_values": requested or ([value] if value not in (None, "") else []),
        "coordinate_bounds": bounds,
        "bounds_required": source.get("must_have_bounds") == "yes",
        "tolerance": _number(source.get("tolerance")),
        "valid_min": _number(source.get("valid_min")),
        "valid_max": _number(source.get("valid_max")),
        "is_climatology": bool(source.get("climatology")),
        "is_generic_model_level_coordinate": bool(generic),
        "generic_level_name": generic,
        "formula": source.get("formula") or "",
        "z_factors": _references(source.get("z_factors")),
        "z_bounds_factors": _references(source.get("z_bounds_factors")),
    }


def catalog_from_cmor(
    branded_variable_id: str,
    variable_name: str,
    variable_entry: dict,
    coordinate_table: dict,
    grid_table: dict,
    formula_table: dict,
):
    """Build the source-neutral coordinate catalog from loaded CMOR tables."""
    coordinate_ids = tuple(_words(variable_entry.get("dimensions")))
    source_coordinates = coordinate_table.get("axis_entry", {})
    data_coordinates = {
        identifier: _coordinate(identifier, source_coordinates[identifier])
        for identifier in coordinate_ids
        if identifier in source_coordinates
    }
    missing = [
        identifier
        for identifier in coordinate_ids
        if identifier not in data_coordinates
    ]
    if missing:
        raise ValueError(
            f"CMOR variable entry {variable_name!r} references coordinate IDs "
            f"absent from the coordinate table: {missing}."
        )
    model_levels = {
        identifier: dict(entry)
        for identifier, entry in data_coordinates.items()
        if entry.get("coordinate_type") == "generic_vertical"
    }
    formula_terms = {
        identifier: {
            "id": identifier,
            "data_type": source.get("type") or "",
            "long_name": source.get("long_name") or "",
            "cf_standard_name": source.get("standard_name") or "",
            "out_name": source.get("out_name") or identifier,
            "units": source.get("units") or "",
            "dimensions": _words(source.get("dimensions")),
        }
        for identifier, source in formula_table.get("formula_entry", {}).items()
    }
    grid_variables = {
        identifier: {
            "id": identifier,
            "data_type": source.get("type") or "",
            "long_name": source.get("long_name") or "",
            "cf_standard_name": source.get("standard_name") or "",
            "out_name": source.get("out_name") or identifier,
            "units": source.get("units") or "",
            "dimensions": _words(source.get("dimensions")),
            "valid_min": _number(source.get("valid_min")),
            "valid_max": _number(source.get("valid_max")),
        }
        for identifier, source in grid_table.get("variable_entry", {}).items()
    }
    grid_axes = {
        identifier: {
            "id": identifier,
            "axis": source.get("axis") or "",
            "data_type": source.get("type") or "",
            "long_name": source.get("long_name") or "",
            "cf_standard_name": source.get("standard_name") or "",
            "out_name": source.get("out_name") or identifier,
            "units": source.get("units") or "",
        }
        for identifier, source in grid_table.get("axis_entry", {}).items()
    }
    return Catalog(
        project_id="cordex-cmip6",
        branded_variable_id=branded_variable_id,
        branded_variable={
            "id": branded_variable_id,
            "out_name": variable_name,
            "dimensions": list(coordinate_ids),
            "cell_methods": variable_entry.get("cell_methods") or None,
        },
        coordinate_ids=coordinate_ids,
        data_coordinates=data_coordinates,
        model_levels=model_levels,
        formula_terms=formula_terms,
        grid_variables=grid_variables,
        grid_axes=grid_axes,
        file_variable_name=variable_name,
    )
