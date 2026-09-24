"""Orchestration for the reusable vocabulary-driven coordinate suite."""

from __future__ import annotations

from checks.coordinate_checks.model import Catalog
from checks.coordinate_checks.ordinary import validate_ordinary
from checks.coordinate_checks.grid import validate_horizontal_grid
from checks.coordinate_checks.utils import coordinate_type
from checks.coordinate_checks.validation import Findings
from checks.coordinate_checks.vertical import validate_model_level


def _data_variable(ds, catalog):
    name = catalog.data_variable_name
    return ds.variables.get(name) if name else None


def _check_dimension_order(findings, data_var, catalog, horizontal_dimensions):
    if data_var is None:
        # Variable presence is deliberately owned by the geophysical-variable
        # checker. Avoid one derivative coordinate finding per family.
        return
    expected = []
    horizontal_added = False
    for identifier in reversed(catalog.coordinate_ids):
        entry = catalog.data_coordinates[identifier]
        kind = coordinate_type(entry)
        if kind == "generic_horizontal":
            if horizontal_dimensions is None:
                # The grid check already reports why topology resolution failed.
                # Do not invent an expected dimension order from file shapes.
                return
            if not horizontal_added:
                expected.extend(horizontal_dimensions)
                horizontal_added = True
            continue
        if kind == "scalar":
            continue
        if kind == "generic_vertical":
            expected.append(str(entry.get("out_name") or "lev"))
            continue
        expected.append(str(entry.get("out_name") or identifier))
    actual = list(data_var.dimensions)
    if expected and actual != expected:
        findings.add(
            "dimension_order",
            f"'{data_var.name}' dimensions in netCDF/C order are {actual}; "
            f"coordinate IDs {list(catalog.coordinate_ids)} require {expected} as "
            f"their {findings.severity_word('dimension_order')} order.",
        )


def check_coordinate_catalog(
    ds,
    catalog: Catalog,
    *,
    severities: dict[str, int],
    grid_topology=None,
    grid_resolution_error=None,
    allow_standard_name_fallback=True,
    bounds_dimension_name="bnds",
    vertices_dimension_name="vertices",
    climatology_bounds_name="climatology_bnds",
    time_bounds_delegated=False,
    check_direct_physical_values=True,
    check_formula_derived_profile=True,
    attributes_allowed_when_unset=(),
):
    """Validate all coordinate IDs required by one known branded variable."""
    findings = Findings(
        severities,
        bounds_dimension_name=bounds_dimension_name,
        vertices_dimension_name=vertices_dimension_name,
        climatology_bounds_name=climatology_bounds_name,
        time_bounds_delegated=time_bounds_delegated,
    )
    data_var = _data_variable(ds, catalog)

    validate_ordinary(
        findings,
        ds,
        catalog,
        data_var,
        check_direct_physical_values=check_direct_physical_values,
        attributes_allowed_when_unset=attributes_allowed_when_unset,
    )
    horizontal_dimensions = validate_horizontal_grid(
        findings,
        ds,
        catalog,
        data_var,
        topology=grid_topology,
        resolution_error=grid_resolution_error,
        allow_standard_name_fallback=allow_standard_name_fallback,
    )
    for identifier in catalog.coordinate_ids:
        entry = catalog.data_coordinates[identifier]
        kind = coordinate_type(entry)
        if kind == "generic_vertical":
            validate_model_level(
                findings,
                ds,
                catalog,
                identifier,
                horizontal_dimensions,
                check_direct_physical_values=check_direct_physical_values,
                check_formula_derived_profile=check_formula_derived_profile,
                attributes_allowed_when_unset=attributes_allowed_when_unset,
            )
        elif kind not in {
            "standard_1d",
            "scalar",
            "auxiliary",
            "site",
            "generic_horizontal",
        }:
            findings.add(
                "identity",
                f"Coordinate ID {identifier!r} has unsupported "
                f"coordinate_type={kind!r}.",
            )

    _check_dimension_order(findings, data_var, catalog, horizontal_dimensions)
    return findings.results()
