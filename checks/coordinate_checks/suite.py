"""Orchestration for the reusable vocabulary-driven coordinate suite."""

from __future__ import annotations

from compliance_checker.base import BaseCheck, TestCtx

from checks.coordinate_checks.grid import (
    check_grid_cell_count,
    check_grid_label_recommendation,
    check_grid_mapping_consistency,
    validate_horizontal_grid,
)
from checks.coordinate_checks.model import Catalog
from checks.coordinate_checks.ordinary import validate_ordinary
from checks.coordinate_checks.utils import coordinate_type
from checks.coordinate_checks.validation import Findings
from checks.utils import severity_ordered


_REGISTRY_FAMILIES = (
    "identity",
    "dimension_order",
    "attributes",
    "recommendations",
    "direction",
    "valid_range",
    "grid_latitude_valid_range",
    "grid_longitude_valid_range",
    "grid_longitude_single_cycle",
    "grid_mapping_consistency",
    "grid_label_recommendation",
    "grid_cell_count_availability",
    "grid_cell_count_consistency",
    "requested_values",
    "bounds",
    "bounds_name",
    "associations",
    "grid",
    "formula",
)


def coordinate_catalog_required(config) -> bool:
    """Whether at least one configured check consumes coordinate metadata."""
    coordinates = getattr(config, "coordinates", None) if config else None
    if coordinates is None:
        return False
    registry = getattr(coordinates, "registry", None)
    if registry is not None and any(
        getattr(registry, family, None) is not None for family in _REGISTRY_FAMILIES
    ):
        return True
    if any(
        getattr(rule, "squareness", None) is not None
        or getattr(rule, "coverage", None) is not None
        for rule in (getattr(coordinates, "variables", {}) or {}).values()
    ):
        return True
    # TIME003 obtains climatology semantics from the coordinate catalogue.
    return getattr(config, "drs", None) is not None


def coordinate_metadata_setup_result(
    *,
    config,
    catalog,
    error,
    project_label,
    get_severity,
):
    """Report catalogue failure once whenever any configured check needs it."""
    if not coordinate_catalog_required(config):
        return []
    coordinates = config.coordinates
    registry = getattr(coordinates, "registry", None)
    setup_rule = getattr(registry, "setup", None) if registry is not None else None

    dependent_rules = []
    if registry is not None:
        dependent_rules.extend(
            rule
            for family in _REGISTRY_FAMILIES
            if (rule := getattr(registry, family, None)) is not None
        )
    for rule in (getattr(coordinates, "variables", {}) or {}).values():
        if rule.squareness is not None:
            dependent_rules.append(rule.squareness)
        if rule.coverage is not None:
            dependent_rules.append(rule.coverage)
    if getattr(config, "drs", None) is not None:
        dependent_rules.append(config.drs.time_range)

    dependent_severities = [
        get_severity(getattr(rule, "severity", None), "HIGH")
        for rule in dependent_rules
    ]
    setup_severity = (
        get_severity(setup_rule.severity, "HIGH")
        if setup_rule is not None
        else None
    )

    if error is not None or catalog is None:
        # A setup failure represents every enabled metadata consumer which can
        # no longer run; do not downgrade that root cause to COORD000's own
        # configured reporting severity.
        severity = max(
            dependent_severities
            + ([setup_severity] if setup_severity is not None else []),
            default=BaseCheck.HIGH,
        )
    else:
        severity = setup_severity or max(
            dependent_severities, default=BaseCheck.HIGH
        )

    if setup_rule is None and error is None and catalog is not None:
        # Disabling COORD000 suppresses its successful status result, not a
        # failure needed to explain why enabled consumers could not run.
        return []

    ctx = TestCtx(
        severity,
        "[COORD000] Coordinate metadata initialization",
    )
    if error:
        ctx.add_failure(
            f"The {project_label} coordinate checks could not be initialized, so "
            "enabled vocabulary-driven coordinate and time checks were skipped "
            f"for this file. Technical reason: {error}"
        )
    elif catalog is None:
        ctx.add_failure(
            f"The {project_label} coordinate catalog is unavailable for an "
            "unknown reason, so enabled vocabulary-driven coordinate and time "
            "checks were skipped."
        )
    else:
        ctx.add_pass()
    return [ctx.to_result()]


def missing_configured_coordinate_result(name, rule, get_severity):
    """Route a missing coordinate to its highest-severity enabled consumer."""
    candidates = []
    if rule.monotonicity is not None:
        candidates.append(
            (
                rule.monotonicity,
                f"[VAR005] Coordinate monotonicity for '{name}'",
            )
        )
    if rule.squareness is not None:
        candidates.append((rule.squareness, "[TIME001] Check Time Squareness "))
    if rule.coverage is not None:
        candidates.append((rule.coverage, "[TIME002] Time bounds"))
    if rule.calendar_recommendation is not None:
        candidates.append(
            (rule.calendar_recommendation, "[TIME003a] Calendar for time coordinate")
        )
    candidates.extend(
        (
            attribute_rule,
            (
                f"[ATTR004] Coordinate variable '{name}' attribute "
                f"'{attribute_rule.attribute_name or key}'"
            ),
        )
        for key, attribute_rule in sorted(rule.attributes.items())
    )
    if not candidates:
        return []
    selected, label = severity_ordered(
        candidates,
        severity=lambda candidate: get_severity(candidate[0].severity, "HIGH"),
    )[0]
    ctx = TestCtx(get_severity(selected.severity, "HIGH"), label)
    ctx.add_failure(
        f"Coordinate variable '{name}' is missing, so this configured check "
        "could not be evaluated."
    )
    return [ctx.to_result()]
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
    registered_grid_metadata=None,
    registered_grid_metadata_error=None,
    grid_topology_config=None,
    allow_standard_name_fallback=True,
    require_explicit_grid_axes=False,
    bounds_dimension_name="bnds",
    vertices_dimension_name="vertices",
    climatology_bounds_name="climatology_bnds",
    time_bounds_delegated=False,
    data_variable_presence_delegated=True,
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
    if data_var is None and not data_variable_presence_delegated:
        expected_name = catalog.data_variable_name or "<unknown>"
        findings.add_blocked(
            ("dimension_order", "associations", "grid_mapping_consistency"),
            f"The data variable {expected_name!r} is absent, so configured "
            "checks which require the data variable could not be evaluated.",
            issue=("missing_data_variable", expected_name),
        )

    if registered_grid_metadata is None and registered_grid_metadata_error:
        findings.add_prerequisite(
            "grid",
            lambda family: (
                f"The horizontal grid could not be verified. "
                f"{registered_grid_metadata_error}"
                if family == "grid"
                else (
                    "The registered grid metadata needed by configured EMD checks "
                    f"could not be used. {registered_grid_metadata_error}"
                )
            ),
            issue=("grid_resolution", registered_grid_metadata_error),
            fallbacks=(
                "grid_mapping_consistency",
                "grid_label_recommendation",
                "grid_cell_count_availability",
                "grid_cell_count_consistency",
            ),
        )

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
        require_explicit_grid_axes=require_explicit_grid_axes,
    )
    check_grid_mapping_consistency(
        findings,
        ds,
        data_var,
        registered_grid_metadata,
    )
    check_grid_label_recommendation(
        findings,
        ds,
        catalog,
        registered_grid_metadata,
        grid_topology_config,
    )
    check_grid_cell_count(
        findings,
        ds,
        catalog,
        data_var,
        topology=grid_topology,
        registered_grid_metadata=registered_grid_metadata,
        grid_topology_config=grid_topology_config,
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
            findings.add_prerequisite(
                "identity",
                f"Coordinate ID {identifier!r} has unsupported "
                f"coordinate_type={kind!r}, so its configured dependent "
                "coordinate checks could not be evaluated.",
                issue=("unsupported_coordinate_type", identifier, kind),
            )

    _check_dimension_order(findings, data_var, catalog, horizontal_dimensions)
    return findings.results()
