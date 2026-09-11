"""Rectilinear, unstructured, and curvilinear horizontal grid checks."""

from __future__ import annotations

from checks.coordinate_checks.model import reference_ids
from checks.coordinate_checks.utils import ncattr, neutral_dtype
from checks.coordinate_checks.validation import (
    Findings,
    check_attributes,
    check_bounds,
    check_direction,
    check_dtype,
    check_trailing_dimension,
    check_valid_range,
)


def _by_standard_name(ds, standard_name):
    return [
        name
        for name, var in ds.variables.items()
        if ncattr(var, "standard_name") == standard_name
    ]


def _identify_coordinate(ds, expected_name, standard_name, allow_fallback):
    """Identify by the ESGVoc name, optionally falling back to standard_name."""
    if expected_name in ds.variables:
        return ds.variables[expected_name]
    if allow_fallback:
        candidates = _by_standard_name(ds, standard_name)
        if candidates:
            return ds.variables[candidates[0]]
    return None


def _grid_entry(catalog, role):
    return catalog.grid_variables.get(role, {})


def _grid_dimensions(catalog, entry, dimensions, vertex_dimension=None):
    """Resolve CV dimension order through the chosen horizontal axis scheme."""
    # Bare indices have no X/Y metadata: their two positions define Y then X.
    roles = {"latitude": dimensions[0], "longitude": dimensions[-1]}
    if len(dimensions) == 2:
        identified = {}
        for identifier, axis in catalog.grid_axes.items():
            name = str(axis.get("out_name") or identifier)
            if name not in dimensions:
                continue
            direction = axis.get("axis") or {"i_index": "X", "j_index": "Y"}.get(
                identifier
            )
            if direction in {"X", "Y"}:
                identified["longitude" if direction == "X" else "latitude"] = name
        if len(identified) == 1:
            other_role = next(role for role in roles if role not in identified)
            roles[other_role] = next(
                (name for name in dimensions if name not in identified.values()),
                dimensions[0],
            )
        roles.update(identified)
    if vertex_dimension is not None:
        roles["vertices"] = vertex_dimension
    references = reference_ids(entry.get("dimensions"))
    if not references:
        return list(dimensions) + (
            [vertex_dimension] if vertex_dimension is not None else []
        )
    expected = []
    for identifier in reversed(references):
        name = roles.get(identifier) or str(
            catalog.grid_axes.get(identifier, {}).get("out_name") or identifier
        )
        # Both horizontal roles share the cell dimension on unstructured grids.
        if name not in expected:
            expected.append(name)
    return expected


def _validate_vertices(findings, ds, coordinate, role, entry, catalog):
    declared = ncattr(coordinate, "bounds")
    vertex_entry = catalog.grid_variables.get(f"vertices_{role}", {})
    recommended = str(vertex_entry.get("out_name") or f"vertices_{role}")
    if not declared:
        findings.add(
            "grid",
            f"It is {findings.severity_word('grid')} for horizontal auxiliary "
            f"coordinate '{coordinate.name}' to have a "
            "bounds attribute naming its vertex variable.",
        )
        return
    if declared != recommended:
        findings.add(
            "bounds_name",
            f"'{coordinate.name}' bounds={declared!r}; the "
            f"{findings.severity_word('bounds_name')} name is "
            f"{recommended!r}.",
        )
    if declared not in ds.variables:
        findings.add(
            "grid",
            f"'{coordinate.name}' bounds attribute names {declared!r}, but that "
            "vertex variable is absent.",
        )
        return
    vertices = ds.variables[declared]
    check_dtype(findings, vertices, declared, vertex_entry, family="grid")
    check_trailing_dimension(
        findings,
        vertices,
        declared,
        findings.vertices_dimension_name,
        description="Vertex variable",
    )
    # Vertex attributes are inherited from the coordinate under CF and are
    # optional. When supplied they must agree with the ESGVoc grid record.
    if vertices.ncattrs():
        check_attributes(
            findings,
            vertices,
            declared,
            vertex_entry,
            fallback_long_name=False,
            missing_ok=True,
        )
    valid = (
        vertices.ndim == coordinate.ndim + 1
        and vertices.dimensions[:-1] == coordinate.dimensions
        and vertices.shape[:-1] == coordinate.shape
        and vertices.shape[-1] >= 3
    )
    if not valid:
        findings.add(
            "grid",
            f"It is {findings.severity_word('grid')} for vertex variable "
            f"'{declared}' to have the coordinate dimensions "
            f"{list(coordinate.dimensions)} followed by a vertex dimension of size "
            f"at least 3; found {list(vertices.dimensions)} with shape {vertices.shape}.",
        )
    if coordinate.ndim in {1, 2} and vertices.ndim:
        expected = _grid_dimensions(
            catalog, vertex_entry, coordinate.dimensions, vertices.dimensions[-1]
        )
        if list(vertices.dimensions) != expected:
            findings.add(
                "grid",
                f"Vertex variable '{declared}' has dimensions {list(vertices.dimensions)}; "
                f"the {findings.severity_word('grid')} dimension order is {expected}.",
            )


def _validate_auxiliary(
    findings,
    ds,
    data_var,
    catalog,
    role,
    ndim,
    allow_standard_name_fallback,
):
    entry = _grid_entry(catalog, role)
    expected_name = str(entry.get("out_name") or role)
    var = _identify_coordinate(
        ds,
        expected_name,
        role,
        allow_standard_name_fallback,
    )
    if var is None:
        findings.add(
            "grid",
            f"The {findings.severity_word('grid')} {ndim}-D {role} grid variable "
            f"'{expected_name}' is absent.",
        )
        return None
    name = var.name
    if name != expected_name:
        findings.add(
            "grid",
            f"{role.capitalize()} grid variable '{name}' was identified by "
            f"standard_name={role!r}, "
            f"but its {findings.severity_word('grid')} variable name is "
            f"'{expected_name}'.",
        )
    if var.ndim != ndim:
        findings.add(
            "grid",
            f"It is {findings.severity_word('grid')} for '{name}' to be {ndim}-D "
            f"for this grid topology; found dimensions "
            f"{list(var.dimensions)}.",
        )
    check_dtype(findings, var, name, entry, family="grid")
    check_attributes(findings, var, name, entry)
    check_valid_range(findings, var, name, entry)
    if var.ndim == ndim:
        expected = _grid_dimensions(catalog, entry, var.dimensions)
        if list(var.dimensions) != expected:
            findings.add(
                "grid",
                f"Grid variable '{name}' has dimensions {list(var.dimensions)}; "
                f"the {findings.severity_word('grid')} dimension order is {expected}.",
            )
    if data_var is not None:
        listed = str(ncattr(data_var, "coordinates") or "").split()
        if name not in listed:
            findings.add(
                "associations",
                f"It is {findings.severity_word('associations')} for "
                f"'{data_var.name}' coordinates attribute to include auxiliary "
                f"{role} coordinate '{name}'.",
            )
    _validate_vertices(findings, ds, var, role, entry, catalog)
    return var


def _validate_grid_axes(findings, ds, dimensions, catalog, data_var):
    """Accept a matching rlon/rlat, x/y, x_deg/y_deg, index, or implicit pair."""
    if not dimensions:
        return
    explicit = [name for name in dimensions if name in ds.variables]
    if not explicit:
        return  # implicit integer indices are explicitly permitted

    matched = {}
    for dimension in explicit:
        var = ds.variables[dimension]
        if var.dimensions != (dimension,):
            findings.add(
                "grid",
                f"It is {findings.severity_word('grid')} for grid axis '{dimension}' "
                f"to be '{dimension}({dimension})'; found dimensions {list(var.dimensions)}.",
            )
        candidates = []
        for identifier, entry in catalog.grid_axes.items():
            out_name = str(entry.get("out_name") or identifier)
            if dimension != out_name:
                continue
            if entry.get("axis") in {"X", "Y"}:
                candidates.append((identifier, entry))
            elif str(entry.get("data_type") or "") == "integer":
                candidates.append((identifier, entry))
        if not candidates:
            findings.add(
                "grid",
                f"Explicit curvilinear dimension coordinate '{dimension}' does not "
                "match any configured grid-axis definition.",
            )
            continue

        def mismatch_score(candidate):
            _, entry = candidate
            return sum(
                (
                    bool(entry.get("axis"))
                    and ncattr(var, "axis") != entry.get("axis"),
                    bool(entry.get("cf_standard_name"))
                    and ncattr(var, "standard_name") != entry.get("cf_standard_name"),
                    bool(entry.get("units"))
                    and ncattr(var, "units") != entry.get("units"),
                    bool(entry.get("data_type"))
                    and neutral_dtype(var) != entry.get("data_type"),
                )
            )

        identifier, entry = min(candidates, key=mismatch_score)
        check_dtype(findings, var, dimension, entry, family="grid")
        check_attributes(findings, var, dimension, entry, fallback_long_name=False)
        check_direction(findings, var, dimension, entry)
        matched[entry.get("axis") or "index"] = identifier

    axis_values = {key for key in matched if key in {"X", "Y"}}
    if axis_values and axis_values != {"X", "Y"}:
        findings.add(
            "grid",
            f"It is {findings.severity_word('grid')} for explicit horizontal axes "
            f"to provide a matching X/Y pair; matched "
            f"{matched} for dimensions {list(dimensions)}.",
        )

    mapping_name = ""
    if data_var is not None:
        mapping_attribute = ncattr(data_var, "grid_mapping")
        mapping_variable = (
            mapping_attribute.split()[0].rstrip(":")
            if isinstance(mapping_attribute, str) and mapping_attribute
            else ""
        )
        if mapping_variable in ds.variables:
            mapping_name = ncattr(ds.variables[mapping_variable], "grid_mapping_name")
    matched_ids = set(matched.values())
    if mapping_name == "rotated_latitude_longitude":
        if matched_ids != {"grid_longitude", "grid_latitude"}:
            findings.add(
                "grid",
                f"The {findings.severity_word('grid')} axis pair for "
                "grid_mapping_name='rotated_latitude_longitude' is the "
                "grid_longitude/grid_latitude pair (rlon/rlat).",
            )
    elif mapping_name and mapping_name != "latitude_longitude":
        if matched_ids not in ({"x", "y"}, {"x_deg", "y_deg"}):
            findings.add(
                "grid",
                f"The {findings.severity_word('grid')} axis pair for "
                f"grid_mapping_name={mapping_name!r} is a matching projected "
                "x/y pair in metres or degrees.",
            )


def validate_horizontal_grid(
    findings: Findings,
    ds,
    catalog,
    data_var,
    *,
    topology=None,
    resolution_error=None,
    allow_standard_name_fallback=True,
):
    requested = [
        identifier
        for identifier in catalog.coordinate_ids
        if identifier in {"latitude", "longitude"}
    ]
    if not requested:
        return []
    if resolution_error or topology is None:
        findings.add(
            "grid",
            "The horizontal grid could not be verified. "
            + (resolution_error or "No grid topology was resolved."),
        )
        return None
    if topology not in {"rectilinear", "curvilinear", "unstructured"}:
        findings.add(
            "grid",
            f"The horizontal grid could not be verified. Unsupported configured "
            f"topology={topology!r}.",
        )
        return None
    discovered = {}
    for role in requested:
        entry = (
            catalog.data_coordinates[role]
            if topology == "rectilinear"
            else _grid_entry(catalog, role)
        )
        expected = str(entry.get("out_name") or role)
        discovered[role] = _identify_coordinate(
            ds,
            expected,
            role,
            allow_standard_name_fallback,
        )
    present = [var for var in discovered.values() if var is not None]
    if len(present) != len(requested):
        missing = [role for role, var in discovered.items() if var is None]
        findings.add(
            "grid",
            f"The {findings.severity_word('grid')} horizontal coordinate(s) are "
            f"absent: {missing}.",
        )
        return []

    if topology == "rectilinear":
        for role, var in discovered.items():
            entry = catalog.data_coordinates[role]
            expected_name = str(entry.get("out_name") or role)
            if var.name != expected_name:
                findings.add(
                    "grid",
                    f"The {findings.severity_word('grid')} coordinate for the "
                    f"configured rectilinear grid is "
                    f"'{expected_name}({expected_name})' for {role}; "
                    f"standard_name={role!r} identified '{var.name}'.",
                )
            if not (
                var.ndim == 1
                and var.name in ds.dimensions
                and var.dimensions == (var.name,)
            ):
                findings.add(
                    "grid",
                    f"It is {findings.severity_word('grid')} for '{expected_name}' "
                    f"on the configured rectilinear grid to be a 1-D coordinate "
                    f"variable; found '{var.name}' with "
                    f"dimensions {list(var.dimensions)}.",
                )
            check_dtype(findings, var, var.name, entry, family="grid")
            check_attributes(findings, var, var.name, entry)
            check_direction(findings, var, var.name, entry)
            check_valid_range(findings, var, var.name, entry)
            check_bounds(
                findings,
                ds,
                var,
                var.name,
                entry,
                recommended_name=f"{entry.get('out_name') or role}_bnds",
            )
        # ESGVoc retains CMOR's longitude/latitude ordering; netCDF/C arrays
        # store the horizontal dimensions latitude/longitude.
        return [
            str(catalog.data_coordinates[role].get("out_name") or role)
            for role in ("latitude", "longitude")
            if role in discovered
        ]

    ndim = 1 if topology == "unstructured" else 2
    validated = {
        role: _validate_auxiliary(
            findings,
            ds,
            data_var,
            catalog,
            role,
            ndim,
            allow_standard_name_fallback,
        )
        for role in requested
    }
    available = [var for var in validated.values() if var is not None]
    if len(available) == len(requested):
        if any(var.ndim != ndim for var in available):
            return None  # Rank errors were reported; no usable grid dimensions.
        dimension_sets = {var.dimensions for var in available}
        if len(dimension_sets) != 1:
            findings.add(
                "grid",
                f"It is {findings.severity_word('grid')} for latitude and longitude "
                "auxiliary coordinates to share identical "
                f"dimensions; found { {role: list(var.dimensions) for role, var in validated.items()} }.",
            )
            return []
        dimensions = list(available[0].dimensions)
        if topology in {"curvilinear", "unstructured"}:
            _validate_grid_axes(findings, ds, dimensions, catalog, data_var)
        return _grid_dimensions(catalog, _grid_entry(catalog, requested[0]), dimensions)
    return []
