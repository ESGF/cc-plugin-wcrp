"""Rectilinear, unstructured, and curvilinear horizontal grid checks."""

from __future__ import annotations

import numpy as np

from checks.coordinate_checks.model import reference_id, reference_ids
from checks.coordinate_checks.utils import ncattr, neutral_dtype
from checks.coordinate_checks.validation import (
    Findings,
    check_attributes,
    check_bounds,
    check_direction,
    check_dtype,
    check_single_longitude_cycle,
    check_trailing_dimension,
    check_valid_range,
    valid_range_slack,
)


# At float64 this is 2 MiB of source values. Temporary arrays keep the
# per-process peak comfortably in the tens of MiB even for multi-million-cell grids.
_VERTEX_CHUNK_VALUES = 262_144


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


def _inferred_horizontal_label(catalog):
    """Infer a horizontal label only for unambiguous reduced layouts."""
    # See https://github.com/CMIP-Data-Request/CMIP7_DReq_Content/issues/36
    coordinate_ids = set(catalog.coordinate_ids)
    if coordinate_ids & {"oline", "siline"}:
        return "ht"
    if "site" in coordinate_ids:
        return "hs"
    if "basin" in coordinate_ids and coordinate_ids & {
        "latitude",
        "gridlatitude",
    }:
        return "hyb"
    if coordinate_ids & {"latitude", "gridlatitude"} and not coordinate_ids & {
        "longitude",
        "gridlongitude",
        "basin",
    }:
        return "hy"
    return ""


def _horizontal_label_for_grid_guidance(catalog, declared_label):
    """Correct known label inconsistencies from the required coordinate layout."""
    return _inferred_horizontal_label(catalog) or declared_label


def _configured_grid_label(config, horizontal_label, region):
    if config is None:
        return ""
    return (
        config.recommended_grid_labels_by_horizontal_label_and_region.get(
            (horizontal_label, region), ""
        )
        or config.recommended_grid_labels_by_horizontal_label.get(
            horizontal_label, ""
        )
    )


def check_grid_label_recommendation(
    findings,
    ds,
    catalog,
    registered_grid_metadata,
    grid_topology_config,
):
    """Recommend a special registered grid independently of cell counting."""
    if registered_grid_metadata is None or grid_topology_config is None:
        return
    declared_label = str(ncattr(ds, "horizontal_label") or "").strip()
    horizontal_label = _horizontal_label_for_grid_guidance(
        catalog, declared_label
    )
    region = str(ncattr(ds, "region") or "").strip()
    recommended_label = _configured_grid_label(
        grid_topology_config, horizontal_label, region
    )
    current_label = str(registered_grid_metadata.get("id") or "").strip()
    if not recommended_label or recommended_label == current_label:
        return
    selectors = f"horizontal_label={horizontal_label!r}"
    if (horizontal_label, region) in (
        grid_topology_config.recommended_grid_labels_by_horizontal_label_and_region
    ):
        selectors += f" and region={region!r}"
    findings.add(
        "grid_label_recommendation",
        f"The file uses grid_label='{current_label}', but for {selectors} it is "
        f"{findings.severity_word('grid_label_recommendation')} to use the "
        f"registered grid_label='{recommended_label}'.",
    )


def _observed_grid_cell_count(
    ds,
    catalog,
    data_var,
    *,
    topology,
    allow_standard_name_fallback,
):
    """Return a reliable horizontal cell count, or ``None`` after bad structure."""
    requested = [
        identifier
        for identifier in catalog.coordinate_ids
        if identifier in {"latitude", "longitude"}
    ]
    if not requested:
        return None

    effective_topology = (
        "rectilinear"
        if requested == ["latitude"] or requested == ["longitude"]
        else topology
    )
    if effective_topology not in {"rectilinear", "curvilinear", "unstructured"}:
        return None

    discovered = {}
    for role in requested:
        entry = (
            catalog.data_coordinates[role]
            if effective_topology == "rectilinear"
            else _grid_entry(catalog, role)
        )
        expected_name = str(entry.get("out_name") or role)
        variable = _identify_coordinate(
            ds,
            expected_name,
            role,
            allow_standard_name_fallback,
        )
        if variable is None:
            return None
        discovered[role] = variable

    variables = list(discovered.values())
    if data_var is not None and any(
        dimension not in data_var.dimensions
        for variable in variables
        for dimension in variable.dimensions
    ):
        return None
    if len(variables) == 1:
        variable = variables[0]
        return int(variable.size) if variable.ndim == 1 else None
    if len(variables) != 2:
        return None

    # Count the coordinate layout actually present in the file. This remains
    # reliable when an incorrect grid_label selects the wrong registered
    # topology and makes the count mismatch useful independent evidence.
    if all(variable.ndim == 1 for variable in variables):
        if variables[0].dimensions == variables[1].dimensions:
            return int(variables[0].size)
        if all(variable.dimensions == (variable.name,) for variable in variables):
            return int(np.prod([variable.size for variable in variables]))
        return None
    if (
        all(variable.ndim == 2 for variable in variables)
        and variables[0].dimensions == variables[1].dimensions
        and variables[0].shape == variables[1].shape
    ):
        return int(variables[0].size)
    return None


def check_grid_cell_count(
    findings,
    ds,
    catalog,
    data_var,
    *,
    topology,
    registered_grid_metadata,
    grid_topology_config,
    allow_standard_name_fallback,
):
    """Compare a registered cell count when the file structure is reliable."""
    if not set(catalog.coordinate_ids) & {"latitude", "longitude"}:
        return
    if registered_grid_metadata is None:
        return
    label = str(registered_grid_metadata.get("id") or "").strip() or "unknown"
    expected = registered_grid_metadata.get("n_cells")
    if expected is None:
        findings.add(
            "grid_cell_count_availability",
            f"Registered grid '{label}' does not define n_cells, so horizontal "
            "cell-count consistency could not be checked.",
        )
        return
    observed = _observed_grid_cell_count(
        ds,
        catalog,
        data_var,
        topology=topology,
        allow_standard_name_fallback=allow_standard_name_fallback,
    )
    if observed is None:
        # Structural COORD011 findings already explain why no reliable count
        # can be obtained; avoid adding a derivative failure.
        return
    if observed != int(expected):
        cell_word = "cell" if observed == 1 else "cells"
        declared_horizontal_label = str(
            ncattr(ds, "horizontal_label") or ""
        ).strip()
        horizontal_label = _horizontal_label_for_grid_guidance(
            catalog, declared_horizontal_label
        )
        reduced_labels = frozenset()
        if grid_topology_config is not None:
            reduced_labels = grid_topology_config.reduced_horizontal_labels
        guidance = (
            "Please verify the horizontal dimensions and use or register an "
            "appropriate grid_label for this grid."
        )
        if horizontal_label in reduced_labels:
            guidance += (
                " For reduced spatial output such as hemispheric, global, or "
                "zonal means, it is "
                f"{findings.severity_word('grid_cell_count_consistency')} to "
                "register or select a separate grid_label."
            )
        findings.add(
            "grid_cell_count_consistency",
            f"Registered grid '{label}' defines n_cells={int(expected)}, but "
            f"the file's horizontal coordinate dimensions contain {observed} "
            f"{cell_word}. {guidance}",
        )


def _grid_entry(catalog, role):
    return catalog.grid_variables.get(role, {})


def _grid_mapping_name(ds, data_var):
    if data_var is None:
        return ""
    mapping_attribute = ncattr(data_var, "grid_mapping")
    mapping_variable = (
        mapping_attribute.split()[0].rstrip(":")
        if isinstance(mapping_attribute, str) and mapping_attribute
        else ""
    )
    if mapping_variable in ds.variables:
        return ncattr(ds.variables[mapping_variable], "grid_mapping_name")
    return ""


def check_grid_mapping_consistency(
    findings,
    ds,
    data_var,
    registered_grid_metadata,
):
    """Compare a supplied file mapping with the registered EMD mapping."""
    if data_var is None or not registered_grid_metadata:
        return
    expected = reference_id(registered_grid_metadata.get("grid_mapping"))
    if not expected:
        return
    actual = _grid_mapping_name(ds, data_var)
    if not actual or actual == expected:
        return
    label = reference_id(registered_grid_metadata.get("id"))
    grid = f" for registered grid {label!r}" if label else ""
    findings.add(
        "grid_mapping_consistency",
        f"File grid_mapping_name={actual!r} does not match the registered "
        f"grid_mapping={expected!r}{grid}; {expected!r} is the "
        f"{findings.severity_word('grid_mapping_consistency')} value.",
    )


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


def _spatial_chunks(variable, *, vertices=False):
    """Yield origins and bounded spatial blocks without loading the full grid."""
    if not variable.size:
        return
    trailing_size = int(variable.shape[-1]) if vertices else 1
    spatial_shape = variable.shape[:-1] if vertices else variable.shape
    target_cells = max(1, _VERTEX_CHUNK_VALUES // trailing_size)
    if not spatial_shape:
        yield (), variable[...]
        return
    if len(spatial_shape) == 1:
        for start in range(0, spatial_shape[0], target_cells):
            selection = (slice(start, start + target_cells),)
            if vertices:
                selection += (slice(None),)
            yield (start,), variable[selection]
        return

    if len(spatial_shape) > 2:
        cells_per_slice = int(np.prod(spatial_shape[1:]))
        leading_block = max(1, target_cells // max(1, cells_per_slice))
        for start in range(0, spatial_shape[0], leading_block):
            selection = (slice(start, start + leading_block),) + tuple(
                slice(None) for _ in spatial_shape[1:]
            )
            if vertices:
                selection += (slice(None),)
            yield (start,) + (0,) * (len(spatial_shape) - 1), variable[selection]
        return

    row_length = int(spatial_shape[1])
    column_block = min(row_length, target_cells)
    row_block = max(1, target_cells // max(1, column_block))
    for row in range(0, spatial_shape[0], row_block):
        for column in range(0, row_length, column_block):
            selection = (
                slice(row, row + row_block),
                slice(column, column + column_block),
            )
            if vertices:
                selection += (slice(None),)
            yield (row, column), variable[selection]


def _global_cell_index(origin, local_shape, flat_index):
    local = np.unravel_index(flat_index, local_shape)
    return tuple(int(start + offset) for start, offset in zip(origin, local))


def _cell_circular_widths(cells: np.ndarray) -> np.ndarray:
    """Return the shortest circular longitude arc containing each cell."""
    finite = np.isfinite(cells)
    counts = np.count_nonzero(finite, axis=1)
    normalized = np.where(finite, np.mod(cells, 360.0), np.inf)
    normalized.sort(axis=1)
    valid_internal = np.arange(max(0, cells.shape[1] - 1))[None, :] < (
        counts[:, None] - 1
    )
    internal_gaps = np.full(valid_internal.shape, -np.inf, dtype="float64")
    np.subtract(
        normalized[:, 1:],
        normalized[:, :-1],
        out=internal_gaps,
        where=valid_internal,
    )
    largest_internal = np.max(
        np.where(valid_internal, internal_gaps, -np.inf), axis=1
    )
    last = np.take_along_axis(
        normalized, np.maximum(counts - 1, 0)[:, None], axis=1
    )[:, 0]
    wrap_gaps = normalized[:, 0] + 360.0 - last
    widths = 360.0 - np.maximum(largest_internal, wrap_gaps)
    widths[counts == 0] = np.nan
    return widths


def _check_grid_variable_valid_range(findings, variable, name, entry, *, family):
    """Count invalid grid-cell centres and retain one extreme indexed example."""
    lower = entry.get("valid_min")
    upper = entry.get("valid_max")
    if lower is None and upper is None:
        return

    total = int(np.prod(variable.shape))
    slack = valid_range_slack(lower, upper)
    below_count = above_count = 0
    lowest = highest = None
    lowest_index = highest_index = None
    saw_finite = False
    saw_nonfinite = False
    try:
        for origin, chunk in _spatial_chunks(variable):
            values = np.ma.asarray(chunk, dtype="float64").filled(np.nan)
            values = np.asarray(values, dtype="float64")
            finite = np.isfinite(values)
            saw_finite = saw_finite or np.any(finite)
            saw_nonfinite = saw_nonfinite or not np.all(finite)
            if lower is not None:
                below = finite & (values < float(lower) - slack)
                below_count += int(np.count_nonzero(below))
                if np.any(below):
                    flat = int(np.argmin(np.where(below, values, np.inf)))
                    value = float(values.flat[flat])
                    if lowest is None or value < lowest:
                        lowest = value
                        lowest_index = _global_cell_index(
                            origin, values.shape, flat
                        )
            if upper is not None:
                above = finite & (values > float(upper) + slack)
                above_count += int(np.count_nonzero(above))
                if np.any(above):
                    flat = int(np.argmax(np.where(above, values, -np.inf)))
                    value = float(values.flat[flat])
                    if highest is None or value > highest:
                        highest = value
                        highest_index = _global_cell_index(
                            origin, values.shape, flat
                        )
    except (TypeError, ValueError, IndexError, OSError, RuntimeError) as exc:
        findings.add(
            family,
            f"Could not check the explicit valid range of grid variable "
            f"'{name}': {exc}.",
        )
        return

    if not saw_finite:
        findings.add(family, f"'{name}' has no finite numeric values to check.")
        return
    if saw_nonfinite:
        findings.add(
            family,
            f"'{name}' contains missing or non-finite grid-cell values, so its "
            "explicit valid range could not be checked completely.",
        )

    qualifier = findings.severity_word(family)
    if below_count:
        findings.add(
            family,
            f"'{name}': {below_count} of {total} grid cells have values below "
            f"the {qualifier} valid_min={lower}. The comparison permits "
            f"numerical tolerance={slack}. Lowest value: {lowest} at index "
            f"{lowest_index}.",
        )
    if above_count:
        findings.add(
            family,
            f"'{name}': {above_count} of {total} grid cells have values above "
            f"the {qualifier} valid_max={upper}. The comparison permits "
            f"numerical tolerance={slack}. Highest value: {highest} at index "
            f"{highest_index}.",
        )


def _check_vertex_valid_range(
    findings,
    vertices,
    name,
    entry,
    *,
    longitude,
    family,
    allow_missing_padding=False,
):
    """Check explicit vertex limits in bounded chunks with local seam allowance."""
    lower = entry.get("valid_min")
    upper = entry.get("valid_max")
    has_explicit_range = lower is not None or upper is not None
    slack = valid_range_slack(lower, upper)
    below_count = above_count = 0
    fully_masked_count = 0
    insufficient_count = 0
    lowest = highest = None
    lowest_width = highest_width = None
    lowest_cell = highest_cell = None
    lowest_index = highest_index = None
    first_fully_masked_index = None
    first_insufficient_index = None
    first_insufficient_finite_count = None
    saw_finite = False
    saw_nonfinite = False
    total = int(np.prod(vertices.shape[:-1]))
    try:
        for origin, chunk in _spatial_chunks(vertices, vertices=True):
            masked_cells = np.ma.asarray(chunk, dtype="float64")
            missing = np.ma.getmaskarray(masked_cells).reshape(
                -1, vertices.shape[-1]
            )
            cells = masked_cells.filled(np.nan)
            cells = np.asarray(cells, dtype="float64").reshape(
                -1, vertices.shape[-1]
            )
            finite = np.isfinite(cells)
            complete_cells = np.all(finite, axis=1)
            if allow_missing_padding:
                finite_counts = np.count_nonzero(finite, axis=1)
                fully_masked = np.all(missing, axis=1)
                fully_masked_count += int(np.count_nonzero(fully_masked))
                if first_fully_masked_index is None and np.any(fully_masked):
                    flat = int(np.flatnonzero(fully_masked)[0])
                    first_fully_masked_index = _global_cell_index(
                        origin, chunk.shape[:-1], flat
                    )
                insufficient = (finite_counts < 3) & ~fully_masked
                insufficient_count += int(np.count_nonzero(insufficient))
                if first_insufficient_index is None and np.any(insufficient):
                    flat = int(np.flatnonzero(insufficient)[0])
                    first_insufficient_index = _global_cell_index(
                        origin, chunk.shape[:-1], flat
                    )
                    first_insufficient_finite_count = int(finite_counts[flat])
            unexpected_nonfinite = ~finite & ~missing
            saw_nonfinite = saw_nonfinite or np.any(unexpected_nonfinite) or (
                not allow_missing_padding and np.any(missing)
            )
            usable_cells = (
                finite_counts >= 3
                if allow_missing_padding
                else complete_cells
            )
            if not np.any(usable_cells):
                continue
            usable_rows = np.flatnonzero(usable_cells)
            cells = cells[usable_cells]
            finite = finite[usable_cells]
            saw_finite = True
            widths = (
                _cell_circular_widths(cells)
                if longitude
                else np.zeros(cells.shape[0], dtype="float64")
            )
            if lower is not None:
                below = finite & (
                    cells < float(lower) - slack - widths[:, None]
                )
                below_cells = np.any(below, axis=1)
                below_count += int(np.count_nonzero(below_cells))
                if np.any(below):
                    row, column = np.unravel_index(
                        np.argmin(np.where(below, cells, np.inf)), cells.shape
                    )
                    value = float(cells[row, column])
                    if lowest is None or value < lowest:
                        lowest = value
                        lowest_width = float(widths[row])
                        lowest_cell = cells[row].tolist()
                        lowest_index = _global_cell_index(
                            origin, chunk.shape[:-1], int(usable_rows[row])
                        )
            if upper is not None:
                above = finite & (
                    cells > float(upper) + slack + widths[:, None]
                )
                above_cells = np.any(above, axis=1)
                above_count += int(np.count_nonzero(above_cells))
                if np.any(above):
                    row, column = np.unravel_index(
                        np.argmax(np.where(above, cells, -np.inf)), cells.shape
                    )
                    value = float(cells[row, column])
                    if highest is None or value > highest:
                        highest = value
                        highest_width = float(widths[row])
                        highest_cell = cells[row].tolist()
                        highest_index = _global_cell_index(
                            origin, chunk.shape[:-1], int(usable_rows[row])
                        )
    except (TypeError, ValueError, IndexError, OSError, RuntimeError) as exc:
        findings.add(
            family,
            f"Could not check the explicit valid range of vertex variable "
            f"'{name}': {exc}.",
        )
        return

    if fully_masked_count:
        findings.add(
            "grid",
            f"'{name}': {fully_masked_count} of {total} grid cells have all "
            "vertex values missing. First incident at index "
            f"{first_fully_masked_index}.",
        )
    if insufficient_count:
        findings.add(
            "grid",
            f"'{name}': {insufficient_count} of {total} grid cells have fewer "
            "than 3 finite vertex values after permitted masked padding. "
            f"First incident at index {first_insufficient_index} has "
            f"{first_insufficient_finite_count} finite vertex values.",
        )
    if not saw_finite:
        if not has_explicit_range or (
            allow_missing_padding
            and fully_masked_count + insufficient_count == total
        ):
            return
        findings.add(
            family, f"'{name}' has no finite numeric values to check."
        )
        return
    if has_explicit_range and saw_nonfinite and not allow_missing_padding:
        findings.add(
            family,
            f"'{name}' contains missing or non-finite vertex values, so its "
            "explicit valid range could not be checked completely.",
        )

    def allowance_text(width):
        return (
            f" and that cell's circular longitude width={width}"
            if longitude
            else ""
        )

    qualifier = findings.severity_word(family)
    if lowest is not None:
        findings.add(
            family,
            f"'{name}': {below_count} of {total} grid cells have at least one "
            f"vertex below the {qualifier} valid_min={lower}. The "
            f"comparison permits numerical tolerance={slack}"
            f"{allowance_text(lowest_width)}. Lowest value {lowest} in cell "
            f"vertices {lowest_cell} at index {lowest_index}.",
        )
    if highest is not None:
        findings.add(
            family,
            f"'{name}': {above_count} of {total} grid cells have at least one "
            f"vertex above the {qualifier} valid_max={upper}. The "
            f"comparison permits numerical tolerance={slack}"
            f"{allowance_text(highest_width)}. Highest value {highest} in cell "
            f"vertices {highest_cell} at index {highest_index}.",
        )


def _validate_vertices(
    findings,
    ds,
    coordinate,
    role,
    entry,
    catalog,
    *,
    allow_missing_padding=False,
):
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
        return

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
    _check_vertex_valid_range(
        findings,
        vertices,
        declared,
        vertex_entry,
        longitude=role == "longitude",
        family=f"grid_{role}_valid_range",
        allow_missing_padding=allow_missing_padding,
    )
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
    topology,
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
        # The selected variable belongs to a different topology. Do not
        # compare it with this topology's coordinate or vertex records.
        return var
    check_dtype(findings, var, name, entry, family="grid")
    check_attributes(findings, var, name, entry)
    _check_grid_variable_valid_range(
        findings,
        var,
        name,
        entry,
        family=f"grid_{role}_valid_range",
    )
    if role == "longitude":
        check_single_longitude_cycle(
            findings,
            var,
            name,
            family="grid_longitude_single_cycle",
        )
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
    _validate_vertices(
        findings,
        ds,
        var,
        role,
        entry,
        catalog,
        allow_missing_padding=topology == "unstructured",
    )
    return var


def _validate_grid_axes(findings, ds, dimensions, catalog, data_var):
    """Accept a matching rlon/rlat, x/y, x_deg/y_deg, index, or implicit pair."""
    if not dimensions:
        return
    mapping_name = _grid_mapping_name(ds, data_var)
    explicit = [name for name in dimensions if name in ds.variables]
    if not explicit:
        return  # implicit integer indices are explicitly permitted

    malformed = [
        dimension
        for dimension in explicit
        if ds.variables[dimension].dimensions != (dimension,)
    ]
    if malformed:
        for dimension in malformed:
            var = ds.variables[dimension]
            findings.add(
                "grid",
                f"It is {findings.severity_word('grid')} for grid axis '{dimension}' "
                f"to be '{dimension}({dimension})'; found dimensions {list(var.dimensions)}.",
            )
        # Axis-pair and metadata checks assume valid coordinate variables.
        return

    matched = {}
    for dimension in explicit:
        var = ds.variables[dimension]
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
    require_explicit_grid_axes=False,
):
    requested = [
        identifier
        for identifier in catalog.coordinate_ids
        if identifier in {"latitude", "longitude"}
    ]
    if not requested:
        return []
    zonal_mean = "latitude" in requested and "longitude" not in requested
    if zonal_mean:
        if resolution_error or topology != "rectilinear":
            detail = resolution_error or f"The resolved topology is {topology!r}."
            findings.add(
                "recommendations",
                "Only latitude is required, so the zonal-mean grid is verified "
                f"as rectilinear regardless of its grid metadata. {detail} It is "
                f"{findings.severity_word('recommendations')} to register or select a grid "
                "with a (zonal-mean) rectilinear topology.",
            )
        topology = "rectilinear"
    elif resolution_error or topology is None:
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
        return None

    if topology == "rectilinear":
        valid_structure = True
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
                valid_structure = False
        if not valid_structure:
            # The variables found by standard_name are not the prescribed 1-D
            # coordinate variables. Do not compare them with unrelated 1-D CV
            # metadata or derive a data-variable dimension order from it.
            return None
        for role, var in discovered.items():
            entry = catalog.data_coordinates[role]
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
    valid_structure = True
    for role, var in discovered.items():
        if var.ndim != ndim:
            findings.add(
                "grid",
                f"It is {findings.severity_word('grid')} for '{var.name}' to be "
                f"{ndim}-D for this grid topology; found dimensions "
                f"{list(var.dimensions)}.",
            )
            valid_structure = False
    if valid_structure:
        dimension_sets = {var.dimensions for var in discovered.values()}
        if len(dimension_sets) != 1:
            findings.add(
                "grid",
                f"It is {findings.severity_word('grid')} for latitude and longitude "
                "auxiliary coordinates to share identical "
                f"dimensions; found { {role: list(var.dimensions) for role, var in discovered.items()} }.",
            )
            valid_structure = False
    if not valid_structure:
        # Rank and shared dimensions determine which grid-variable records are
        # applicable. Stop the complete pair before metadata, range, vertex,
        # association, or data-variable dimension-order checks are attempted.
        return None

    validated = {
        role: _validate_auxiliary(
            findings,
            ds,
            data_var,
            catalog,
            role,
            ndim,
            topology,
            allow_standard_name_fallback,
        )
        for role in requested
    }
    available = [var for var in validated.values() if var is not None]
    if len(available) == len(requested):
        dimensions = list(available[0].dimensions)
        if topology in {"curvilinear", "unstructured"}:
            mapping_name = _grid_mapping_name(ds, data_var)
            explicit = [name for name in dimensions if name in ds.variables]
            if (
                require_explicit_grid_axes
                and mapping_name
                and mapping_name != "latitude_longitude"
                and not explicit
            ):
                expected = (
                    "rlon/rlat"
                    if mapping_name == "rotated_latitude_longitude"
                    else "x/y"
                )
                findings.add(
                    "grid",
                    f"The {findings.severity_word('grid')} native grid axes for "
                    f"grid_mapping_name={mapping_name!r} are explicit 1-D "
                    f"{expected} coordinate variables; dimensions {dimensions} "
                    "have only implicit integer indices.",
                )
            else:
                _validate_grid_axes(findings, ds, dimensions, catalog, data_var)
        return _grid_dimensions(catalog, _grid_entry(catalog, requested[0]), dimensions)
    return None
