"""Rectilinear, unstructured, and curvilinear horizontal grid checks."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager

import numpy as np

from checks.coordinate_checks.model import as_dict, reference_id, reference_ids
from checks.coordinate_checks.utils import ncattr, neutral_dtype
from checks.coordinate_checks.validation import (
    Findings,
    check_attributes,
    check_bounds,
    check_direction,
    check_dtype,
    check_missing_value_attributes,
    check_single_longitude_cycle,
    check_trailing_dimension,
    check_valid_range,
    valid_range_slack,
)

# At float64 this is 2 MiB of source values. Temporary arrays keep the
# per-process peak comfortably in the tens of MiB even for multi-million-cell grids.
_VERTEX_CHUNK_VALUES = 262_144
_GRID_SCAN_CACHE_BYTES = 32 * 2**20


def _vertex_chunk_working_set_bytes(variable):
    """Estimate cache needed for all storage chunks across one vertex cell."""
    try:
        chunks = variable.chunking()
        if not isinstance(chunks, (list, tuple)) or not chunks:
            return 0
        trailing_chunks = (
            int(variable.shape[-1]) + int(chunks[-1]) - 1
        ) // int(chunks[-1])
        return int(np.prod(chunks)) * int(variable.dtype.itemsize) * trailing_chunks
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0


@contextmanager
def _bounded_variable_chunk_cache(
    variable,
    maximum_bytes=_GRID_SCAN_CACHE_BYTES,
    *,
    minimum_working_set_bytes=0,
):
    """Temporarily cap a netCDF variable's HDF5 cache during full-grid scans."""
    try:
        size, nelems, preemption = variable.get_var_chunk_cache()
    except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
        yield
        return
    target_size = min(size, max(maximum_bytes, minimum_working_set_bytes))
    if size <= target_size:
        yield
        return
    try:
        variable.set_var_chunk_cache(target_size, nelems, preemption)
    except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
        yield
        return
    try:
        yield
    finally:
        try:
            # Resetting also releases the scan cache. netCDF4 allocates the
            # restored cache lazily if a later consumer accesses the variable.
            variable.set_var_chunk_cache(size, nelems, preemption)
        except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
            pass


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
    if "basin" in coordinate_ids and "latitude" in coordinate_ids:
        return "hyb"
    if "basin" in coordinate_ids and "gridlatitude" in coordinate_ids:
        return "ht"
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


def _basin_by_latitude_axis(catalog):
    """Return the sole latitude-like axis in a basin-by-latitude layout."""
    coordinate_ids = set(catalog.coordinate_ids)
    if "basin" not in coordinate_ids:
        return ""
    axes = coordinate_ids & {"latitude", "gridlatitude"}
    return next(iter(axes)) if len(axes) == 1 else ""


def _configured_grid_type(config, catalog, horizontal_label):
    if config is None:
        return ""
    # Both latitude+basin (hyb) and gridlatitude+basin (ht) describe a
    # basin-by-latitude output grid. Keep their horizontal labels distinct
    # while using the shared configurable grid-type recommendation.
    selector = "hyb" if _basin_by_latitude_axis(catalog) else horizontal_label
    return config.recommended_grid_types_by_horizontal_label.get(selector, "")


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
    if _basin_by_latitude_axis(catalog):
        # A declared ht with gridlatitude+basin is not the oline/siline
        # transect represented by the fixed ht grid-label recommendation.
        recommended_label = ""
    current_label = str(registered_grid_metadata.get("id") or "").strip()
    accepted_labels = (
        grid_topology_config.accepted_grid_labels_by_horizontal_label_and_region.get(
            (horizontal_label, region), frozenset()
        )
    )
    if current_label in accepted_labels:
        return
    if recommended_label:
        if recommended_label == current_label:
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
        return

    recommended_type = _configured_grid_type(
        grid_topology_config, catalog, horizontal_label
    )
    current_type = reference_id(registered_grid_metadata.get("grid_type"))
    if not recommended_type or recommended_type == current_type:
        return
    current_description = (
        f"registered grid_type='{current_type}'"
        if current_type
        else "no registered grid_type"
    )
    findings.add(
        "grid_label_recommendation",
        f"The file uses grid_label='{current_label}' with {current_description}, "
        f"but for horizontal_label={horizontal_label!r} it is "
        f"{findings.severity_word('grid_label_recommendation')} to register and/or "
        f"use a grid_label with grid_type='{recommended_type}'.",
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


def _observed_basin_by_latitude_cell_counts(
    ds,
    catalog,
    data_var,
    *,
    allow_standard_name_fallback,
):
    """Return both permitted interpretations of basin-by-latitude n_cells."""
    axis_identifier = _basin_by_latitude_axis(catalog)
    if not axis_identifier or data_var is None:
        return None

    entries = {
        identifier: catalog.data_coordinates.get(identifier, {})
        for identifier in (axis_identifier, "basin")
    }
    variables = {}
    for identifier, entry in entries.items():
        expected_name = str(entry.get("out_name") or identifier)
        standard_name = str(entry.get("cf_standard_name") or identifier)
        variable = _identify_coordinate(
            ds,
            expected_name,
            standard_name,
            allow_standard_name_fallback,
        )
        if variable is None:
            return None
        variables[identifier] = variable

    axis = variables[axis_identifier]
    if axis.ndim != 1 or axis.dimensions[0] not in data_var.dimensions:
        return None
    axis_dimension = axis.dimensions[0]
    basin_dimensions = [
        dimension
        for dimension in variables["basin"].dimensions
        if dimension in data_var.dimensions
    ]
    if len(basin_dimensions) != 1 or basin_dimensions[0] == axis_dimension:
        return None

    axis_size = len(ds.dimensions[axis_dimension])
    basin_size = len(ds.dimensions[basin_dimensions[0]])
    return frozenset({axis_size, axis_size * basin_size})


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
    if registered_grid_metadata is None:
        return
    declared_horizontal_label = str(ncattr(ds, "horizontal_label") or "").strip()
    horizontal_label = _horizontal_label_for_grid_guidance(
        catalog, declared_horizontal_label
    )
    basin_axis = _basin_by_latitude_axis(catalog)
    if basin_axis:
        recommended_type = _configured_grid_type(
            grid_topology_config, catalog, horizontal_label
        )
        registered_type = reference_id(registered_grid_metadata.get("grid_type"))
        if not recommended_type or registered_type != recommended_type:
            # Parent grid labels are permitted for reduced output. Their cell
            # count describes the parent grid and is not comparable here.
            return
    elif not set(catalog.coordinate_ids) & {"latitude", "longitude"}:
        return
    label = str(registered_grid_metadata.get("id") or "").strip() or "unknown"
    expected = registered_grid_metadata.get("n_cells")
    if expected is None:
        findings.add_prerequisite(
            "grid_cell_count_availability",
            f"Registered grid '{label}' does not define n_cells, so horizontal "
            "cell-count consistency could not be checked.",
            issue=("missing_registered_cell_count", label),
            fallbacks=("grid_cell_count_consistency",),
        )
        return
    if basin_axis:
        observed_counts = _observed_basin_by_latitude_cell_counts(
            ds,
            catalog,
            data_var,
            allow_standard_name_fallback=allow_standard_name_fallback,
        )
    else:
        observed = _observed_grid_cell_count(
            ds,
            catalog,
            data_var,
            topology=topology,
            allow_standard_name_fallback=allow_standard_name_fallback,
        )
        observed_counts = None if observed is None else frozenset({observed})
    if observed_counts is None:
        # Structural COORD011 findings already explain why no reliable count
        # can be obtained; avoid adding a derivative failure.
        return
    if int(expected) not in observed_counts:
        if basin_axis:
            permitted = sorted(observed_counts)
            if len(permitted) == 1:
                observed_description = f"{permitted[0]} cells"
            else:
                observed_description = (
                    f"either {permitted[0]} latitude-axis cells or "
                    f"{permitted[1]} latitude-by-basin/sector positions"
                )
            comparison = (
                "the file's horizontal coordinate dimensions represent "
                f"{observed_description}"
            )
        else:
            observed = next(iter(observed_counts))
            cell_word = "cell" if observed == 1 else "cells"
            comparison = (
                "the file's horizontal coordinate dimensions contain "
                f"{observed} {cell_word}"
            )
        reduced_labels = frozenset()
        if grid_topology_config is not None:
            reduced_labels = grid_topology_config.reduced_horizontal_labels
        guidance = (
            "Please verify the horizontal dimensions and register and/or use an "
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
            f"{comparison}. {guidance}",
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


def check_grid_region_consistency(findings, ds, registered_grid_metadata):
    """Compare the file region with the registered grid's DRS region."""
    if not registered_grid_metadata:
        return
    actual = str(ncattr(ds, "region") or "").strip()
    registered_region = registered_grid_metadata.get("region")
    if isinstance(registered_region, str):
        expected = registered_region.strip()
    else:
        expected = str(as_dict(registered_region).get("drs_name") or "").strip()
    if not actual or not expected or actual == expected:
        return
    label = reference_id(registered_grid_metadata.get("id"))
    grid = f" for registered grid {label!r}" if label else ""
    findings.add(
        "grid_region_consistency",
        f"File region={actual!r} does not match the registered grid "
        f"region={expected!r}{grid}; {expected!r} is the "
        f"{findings.severity_word('grid_region_consistency')} value.",
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


def _bounded_spatial_chunks(variable, *, vertices=False):
    """Yield grid blocks while bounding and subsequently releasing I/O cache."""
    working_set = _vertex_chunk_working_set_bytes(variable) if vertices else 0
    with _bounded_variable_chunk_cache(
        variable,
        minimum_working_set_bytes=working_set,
    ):
        yield from _spatial_chunks(variable, vertices=vertices)


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


def _check_grid_variable_valid_range(
    findings,
    variable,
    name,
    entry,
    *,
    family,
):
    """Check explicit limits for the finite grid-cell centres."""
    lower = entry.get("valid_min")
    upper = entry.get("valid_max")
    has_explicit_range = lower is not None or upper is not None
    check_range = has_explicit_range and family in findings.messages
    if not check_range:
        return

    total = int(np.prod(variable.shape))
    slack = valid_range_slack(lower, upper)
    below_count = above_count = 0
    lowest = highest = None
    lowest_index = highest_index = None
    saw_finite = False
    try:
        for origin, chunk in _bounded_spatial_chunks(variable):
            values = np.ma.asarray(chunk, dtype="float64").filled(np.nan)
            values = np.asarray(values, dtype="float64")
            finite = np.isfinite(values)
            saw_finite = saw_finite or np.any(finite)
            if check_range and lower is not None:
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
            if check_range and upper is not None:
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
            f"Could not inspect auxiliary grid coordinate '{name}' for "
            f"its explicit valid range: {exc}.",
        )
        return

    if not saw_finite:
        return

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
):
    """Check explicit limits for the finite vertex values in bounded chunks."""
    lower = entry.get("valid_min")
    upper = entry.get("valid_max")
    has_explicit_range = lower is not None or upper is not None
    check_range = has_explicit_range and family in findings.messages
    if not check_range:
        return
    slack = valid_range_slack(lower, upper)
    below_count = above_count = 0
    lowest = highest = None
    lowest_width = highest_width = None
    lowest_cell = highest_cell = None
    lowest_index = highest_index = None
    saw_finite = False
    total = int(np.prod(vertices.shape[:-1]))
    try:
        for origin, chunk in _bounded_spatial_chunks(vertices, vertices=True):
            masked_cells = np.ma.asarray(chunk, dtype="float64")
            cells = masked_cells.filled(np.nan)
            cells = np.asarray(cells, dtype="float64").reshape(
                -1, vertices.shape[-1]
            )
            finite = np.isfinite(cells)
            finite_counts = np.count_nonzero(finite, axis=1)
            usable_cells = finite_counts > 0
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
            if check_range and lower is not None:
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
            if check_range and upper is not None:
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
            f"Could not inspect vertex variable '{name}' for its explicit "
            f"valid range: {exc}.",
        )
        return

    if not saw_finite:
        return

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


def _check_grid_geometry_missing_values(
    findings,
    latitude,
    longitude,
    latitude_vertices,
    longitude_vertices,
    *,
    topology,
):
    """Require each grid position to be fully void or coherently geolocated."""
    family = "grid_geometry_missing_values"
    if family not in findings.messages:
        return
    if (
        latitude.shape != longitude.shape
        or latitude_vertices.shape != longitude_vertices.shape
        or latitude_vertices.shape[:-1] != latitude.shape
    ):
        # Shape ownership remains with the structural grid check.
        return

    total = int(np.prod(latitude.shape))
    invalid_count = 0
    first_index = None
    first_detail = None
    nonfinite_count = 0
    first_nonfinite_index = None
    try:
        variables = (
            latitude,
            longitude,
            latitude_vertices,
            longitude_vertices,
        )
        with ExitStack() as stack:
            for variable in variables:
                stack.enter_context(
                    _bounded_variable_chunk_cache(
                        variable,
                        minimum_working_set_bytes=(
                            _vertex_chunk_working_set_bytes(variable)
                            if variable.ndim == latitude.ndim + 1
                            else 0
                        ),
                    )
                )
            for origin, latitude_vertex_chunk in _spatial_chunks(
                latitude_vertices, vertices=True
            ):
                spatial_shape = latitude_vertex_chunk.shape[:-1]
                selection = tuple(
                    slice(start, start + length)
                    for start, length in zip(origin, spatial_shape)
                )
                vertex_selection = selection + (slice(None),)
                latitude_values = np.ma.asarray(
                    latitude[selection], dtype="float64"
                )
                longitude_values = np.ma.asarray(
                    longitude[selection], dtype="float64"
                )
                latitude_vertex_values = np.ma.asarray(
                    latitude_vertex_chunk, dtype="float64"
                )
                longitude_vertex_values = np.ma.asarray(
                    longitude_vertices[vertex_selection], dtype="float64"
                )

                latitude_data = np.asarray(
                    np.ma.getdata(latitude_values), dtype="float64"
                )
                longitude_data = np.asarray(
                    np.ma.getdata(longitude_values), dtype="float64"
                )
                latitude_vertex_data = np.asarray(
                    np.ma.getdata(latitude_vertex_values), dtype="float64"
                )
                longitude_vertex_data = np.asarray(
                    np.ma.getdata(longitude_vertex_values), dtype="float64"
                )
                latitude_mask = np.ma.getmaskarray(latitude_values)
                longitude_mask = np.ma.getmaskarray(longitude_values)
                latitude_vertex_mask = np.ma.getmaskarray(latitude_vertex_values)
                longitude_vertex_mask = np.ma.getmaskarray(longitude_vertex_values)

                latitude_finite = (
                    np.isfinite(latitude_data) & ~latitude_mask
                ).reshape(-1)
                longitude_finite = (
                    np.isfinite(longitude_data) & ~longitude_mask
                ).reshape(-1)
                latitude_vertex_finite = (
                    np.isfinite(latitude_vertex_data) & ~latitude_vertex_mask
                ).reshape(-1, latitude_vertices.shape[-1])
                longitude_vertex_finite = (
                    np.isfinite(longitude_vertex_data) & ~longitude_vertex_mask
                ).reshape(-1, longitude_vertices.shape[-1])
                unmasked_nonfinite = (
                    (~np.isfinite(latitude_data) & ~latitude_mask).reshape(-1)
                    | (~np.isfinite(longitude_data) & ~longitude_mask).reshape(-1)
                    | np.any(
                        (~np.isfinite(latitude_vertex_data) & ~latitude_vertex_mask)
                        .reshape(-1, latitude_vertices.shape[-1]),
                        axis=1,
                    )
                    | np.any(
                        (
                            ~np.isfinite(longitude_vertex_data)
                            & ~longitude_vertex_mask
                        ).reshape(-1, longitude_vertices.shape[-1]),
                        axis=1,
                    )
                )
                nonfinite_count += int(np.count_nonzero(unmasked_nonfinite))
                if first_nonfinite_index is None and np.any(unmasked_nonfinite):
                    flat = int(np.flatnonzero(unmasked_nonfinite)[0])
                    first_nonfinite_index = _global_cell_index(
                        origin, spatial_shape, flat
                    )

                vertex_masks_match = np.all(
                    latitude_vertex_finite == longitude_vertex_finite, axis=1
                )
                paired_vertices = latitude_vertex_finite & longitude_vertex_finite
                paired_counts = np.count_nonzero(paired_vertices, axis=1)
                all_vertices_void = ~np.any(
                    latitude_vertex_finite | longitude_vertex_finite, axis=1
                )
                centres_complete = latitude_finite & longitude_finite
                centres_void = ~latitude_finite & ~longitude_finite
                void = centres_void & all_vertices_void
                if topology == "unstructured":
                    complete = (
                        centres_complete
                        & vertex_masks_match
                        & (paired_counts >= 3)
                    )
                    vertex_requirement = "at least 3 paired finite vertices"
                else:
                    complete = (
                        centres_complete
                        & vertex_masks_match
                        & (paired_counts == latitude_vertices.shape[-1])
                    )
                    vertex_requirement = "all vertex pairs finite"
                invalid = ~(void | complete) & ~unmasked_nonfinite
                invalid_count += int(np.count_nonzero(invalid))
                if first_index is None and np.any(invalid):
                    flat = int(np.flatnonzero(invalid)[0])
                    first_index = _global_cell_index(origin, spatial_shape, flat)
                    mismatched_slots = int(
                        np.count_nonzero(
                            latitude_vertex_finite[flat]
                            != longitude_vertex_finite[flat]
                        )
                    )
                    first_detail = (
                        f"latitude centre is "
                        f"{'finite' if latitude_finite[flat] else 'missing'}, "
                        f"longitude centre is "
                        f"{'finite' if longitude_finite[flat] else 'missing'}, "
                        f"{int(paired_counts[flat])} of "
                        f"{latitude_vertices.shape[-1]} vertex pairs are finite, "
                        f"and {mismatched_slots} vertex slots define only one "
                        "of latitude or longitude"
                    )
    except (TypeError, ValueError, IndexError, OSError, RuntimeError) as exc:
        findings.add(
            family,
            "Could not inspect horizontal grid geometry for missing values: "
            f"{exc}.",
        )
        return

    if nonfinite_count:
        findings.add(
            family,
            f"Horizontal grid geometry contains unmasked NaN or infinite values "
            f"at {nonfinite_count} of {total} grid positions. First incident at "
            f"index {first_nonfinite_index}. Missing grid geometry must use the "
            "configured finite fill value rather than NaN or infinity.",
        )
    if invalid_count:
        findings.add(
            family,
            f"Horizontal grid geometry is partially defined for {invalid_count} "
            f"of {total} grid positions. First incident at index {first_index}: "
            f"{first_detail}. Each position must either be fully void or define "
            f"both centres with {vertex_requirement}.",
        )


def _validate_vertices(
    findings,
    ds,
    coordinate,
    role,
    entry,
    catalog,
):
    declared = ncattr(coordinate, "bounds")
    vertex_entry = catalog.grid_variables.get(f"vertices_{role}", {})
    recommended = str(vertex_entry.get("out_name") or f"vertices_{role}")
    if not declared:
        findings.add_prerequisite(
            "grid",
            lambda family: (
                f"It is {findings.severity_word(family)} for horizontal auxiliary "
                f"coordinate '{coordinate.name}' to have a bounds attribute naming "
                "its vertex variable; dependent vertex checks could not be evaluated."
            ),
            issue=("missing_vertices_attribute", coordinate.name),
            fallbacks=(f"grid_{role}_valid_range",),
        )
        return None
    if declared != recommended:
        findings.add(
            "bounds_name",
            f"'{coordinate.name}' bounds={declared!r}; the "
            f"{findings.severity_word('bounds_name')} name is "
            f"{recommended!r}.",
        )
    if declared not in ds.variables:
        findings.add_prerequisite(
            "grid",
            f"'{coordinate.name}' bounds attribute names {declared!r}, but that "
            "vertex variable is absent, so dependent vertex checks could not be "
            "evaluated.",
            issue=("missing_vertices_variable", declared),
            fallbacks=(f"grid_{role}_valid_range",),
        )
        return None
    vertices = ds.variables[declared]
    valid = (
        vertices.ndim == coordinate.ndim + 1
        and vertices.dimensions[:-1] == coordinate.dimensions
        and vertices.shape[:-1] == coordinate.shape
        and vertices.shape[-1] >= 3
    )
    if not valid:
        findings.add_prerequisite(
            "grid",
            lambda family: (
                f"It is {findings.severity_word(family)} for vertex variable "
                f"'{declared}' to have the coordinate dimensions "
                f"{list(coordinate.dimensions)} followed by a vertex dimension of "
                f"size at least 3; found {list(vertices.dimensions)} with shape "
                f"{vertices.shape}. Dependent vertex checks could not be evaluated."
            ),
            issue=("invalid_vertices_structure", declared),
            fallbacks=(f"grid_{role}_valid_range",),
        )
        return None

    check_dtype(findings, vertices, declared, vertex_entry, family="grid")
    check_missing_value_attributes(
        findings,
        vertices,
        declared,
        vertex_entry,
        role="vertex",
    )
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
    return vertices


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
        return None, None
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
        return var, None
    check_dtype(findings, var, name, entry, family="grid")
    check_missing_value_attributes(findings, var, name, entry, role="auxiliary")
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
            value_chunks=(chunk for _, chunk in _bounded_spatial_chunks(var)),
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
    vertices = _validate_vertices(
        findings,
        ds,
        var,
        role,
        entry,
        catalog,
    )
    return var, vertices


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
        check_missing_value_attributes(
            findings,
            var,
            dimension,
            entry,
            role="coordinate",
        )
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
        findings.add_prerequisite(
            "grid",
            "The horizontal grid could not be verified. "
            + (resolution_error or "No grid topology was resolved."),
            issue=(
                "grid_resolution",
                resolution_error or "No grid topology was resolved.",
            ),
        )
        return None
    if topology not in {"rectilinear", "curvilinear", "unstructured"}:
        findings.add_prerequisite(
            "grid",
            f"The horizontal grid could not be verified. Unsupported configured "
            f"topology={topology!r}.",
            issue="horizontal_grid_unresolved",
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
        findings.add_prerequisite(
            "grid",
            lambda family: (
                f"The {findings.severity_word(family)} horizontal coordinate(s) "
                f"are absent: {missing}."
                + (
                    " Dependent horizontal-grid checks could not be evaluated."
                    if family != "grid"
                    else ""
                )
            ),
            issue=("missing_horizontal_coordinates", tuple(missing)),
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
                findings.add_prerequisite(
                    "grid",
                    lambda family, expected_name=expected_name, var=var: (
                        f"It is {findings.severity_word(family)} for "
                        f"'{expected_name}' on the configured rectilinear grid to "
                        f"be a 1-D coordinate variable; found '{var.name}' with "
                        f"dimensions {list(var.dimensions)}. Dependent "
                        "horizontal-grid checks could not be evaluated."
                    ),
                    issue=("invalid_horizontal_structure", var.name),
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
            check_missing_value_attributes(
                findings,
                var,
                var.name,
                entry,
                role="coordinate",
            )
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
            findings.add_prerequisite(
                "grid",
                lambda family, var=var: (
                    f"It is {findings.severity_word(family)} for '{var.name}' to "
                    f"be {ndim}-D for this grid topology; found dimensions "
                    f"{list(var.dimensions)}. Dependent horizontal-grid checks "
                    "could not be evaluated."
                ),
                issue=("invalid_horizontal_structure", var.name),
            )
            valid_structure = False
    if valid_structure:
        dimension_sets = {var.dimensions for var in discovered.values()}
        if len(dimension_sets) != 1:
            findings.add_prerequisite(
                "grid",
                lambda family: (
                    f"It is {findings.severity_word(family)} for latitude and "
                    "longitude auxiliary coordinates to share identical dimensions; "
                    f"found { {role: list(var.dimensions) for role, var in discovered.items()} }. "
                    "Dependent horizontal-grid checks could not be evaluated."
                ),
                issue="mismatched_horizontal_dimensions",
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
    available = [pair[0] for pair in validated.values() if pair[0] is not None]
    if len(available) == len(requested):
        latitude, latitude_vertices = validated["latitude"]
        longitude, longitude_vertices = validated["longitude"]
        if latitude_vertices is not None and longitude_vertices is not None:
            if (
                latitude_vertices.shape != longitude_vertices.shape
                or latitude_vertices.dimensions != longitude_vertices.dimensions
            ):
                findings.add(
                    "grid",
                    "Latitude and longitude vertex variables must have identical "
                    "dimensions and shapes; found "
                    f"{latitude_vertices.name!r} with "
                    f"{list(latitude_vertices.dimensions)} and "
                    f"{longitude_vertices.name!r} with "
                    f"{list(longitude_vertices.dimensions)}.",
                )
            else:
                _check_grid_geometry_missing_values(
                    findings,
                    latitude,
                    longitude,
                    latitude_vertices,
                    longitude_vertices,
                    topology=topology,
                )
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
