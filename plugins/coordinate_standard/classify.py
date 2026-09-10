"""File-side classifier for open netCDF4 datasets."""
from __future__ import annotations

import re

from compliance_checker.cf import util as cfutil

from plugins.coordinate_standard.model import (
    Coordinate,
    Representation,
    Role,
    VariableKind,
)
from plugins.coordinate_standard.rules import (
    PARAMETRIC_STANDARD_NAMES,
    ROLE_STANDARD_NAMES,
    VERTICAL_STANDARD_NAMES,
    classify_representation,
    classify_role,
)


def neutral_dtype(var) -> str:
    kind = getattr(getattr(var, "dtype", None), "kind", "")
    if kind in ("S", "U") or getattr(var, "dtype", None) is str:
        return "character"
    if kind in ("i", "u"):
        return "integer"
    if kind == "f":
        return "double"
    return ""


def attr(var, name, default=""):
    return var.getncattr(name) if name in var.ncattrs() else default


def read_scalar(var):
    try:
        return float(var[...].ravel()[0])
    except Exception:
        return var[...]


def read_1d_values(var):
    """Return a coordinate's values as a list, or None if not a 1-D numeric axis."""
    try:
        if var.ndim == 1 and neutral_dtype(var) != "character":
            return list(var[:])
    except Exception:
        pass
    return None


def classify_dataset(ds) -> tuple[dict, dict]:
    """Classify an open netCDF4 dataset.

    Returns (kinds, candidates): kinds maps each variable name to a
    VariableKind, candidates maps each coordinate name to a Coordinate.
    Structural detection is delegated to compliance_checker.cf.util
    (dimension and auxiliary coordinates, bounds, grid mappings). The
    name-suffix fallbacks catch malformed files that drop the bounds
    attribute.
    """
    dims = set(ds.dimensions)

    dim_coord_names = set(cfutil.get_coordinate_variables(ds))
    aux_coord_names = set(cfutil.get_auxiliary_coordinate_variables(ds))
    bounds_names = {v.name for v in cfutil.get_bounds_variables(ds)}
    grid_mapping_names = set(cfutil.get_grid_mapping_variables(ds))

    # Variables consumed by a formula_terms attribute ("term: var term: var").
    # No cf.util getter resolves these to variable names. The regex tolerates
    # nonstandard spacing around the colon ("term : var").
    formula_refs: set[str] = set()
    for var in ds.variables.values():
        if "formula_terms" in var.ncattrs():
            s = str(var.getncattr("formula_terms"))
            formula_refs.update(
                m.group(1) for m in re.finditer(r"\w+\s*:\s*(\w+)", s))

    kinds: dict[str, VariableKind] = {}
    candidates: dict[str, Coordinate] = {}

    for name, var in ds.variables.items():
        sn = attr(var, "standard_name", "")
        axis = attr(var, "axis", "")
        dtype = neutral_dtype(var)

        if name in grid_mapping_names or "grid_mapping_name" in var.ncattrs():
            kinds[name] = VariableKind.GRID_MAPPING
            continue
        if (name in bounds_names or name.endswith(("_bnds", "_bounds"))
                or "vertices" in name):
            kinds[name] = VariableKind.BOUNDS
            continue
        # A parametric coordinate can appear in its own formula_terms
        # (CF appendix D, e.g. ocean_sigma: "sigma: sigma ..."), so a
        # variable that carries formula_terms itself is a coordinate,
        # not a term.
        if name in formula_refs and "formula_terms" not in var.ncattrs():
            kinds[name] = VariableKind.FORMULA_TERM
            continue

        parametric = ("formula_terms" in var.ncattrs()) or (sn in PARAMETRIC_STANDARD_NAMES)
        is_dim_coord = name in dim_coord_names
        is_aux = (name in aux_coord_names) and not is_dim_coord
        # A 1-D length-1 variable referenced in a coordinates attribute is
        # still a scalar coordinate (CF: scalar coordinates are degenerate
        # auxiliaries), so being auxiliary does not veto scalar treatment.
        is_scalar = (var.ndim == 0) or (
            var.size == 1 and name not in dims
            and (axis or sn in VERTICAL_STANDARD_NAMES or sn in ("latitude", "longitude")))

        looks_coord = bool(is_dim_coord or is_aux or is_scalar or parametric
                           or axis or sn in ROLE_STANDARD_NAMES)
        if not looks_coord:
            kinds[name] = VariableKind.DATA
            continue

        role = classify_role(sn, axis, dtype, parametric,
                             units=attr(var, "units"), long_name=attr(var, "long_name"),
                             name=name, var=var)
        rep = classify_representation(parametric=parametric, has_value=is_scalar,
                                      dtype=dtype, is_aux=is_aux)

        if rep in (Representation.AUXILIARY, Representation.SCALAR):
            # A scalar coordinate is a degenerate auxiliary coordinate.
            kind = VariableKind.AUXILIARY_COORDINATE
        elif rep is Representation.INDEX:
            kind = VariableKind.INDEX_COORDINATE
        elif role in (Role.GRID_X, Role.GRID_Y):
            kind = VariableKind.GRID_COORDINATE
        else:
            kind = VariableKind.DIMENSION_COORDINATE

        kinds[name] = kind
        candidates[name] = Coordinate(
            name=name, role=role, representation=rep, kind=kind,
            standard_name=sn, units=attr(var, "units"), axis=axis,
            positive=attr(var, "positive"), out_name=name,
            must_have_bounds=("bounds" in var.ncattrs()),
            parametric=parametric,
            rank=(0 if is_scalar else var.ndim),
            value=(str(read_scalar(var)) if is_scalar else ""),
        )
    return kinds, candidates
