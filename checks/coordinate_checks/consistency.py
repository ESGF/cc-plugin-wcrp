"""Coordinate identities for cross-file summaries, without further CV reads."""

from checks.coordinate_checks.utils import (
    TEXT_AUXILIARY_COORDINATE_NAME,
    coordinate_type,
)
from checks.coordinate_checks.validation import Findings
from checks.coordinate_checks.vertical import select_model_level


def _text_coordinate_names(ds, entry):
    """Locate text coordinates by the expected name or their declared axis."""
    dimension = str(entry.get("out_name") or entry.get("id") or "")
    return [
        name for name, var in ds.variables.items()
        if (getattr(var.dtype, "kind", None) in {"S", "U"} or var.dtype is str)
        and (name == TEXT_AUXILIARY_COORDINATE_NAME
             or (var.dimensions and var.dimensions[0] == dimension))
    ]


def coordinate_ids_for_summary(ds, catalog, names):
    """Map existing summary variables to unambiguous catalogue identities.

    Unresolved names remain absent so consumers can use their legacy fallback.
    This records identity, not successful validation of the coordinate values.
    """
    if catalog is None:
        return {}
    names = set(names)
    candidates = {}

    def add(name, identifier):
        if name in names:
            candidates.setdefault(name, set()).add(identifier)

    for identifier in catalog.coordinate_ids:
        entry = catalog.data_coordinates.get(identifier, {})
        kind = coordinate_type(entry)
        name = str(entry.get("out_name") or identifier)
        if kind == "generic_vertical":
            selected = select_model_level(Findings({}), ds, catalog, identifier)
            if selected is not None:
                add(name, selected[0])
        elif kind == "auxiliary":
            # 'sector' represents the requested character coordinate, not a
            # universal identity shared by soil pools, vegetation and basins.
            for text_name in _text_coordinate_names(ds, entry):
                add(text_name, identifier)
        elif kind in {"standard_1d", "scalar"}:
            add(name, identifier)

    for collection in (catalog.grid_variables, catalog.grid_axes):
        for identifier, entry in collection.items():
            add(str(entry.get("out_name") or identifier), identifier)

    return {
        name: next(iter(identifiers))
        for name, identifiers in sorted(candidates.items())
        if len(identifiers) == 1
    }


def dimension_metadata_for_summary(ds, coordinate_ids):
    """Describe actual axes; ambiguous or malformed metadata is left unresolved."""
    candidates = {}
    string_dimensions = set()
    scientific_dimensions = set()
    for name, var in ds.variables.items():
        dims = var.dimensions
        is_char = getattr(var.dtype, "kind", None) == "S" and var.dtype.itemsize == 1
        if is_char and len(dims) >= 2:
            string_dimensions.add(dims[-1])
            scientific_dimensions.update(dims[:-1])
        else:
            scientific_dimensions.update(dims)
        identity = coordinate_ids.get(name)
        if identity and (len(dims) == 1 or (is_char and len(dims) == 2)):
            candidates.setdefault(dims[0], set()).add(identity)
    # Never hide a dimension also used as a scientific/data axis.
    string_dimensions -= scientific_dimensions
    return (
        {name: next(iter(ids)) for name, ids in sorted(candidates.items()) if len(ids) == 1},
        sorted(string_dimensions),
    )
