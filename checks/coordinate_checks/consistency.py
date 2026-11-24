"""Coordinate identities for cross-file summaries, without further CV reads."""

from checks.coordinate_checks.utils import (
    TEXT_AUXILIARY_COORDINATE_NAME,
    coordinate_type,
)
from checks.coordinate_checks.validation import Findings
from checks.coordinate_checks.vertical import select_model_level


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
            add(TEXT_AUXILIARY_COORDINATE_NAME, identifier)
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
