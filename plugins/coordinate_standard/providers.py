"""Provider seam (CMOR tables today, esgvoc later) and esgvoc helpers."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from plugins.coordinate_standard.classify import attr
from plugins.coordinate_standard.model import (
    Coordinate,
    Standard,
    VariableKind,
)
from plugins.coordinate_standard.rules import (
    classify_representation,
    classify_role,
)


class StandardProvider(ABC):
    """Builds a Standard. Swap the implementation to change the source."""

    @abstractmethod
    def build(self) -> Standard:
        ...


class CmorTableProvider(StandardProvider):
    """Build the standard from the CMOR tables.

    Tables are read from tables_dir (<prefix>_coordinate.json and friends)
    or handed in directly as dicts via tables=... . Inside the checker, pass
    the already-loaded self.CTcoords etc.
    """

    def __init__(self, tables_dir: Path | str = ".", prefix: str = "CMIP7",
                 *, tables: dict | None = None):
        self.dir = Path(tables_dir)
        self.prefix = prefix
        self._injected = tables   # {"coordinate":..., "grids":..., "formula_terms":...}

    def _table(self, name: str) -> dict:
        if self._injected is not None:
            return self._injected[name]
        return json.loads((self.dir / f"{self.prefix}_{name}.json").read_text())

    @staticmethod
    def _is_parametric(entry: dict) -> bool:
        return bool(entry.get("generic_level_name") or entry.get("z_factors")
                    or entry.get("formula"))

    def _coordinate(self, name: str, entry: dict, kind: VariableKind, *,
                    is_aux: bool = False) -> Coordinate:
        parametric = self._is_parametric(entry)
        has_value = str(entry.get("value", "")) != ""
        dtype = entry.get("type", "")
        # Rank: scalar entries are 0-D; grids variable_entry carries a
        # "dimensions" string ("longitude latitude" is 2-D); axis entries
        # are 1-D by definition.
        if has_value:
            rank = 0
        elif entry.get("dimensions"):
            rank = len(str(entry["dimensions"]).split())
        else:
            rank = 1
        return Coordinate(
            name=name,
            role=classify_role(entry.get("standard_name", ""), entry.get("axis", ""),
                               dtype, parametric, units=entry.get("units", ""),
                               long_name=entry.get("long_name", ""), name=name),
            representation=classify_representation(
                parametric=parametric, has_value=has_value, dtype=dtype, is_aux=is_aux),
            kind=kind,
            standard_name=entry.get("standard_name", ""),
            units=entry.get("units", ""),
            axis=entry.get("axis", ""),
            positive=entry.get("positive", ""),
            out_name=entry.get("out_name", ""),
            must_have_bounds=str(entry.get("must_have_bounds", "")).lower() == "yes",
            stored_direction=entry.get("stored_direction", ""),
            valid_min=str(entry.get("valid_min", "")),
            valid_max=str(entry.get("valid_max", "")),
            tolerance=str(entry.get("tolerance", "")),
            parametric=parametric,
            rank=rank,
            value=str(entry.get("value", "")),
            requested=list(entry.get("requested", [])),
        )

    def build(self) -> Standard:
        coordinate = self._table("coordinate")
        grids = self._table("grids")
        formula = self._table("formula_terms")

        version = coordinate.get("Header", {}).get("table_date", "") or \
            grids.get("Header", {}).get("table_date", "")
        std = Standard(source=f"cmor-tables:{self.prefix}", version=version)

        for name, entry in coordinate["axis_entry"].items():
            kind = (VariableKind.INDEX_COORDINATE if entry.get("type") == "integer"
                    else VariableKind.DIMENSION_COORDINATE)
            std.coordinates.append(self._coordinate(name, entry, kind))

        for name, entry in grids["axis_entry"].items():
            if name == "vertices":
                std.bounds_vars[name] = entry
                continue
            kind = (VariableKind.INDEX_COORDINATE if entry.get("type") == "integer"
                    else VariableKind.GRID_COORDINATE)
            std.coordinates.append(self._coordinate(name, entry, kind))

        for name, entry in grids["variable_entry"].items():
            if name.startswith("vertices_"):
                std.bounds_vars[name] = entry
                continue
            std.coordinates.append(
                self._coordinate(name, entry, VariableKind.AUXILIARY_COORDINATE,
                                 is_aux=True))

        std.grid_mappings.update(grids.get("mapping_entry", {}))
        std.formula_terms.update(formula["formula_entry"])
        return std


class EsgvocProvider(StandardProvider):
    """Future backend sourced from esgvoc. Stub until esgvoc exposes the
    value-level coordinate metadata; see esgvoc_readiness()."""

    def build(self) -> Standard:
        raise NotImplementedError(
            "esgvoc provider not ready yet: the universe does not expose "
            "value-level coordinate metadata (requested/units/positive/...). "
            "Check esgvoc_readiness() for what is currently available.")


# --- esgvoc helpers (optional; needs a configured esgvoc DB) ---

# Fields a standard-side Coordinate needs from esgvoc before EsgvocProvider
# can replace the CMOR tables.
_ESGVOC_REQUIRED_FIELDS = (
    "standard_name", "units", "axis", "positive", "out_name",
    "requested", "value", "valid_min", "valid_max", "stored_direction",
    "tolerance", "must_have_bounds",
)


def esgvoc_readiness() -> dict:
    """Probe the active esgvoc universe (needs esgvoc >= 4.2.0).

    For each coordinate-relevant data descriptor, report the model fields it
    exposes and which required fields are still missing. Returns
    {descriptor: {"model": name, "fields": [...], "missing": [...]}}, or
    {"error": ...} entries when a descriptor is absent or unreadable. The
    provider can be implemented once "missing" is empty for a coordinate-like
    descriptor.
    """
    result: dict = {}
    try:
        import esgvoc.api as ev
    except ImportError as e:
        return {"error": f"esgvoc not importable: {e}"}

    for dd in ("coordinate", "vertical_coordinate", "grid", "grid_mapping",
               "formula_terms"):
        try:
            cls = ev.get_model_from_data_descriptor(dd)
            fields = list(getattr(cls, "model_fields", {}))
            result[dd] = {
                "model": getattr(cls, "__name__", str(cls)),
                "fields": fields,
                "missing": [f for f in _ESGVOC_REQUIRED_FIELDS if f not in fields],
            }
        except Exception as e:
            result[dd] = {"error": f"{type(e).__name__}: {e}"}
    return result


_ESGVOC_TERM_CACHE: dict = {}


def _esgvoc_terms(data_descriptor: str) -> frozenset | None:
    """Term ids of a universe data descriptor, or None if esgvoc is unavailable."""
    if data_descriptor not in _ESGVOC_TERM_CACHE:
        try:
            import esgvoc.api as ev
            _ESGVOC_TERM_CACHE[data_descriptor] = frozenset(
                t.id for t in ev.get_all_terms_in_data_descriptor(data_descriptor))
        except Exception:
            _ESGVOC_TERM_CACHE[data_descriptor] = None
    return _ESGVOC_TERM_CACHE[data_descriptor]


def esgvoc_name_findings(ds, kinds: dict, candidates: dict) -> dict[str, str]:
    """Name-level vocabulary checks against the esgvoc universe.

    Complements the CMOR-table match with what esgvoc can already validate:
    grid_mapping_name must be a known grid_mapping term, and a parametric
    vertical standard_name must be a known vertical_coordinate term. Returns
    {varname: finding}. Empty when everything is known or esgvoc is not
    available (soft dependency).
    """
    findings: dict[str, str] = {}

    gm_terms = _esgvoc_terms("grid_mapping")
    if gm_terms is not None:
        for name, kind in kinds.items():
            if kind is VariableKind.GRID_MAPPING:
                gmn = attr(ds.variables[name], "grid_mapping_name")
                if gmn and gmn not in gm_terms:
                    findings[name] = (f"grid_mapping_name {gmn!r} not in the "
                                      "esgvoc grid_mapping vocabulary")

    vert_terms = _esgvoc_terms("vertical_coordinate")
    if vert_terms is not None:
        for name, cand in candidates.items():
            if (cand.parametric and cand.standard_name
                    and cand.standard_name not in vert_terms):
                findings[name] = (f"standard_name {cand.standard_name!r} not in "
                                  "the esgvoc vertical_coordinate vocabulary")
    return findings
