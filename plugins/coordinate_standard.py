"""Coordinate validation engine built on compliance-checker and cf-xarray.

Builds a standard from the CMOR tables (later esgvoc), classifies a netCDF
file the same way, then matches the two and reports differences. Structural
detection, role detectors and units logic come from compliance_checker.cf.util
(and cf.appendix_d for parametric vertical coordinates). Name vocabularies
come from cf_xarray.criteria. Result objects are created in
checks/coordinate_checks/.

Example:
    from plugins.coordinate_standard import (
        CmorTableProvider, classify_dataset, match, read_1d_values,
    )

    std = CmorTableProvider(tables={
        "coordinate": CTcoords, "grids": CTgrids, "formula_terms": CTformulas,
    }).build()
    kinds, candidates = classify_dataset(ds)
    outcome, detail = match(candidate, read_1d_values(ds.variables[name]), std)
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from cf_xarray.criteria import coordinate_criteria as _CF_CRITERIA
from cf_xarray.criteria import regex as _CF_NAME_REGEX
from compliance_checker.cf import util as cfutil
from compliance_checker.cf.appendix_d import dimless_vertical_coordinates_1_7

__all__ = [
    # model
    "Role", "Representation", "VariableKind", "Coordinate", "Standard",
    # provider
    "StandardProvider", "CmorTableProvider", "EsgvocProvider",
    # esgvoc helpers (optional)
    "esgvoc_readiness", "esgvoc_name_findings",
    # classification
    "classify_dataset", "classify_role", "classify_representation", "read_1d_values",
    # match
    "match", "diff", "missing_coordinates", "GOOD_OUTCOMES", "WARN_OUTCOMES",
]


# === 1. Model ===
class Role(str, Enum):
    """Semantic axis of a coordinate."""
    LATITUDE = "LATITUDE"
    LONGITUDE = "LONGITUDE"
    TIME = "TIME"
    VERTICAL = "VERTICAL"
    GRID_X = "GRID_X"                 # native/projected x: rlon, projection_x
    GRID_Y = "GRID_Y"                 # native/projected y: rlat, projection_y
    INDEX = "INDEX"                   # i, j, k, l, m, site
    CATEGORY = "CATEGORY"             # character/label axes: region, area_type
    PHYSICAL_AXIS = "PHYSICAL_AXIS"   # diagnostic axes: wavelength, optical depth
    UNKNOWN = "UNKNOWN"


class Representation(str, Enum):
    """Storage form of a coordinate."""
    DIMENSION = "DIMENSION_COORDINATE"
    AUXILIARY = "AUXILIARY_COORDINATE"
    SCALAR = "SCALAR_COORDINATE"
    INDEX = "INDEX_COORDINATE"
    FORMULA = "FORMULA_COORDINATE"


class VariableKind(str, Enum):
    """Kind of a netCDF variable (file side)."""
    DATA = "DATA"
    DIMENSION_COORDINATE = "DIMENSION_COORDINATE"
    GRID_COORDINATE = "GRID_COORDINATE"
    AUXILIARY_COORDINATE = "AUXILIARY_COORDINATE"
    INDEX_COORDINATE = "INDEX_COORDINATE"
    BOUNDS = "BOUNDS"
    GRID_MAPPING = "GRID_MAPPING"
    FORMULA_TERM = "FORMULA_TERM"
    UNKNOWN = "UNKNOWN"


@dataclass
class Coordinate:
    """One coordinate, source-neutral (standard side or file side)."""
    name: str
    role: Role
    representation: Representation
    kind: VariableKind
    standard_name: str = ""
    units: str = ""
    axis: str = ""
    positive: str = ""
    out_name: str = ""
    must_have_bounds: bool = False
    stored_direction: str = ""
    valid_min: str = ""
    valid_max: str = ""
    tolerance: str = ""
    parametric: bool = False                       # parametric (formula) vertical
    rank: int = -1                                 # ndim; 0 = scalar, -1 = unknown
    value: str = ""                                # scalar coords (single value)
    requested: list = field(default_factory=list)  # multi-value / label axes

    def __repr__(self) -> str:
        v = f" value={self.value}" if self.value else ""
        r = f" requested[{len(self.requested)}]" if self.requested else ""
        return (f"<{self.name} {self.role.value} {self.representation.value}"
                f" sn={self.standard_name!r}{v}{r}>")


@dataclass
class Standard:
    """A built standard: coordinate catalogue, provenance and non-coordinate vars."""
    source: str
    version: str = ""
    coordinates: list[Coordinate] = field(default_factory=list)
    formula_terms: dict = field(default_factory=dict)
    grid_mappings: dict = field(default_factory=dict)
    bounds_vars: dict = field(default_factory=dict)


# === 2. Classification rules (standard side and file side) ===

# Discrete (non-parametric) vertical standard names, from cf-xarray's public
# criteria table. model_level_number is a CMOR name cf-xarray does not list.
VERTICAL_STANDARD_NAMES = (
    frozenset(_CF_CRITERIA["vertical"]["standard_name"]) | {"model_level_number"}
)

# Parametric (dimensionless) vertical coordinates, from CF appendix D.
PARAMETRIC_STANDARD_NAMES = frozenset(dimless_vertical_coordinates_1_7)

ROLE_STANDARD_NAMES = VERTICAL_STANDARD_NAMES | {
    "latitude", "longitude", "time", "grid_latitude", "grid_longitude",
    "projection_x_coordinate", "projection_y_coordinate", "region", "area_type",
}


def _role_from_weak_signals(units: str, name: str, long_name: str,
                            var=None, dtype: str = "") -> Role | None:
    """Best-effort role guess for variables lacking standard_name and axis.

    Unit and attribute signals come from compliance-checker (is_time_variable,
    is_vertical_coordinate, VALID_LAT_UNITS/VALID_LON_UNITS, units_convertible).
    Name signals come from cf_xarray.criteria.regex, which covers rotated-pole
    and ocean-model spellings (rlon/rlat, nlon/nlat, nav_lon, lev, sigma).
    Name regexes are skipped for integer and character variables so index and
    label axes fall through to INDEX/CATEGORY. Returns None if nothing is a
    confident match.
    """
    u = str(units or "").strip()
    ul = u.lower()
    n = str(name or "").strip()
    nl = n.lower()

    # Units are the strongest weak signal.
    if ul in cfutil.VALID_LON_UNITS:
        return Role.LONGITUDE
    if ul in cfutil.VALID_LAT_UNITS:
        return Role.LATITUDE

    # Time: cf.util detector, then "<unit> since <date>" units.
    if var is not None and cfutil.is_time_variable(n, var):
        return Role.TIME
    if " since " in ul:
        return Role.TIME

    # Vertical: cf.util detector, then pressure-convertible units.
    if var is not None and cfutil.is_vertical_coordinate(n, var):
        return Role.VERTICAL
    if u and cfutil.units_convertible(u, "dbar"):
        return Role.VERTICAL

    # Name regexes apply to plain numeric variables only.
    if dtype in ("integer", "character"):
        return None

    if _CF_NAME_REGEX["time"].fullmatch(nl) or _CF_NAME_REGEX["T"].fullmatch(nl):
        return Role.TIME
    # Native/projected grid axes before the geographic buckets.
    if _CF_NAME_REGEX["X"].fullmatch(nl):
        return Role.GRID_X
    if _CF_NAME_REGEX["Y"].fullmatch(nl):
        return Role.GRID_Y
    if _CF_NAME_REGEX["Z"].fullmatch(nl):
        return Role.VERTICAL
    # Geographic lon/lat by name, unless units contradict it (length units
    # mean a projected axis).
    degreeish = not u or "degree" in ul
    if _CF_NAME_REGEX["longitude"].fullmatch(nl) and degreeish:
        return Role.LONGITUDE
    if _CF_NAME_REGEX["latitude"].fullmatch(nl) and degreeish:
        return Role.LATITUDE
    return None


def classify_role(standard_name: str, axis: str, dtype: str, parametric: bool,
                  units: str = "", long_name: str = "", name: str = "",
                  var=None) -> Role:
    """Work out a coordinate's Role.

    The standard_name drives the decision; the axis alone is ambiguous (lon,
    rlon and projection_x all carry axis=X). Without a standard_name the
    weak-signal fallback is used, so non-CF files still get classified. On
    the file side, pass the netCDF variable as var to enable the cf.util
    detectors.
    """
    sn = standard_name
    if sn == "longitude":
        return Role.LONGITUDE
    if sn == "latitude":
        return Role.LATITUDE
    if sn in ("grid_longitude", "projection_x_coordinate"):
        return Role.GRID_X
    if sn in ("grid_latitude", "projection_y_coordinate"):
        return Role.GRID_Y
    if axis == "T" or sn == "time":
        return Role.TIME
    if axis == "Z" or parametric or sn in VERTICAL_STANDARD_NAMES:
        return Role.VERTICAL

    if not sn:
        weak = _role_from_weak_signals(units, name, long_name, var=var, dtype=dtype)
        if weak is not None:
            return weak

    if dtype == "integer":
        return Role.INDEX
    if dtype == "character":
        return Role.CATEGORY
    if axis == "" and dtype in ("double", "float", "real"):
        return Role.PHYSICAL_AXIS
    return Role.UNKNOWN


def classify_representation(*, parametric: bool, has_value: bool,
                            dtype: str, is_aux: bool) -> Representation:
    """Assign a Representation from structural signals."""
    if parametric:
        return Representation.FORMULA
    if has_value:
        # A single value means scalar, even if the variable is also named
        # in a coordinates attribute.
        return Representation.SCALAR
    if is_aux:
        return Representation.AUXILIARY
    if dtype == "integer":
        return Representation.INDEX
    return Representation.DIMENSION


# === 3. File-side classifier ===
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


# === 4. Provider seam (CMOR tables today, esgvoc later) ===
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


def _esgvoc_terms(data_descriptor: str) -> frozenset | None:
    """Term ids of a universe data descriptor, or None if esgvoc is unavailable."""
    global _ESGVOC_TERM_CACHE
    try:
        cache = _ESGVOC_TERM_CACHE
    except NameError:
        cache = _ESGVOC_TERM_CACHE = {}
    if data_descriptor not in cache:
        try:
            import esgvoc.api as ev
            cache[data_descriptor] = frozenset(
                t.id for t in ev.get_all_terms_in_data_descriptor(data_descriptor))
        except Exception:
            cache[data_descriptor] = None
    return cache[data_descriptor]


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


# === 5. Match and diff ===
GOOD_OUTCOMES = ("MATCH", "PINNED")
WARN_OUTCOMES = ("AMBIGUOUS", "AMBIGUOUS_UNPINNABLE", "MATCH_WITH_WARNINGS")


def _to_floats(values):
    out = []
    for x in values:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return None
    return out


def _values_match(file_values, ref_values, tolerance) -> bool:
    """Compare value lists. An explicit table tolerance (including 0) is
    used as-is; an absent tolerance falls back to a small relative one."""
    try:
        tol = float(tolerance) if str(tolerance).strip() else None
    except (TypeError, ValueError):
        tol = None
    if len(file_values) != len(ref_values):
        return False
    return all(
        abs(a - b) <= (tol if tol is not None else 1e-6 * max(1.0, abs(b)))
        for a, b in zip(file_values, ref_values))


def _compare_units(candidate_units: str, entry_units: str) -> tuple[str, str]:
    """Tiered units comparison backed by udunits.

    Returns (level, message): "ok", "warn" (convertible but not identical,
    e.g. hPa vs Pa) or "fail" (not convertible or invalid). CMOR time
    templates like "days since ?" accept any reference date.
    """
    if not entry_units:
        return "ok", ""
    if not candidate_units:
        return "warn", f"units missing, table expects {entry_units!r}"
    if candidate_units == entry_units:
        return "ok", ""
    if "?" in entry_units:      # CMOR template, e.g. "days since ?"
        pattern = re.escape(entry_units).replace(r"\?", ".+")
        if re.fullmatch(pattern, candidate_units):
            return "ok", ""
        # Compare the base units of "<unit> since <?>" (hours vs days).
        if " since " in entry_units and " since " in candidate_units:
            entry_base = entry_units.split(" since ")[0]
            cand_base = candidate_units.split(" since ")[0]
            if cfutil.units_convertible(cand_base, entry_base):
                return "warn", (f"units {candidate_units!r} use base unit "
                                f"{cand_base!r}, table expects {entry_base!r}")
            return "fail", f"units {candidate_units!r} not convertible to {entry_units!r}"
        # A template can never be parsed by udunits, so do not fall through.
        return "fail", (f"units {candidate_units!r} do not match the table "
                        f"template {entry_units!r}")
    if cfutil.units_convertible(candidate_units, entry_units):
        return "warn", (f"units {candidate_units!r} convertible to, but not "
                        f"identical to, table units {entry_units!r}")
    return "fail", f"units {candidate_units!r} not convertible to {entry_units!r}"


def diff(candidate: Coordinate, file_values, entry: Coordinate) -> tuple[list[str], list[str]]:
    """Diff a candidate against its matched standard entry.

    Returns (failures, warnings). Units are compared with udunits semantics
    (see _compare_units), not raw string equality.
    """
    failures: list[str] = []
    warnings: list[str] = []

    # Unit-string validity is udunits' call. "since" templates are handled
    # by _compare_units.
    if (candidate.units and " since " not in candidate.units
            and not cfutil.units_known(candidate.units)):
        failures.append(f"units {candidate.units!r} not recognized by udunits")

    level, msg = _compare_units(candidate.units, entry.units)
    if level == "fail":
        failures.append(msg)
    elif level == "warn":
        warnings.append(msg)

    if entry.positive:
        if not candidate.positive:
            failures.append(f"missing positive (table expects {entry.positive!r})")
        elif entry.positive != candidate.positive:
            failures.append(f"positive {candidate.positive!r}!={entry.positive!r}")
    if entry.must_have_bounds and not candidate.must_have_bounds:
        failures.append("missing bounds")
    if (entry.rank >= 0 and candidate.rank >= 0
            and entry.rank != candidate.rank):
        msg = f"rank {candidate.rank} != table rank {entry.rank}"
        # Unstructured/HEALPix files legitimately store aux lat/lon as 1-D
        # (lat(cells)), while the grids table only knows the 2-D curvilinear
        # form. Rank deviations on auxiliaries are therefore advisory.
        if candidate.representation is Representation.AUXILIARY:
            warnings.append(msg)
        else:
            failures.append(msg)
    # stored_direction is not diffed here: the atomic check [VAR005]
    # (checks/variable_checks/check_coordinate_monotonicity.py) owns
    # direction checking; the glue feeds it the pinned entry's direction.
    fv = _to_floats(file_values) if file_values is not None else None
    if fv:
        if entry.valid_min and min(fv) < float(entry.valid_min):
            failures.append(f"min {min(fv)} < valid_min {entry.valid_min}")
        if entry.valid_max and max(fv) > float(entry.valid_max):
            failures.append(f"max {max(fv)} > valid_max {entry.valid_max}")
    # Pinned values: when the entry prescribes exact values (requested list
    # or scalar value), the file must match them even when the entry was
    # resolved without value-pinning (a pool of one). Label (character)
    # axes are skipped: their requested lists are not numeric.
    if entry.requested or entry.value:
        ref = (_to_floats(entry.requested) if entry.requested
               else _to_floats([entry.value]))
        cand = fv
        if cand is None and candidate.value:
            cand = _to_floats([candidate.value])
        if (ref is not None and cand is not None
                and not _values_match(cand, ref, entry.tolerance)):
            failures.append(
                f"values do not match the table's "
                f"{'requested list' if entry.requested else 'value'}")
    return failures, warnings


def missing_coordinates(ds, kinds: dict, std: Standard,
                        ct_dimensions=None) -> dict[str, str]:
    """Report expected coordinate variables that do not exist in the file.

    Deliberately plain: if a dimension used by a data variable matches a
    standard entry's out_name but no variable of that name exists, the
    coordinate variable does not exist. No inference. Returns
    {expected_variable_name: finding}.

    ct_dimensions, when given, is the token list from a CMOR variable table
    entry's "dimensions" string (e.g. ["longitude", "latitude", "time",
    "height2m"]). Each token names a coordinate-table entry and is satisfied
    if a variable matching any of its entries' out_names exists ("latitude"
    accepts the 1-D lat axis or the 2-D latitude grids auxiliary). This
    catches missing scalar coordinates like height2m, which are never netCDF
    dimensions. Unknown tokens are skipped: a table problem is not the
    file's fault.
    """
    out_names: dict[str, list[str]] = {}
    for c in std.coordinates:
        out_names.setdefault(c.out_name or c.name, []).append(c.name)

    data_dims: set[str] = set()
    for name, kind in kinds.items():
        if kind is VariableKind.DATA:
            data_dims.update(ds.variables[name].dimensions)

    findings: dict[str, str] = {}
    for dim in sorted(data_dims):
        if dim in out_names and dim not in ds.variables:
            entries = ", ".join(sorted(set(out_names[dim])))
            findings[dim] = (f"coordinate variable {dim!r} does not exist "
                             f"(dimension present; table entries: {entries})")

    if ct_dimensions:
        by_entry_name: dict[str, list[Coordinate]] = {}
        for c in std.coordinates:
            by_entry_name.setdefault(c.name, []).append(c)
        for token in ct_dimensions:
            entries = by_entry_name.get(str(token))
            if not entries:
                continue                    # unknown token: not the file's fault
            accepted = {c.out_name or c.name for c in entries}
            if accepted & set(ds.variables):
                continue
            if accepted & set(findings):    # already reported by the pass above
                continue
            key = sorted(accepted)[0]
            names = " / ".join(sorted(accepted))
            findings[key] = (
                f"coordinate variable {names!r} does not exist "
                f"(required by the variable table: dimensions token {token!r})")
    return findings


def match(candidate: Coordinate, file_values, std: Standard) -> tuple[str, object]:
    """Match a candidate to its standard entry and diff. Returns (outcome, detail).

    Outcomes: MATCH / PINNED (good); MATCH_WITH_WARNINGS (matched, soft
    issues); MISMATCH:<why> (attributes or values disagree); NO_VALUE_MATCH /
    NO_STANDARD_ENTRY (no standard fit); AMBIGUOUS / AMBIGUOUS_UNPINNABLE
    (cannot resolve which entry, e.g. time/lat/lon).
    """
    pool = [c for c in std.coordinates
            if c.role == candidate.role
            and c.representation == candidate.representation
            and (not candidate.standard_name
                 or c.standard_name == candidate.standard_name)]
    if not pool:
        return "NO_STANDARD_ENTRY", None
    if len(pool) == 1:
        failures, warnings = diff(candidate, file_values, pool[0])
        if failures:
            return "MISMATCH:" + "; ".join(failures), pool[0].name
        if warnings:
            return "MATCH_WITH_WARNINGS:" + "; ".join(warnings), pool[0].name
        return "MATCH", pool[0].name

    fv = _to_floats(file_values) if file_values is not None else None
    if candidate.value:
        fv = _to_floats([candidate.value])
    pinnable = [c for c in pool if c.requested or c.value]
    if not pinnable:
        return "AMBIGUOUS_UNPINNABLE", [c.name for c in pool]

    pinned = []
    for c in pinnable:
        ref = _to_floats(c.requested) if c.requested else _to_floats([c.value])
        if fv is not None and ref is not None and _values_match(fv, ref, c.tolerance):
            pinned.append(c.name)
    if len(pinned) == 1:
        return "PINNED", pinned[0]
    if pinned:
        return "AMBIGUOUS", pinned
    return "NO_VALUE_MATCH", [c.name for c in pool]


# === 6. Plain-text report (CLI lives in checks/coordinate_checks/classify_file.py) ===
_STATUS = {                       # outcome prefix -> (label, is_fail)
    "MATCH": ("PASS", False), "PINNED": ("PASS", False),
    "MATCH_WITH_WARNINGS": ("WARN", False),
    "MISMATCH": ("FAIL", True), "NO_VALUE_MATCH": ("FAIL", True),
    "NO_STANDARD_ENTRY": ("FAIL", True),
    "AMBIGUOUS_UNPINNABLE": ("WARN", False), "AMBIGUOUS": ("WARN", False),
}


def report(ds, std: Standard) -> int:
    """Classify ds, match against std, print a report, return the FAIL count."""
    kinds, candidates = classify_dataset(ds)

    by_kind: dict[str, list[str]] = {}
    for name, kind in kinds.items():
        by_kind.setdefault(kind.value, []).append(name)
    print(f"\nVariables ({len(kinds)}):")
    for kind, names in sorted(by_kind.items()):
        print(f"  {kind:22} {', '.join(names)}")

    print("\nCoordinate checks:")
    fails = warns = passes = 0
    for name, cand in candidates.items():
        outcome, detail = match(cand, read_1d_values(ds.variables[name]), std)
        label, is_fail = _STATUS.get(outcome.split(":", 1)[0], ("WARN", False))
        fails += is_fail
        warns += label == "WARN"
        passes += label == "PASS"
        var = ds.variables[name]
        cf_missing = ("standard_name" not in var.ncattrs()
                      and "axis" not in var.ncattrs())
        note = "  (no standard_name/axis - role inferred)" if cf_missing else ""
        detail_str = f" [{detail}]" if detail else ""
        print(f"  {label:4} {name:18} {cand.role.value:13} "
              f"{cand.representation.value:22} {outcome}{detail_str}{note}")

    missing = missing_coordinates(ds, kinds, std)
    if missing:
        print("\nMissing coordinates:")
        for name, finding in missing.items():
            print(f"  FAIL {name:18} {finding}")
        fails += len(missing)

    vocab = esgvoc_name_findings(ds, kinds, candidates)
    if vocab:
        print("\nesgvoc vocabulary findings:")
        for name, finding in vocab.items():
            print(f"  WARN {name:18} {finding}")

    print(f"\nSummary: {len(candidates)} coordinates  |  "
          f"PASS {passes}  WARN {warns}  FAIL {fails}"
          + (f"  |  missing {len(missing)}" if missing else "")
          + (f"  |  esgvoc vocab WARN {len(vocab)}" if vocab else ""))
    return fails
