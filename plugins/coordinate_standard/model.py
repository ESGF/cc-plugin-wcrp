"""Model: the vocabulary enums and the source-neutral containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
