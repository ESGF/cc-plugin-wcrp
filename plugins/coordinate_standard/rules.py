"""Classification rules, shared by the standard side and the file side."""
from __future__ import annotations

from cf_xarray.criteria import coordinate_criteria as _CF_CRITERIA
from cf_xarray.criteria import regex as _CF_NAME_REGEX
from compliance_checker.cf import util as cfutil
from compliance_checker.cf.appendix_d import dimless_vertical_coordinates_1_7

from plugins.coordinate_standard.model import Representation, Role

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
