"""Temporary fixes for bugs in compliance-checker's own CF checks.

The CMIP7 QA workflow runs the compliance-checker ``cf`` suite next to the
WCRP plugins. A crash in a ``cf`` check is reported as an error, which blocks
publication even though the dataset itself is fine. Until the fixes below are
released upstream, they are applied when ``plugins.wcrp_base`` is imported.
compliance-checker loads every installed plugin at start-up, so the fixes
also take effect for ``cf`` runs whenever cc-plugin-wcrp is installed.

Which fix is applied for which compliance-checker release is configured in
``cc_patches.toml`` next to this module (enable switch and version range per
fix). The configuration ships with the package, so the cc-plugin-wcrp version
and the compliance-checker version together determine which fixes were used.
Nothing is written to the checker report or to stderr; the outcome is logged
at debug level and kept in ``STATUS``.

``tests/test_cc_patches.py`` checks for the installed compliance-checker
that every active fix replaces reviewed upstream code that still shows the
bug, and that upstream works wherever a fix is not active.

To add a fix: write the replacement here, register it in ``PATCHES`` with the
fingerprint of the broken upstream source (``source_sha256``), add a section
to ``cc_patches.toml`` and tests. To retire a fix for newer releases, lower
its ``max_version``; to drop it entirely, remove it from all three places.
"""

import hashlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import toml
from compliance_checker.base import BaseCheck, Result, TestCtx
from compliance_checker.cf.appendix_d import dimless_vertical_coordinates_1_6
from compliance_checker.cf.cf_1_6 import CF1_6Check
from compliance_checker.cf.cf_1_7 import CF1_7Check
from netCDF4 import chartostring
from packaging.version import InvalidVersion, Version

try:
    import compliance_checker.cf.util as cfutil
except ImportError:  # pragma: no cover - compliance-checker < 6
    from compliance_checker import cfutil

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).with_name("cc_patches.toml")

# --------------------------------------------------------------------------
# CF 6.1.1: check_geographic_region
# --------------------------------------------------------------------------
# Upstream joins all characters of the region variable into one string. For a
# labeled axis with several regions (a 2-D char array with one name per row,
# e.g. the basin axis of hfbasin, htovgyre, sltovgyre, ...) this raises a
# TypeError. Padding with NUL characters (the netCDF default fill) is also not
# removed, so valid fixed-width names are reported as invalid regions.
#
# The list of valid names in compliance-checker dates from 2014 and lacks six
# names of the current CF standardized region list, among them
# atlantic_arctic_ocean and indian_pacific_ocean, which CMIP uses on the basin
# axis. The list below is the CF standardized region list, version 5
# (12 November 2024):
# https://cfconventions.org/Data/standardized-region-list/standardized-region-list.current.xml

_CF_REGION_NAMES = frozenset(
    [
        "africa",
        "antarctica",
        "arabian_sea",
        "aral_sea",
        "arctic_ocean",
        "asia",
        "atlantic_arctic_ocean",
        "atlantic_ocean",
        "australia",
        "baltic_sea",
        "barents_opening",
        "barents_sea",
        "beaufort_sea",
        "bellingshausen_sea",
        "bering_sea",
        "bering_strait",
        "black_sea",
        "canadian_archipelago",
        "caribbean_sea",
        "caspian_sea",
        "central_america",
        "chukchi_sea",
        "contiguous_united_states",
        "davis_strait",
        "denmark_strait",
        "drake_passage",
        "east_china_sea",
        "english_channel",
        "eurasia",
        "europe",
        "faroe_scotland_channel",
        "florida_bahamas_strait",
        "fram_strait",
        "gibraltar_strait",
        "global",
        "global_land",
        "global_ocean",
        "great_lakes",
        "greenland",
        "gulf_of_alaska",
        "gulf_of_mexico",
        "hudson_bay",
        "iceland_faroe_channel",
        "indian_ocean",
        "indian_pacific_ocean",
        "indo_pacific_ocean",
        "indonesian_throughflow",
        "irish_sea",
        "lake_baykal",
        "lake_chad",
        "lake_malawi",
        "lake_tanganyika",
        "lake_victoria",
        "mediterranean_sea",
        "mozambique_channel",
        "north_america",
        "north_sea",
        "northern_hemisphere",
        "norwegian_sea",
        "pacific_equatorial_undercurrent",
        "pacific_ocean",
        "persian_gulf",
        "red_sea",
        "ross_sea",
        "sea_of_japan",
        "sea_of_okhotsk",
        "south_america",
        "south_china_sea",
        "southern_hemisphere",
        "southern_ocean",
        "taiwan_luzon_straits",
        "weddell_sea",
        "windward_passage",
        "yellow_sea",
    ]
)


def _region_labels(variable):
    """Return the region names stored in a ``standard_name = region`` variable.

    Handles a single name or several names, stored either as a char array
    (last dimension is the string length) or as a string variable. Trailing
    spaces and NUL characters used to fill fixed-width labels are removed.
    """
    regions = np.ma.filled(variable[:], "")
    # Join the final character dimension into labels, leaving any preceding
    # dimensions intact so each region is checked separately.
    if regions.dtype == np.dtype("S1"):
        regions = chartostring(regions)
    labels = []
    for region in np.atleast_1d(regions).ravel():
        if isinstance(region, bytes):
            region = region.decode("utf-8")
        labels.append(str(region).rstrip("\x00 "))
    return labels


def check_geographic_region(self, ds):
    """
    6.1.1 When data is representative of geographic regions which can be identified by names but which have complex
    boundaries that cannot practically be specified using longitude and latitude boundary coordinates, a labeled
    axis should be used to identify the regions.

    Recommend that the names be chosen from the list of standardized region names whenever possible. To indicate
    that the label values are standardized the variable that contains the labels must be given the standard_name
    attribute with the value region.

    (Patched by cc-plugin-wcrp: supports several region names per variable.)

    :param netCDF4.Dataset ds: An open netCDF dataset
    :rtype: list
    :return: List of results
    """
    ret_val = []
    for var in ds.get_variables_by_attributes(standard_name="region"):
        valid_region = TestCtx(BaseCheck.MEDIUM, self.section_titles["6.1"])
        for region in _region_labels(var):
            valid_region.assert_true(
                region.lower() in _CF_REGION_NAMES,
                f"6.1.1 '{region}' specified by '{var.name}' is not a valid region",
            )
        ret_val.append(valid_region.to_result())
    return ret_val


# --------------------------------------------------------------------------
# CF 7.1: check_cell_boundaries_interval
# --------------------------------------------------------------------------
# Upstream reads each coordinate and its complete boundary variable into
# memory at once and builds several temporary arrays of the same size. For
# large unstructured grids (millions of cells, many vertices) this needs
# several GiB and the process may be killed. Here the check reads and compares
# the data in bounded blocks. The result is the same as upstream's.

# Maximum number of boundary values processed at once to limit memory used by
# temporary arrays. For float64 data, 262,144 values use about 2 MiB.
_CELL_BOUNDARY_CHUNK_VALUES = 262_144


def _cell_boundary_slices(shape, vertices):
    """Yield slices that limit the number of boundary values read at once."""
    if not shape:
        yield ()
        return

    cells_per_chunk = max(1, _CELL_BOUNDARY_CHUNK_VALUES // vertices)
    inner_cells = 1
    chunk_axis = len(shape) - 1
    for axis in range(len(shape) - 1, -1, -1):
        if inner_cells > cells_per_chunk:
            break
        chunk_axis = axis
        inner_cells *= shape[axis]

    trailing_cells = max(1, int(np.prod(shape[chunk_axis + 1 :])))
    chunk_length = max(1, cells_per_chunk // trailing_cells)
    for prefix in np.ndindex(shape[:chunk_axis]):
        for start in range(0, shape[chunk_axis], chunk_length):
            yield (
                *(slice(index, index + 1) for index in prefix),
                slice(start, min(start + chunk_length, shape[chunk_axis])),
                *(slice(None) for _ in shape[chunk_axis + 1 :]),
            )


def _count_points_outside_cells(centers, bnds, is_lon):
    """Count points that lie outside the bounding box of their cell."""
    # Bounding-box test: center must lie within [min(bnds), max(bnds)] along
    # the last (vertex) axis. Works for intervals (N, 2) as well as for
    # polygonal / unstructured cell bounds (N, nvertices).
    bmin = bnds.min(axis=-1)
    bmax = bnds.max(axis=-1)

    # Longitude-wrap awareness: for longitude coordinates, cells crossing the
    # antimeridian have vertices on both sides of +/-180 and the raw bounding
    # box spans almost the whole globe. Unwrap vertices relative to the cell
    # center so the min/max reflects the true (wrapped) cell.
    if is_lon:
        wrap = np.asarray(bmax - bmin > 180.0)
        if wrap.any():
            c_col = np.asarray(centers)[..., None]
            delta = np.asarray(bnds) - c_col
            delta = np.where(delta > 180.0, delta - 360.0, delta)
            delta = np.where(delta < -180.0, delta + 360.0, delta)
            bnds = np.where(wrap[..., None], c_col + delta, np.asarray(bnds))
            bmin = bnds.min(axis=-1)
            bmax = bnds.max(axis=-1)

    # A float tolerance accommodates common float32/float64 precision
    # mismatches between coordinate and bounds variables.
    tol = 0.0
    if centers.dtype.kind == "f" or bnds.dtype.kind == "f":
        rtol = 1e-5
        atol = 1e-8
        tol = atol + rtol * np.maximum(np.abs(bmin), np.abs(bmax))
    bad = (centers < bmin - tol) | (centers > bmax + tol)
    if np.ma.is_masked(bad):
        bad = bad.filled(False)
    return int(np.asarray(bad, dtype=bool).sum())


def _enlarge_chunk_cache(boundary_variable):
    """Make sure one complete HDF5 chunk of the bounds fits in the cache.

    Otherwise a large compressed chunk is decompressed again for every block
    that is read from it. Returns the previous cache settings, or ``None`` if
    nothing was changed.
    """
    chunks = boundary_variable.chunking()
    if not isinstance(chunks, (list, tuple)):
        return None
    original_cache = boundary_variable.get_var_chunk_cache()
    trailing_chunks = int(np.ceil(boundary_variable.shape[-1] / chunks[-1]))
    cache_size = int(np.prod(chunks)) * boundary_variable.dtype.itemsize * trailing_chunks
    if original_cache[0] >= cache_size:
        return None
    boundary_variable.set_var_chunk_cache(cache_size, original_cache[1], original_cache[2])
    return original_cache


def check_cell_boundaries_interval(self, ds):
    """
    7.1 Cell Boundaries
    Recommendations: (1/2)
    The points specified by a coordinate or auxiliary coordinate variable
    should lie within, or on the boundary, of the cells specified by the
    associated boundary variable.

    Works for interval bounds ``(N, 2)`` and for polygonal / unstructured
    bounds ``(N, nvertices)`` via a bounding-box test (center lies within
    the [min, max] of the vertex set). Processes bounded blocks and emits
    a single Result per coordinate.

    (Patched by cc-plugin-wcrp: limits memory use for large grids.)
    """
    ret_val = []
    for variable_name, boundary_variable_name in cfutil.get_cell_boundary_map(
        ds,
    ).items():
        variable = ds.variables[variable_name]
        boundary_variable = ds.variables[boundary_variable_name]

        # ``bnds`` must add exactly one trailing vertex axis to ``centers``:
        # 1-D coord ``(N,)`` -> bounds ``(N, k)``; 2-D curvilinear coord
        # ``(J, I)`` -> bounds ``(J, I, k)``. Anything else is handled by
        # check_cell_boundaries and the recommendation check cannot
        # meaningfully proceed.
        if (
            boundary_variable.ndim != variable.ndim + 1
            or boundary_variable.shape[-1] < 2
            or boundary_variable.shape[:-1] != variable.shape
        ):
            continue

        is_lon = (
            str(getattr(variable, "units", "")).strip() == "degrees_east"
            or str(getattr(variable, "standard_name", "")).strip() == "longitude"
        )

        # As upstream, a coordinate whose data cannot be read is skipped here;
        # unreadable or malformed boundary variables are reported by other
        # checks. Only reading is guarded, so an error in the comparison
        # itself is still reported instead of silently passing.
        n_bad = 0
        readable = True
        original_cache = None
        try:
            try:
                original_cache = _enlarge_chunk_cache(boundary_variable)
            except Exception:
                logger.debug("Could not adjust chunk cache of '%s'", boundary_variable_name, exc_info=True)

            for selection in _cell_boundary_slices(variable.shape, boundary_variable.shape[-1]):
                try:
                    centers = np.ma.asanyarray(variable[selection])
                    bnds = np.ma.asanyarray(boundary_variable[selection + (slice(None),)])
                except Exception:
                    readable = False
                    break
                n_bad += _count_points_outside_cells(centers, bnds, is_lon)
        finally:
            if original_cache is not None:
                boundary_variable.set_var_chunk_cache(*original_cache)
        if not readable:
            continue

        reasoning = []
        if n_bad:
            reasoning.append(
                f"{n_bad} point(s) specified by the coordinate variable '{variable_name}' lie outside the bounding box of the associated boundary variable '{boundary_variable_name}'",
            )

        ret_val.append(
            Result(
                BaseCheck.MEDIUM,
                not n_bad,
                self.section_titles["7.1"],
                reasoning,
            ),
        )
    return ret_val


# --------------------------------------------------------------------------
# CF 4.3: check_dimensional_vertical_coordinate
# --------------------------------------------------------------------------
# Upstream treats every variable with standard_name depth/height/altitude as
# an auxiliary coordinate and, being vertical, requires a ``positive``
# attribute (HIGH). For data files whose main variable carries such a
# standard_name -- e.g. CMIP ``depthsl`` (standard_name = depth) -- this is a
# false positive: the variable is the data, not a coordinate, and must not
# have ``positive``. cc-plugin-wcrp already excludes the main variable from
# its own coordinate discovery (WCRPBaseCheck._initialize_coords_info).
#
# The cf checker has no project context (no plugin, no esgvoc), so the
# exception is decided from file metadata and cc_patches.toml only. The
# ``positive`` requirement is dropped only for a variable that is listed in
# cc_patches.toml, is the file's main variable (global attribute
# ``variable_id``), is in a file whose global attributes match the configured
# ones (e.g. mip_era = CMIP7), and is not used as a coordinate anywhere in the
# file. All other §4.3 requirements, including units, still apply to it.

# Set from cc_patches.toml: variable names the exception may apply to, and
# global attributes the file must have (all must match).
_MAIN_VARIABLES_NOT_VERTICAL_COORDINATES = frozenset()
_MAIN_VARIABLES_REQUIRED_GLOBAL_ATTRIBUTES = {}


def _configure_main_variables_not_vertical_coordinates(settings):
    """Take over the patch options; invalid options disable the exception."""
    global _MAIN_VARIABLES_NOT_VERTICAL_COORDINATES, _MAIN_VARIABLES_REQUIRED_GLOBAL_ATTRIBUTES
    variables = settings.get("variables", [])
    attributes = settings.get("global_attributes", {})
    valid = (
        isinstance(variables, list)
        and all(isinstance(name, str) for name in variables)
        and isinstance(attributes, dict)
        and all(isinstance(value, str) for value in attributes.values())
    )
    _MAIN_VARIABLES_NOT_VERTICAL_COORDINATES = frozenset(variables) if valid else frozenset()
    _MAIN_VARIABLES_REQUIRED_GLOBAL_ATTRIBUTES = dict(attributes) if valid else {}


def _is_main_variable_not_coordinate(ds, name):
    """Whether ``name`` is the file's main variable and not a coordinate.

    True only if the variable is listed in the configuration, equals the
    ``variable_id`` global attribute, the file has the configured global
    attributes, the variable is not a coordinate variable, has no ``axis`` or
    ``positive`` attribute and is not referenced by another variable's
    ``coordinates``, ``bounds``, ``climatology`` or ``formula_terms``.
    """
    if name not in _MAIN_VARIABLES_NOT_VERTICAL_COORDINATES:
        return False
    if str(getattr(ds, "variable_id", "")).strip() != name:
        return False
    for attribute, expected in _MAIN_VARIABLES_REQUIRED_GLOBAL_ATTRIBUTES.items():
        if str(getattr(ds, attribute, "")).strip() != expected:
            return False
    variable = ds.variables[name]
    if variable.dimensions == (name,):
        return False
    if hasattr(variable, "axis") or hasattr(variable, "positive"):
        return False
    for other in ds.variables.values():
        if other.name == name:
            continue
        references = []
        for attribute in ("coordinates", "bounds", "climatology"):
            value = getattr(other, attribute, None)
            if isinstance(value, str):
                references.extend(value.split())
        formula_terms = getattr(other, "formula_terms", None)
        if isinstance(formula_terms, str):
            references.extend(formula_terms.split()[1::2])
        if name in references:
            return False
    return True


def check_dimensional_vertical_coordinate(
    self,
    ds,
    dimless_vertical_coordinates=dimless_vertical_coordinates_1_6,
):
    """
    Check units for variables defining vertical position are valid under
    CF.

    CF §4.3.1 The units attribute for dimensional coordinates will be a string
    formatted as per the udunits.dat file.

    The acceptable units for vertical (depth or height) coordinate variables
    are:
    - units of pressure as listed in the file udunits.dat. For vertical axes
      the most commonly used of these include include bar, millibar,
      decibar, atmosphere (atm), pascal (Pa), and hPa.
    - units of length as listed in the file udunits.dat. For vertical axes
      the most commonly used of these include meter (metre, m), and
      kilometer (km).
    - other units listed in the file udunits.dat that may under certain
      circumstances reference vertical position such as units of density or
      temperature.

    Plural forms are also acceptable.

    (Patched by cc-plugin-wcrp: no ``positive`` requirement for a main data
    variable that is not a coordinate, see plugins/cc_patches.toml.)

    :param netCDF4.Dataset ds: An open netCDF dataset
    :rtype: list
    :return: List of results
    """
    ret_val = []
    z_variables = cfutil.get_z_variables(ds)
    for name in z_variables:
        variable = ds.variables[name]
        standard_name = getattr(variable, "standard_name", None)
        units = getattr(variable, "units", None)
        positive = getattr(variable, "positive", None)
        # Skip the variable if it's dimensionless
        if hasattr(variable, "formula_terms") or standard_name in dimless_vertical_coordinates:
            continue

        valid_vertical_coord = TestCtx(BaseCheck.HIGH, self.section_titles["4.3"])
        valid_vertical_coord.assert_true(
            isinstance(units, str) and units,
            f"§4.3.1 {name}'s units must be defined for vertical coordinates, there is no default",
        )

        if not cfutil.units_convertible("bar", units) and not _is_main_variable_not_coordinate(ds, name):
            valid_vertical_coord.assert_true(
                positive in ("up", "down"),
                f"{name}: vertical coordinates not defining pressure must include a positive attribute that is either 'up' or 'down'",
            )

        # _check_valid_standard_units, part of the Chapter 3 checks,
        # already verifies that this coordinate has valid units

        ret_val.append(valid_vertical_coord.to_result())

    return ret_val


# --------------------------------------------------------------------------
# Patch registry and application
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UpstreamPatch:
    """One upstream method replaced by a fixed implementation."""

    # Section name in cc_patches.toml.
    name: str
    owner: type
    attribute: str
    replacement: Callable
    # SHA-256 of ``inspect.getsource`` of each reviewed, broken upstream
    # version of the method. Only used by the tests, to notice upstream
    # changes within a configured version range.
    reviewed_upstream_sha256: tuple
    # Patch-specific settings in cc_patches.toml besides enabled/min_version/
    # max_version, and the function that takes them over when applied.
    option_keys: tuple = ()
    configure: Callable = None


PATCHES = (
    UpstreamPatch(
        name="geographic_region",
        owner=CF1_6Check,
        attribute="check_geographic_region",
        replacement=check_geographic_region,
        reviewed_upstream_sha256=(
            # 6.0.0 - 6.1.0
            "8293061c01d6be4a818614486b820aeb6e712033f8ff47dfcfae1bf8f43fca20",
        ),
    ),
    UpstreamPatch(
        name="cell_boundaries_interval",
        owner=CF1_7Check,
        attribute="check_cell_boundaries_interval",
        replacement=check_cell_boundaries_interval,
        reviewed_upstream_sha256=(
            # 6.1.0
            "4d4073be32a516db4197ecdfe960a144613baa5b591f3b73a29d37a9c0ad9de3",
        ),
    ),
    UpstreamPatch(
        name="main_variable_vertical_positive",
        owner=CF1_6Check,
        attribute="check_dimensional_vertical_coordinate",
        replacement=check_dimensional_vertical_coordinate,
        reviewed_upstream_sha256=(
            # 6.1.0
            "6353bb303998b31beec36f8b3abc1f0b402c17d08714ffa5010bd9222d0c8ca8",
        ),
        option_keys=("variables", "global_attributes"),
        configure=_configure_main_variables_not_vertical_coordinates,
    ),
)

# Upstream implementations that were replaced, by patch name.
ORIGINALS = {}

# Outcome per patch name, e.g. "applied", "disabled",
# "not applicable to compliance-checker 6.3.0".
STATUS = {}


def source_sha256(function):
    """Fingerprint of a function's source code."""
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def installed_compliance_checker_version():
    import compliance_checker

    return Version(compliance_checker.__version__)


def load_config(path=None):
    """Read the patch configuration; returns ``{patch name: settings}``."""
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return toml.load(f).get("patches", {})


def is_active(settings, cc_version):
    """Whether a patch with these settings applies to ``cc_version``.

    ``min_version`` and ``max_version`` are inclusive; either may be omitted.
    Development builds count as newer than the release they start from
    (6.1.1.dev3 > 6.1.0), so they are only covered by ranges that reach the
    next release.
    """
    if not settings.get("enabled", False):
        return False
    if "min_version" in settings and cc_version < Version(settings["min_version"]):
        return False
    if "max_version" in settings and cc_version > Version(settings["max_version"]):
        return False
    return True


def apply_patches(config=None, cc_version=None):
    """Apply the configured patches. Safe to call repeatedly.

    Never raises and never prints: if the configuration or the version cannot
    be read, no patch is applied and compliance-checker runs unchanged.
    """
    try:
        config = load_config() if config is None else config
        cc_version = installed_compliance_checker_version() if cc_version is None else cc_version
    except (OSError, ValueError, InvalidVersion, toml.TomlDecodeError):
        logger.debug("compliance-checker patches not applied", exc_info=True)
        config, cc_version = {}, None

    for patch in PATCHES:
        settings = config.get(patch.name, {})
        current = patch.owner.__dict__.get(patch.attribute)
        try:
            active = cc_version is not None and is_active(settings, cc_version)
        except InvalidVersion:
            active = False
        if active:
            if patch.configure is not None:
                patch.configure(settings)
            if current is not patch.replacement:
                ORIGINALS[patch.name] = current
                setattr(patch.owner, patch.attribute, patch.replacement)
            STATUS[patch.name] = "applied"
        else:
            # Undo an earlier application (only relevant when called again
            # with a different configuration, e.g. in tests).
            if current is patch.replacement and patch.name in ORIGINALS:
                setattr(patch.owner, patch.attribute, ORIGINALS.pop(patch.name))
            if not settings.get("enabled", False):
                STATUS[patch.name] = "disabled"
            else:
                STATUS[patch.name] = f"not applicable to compliance-checker {cc_version}"
        logger.debug("compliance-checker patch %s: %s", patch.name, STATUS[patch.name])
    return dict(STATUS)
