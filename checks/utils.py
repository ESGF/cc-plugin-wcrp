import os
import re
from datetime import timedelta

import cftime
import numpy as np
from compliance_checker.base import BaseCheck

from checks.time_checks.time_constants import FREQ_INC

try:
    from esgvoc.api.universe import find_terms_in_data_descriptor

    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False


# === Map severity constants to textual qualifiers ===

SEVERITY_WORDING_MAP = {
    BaseCheck.HIGH: "required",
    BaseCheck.MEDIUM: "recommended",
    BaseCheck.LOW: "suggested",
}
SEVERITY_WORDING_MAP_NOUN = {
    BaseCheck.HIGH: "requirement",
    BaseCheck.MEDIUM: "recommendation",
    BaseCheck.LOW: "suggestion",
}


def severity_word(severity, noun=False):
    """
    Return a human-readable qualifier ("required", "recommended", "suggested")
    for a given severity constant or string.
    """
    if isinstance(severity, str):
        s = severity.upper()[0]
        if s == "H":
            if noun:
                return SEVERITY_WORDING_MAP_NOUN[BaseCheck.HIGH]
            return SEVERITY_WORDING_MAP[BaseCheck.HIGH]
        elif s == "M":
            if noun:
                return SEVERITY_WORDING_MAP_NOUN[BaseCheck.MEDIUM]
            return SEVERITY_WORDING_MAP[BaseCheck.MEDIUM]
        elif s == "L":
            if noun:
                return SEVERITY_WORDING_MAP_NOUN[BaseCheck.LOW]
            return SEVERITY_WORDING_MAP[BaseCheck.LOW]
    if noun:
        return SEVERITY_WORDING_MAP_NOUN.get(severity, "recommendation")
    return SEVERITY_WORDING_MAP.get(severity, "recommended")


# === Mapping CMOR<-->python datatypes
dtypesdict = {
    "integer": np.int32,
    "long": np.int64,
    "real": np.float32,
    "double": np.float64,
    "character": "S",
}
_dtypesdict = {
    **dtypesdict,
    "character": str,
}


# === cc_plugin_cc6 utils and constants ===


def convert_posix_to_python(posix_regex):
    """
    Convert common POSIX regular expressions to Python regular expressions.

    Args:
        posix_regex (str): The POSIX regular expression to convert.

    Returns:
        str: The converted Python regular expression.

    Raises:
        ValueError: If the input is not a string or contains invalid POSIX character classes.
    """
    if not isinstance(posix_regex, str):
        raise ValueError("Input must be a string")

    # Dictionary of POSIX to Python character class conversions
    posix_to_python_classes = {
        r"[[:alnum:]]": r"[a-zA-Z0-9]",
        r"[[:alpha:]]": r"[a-zA-Z]",
        r"[[:digit:]]": r"\d",
        r"[[:xdigit:]]": r"[0-9a-fA-F]",
        r"[[:lower:]]": r"[a-z]",
        r"[[:upper:]]": r"[A-Z]",
        r"[[:blank:]]": r"[ \t]",
        r"[[:space:]]": r"\s",
        r"[[:punct:]]": r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]',
        r"[[:word:]]": r"\w",
    }

    # Replace POSIX character classes with Python equivalents
    for posix_class, python_class in posix_to_python_classes.items():
        posix_regex = posix_regex.replace(posix_class, python_class)

    # Replace POSIX quantifiers with Python equivalents
    posix_regex = posix_regex.replace(r"\{", "{").replace(r"\}", "}")

    return posix_regex


def match_pattern_or_string(pattern, target):
    """
    Compare a regex pattern or a string with the target string.

    Args:
        pattern (str): The regex pattern or string to compare.
        target (str): The string to compare against.

    Returns:
        bool: True if the target matches the regex pattern or is equal to the string.
    """
    return bool(
        re.fullmatch(convert_posix_to_python(pattern), target, flags=re.ASCII)
    ) or (
        pattern == target
        and convert_posix_to_python(target) == target
        and ".*" not in target
    )


def to_str(val):
    """
    Decode byte strings to utf-8 if possible and leave other typed input unchanged.
    """
    if isinstance(val, (bytes, np.bytes_)):
        try:
            return val.decode("utf-8")
        except UnicodeDecodeError:
            return val
    return str(val)


def sanitize(obj):
    """
    Make sure all values are json-serializable.
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def printtimedelta(d):
    """Return timedelta (s) as either min, hours, days, whatever fits best."""
    if d > 86000:
        return f"{d/86400.} days"
    if d > 3500:
        return f"{d/3600.} hours"
    if d > 50:
        return f"{d/60.} minutes"
    else:
        return f"{d} seconds"


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


# Only base labels whose increments are unique can be inferred. Variants such
# as monPt and monClim intentionally remain ambiguous.
_INFERABLE_FREQUENCIES = (
    "subhr",
    "1hr",
    "3hr",
    "6hr",
    "day",
    "mon",
    "sem",
    "yr",
    "dec",
    "cen",
)
_FIXED_INCREMENT_TOLERANCE_SECONDS = 1.0
_CALENDAR_COORDINATE_TOLERANCE_SECONDS = timedelta(days=2).total_seconds()


def _read_unmasked_values(variable):
    """Read a variable as an ndarray, rejecting missing or unreadable values."""
    try:
        values = np.ma.asarray(variable[:])
    except Exception:
        return None
    if values.size == 0 or np.ma.getmaskarray(values).any():
        return None
    return np.asarray(values)


def _decoded_datetimes(values, units, calendar):
    """Decode numeric time values, returning None for invalid metadata/data."""
    if not isinstance(units, str) or not units.strip():
        return None
    try:
        decoded = cftime.num2date(
            values,
            units=units,
            calendar=calendar or "standard",
            only_use_cftime_datetimes=True,
        )
    except Exception:
        return None
    return np.asarray(decoded, dtype=object)


def _coordinate_date_pairs(time, units, calendar):
    """Return consecutive decoded time-coordinate pairs."""
    values = _read_unmasked_values(time)
    if values is None or values.ndim > 1:
        return []
    values = values.reshape(-1)
    if values.size < 2:
        return []
    dates = _decoded_datetimes(values, units, calendar)
    if dates is None:
        return []
    pairs = list(zip(dates[:-1], dates[1:]))
    for start, end in pairs:
        try:
            increasing = (end - start).total_seconds() > 0
        except Exception:
            return []
        if not increasing:
            return []
    return pairs


def _bounds_date_pairs(time, time_bounds, units, calendar):
    """Return decoded start/end pairs from two-dimensional time bounds."""
    if time_bounds is None or getattr(time_bounds, "ndim", None) != 2:
        return []
    values = _read_unmasked_values(time_bounds)
    if values is None:
        return []

    time_dimensions = getattr(time, "dimensions", ())
    bounds_dimensions = getattr(time_bounds, "dimensions", ())
    time_dimension = time_dimensions[0] if time_dimensions else None
    if time_dimension in bounds_dimensions:
        time_axis = bounds_dimensions.index(time_dimension)
    elif values.shape[0] == getattr(time, "size", None):
        time_axis = 0
    elif values.shape[1] == getattr(time, "size", None):
        time_axis = 1
    else:
        return []

    values = np.moveaxis(values, time_axis, 0)
    if values.shape[0] != getattr(time, "size", values.shape[0]):
        return []
    if values.shape[1] < 2:
        return []

    starts = _decoded_datetimes(values[:, 0], units, calendar)
    ends = _decoded_datetimes(values[:, -1], units, calendar)
    if starts is None or ends is None:
        return []

    pairs = list(zip(starts, ends))
    for start, end in pairs:
        try:
            increasing = (end - start).total_seconds() > 0
        except Exception:
            return []
        if not increasing:
            return []
    return pairs


def add_time_increment(date, value, unit, calendar):
    """Add a fixed or calendar-dependent FREQ_INC increment to a CF date."""
    value = int(value)
    unit = str(unit)
    calendar = calendar or "standard"
    if unit in ("seconds", "minutes", "hours", "days"):
        return date + timedelta(**{unit: value})

    year = int(date.year)
    month = int(date.month)
    day = int(date.day)
    if unit == "years":
        year += value
    elif unit == "months":
        total_months = month + value
        year_delta, zero_based_month = divmod(total_months - 1, 12)
        year += year_delta
        month = zero_based_month + 1
    else:
        raise ValueError(f"Unsupported time increment unit: {unit}")

    date_components = (
        year,
        month,
        day,
        date.hour,
        date.minute,
        date.second,
        getattr(date, "microsecond", 0),
    )
    for candidate_day in range(day, 0, -1):
        try:
            return cftime.datetime(
                date_components[0],
                date_components[1],
                candidate_day,
                *date_components[3:],
                calendar=calendar,
            )
        except ValueError:
            continue
    raise ValueError(f"Could not add {value} {unit} to {date}")


def _matches_subhourly_frequency(pairs, calendar, increment):
    """Check for a regular step that divides one hour and is within the limit."""
    value, unit = increment
    observed_intervals = []
    for start, end in pairs:
        try:
            observed = (end - start).total_seconds()
            maximum_end = add_time_increment(start, value, unit, calendar)
            maximum = (maximum_end - start).total_seconds()
        except Exception:
            return False
        if (
            not np.isfinite(observed)
            or not np.isfinite(maximum)
            or observed <= 0
            or observed > maximum + _FIXED_INCREMENT_TOLERANCE_SECONDS
        ):
            return False
        steps_per_hour = round(timedelta(hours=1).total_seconds() / observed)
        if steps_per_hour < 1 or (
            abs(steps_per_hour * observed - timedelta(hours=1).total_seconds())
            > _FIXED_INCREMENT_TOLERANCE_SECONDS
        ):
            return False
        observed_intervals.append(observed)

    return (
        max(observed_intervals) - min(observed_intervals)
        <= _FIXED_INCREMENT_TOLERANCE_SECONDS
    )


def _frequency_for_date_pairs(pairs, calendar, *, coordinate_values):
    """Return the unique FREQ_INC base label matching all decoded pairs."""
    if not pairs:
        return "unknown"
    matches = []
    for frequency in _INFERABLE_FREQUENCIES:
        increment = FREQ_INC.get(("None", frequency))
        if not increment:
            continue
        value, unit = increment
        if frequency == "subhr":
            if _matches_subhourly_frequency(pairs, calendar, increment):
                matches.append(frequency)
            continue
        tolerance = _FIXED_INCREMENT_TOLERANCE_SECONDS
        if coordinate_values and unit in ("months", "years"):
            # Monthly/annual means can shift by half the change in adjacent
            # cell widths while still representing a regular calendar step.
            tolerance = _CALENDAR_COORDINATE_TOLERANCE_SECONDS
        matched = True
        for start, end in pairs:
            try:
                expected = add_time_increment(start, value, unit, calendar)
                difference = abs((end - expected).total_seconds())
            except Exception:
                matched = False
                break
            if not np.isfinite(difference) or difference > tolerance:
                matched = False
                break
        if matched:
            matches.append(frequency)
    return matches[0] if len(matches) == 1 else "unknown"


def infer_frequency(time, time_bounds=None, units=None, calendar=None):
    """Infer a WCRP base frequency from a numeric CF time coordinate.

    Time-cell widths and consecutive coordinate intervals are evaluated
    independently. If both are available, they must resolve to the same
    frequency. A single time value can therefore only be inferred when valid
    bounds are available. Missing, irregular, decreasing, contradictory, or
    otherwise unsupported input returns ``"unknown"`` rather than raising.

    Specialized labels such as ``monPt``, ``monClim``, and ``1hrCM`` are not
    inferred because interval lengths alone cannot identify their semantics.
    ``subhr`` denotes any regular positive interval up to the maximum stored
    in its generic ``FREQ_INC`` entry that divides one hour exactly.

    Parameters
    ----------
    time : netCDF4.Variable
        Numeric CF time coordinate.
    time_bounds : netCDF4.Variable, optional
        Associated two-dimensional time bounds variable.
    units : str, optional
        CF time units. Defaults to the time variable's ``units`` attribute.
    calendar : str, optional
        CF calendar. Defaults to the time variable's ``calendar`` attribute or
        ``"standard"``.

    Returns
    -------
    str
        One of the supported base-frequency labels, or ``"unknown"``.
    """
    if time is None:
        return "unknown"
    units = units if units is not None else getattr(time, "units", None)
    calendar = (
        calendar if calendar is not None else getattr(time, "calendar", "standard")
    )

    coordinate_pairs = _coordinate_date_pairs(time, units, calendar)
    bounds_pairs = _bounds_date_pairs(
        time,
        time_bounds,
        units,
        calendar,
    )
    coordinate_frequency = _frequency_for_date_pairs(
        coordinate_pairs,
        calendar,
        coordinate_values=True,
    )
    bounds_frequency = _frequency_for_date_pairs(
        bounds_pairs,
        calendar,
        coordinate_values=False,
    )

    if coordinate_pairs and bounds_pairs:
        if coordinate_frequency == bounds_frequency:
            return coordinate_frequency
        return "unknown"
    if bounds_pairs:
        return bounds_frequency
    if coordinate_pairs:
        return coordinate_frequency
    return "unknown"


# Nominal durations retained for the legacy CORDEX file-chunking check. They
# are not frequency-inference tolerances; inference uses FREQ_INC above.
deltdic = {
    "subhr": timedelta(minutes=10).total_seconds(),
    "1hr": timedelta(hours=1).total_seconds(),
    "3hr": timedelta(hours=3).total_seconds(),
    "6hr": timedelta(hours=6).total_seconds(),
    "day": timedelta(days=1).total_seconds(),
    "mon": timedelta(days=31).total_seconds(),
    "yr": timedelta(days=360).total_seconds(),
    "dec": timedelta(days=3600).total_seconds(),
    "cen": timedelta(days=36000).total_seconds(),
}
# CMIP-style frequencies for "time: point" share these nominal durations.
for l_freq in ("subhr", "1hr", "3hr", "6hr", "day", "mon", "yr"):
    deltdic[l_freq + "Pt"] = deltdic[l_freq]


def retrieve(url, fname, path, force=False):
    """
    Retrieve a file from a given URL and save it to a local path.
    """
    import pooch

    # Create the full path to the file
    full_path = os.path.join(os.path.expanduser(path), fname)
    # Check if the file exists locally and delete if redownload is forced
    if os.path.isfile(full_path) and force:
        print(f"Removing existing file '{full_path}'")
        os.remove(full_path)

    filename = pooch.retrieve(
        url=url,
        fname=fname,
        known_hash=None,
        path=path,
    )
    return filename


def _compare_CV_element(el, val):
    """Compares value of a CV entry to a given value."""
    # ########################################################################################
    # 5-6 Types of CV entries ('*' is the element that is the value for comparison):
    # 0 # value
    # 1 # key -> *list of values
    # 2 # key -> *list of length 1 (regex)
    # 3 # key -> *dict key -> value
    # 4 # key -> *dict key -> dict key -> *value
    # 5 # key -> *dict key -> dict key -> *list of values
    # CMIP6 only and not considered here:
    # 6 # key (source_id) -> *dict key -> dict key (license_info) -> dict key (id, license) -> value
    # ########################################################################################
    # 0 (2nd+ level comparison) #
    if isinstance(el, str):
        return (match_pattern_or_string(el, str(val)), [], [el])
    # 1 and 2 #
    elif isinstance(el, list):
        return (any([match_pattern_or_string(eli, str(val)) for eli in el]), [], el)
    # 3 to 6 #
    elif isinstance(el, dict):
        if val in el.keys():
            # 3 #
            if isinstance(el[val], str):
                return True, [], []
            # 4 to 6 #
            elif isinstance(el[val], dict):
                return True, list(el[val].keys()), []
            else:
                raise ValueError(
                    f"Unknown CV structure for element: {el} and value {val}."
                )
        else:
            return False, [], list(el.keys())
    # (Yet) unknown
    else:
        raise ValueError(f"Unknown CV structure for element: {el} and value: {val}.")


def _compare_CV(CheckerObject, dic2comp, errmsg_prefix):
    """Compares dictionary of key-val pairs with CV."""
    checked = {key: False for key in dic2comp.keys()}
    messages = []
    for attr in dic2comp.keys():
        if attr in CheckerObject.CV:
            errmsg = f"""{errmsg_prefix}'{attr}' does not comply with the CV: '{dic2comp[attr] if dic2comp[attr] else 'unset'}'."""
            checked[attr] = True
            test, attrs_lvl2, allowed_vals = _compare_CV_element(
                CheckerObject.CV[attr], dic2comp[attr]
            )
            # If comparison fails
            if not test:
                if len(allowed_vals) == 1:
                    errmsg += f""" Expected value/pattern: '{allowed_vals[0]}'."""
                elif len(allowed_vals) > 3:
                    errmsg += f""" Allowed values: {", ".join(f"'{av}'" for av in allowed_vals[0:3])}, ..."""
                elif len(allowed_vals) > 1:
                    errmsg += f""" Allowed values: {", ".join(f"'{av}'" for av in allowed_vals)}."""
                messages.append(errmsg)
            # If comparison could not be processed completely, as the CV element is another dictionary
            else:
                for attr_lvl2 in attrs_lvl2:
                    if attr_lvl2 in dic2comp.keys():
                        errmsg_lvl2 = f"""{errmsg_prefix}'{attr_lvl2}' does not comply with the CV: '{dic2comp[attr_lvl2] if dic2comp[attr_lvl2] else 'unset'}'."""
                        checked[attr_lvl2] = True
                        try:
                            test, attrs_lvl3, allowed_vals = _compare_CV_element(
                                CheckerObject.CV[attr][dic2comp[attr]][attr_lvl2],
                                dic2comp[attr_lvl2],
                            )
                        except ValueError:
                            raise ValueError(
                                f"Unknown CV structure for element {attr} -> {CheckerObject.CV[attr][dic2comp[attr]][attr_lvl2]} / {attr_lvl2} -> {dic2comp[attr_lvl2]}."
                            )
                        if not test:
                            if len(allowed_vals) == 1:
                                errmsg_lvl2 += (
                                    f""" Expected value/pattern: '{allowed_vals[0]}'."""
                                )
                            elif len(allowed_vals) > 3:
                                errmsg_lvl2 += f""" Allowed values: {", ".join(f"'{av}'" for av in allowed_vals[0:3])}, ..."""
                            elif len(allowed_vals) > 1:
                                errmsg_lvl2 += f""" Allowed values: {", ".join(f"'{av}'" for av in allowed_vals)}."""
                            messages.append(errmsg_lvl2)
                        else:
                            if len(attrs_lvl3) > 0:
                                raise ValueError(
                                    f"Unknown CV structure for element {attr} -> {dic2comp[attr]} -> {attr_lvl2}."
                                )
    return checked, messages


# === Further utils ===


def _find_drs_directory_and_filename(filepath, project_id="cmip6"):
    """
    Intelligently finds the DRS directory path by locating the project_id.
    """
    try:
        path_parts = os.path.dirname(filepath).lower().split(os.sep)
        original_parts = os.path.dirname(filepath).split(os.sep)
        # Using last occurence of <project_id> as start index of DRS path
        start_index = len(path_parts) - 1 - path_parts[::-1].index(project_id.lower())
        drs_directory = os.path.join(*original_parts[start_index:])
        filename = os.path.basename(filepath)
        return drs_directory, filename, None
    except (ValueError, TypeError):
        return (
            None,
            None,
            f"DRS project root '{project_id}' not found in the file path '{filepath}'.",
        )


def _parse_filename_components(filename, filename_template_keys):
    """
    Parses filename to extract its components.
    Returns a dictionary of the components or an error message.
    """
    # Remove the .nc extension and split by the underscore separator
    filename_parts = filename.replace(".nc", "").split("_")

    # If filename has fewer parts than expected, try to handle missing 'time_range'
    if len(filename_parts) == len(filename_template_keys):
        filename_facets = dict(zip(filename_template_keys, filename_parts))
    elif len(filename_parts) == len(filename_template_keys) - 1:
        # Set 'time_range' to 'UNSET' if missing
        filename_facets = {}
        part_idx = 0
        for key in filename_template_keys:
            if key == "time_range":
                filename_facets[key] = "UNSET"
            else:
                filename_facets[key] = filename_parts[part_idx]
                part_idx += 1
    else:
        return None, (
            f"Filename '{filename}' does not have the expected {len(filename_template_keys)} "
            f"components (or {len(filename_template_keys)-1} for time invariant variables)."
        )

    return filename_facets, None


def _get_drs_facets(filepath, project_id, dir_template_keys, filename_template_keys):
    """
    Parses a full filepath to extract DRS components from both the directory path and the filename.
    """
    try:
        drs_directory, filename, error_msg = _find_drs_directory_and_filename(
            filepath, project_id
        )
        if error_msg:
            return None, None, error_msg

        # --- Directory handling ---
        drs_path_parts = drs_directory.split(os.sep)
        if len(drs_path_parts) != len(dir_template_keys):
            return (
                None,
                None,
                (
                    f"Directory path does not match expected DRS depth. "
                    f"Found {len(drs_path_parts)}, expected {len(dir_template_keys)}."
                ),
            )
        dir_facets = dict(zip(dir_template_keys, drs_path_parts))

        # --- Filename handling ---
        filename_facets, error_msg = _parse_filename_components(
            filename, filename_template_keys
        )
        if error_msg:
            return None, None, error_msg

        return dir_facets, filename_facets, None

    except Exception as e:
        return None, None, f"An unexpected error occurred during DRS parsing: {e}"


###################################
# Coordinate Utility Functions
###################################


def convert_lon_360(lon):
    """Convert longitude to [0, 360)."""
    lon = np.asarray(lon)
    return lon % 360.0


def convert_lon_180(lon):
    """Convert longitude to [-180, 180)."""
    lon = np.asarray(lon)
    return ((lon + 180.0) % 360.0) - 180.0


def crosses_zero_meridian(lon, intv=5.0):
    """
    Check if longitude crosses 0-meridian.

    Args:
        lon (numpy.ndarray): Array of longitudes.
        intv (float, optional): Requiring longitude in interval [-intv, 0] and [0,intv]
                                to be classified as crossing 0-meridian. Default is 5.

    Returns:
        bool: True if longitude crosses 0-meridian.
    """
    lon180 = convert_lon_180(lon)
    return bool(
        np.any((lon180 > -intv) & (lon180 < 0))
        and np.any((lon180 > 0) & (lon180 < intv))
    )


def crosses_anti_meridian(lon, intv=5.0):
    """
    Check if longitude crosses anti-meridian.

    Args:
        lon (numpy.ndarray): Array of longitudes.
        intv (float, optional): Requiring longitude in interval [-intv, 0] and [0,intv]
                                to be classified as crossing 0-meridian. Default is 5.

    Returns:
        bool: True if longitude crosses anti-meridian.
    """
    lon360 = convert_lon_360(lon)
    return bool(
        np.any((lon360 > 180 - intv) & (lon360 < 180))
        and np.any((lon360 > 180) & (lon360 < 180 + intv))
    )


def resolve_member_id(ds):
    """
    Compute the CMIP DRS `member_id` from global attributes.

    Per the CMIP6 / CMIP6Plus DRS specification, the member token used in
    both the filename and the directory path is:

        member_id = <sub_experiment_id>-<variant_label>   if sub_experiment_id
                                                            is set and not "none"
        member_id = <variant_label>                       otherwise

    The DRS templates label this position "variant_label", but a file that
    belongs to a sub-experiment carries the composite token there
    (e.g. "s1960-r1i1p1f2", "f2023-r2i1p1f3").

    Parameters
    ----------
    ds : netCDF4.Dataset

    Returns
    -------
    str
        The expected member_id token. Falls back to variant_label alone if
        sub_experiment_id is absent, empty, or "none".
    """
    variant = ""
    if "variant_label" in ds.ncattrs():
        variant = str(ds.getncattr("variant_label"))

    sub_exp = ""
    if "sub_experiment_id" in ds.ncattrs():
        sub_exp = str(ds.getncattr("sub_experiment_id")).strip()

    if sub_exp and sub_exp.lower() != "none":
        return f"{sub_exp}-{variant}"
    return variant


# =============================================================================
# Experiment term resolution + parent/sub presence helpers (esgvoc-backed)
# =============================================================================
# Grounded on ACTUAL esgvoc output:
#
#   CMIP6 / cmip6plus (ExperimentLegacy):
#     parent_experiment_id -> list[str]  (attribute ABSENT means "no parent")
#     sub_experiment_id    -> list[str]  (['none'] means "no sub-experiment")
#     e.g. historical: parent_experiment_id=['picontrol'], sub_experiment_id=['none']
#
#   CMIP7 (ExperimentCMIP7):
#     parent_experiment -> nested Experiment object (or absent) == has parent
#     (there is NO sub_experiment_id in CMIP7)
#
# Term ids are stored lowercase internally while NetCDF files carry DRS casing
# ('piControl'). get_term_in_collection(term_id='piControl') can return an
# EMPTY shell term in CMIP7 (casing mismatch), hence the drs_name fallback.

try:
    import esgvoc.api as _voc_api

    _ESG_VOCAB_PROJECT_API = True
except ImportError:
    _voc_api = None
    _ESG_VOCAB_PROJECT_API = False

# experiment collection name differs per project
_EXPERIMENT_COLLECTION = {
    "cmip6": "experiment_id",
    "cmip6plus": "experiment_id",
    "cmip7": "experiment",
}

# tokens that mean "no value" inside a CV list or a file attribute
NO_VALUE_TOKENS = {"none", "no parent", "no_parent", ""}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _lower_str_list(values):
    return [str(v).strip().lower() for v in _as_list(values) if v is not None]


def _experiment_collection(project_id):
    return _EXPERIMENT_COLLECTION.get(str(project_id).strip().lower(), "experiment_id")


def _term_is_empty(term):
    """A resolved term is unusable if it exposes neither id nor drs_name
    (the CMIP7 casing-mismatch case returns such a shell object)."""
    if term is None:
        return True
    return not (getattr(term, "id", None) or getattr(term, "drs_name", None))


def resolve_experiment_term(ds, project_id):
    """
    Resolve the CV experiment term for the file's experiment_id, robust to DRS
    casing (direct -> lowercase -> exact drs_name match). Returns term or None.
    """
    if not _ESG_VOCAB_PROJECT_API:
        return None
    if "experiment_id" not in ds.ncattrs():
        return None

    exp_id = str(ds.getncattr("experiment_id")).strip()
    collection_id = _experiment_collection(project_id)

    term = _voc_api.get_term_in_collection(
        project_id=project_id, collection_id=collection_id, term_id=exp_id
    )
    if not _term_is_empty(term):
        return term

    term = _voc_api.get_term_in_collection(
        project_id=project_id, collection_id=collection_id, term_id=exp_id.lower()
    )
    if not _term_is_empty(term):
        return term

    try:
        candidates = _voc_api.find_terms_in_collection(
            project_id=project_id,
            collection_id=collection_id,
            expression=exp_id,
            selected_term_fields=["id", "drs_name"],
        )
    except Exception:
        candidates = []

    for item in candidates or []:
        if str(getattr(item, "drs_name", "")).strip() == exp_id:
            resolved_id = str(getattr(item, "id", "")).strip()
            if resolved_id:
                term = _voc_api.get_term_in_collection(
                    project_id=project_id,
                    collection_id=collection_id,
                    term_id=resolved_id,
                )
                if not _term_is_empty(term):
                    return term
    return None


def has_parent_experiment(term, project_id):
    """Does the CV declare a parent experiment for this experiment?"""
    if term is None:
        return False
    if str(project_id).strip().lower() == "cmip7":
        return getattr(term, "parent_experiment", None) is not None
    parents = getattr(term, "parent_experiment_id", None)
    norm = [p for p in _lower_str_list(parents) if p not in NO_VALUE_TOKENS]
    return bool(norm)


def has_sub_experiment(term, project_id):
    """Does the CV declare a sub-experiment? (CMIP6/plus only; CMIP7 -> False)"""
    if term is None:
        return False
    if str(project_id).strip().lower() == "cmip7":
        return False
    subs = getattr(term, "sub_experiment_id", None)
    norm = [s for s in _lower_str_list(subs) if s not in NO_VALUE_TOKENS]
    return bool(norm)


def has_parent_experiment_for_ds(ds, project_id):
    """Convenience: resolve term from ds then evaluate has_parent_experiment."""
    return has_parent_experiment(resolve_experiment_term(ds, project_id), project_id)


def has_sub_experiment_for_ds(ds, project_id):
    """Convenience: resolve term from ds then evaluate has_sub_experiment."""
    return has_sub_experiment(resolve_experiment_term(ds, project_id), project_id)


def has_parent_activity(term, project_id):
    """
    Does the CV declare a parent ACTIVITY for this experiment?

      CMIP7: parent_activity is a nested Activity object (or None).
      CMIP6/plus: parent_activity_id is its own list field (separate from
      parent_experiment_id in the model) -- read independently rather than
      reusing has_parent_experiment(), in case the two ever diverge in the CV data.
    """
    if term is None:
        return False
    if str(project_id).strip().lower() == "cmip7":
        return getattr(term, "parent_activity", None) is not None
    parents = getattr(term, "parent_activity_id", None)
    norm = [p for p in _lower_str_list(parents) if p not in NO_VALUE_TOKENS]
    return bool(norm)


def has_parent_activity_for_ds(ds, project_id):
    """Convenience: resolve term from ds then evaluate has_parent_activity."""
    return has_parent_activity(resolve_experiment_term(ds, project_id), project_id)


# parent_mip_era: only modeled in CV for CMIP7 (ExperimentCMIP7.parent_mip_era).
# The CMIP6/plus legacy model has NO such field -- for those projects there is
# no CV data to read, so this keyword must not be used in cmip6/cmip6plus TOMLs
# (no inference, no fallback to has_parent: that would not be a CV-derived value).
def has_parent_mip_era(term, project_id):
    if term is None:
        return False
    if str(project_id).strip().lower() == "cmip7":
        return getattr(term, "parent_mip_era", None) is not None
    # CMIP6/plus: no CV field for this -> cannot resolve, fail safe to required.
    return True


def has_parent_mip_era_for_ds(ds, project_id):
    return has_parent_mip_era(resolve_experiment_term(ds, project_id), project_id)


# has_parent_variant_label REMOVED: no CV field exists for this attribute in either CMIP6 or CMIP7 (verified against esgvoc output and the ExperimentLegacy/ExperimentCMIP7 models). No dynamic rule without a real CV field to read.
