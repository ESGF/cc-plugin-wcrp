import os
import re
from datetime import timedelta

import numpy as np
from compliance_checker.base import BaseCheck

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


# Definition of maximum deviations from the given frequency
deltdic = {}
deltdic["monmax"] = timedelta(days=31.5).total_seconds()
deltdic["monmin"] = timedelta(days=27.5).total_seconds()
deltdic["mon"] = timedelta(days=31).total_seconds()
deltdic["daymax"] = timedelta(days=1.1).total_seconds()
deltdic["daymin"] = timedelta(days=0.9).total_seconds()
deltdic["day"] = timedelta(days=1).total_seconds()
deltdic["1hrmin"] = timedelta(hours=0.9).total_seconds()
deltdic["1hrmax"] = timedelta(hours=1.1).total_seconds()
deltdic["1hr"] = timedelta(hours=1).total_seconds()
deltdic["3hrmin"] = timedelta(hours=2.9).total_seconds()
deltdic["3hrmax"] = timedelta(hours=3.1).total_seconds()
deltdic["3hr"] = timedelta(hours=3).total_seconds()
deltdic["6hrmin"] = timedelta(hours=5.9).total_seconds()
deltdic["6hrmax"] = timedelta(hours=6.1).total_seconds()
deltdic["6hr"] = timedelta(hours=6).total_seconds()
deltdic["yrmax"] = timedelta(days=366.1).total_seconds()
deltdic["yrmin"] = timedelta(days=359.9).total_seconds()
deltdic["yr"] = timedelta(days=360).total_seconds()
deltdic["subhr"] = timedelta(seconds=600).total_seconds()
deltdic["subhrmax"] = timedelta(seconds=601).total_seconds()
deltdic["subhrmin"] = timedelta(seconds=599).total_seconds()
deltdic["dec"] = timedelta(days=3600).total_seconds()
deltdic["decmax"] = timedelta(days=3599.99).total_seconds()
deltdic["decmin"] = timedelta(days=3660.01).total_seconds()
deltdic["cen"] = timedelta(days=36000).total_seconds()
deltdic["cenmax"] = timedelta(days=35999.99).total_seconds()
deltdic["cenmin"] = timedelta(days=36600.01).total_seconds()
# CMIP-style frequencies for "time: point":
for l_freq in ["subhr", "1hr", "3hr", "6hr", "day", "mon", "yr"]:
    deltdic[l_freq + "Pt"] = deltdic[l_freq]
    deltdic[l_freq + "Ptmax"] = deltdic[l_freq + "max"]
    deltdic[l_freq + "Ptmin"] = deltdic[l_freq + "min"]


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
