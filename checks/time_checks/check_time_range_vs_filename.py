#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[TIME003] Check that the dataset time axis matches the time range declared in the
filename.

Time-range token detection is delegated to esgvoc's DRS validator rather than a
hard-coded date-format regex. esgvoc knows, per project and per frequency, which
time-range layouts are valid (YYYY, YYYYMM, YYYYMMDD, YYYYMMDDHH, YYYYMMDDHHMM,
YYYYMMDDHHMMSS).

If esgvoc (or the project vocabulary) is unavailable, the check falls back to a
relaxed structural parse that accepts any even-length date token from 4 to 14
digits.
"""

import os
import re
from datetime import timedelta

from compliance_checker.base import BaseCheck, TestCtx
from netCDF4 import num2date

# -----------------------------------------------------------------------------
# Optional esgvoc DRS validator
# -----------------------------------------------------------------------------
try:
    from esgvoc.apps.drs.validator import DrsValidator

    _ESGVOC_AVAILABLE = True
except Exception:
    DrsValidator = None
    _ESGVOC_AVAILABLE = False

# Cache of DrsValidator instances per project_id (build is relatively expensive)
_VALIDATOR_CACHE: dict = {}

# Relaxed fallback: a time-range token is two even-length digit runs (4..14)
# separated by a hyphen. esgvoc is the authority; this is only a safety net.
_FALLBACK_TOKEN_RE = re.compile(r"^(?P<start>\d{4,14})-(?P<end>\d{4,14})$")

# Frequency -> (<label format>, <token length>, <tuple length>)
# tuple layout: (Y, M, D, H, Min, S)
_DEFAULT_TIME_LABEL_PRECISION = {
    "yr": "yyyy",
    "dec": "yyyy",
    "mon": "yyyyMM",
    "day": "yyyyMMdd",
    "6hr": "yyyyMMddhhmm",
    "3hr": "yyyyMMddhhmm",
    "1hr": "yyyyMMddhhmm",
    "subhr": "yyyyMMddhhmmss",
}

_LABEL_TO_PRECISION = {
    "yyyy": (4, 1),
    "yyyyMM": (6, 2),
    "yyyyMMdd": (8, 3),
    "yyyyMMddhh": (10, 4),
    "yyyyMMddhhmm": (12, 5),
    "yyyyMMddhhmmss": (14, 6),
}

_FALLBACK_PRECISION_KEY = "default"


def _project_id_from_ds(ds):
    """Resolve esgvoc project id ('cmip6'/'cmip7'/...) from global attributes."""
    try:
        mip_era = str(ds.getncattr("mip_era")).strip().lower()
        if mip_era in ("cmip6", "cmip7"):
            return mip_era
    except AttributeError:
        pass
    try:
        proj = str(ds.getncattr("project_id")).strip().lower()
        if proj:
            return proj
    except AttributeError:
        pass
    return None


def _get_validator(project_id):
    """Return a cached DrsValidator for the project, or None if unavailable."""
    if not _ESGVOC_AVAILABLE or not project_id:
        return None
    if project_id in _VALIDATOR_CACHE:
        return _VALIDATOR_CACHE[project_id]
    try:
        validator = DrsValidator(project_id=project_id)
    except Exception:
        validator = None
    _VALIDATOR_CACHE[project_id] = validator
    return validator


def _esgvoc_filename_ok(ds, filename_no_ext):
    """
    Ask esgvoc whether the filename is a structurally valid DRS expression.

    Returns True (valid), False (structural error), or None (esgvoc unavailable).
    """
    project_id = _project_id_from_ds(ds)
    validator = _get_validator(project_id)
    if validator is None:
        return None
    try:
        report = validator.validate_file_name(filename_no_ext + ".nc")
    except Exception:
        return None
    errors = getattr(report, "errors", None)
    return not errors


def _extract_time_range_token(filename):
    """
    Return (start_str, end_str) of the time-range token, or (None, None).
    The token is the last underscore-separated segment of the stem.
    """
    stem = filename[:-3] if filename.endswith(".nc") else filename
    last_token = stem.split("_")[-1]
    match = _FALLBACK_TOKEN_RE.match(last_token)
    if not match:
        return None, None
    return match.group("start"), match.group("end")


def _fields_from_datestr(s):
    """
    Parse a CMIP date token into integer fields, variable length:
      YYYY .. YYYYMMDDHHMMSS -> (Y,) .. (Y,M,D,H,Min,S)
    """
    n = len(s)
    if n not in (4, 6, 8, 10, 12, 14):
        raise ValueError(f"Unrecognized time range token length: '{s}'")
    fields = [int(s[0:4])]
    for start in range(4, n, 2):
        fields.append(int(s[start : start + 2]))
    return tuple(fields)


def _infer_is_climatology(ds):
    """
    Placeholder for climatology detection.

    TODO: implement robust climatology detection.
    """
    return False


def _build_precision_map(precision_by_frequency=None):
    """Build and normalize frequency -> label precision mapping."""
    mapping = dict(_DEFAULT_TIME_LABEL_PRECISION)
    if not precision_by_frequency:
        return mapping

    normalized = {}
    for key, value in precision_by_frequency.items():
        if key is None or value is None:
            continue
        normalized[str(key).strip()] = str(value).strip()

    default_label = normalized.get(_FALLBACK_PRECISION_KEY)
    if default_label in _LABEL_TO_PRECISION:
        for freq in list(mapping.keys()):
            if freq not in normalized:
                mapping[freq] = default_label

    for key, value in normalized.items():
        if key == _FALLBACK_PRECISION_KEY:
            continue
        mapping[key] = value

    if default_label:
        mapping[_FALLBACK_PRECISION_KEY] = default_label
    return mapping


def _expected_precision_from_frequency(freq, is_climatology=False, precision_by_frequency=None):
    """
    Return (label_format, token_length, tuple_length) expected for frequency.

    When climatology is requested, monthly precision (yyyyMM) is expected.
    """
    mapping = _build_precision_map(precision_by_frequency)
    fallback_label = mapping.get(_FALLBACK_PRECISION_KEY)

    if is_climatology:
        label = mapping.get("climatology") or fallback_label or "yyyyMM"
    else:
        label = mapping.get(freq) or fallback_label

    if label not in _LABEL_TO_PRECISION:
        return None

    token_len, tuple_len = _LABEL_TO_PRECISION[label]
    return label, token_len, tuple_len


def _full_tuple(dt):
    return (
        dt.year,
        dt.month,
        dt.day,
        getattr(dt, "hour", 0),
        getattr(dt, "minute", 0),
        getattr(dt, "second", 0),
    )


def _round_datetime(dt, nearest):
    """Round date-like objects to nearest minute or second when possible."""
    if nearest == "minute":
        try:
            add_minute = (
                getattr(dt, "second", 0) >= 30
                or getattr(dt, "microsecond", 0) >= 500000
            )
            if add_minute:
                dt = dt + timedelta(minutes=1)
            return dt.replace(second=0, microsecond=0)
        except Exception:
            return dt

    if nearest == "second":
        try:
            if getattr(dt, "microsecond", 0) >= 500000:
                dt = dt + timedelta(seconds=1)
            return dt.replace(microsecond=0)
        except Exception:
            return dt

    return dt


def _coverage_from_climatology_bounds(ds):
    """
    Return climatology coverage from climatology bounds if available.
    """
    if "time" not in ds.variables:
        return None, None, "Missing 'time' variable."

    tvar = ds.variables["time"]
    bname = getattr(tvar, "climatology", None)
    if not bname or bname not in ds.variables:
        return None, None, "Missing climatology bounds variable referenced by 'time:climatology'."

    try:
        bvar = ds.variables[bname]
        units = tvar.units
        calendar = getattr(tvar, "calendar", "standard")
        bvals = bvar[:]
        start_dt = num2date(bvals[0, 0], units=units, calendar=calendar)
        end_dt = num2date(bvals[-1, -1], units=units, calendar=calendar)
        return _full_tuple(start_dt), _full_tuple(end_dt), None
    except Exception as e:
        return None, None, f"Error converting climatology bounds values: {e}"


def _coverage_from_time_coordinate(ds):
    """
    Return data coverage from first and last values in the time coordinate.
    """
    if "time" not in ds.variables:
        return None, None, "Missing 'time' variable."

    tvar = ds.variables["time"]

    try:
        tvals = tvar[:]
        if hasattr(tvals, "compressed"):
            tvals = tvals.compressed()
        if tvals.size == 0:
            return None, None, "The 'time' variable is empty."

        units = tvar.units
        calendar = getattr(tvar, "calendar", "standard")
        dts = num2date([tvals[0], tvals[-1]], units=units, calendar=calendar)
        return _full_tuple(dts[0]), _full_tuple(dts[1]), None
    except Exception as e:
        return None, None, f"Error converting time values: {e}"


def _coverage_at_precision(ds, tuple_length, freq, is_climatology=False):
    """
    Return start/end coverage tuples at expected precision.
    """
    if is_climatology:
        cov_start_full, cov_end_full, err = _coverage_from_climatology_bounds(ds)
    else:
        cov_start_full, cov_end_full, err = _coverage_from_time_coordinate(ds)

    if err:
        return None, None, err

    # Apply rounding requirements for sub-daily frequencies.
    if not is_climatology and freq in {"1hr", "3hr", "6hr", "subhr"}:
        try:
            tvar = ds.variables["time"]
            tvals = tvar[:]
            if hasattr(tvals, "compressed"):
                tvals = tvals.compressed()
            units = tvar.units
            calendar = getattr(tvar, "calendar", "standard")
            start_dt, end_dt = num2date([tvals[0], tvals[-1]], units=units, calendar=calendar)
            if freq in {"1hr", "3hr", "6hr"}:
                start_dt = _round_datetime(start_dt, "minute")
                end_dt = _round_datetime(end_dt, "minute")
            else:
                start_dt = _round_datetime(start_dt, "second")
                end_dt = _round_datetime(end_dt, "second")
            cov_start_full = _full_tuple(start_dt)
            cov_end_full = _full_tuple(end_dt)
        except Exception:
            pass

    return cov_start_full[:tuple_length], cov_end_full[:tuple_length], None


def check_time_range_vs_filename(ds, severity=BaseCheck.MEDIUM, precision_by_frequency=None):
    """
    [TIME003] Compare filename time range with actual data coverage.
    """
    check_id = "TIME003"
    ctx = TestCtx(severity, f"[{check_id}] Check Time Range vs Filename")

    # Timeless frequencies (fx, fixed fields) have no time-range token.
    try:
        freq = str(ds.getncattr("frequency")).strip()
    except AttributeError:
        freq = ""

    if freq == "fx" or "time" not in ds.variables:
        ctx.add_pass()
        return [ctx.to_result()]

    is_climatology = _infer_is_climatology(ds)
    expected_precision = _expected_precision_from_frequency(
        freq,
        is_climatology=is_climatology,
        precision_by_frequency=precision_by_frequency,
    )
    if expected_precision is None:
        ctx.add_failure(f"Unsupported frequency for time range precision: '{freq}'.")
        return [ctx.to_result()]

    _label_fmt, expected_len, tuple_len = expected_precision

    filename = os.path.basename(ds.filepath())
    stem = filename[:-3] if filename.endswith(".nc") else filename

    # 1) Delegate structural validation (incl. time-range part) to esgvoc.
    esgvoc_ok = _esgvoc_filename_ok(ds, stem)

    # 2) Extract trailing time-range token for numeric comparison.
    start_str, end_str = _extract_time_range_token(filename)

    if start_str is None:
        if esgvoc_ok is True:
            ctx.add_pass()
            return [ctx.to_result()]
        ctx.add_failure(
            "No time range token found at the end of the filename "
            "(expected a trailing '_<start>-<end>' segment)."
        )
        return [ctx.to_result()]

    if len(start_str) != expected_len or len(end_str) != expected_len:
        ctx.add_failure(
            f"Time range precision mismatch for frequency '{freq}': "
            f"expected {expected_len}-digit labels but got '{start_str}-{end_str}'."
        )
        return [ctx.to_result()]

    try:
        expected_start = _fields_from_datestr(start_str)
        expected_end = _fields_from_datestr(end_str)
    except Exception as e:
        ctx.add_failure(f"Error parsing time range from filename: {e}")
        return [ctx.to_result()]

    cov_start, cov_end, err = _coverage_at_precision(
        ds,
        tuple_length=tuple_len,
        freq=freq,
        is_climatology=is_climatology,
    )
    if err:
        ctx.add_failure(err)
        return [ctx.to_result()]

    issues = []
    if cov_start > expected_start:
        issues.append(f"Data starts at {cov_start}, later than filename start {start_str}.")
    elif cov_start < expected_start:
        issues.append(f"Data starts at {cov_start}, earlier than filename start {start_str}.")

    if cov_end < expected_end:
        issues.append(f"Data ends at {cov_end}, earlier than filename end {end_str}.")
    elif cov_end > expected_end:
        issues.append(f"Data ends at {cov_end}, later than filename end {end_str}.")

    if issues:
        for msg in issues:
            ctx.add_failure(msg)
    else:
        ctx.add_pass()

    return [ctx.to_result()]
