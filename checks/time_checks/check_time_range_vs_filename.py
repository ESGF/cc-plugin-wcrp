#!/usr/bin/env python

import os
import re
from compliance_checker.base import BaseCheck, TestCtx
from netCDF4 import num2date


# CMIP7 DRS time-range tokens by frequency. Lengths come from the
# CMIP7 DRS specification ("Date/Time element specifications"):
#   yr / yrPt / dec      -> YYYY              (4)
#   mon / monPt / monC   -> YYYYMM            (6)
#   day / dayPt          -> YYYYMMDD          (8)
#   6hr / 6hrPt / 3hr…   -> YYYYMMDDhh        (10)
#   1hrCM / sub-daily Pt -> YYYYMMDDhhmm      (12)
#   subhrPt              -> YYYYMMDDhhmmss    (14)
# Pre-CMIP6 we only ever saw 6 or 8. Accept all valid lengths so the
# check applies cleanly to sub-daily CMIP7 files.
_TIME_RANGE_RE = re.compile(
    r"^(?P<start>\d{4}|\d{6}|\d{8}|\d{10}|\d{12}|\d{14})"
    r"-(?P<end>\d{4}|\d{6}|\d{8}|\d{10}|\d{12}|\d{14})"
    r"(?:-clim)?$"
)


def _extract_time_range_from_filename(filename: str):
    """
    Extract time range token from filename.
    Works for CMIP6 and CMIP7:
      ..._YYYY-YYYY.nc                   (yearly)
      ..._YYYYMM-YYYYMM.nc               (monthly)
      ..._YYYYMMDD-YYYYMMDD.nc           (daily)
      ..._YYYYMMDDhh-YYYYMMDDhh.nc       (hourly point)
      ..._YYYYMMDDhhmm-YYYYMMDDhhmm.nc   (sub-hourly point)
      ..._YYYYMMDDhhmmss-YYYYMMDDhhmmss.nc
      The optional ``-clim`` suffix on climatology files is also accepted.
    Returns (start_str, end_str, precision_len) or (None, None, None)
    if not found. ``precision_len`` is the digit-count (4/6/8/10/12/14)
    so callers can decide how granular the coverage comparison should be.
    """
    stem = filename[:-3] if filename.endswith(".nc") else filename
    last_token = stem.split("_")[-1]
    m = _TIME_RANGE_RE.match(last_token)
    if not m:
        return None, None, None

    start_str = m.group("start")
    end_str = m.group("end")
    return start_str, end_str, len(start_str)


def _tuple_from_datestr(s: str):
    """YYYY / YYYYMM / YYYYMMDD / YYYYMMDDhh / YYYYMMDDhhmm /
    YYYYMMDDhhmmss -> tuple at the matching precision."""
    if len(s) == 4:
        return (int(s[:4]),)
    if len(s) == 6:
        return (int(s[:4]), int(s[4:6]))
    if len(s) == 8:
        return (int(s[:4]), int(s[4:6]), int(s[6:8]))
    if len(s) == 10:
        return (int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]))
    if len(s) == 12:
        return (int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), int(s[10:12]))
    if len(s) == 14:
        return (
            int(s[:4]), int(s[4:6]), int(s[6:8]),
            int(s[8:10]), int(s[10:12]), int(s[12:14]),
        )
    raise ValueError(f"Unrecognized time range token: {s}")


def _coverage_from_time(ds):
    """
    Prefer bounds if available:
      time:bounds="time_bnds" and time_bnds(time, bnds)
    Returns (start_tuple, end_tuple, err). Tuples are at second precision
    (Y, M, D, h, m, s); callers truncate to the filename's precision.
    """
    if "time" not in ds.variables:
        return None, None, "Missing 'time' variable."

    tvar = ds.variables["time"]

    # If bounds exist, use them (more accurate than midpoints)
    bname = getattr(tvar, "bounds", None)
    if bname and bname in ds.variables:
        bvar = ds.variables[bname]
        try:
            units = tvar.units
            calendar = getattr(tvar, "calendar", "standard")

            bvals = bvar[:]
            # expected shape (n,2)
            start_val = bvals[0, 0]
            end_val = bvals[-1, 1]

            start_dt = num2date(start_val, units=units, calendar=calendar)
            end_dt = num2date(end_val, units=units, calendar=calendar)

            return (
                (start_dt.year, start_dt.month, start_dt.day,
                 start_dt.hour, start_dt.minute, start_dt.second),
                (end_dt.year, end_dt.month, end_dt.day,
                 end_dt.hour, end_dt.minute, end_dt.second),
                None,
            )
        except Exception:
            # fallback to time points if bounds conversion fails
            pass

    # Fallback: use time points
    try:
        tvals = tvar[:]
        if hasattr(tvals, "compressed"):
            tvals = tvals.compressed()
        if tvals.size == 0:
            return None, None, "The 'time' variable is empty."

        units = tvar.units
        calendar = getattr(tvar, "calendar", "standard")
        dts = num2date(tvals, units=units, calendar=calendar)

        first = dts[0]
        last = dts[-1]
        return (
            (first.year, first.month, first.day,
             first.hour, first.minute, first.second),
            (last.year, last.month, last.day,
             last.hour, last.minute, last.second),
            None,
        )
    except Exception as e:
        return None, None, f"Error converting time values: {e}"


# Number of (Y, M, D, h, m, s) tuple components for each filename
# token precision. e.g. precision=8 (YYYYMMDD) -> 3 components.
_PRECISION_TO_COMPONENTS = {
    4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6,
}


# Frequencies for which no time range token is expected in the filename.
_TIMELESS_FREQUENCIES = {"fx", "ofx"}


def check_time_range_vs_filename(ds, severity=BaseCheck.MEDIUM):
    """
    [TIME003] Check that the dataset time axis covers the time range declared in the filename.

    Files with frequency 'fx' or 'ofx' are time-invariant: no time range token is expected
    in their filename, and this check is skipped with a pass.
    For all other frequencies, the absence of a time range token in the filename is itself
    an error (the file is mis-named).
    """
    check_id = "TIME003"
    ctx = TestCtx(severity, f"[{check_id}] Check Time Range vs Filename")

    # Read frequency from global attributes
    frequency = getattr(ds, "frequency", None)

    filename = os.path.basename(ds.filepath())
    start_str, end_str, precision = _extract_time_range_from_filename(filename)

    if not start_str or not end_str:
        # No time range token found in filename.
        if frequency in _TIMELESS_FREQUENCIES:
            # Expected for fx/ofx — not a failure.
            ctx.add_pass()
        elif frequency is None:
            # Cannot determine if time range is required without knowing frequency.
            ctx.add_failure(
                "No time range token found in filename and 'frequency' global attribute is missing; "
                "cannot determine whether a time range is required."
            )
        else:
            # Any other frequency should have a time range token — this is a real error.
            ctx.add_failure(
                f"No time range token found in filename, but frequency='{frequency}' "
                "requires a time range (e.g. '_YYYY-YYYY.nc', '_YYYYMM-YYYYMM.nc', "
                "'_YYYYMMDD-YYYYMMDD.nc', or '_YYYYMMDDhhmm-YYYYMMDDhhmm.nc')."
            )
        return [ctx.to_result()]

    try:
        expected_start = _tuple_from_datestr(start_str)
        expected_end = _tuple_from_datestr(end_str)
    except Exception as e:
        ctx.add_failure(f"Error parsing time range from filename: {e}")
        return [ctx.to_result()]

    cov_start, cov_end, err = _coverage_from_time(ds)
    if err:
        ctx.add_failure(err)
        return [ctx.to_result()]

    # Truncate the (Y, M, D, h, m, s) coverage tuple to the filename's
    # precision so comparisons line up. e.g. YYYYMM filename → compare
    # only (Y, M).
    ncomp = _PRECISION_TO_COMPONENTS.get(precision, 3)
    cov_start = cov_start[:ncomp]
    cov_end = cov_end[:ncomp]

    # Fail if dataset starts after expected start OR ends before expected end
    if cov_start > expected_start or cov_end < expected_end:
        ctx.add_failure(
            f"Time coverage [{cov_start} - {cov_end}] does not fully cover filename range [{start_str} - {end_str}]."
        )
    else:
        ctx.add_pass()

    return [ctx.to_result()]
