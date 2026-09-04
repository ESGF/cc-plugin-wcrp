#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import cftime
from compliance_checker.base import BaseCheck, TestCtx
from datetime import timedelta as py_timedelta
from checks.time_checks.time_constants import FREQ_INC, AVERAGE_CORRECTION_FREQ

TOL_SECONDS = 0.1
NDECIMALS = 8  # display precision for failure messages; must resolve well below TOL_SECONDS
_TIME_RANGE_RE = re.compile(r"_(\d{4,14})-(\d{4,14})(?:-clim)?\.nc$", re.IGNORECASE)

def _tolerance(start: cftime.datetime, units: str, calendar: str) -> float:
    """TOL_SECONDS in the file's original units via cftime."""
    # start time is only used to get the correct calendar for cftime.date2num
    n0 = cftime.date2num(start, units=units, calendar=calendar)
    # Add TOL_SECONDS to start time, convert to numeric, and return the difference in the file's units.
    n1 = cftime.date2num(start + py_timedelta(seconds=TOL_SECONDS), units=units, calendar=calendar)
    return float(n1 - n0)


def _to_seconds(diff_units: float, tol_units: float) -> float:
    """Convert a difference in the file's own time units back to seconds,
    using the same tol/TOL_SECONDS ratio ``_tolerance`` computed -- so a
    failure message can report a physically meaningful offset instead of raw
    axis values, which can print identically at NDECIMALS while still
    differing by more than the tolerance."""
    return diff_units * TOL_SECONDS / tol_units


def _round(arr: np.ndarray, ndecs: int) -> np.ndarray:
    # Round (not truncate) to ``ndecs`` decimal places before comparison.
    # ``np.trunc`` was used here previously, but it floors toward zero and
    # is sensitive to round-trip float64 precision: a theoretical value
    # computed as ``first + i*step`` accumulates a per-index epsilon
    # (~5e-17) that, after multiplying by 1e6, falls just below the next
    # integer and trunc drops it. The file's stored value can be exact at
    # the same precision, so the two land in different 1e-6 buckets and a
    # phantom mismatch fires even though the values agree at well below
    # the check's resolution. ``np.round`` (banker's rounding to nearest
    # even) is unaffected by ulp-level precision drift.
    f = 10.0 ** int(ndecs)
    return np.round(arr * f) / f


def _get_ds_path(ds) -> str:
    if hasattr(ds, "filepath"):
        try:
            return ds.filepath()
        except Exception:
            pass
    return getattr(ds, "filename", "") or ""


def _parse_filename_start(filepath: str):
    """
    Extract start boundary from filename time range.
    """
    fname = os.path.basename(filepath or "")
    m = _TIME_RANGE_RE.search(fname)
    if not m:
        return None
    s = m.group(1)

    y = int(s[0:4])
    mo = int(s[4:6]) if len(s) >= 6 else 1
    d = int(s[6:8]) if len(s) >= 8 else 1
    hh = int(s[8:10]) if len(s) >= 10 else 0
    mm = int(s[10:12]) if len(s) >= 12 else 0
    ss = int(s[12:14]) if len(s) >= 14 else 0
    return (y, mo, d, hh, mm, ss)


def _resolve_table_id(ds) -> str:
    try:
        t = str(ds.getncattr("table_id"))
        return t.split()[-1] if " " in t else t
    except Exception:
        return "None"


def _resolve_frequency(ds) -> str | None:
    try:
        return str(ds.getncattr("frequency"))
    except Exception:
        return None


def _resolve_target_variable(ds) -> str | None:
    try:
        vid = ds.getncattr("variable_id")
        if vid in ds.variables:
            return str(vid)
    except Exception:
        pass

    base = os.path.basename(_get_ds_path(ds))
    if "_" in base:
        cand = base.split("_")[0]
        if cand in ds.variables:
            return cand

    for vname, var in ds.variables.items():
        if vname == "time":
            continue
        if "time" in getattr(var, "dimensions", ()):
            return vname

    return None


def _is_instantaneous(ds, target_var: str | None, freq_id: str) -> bool:
    cm = ""
    if target_var and target_var in ds.variables:
        cm = str(getattr(ds.variables[target_var], "cell_methods", "") or "").lower()

    if "time: point" in cm:
        return True
    if freq_id in set(AVERAGE_CORRECTION_FREQ):
        return False
    return True


def _add_time_increment(
    date: cftime.datetime, value: int, unit: str, calendar: str
) -> cftime.datetime:
    if unit in ("seconds", "minutes", "hours", "days"):
        return date + py_timedelta(**{unit: int(value)})

    y = int(date.year)
    m = int(date.month)
    d = int(date.day)

    if unit == "years":
        y += int(value)
    elif unit == "months":
        tot = m + int(value)
        div, mod = divmod(tot - 1, 12)
        y += div
        m = mod + 1
    else:
        return date

    if calendar == "360_day":
        d = min(d, 30)
        return cftime.datetime(
            y, m, d, date.hour, date.minute, date.second, calendar=calendar
        )

    for dd in range(d, 0, -1):
        try:
            return cftime.datetime(
                y, m, dd, date.hour, date.minute, date.second, calendar=calendar
            )
        except ValueError:
            continue

    return cftime.datetime(
        y, m, 1, date.hour, date.minute, date.second, calendar=calendar
    )


def _midpoint_num(d0, d1, units: str, calendar: str) -> float:
    n0 = float(cftime.date2num(d0, units=units, calendar=calendar))
    n1 = float(cftime.date2num(d1, units=units, calendar=calendar))
    return 0.5 * (n0 + n1)


def _parse_freq_token(token: str):
    """
    Parse TOML fallback tokens (e.g. 30m, 1h, 1D, 1M, 1Y).
    'm' = minutes, 'M' = months.
    """
    if not token:
        return None
    m = re.match(r"^\s*(\d+)\s*([smhDMY])\s*$", str(token).strip())
    if not m:
        return None
    val = int(m.group(1))
    u = m.group(2)

    if u == "s":
        return val, "seconds"
    if u == "m":
        return val, "minutes"
    if u == "h":
        return val, "hours"
    if u == "D":
        return val, "days"
    if u == "M":
        return val, "months"
    if u == "Y":
        return val, "years"
    return None


def _resolve_increment(table_id: str, freq_id: str, fallback_freq: dict | None):
    """
    Resolution order:
      1) local nctime mapping: (table_id, freq_id)
      2) local nctime mapping: ('None', freq_id)
      3) TOML fallback: frequency[freq_id] -> token -> (val, unit)
    """
    if (table_id, freq_id) in FREQ_INC:
        return FREQ_INC[(table_id, freq_id)]
    if ("None", freq_id) in FREQ_INC:
        return FREQ_INC[("None", freq_id)]
    if fallback_freq and freq_id in fallback_freq:
        return _parse_freq_token(fallback_freq[freq_id])
    return None


def check_time_squareness(
    ds, severity=BaseCheck.HIGH, calendar="", ref_time_units="", frequency=None
):
    """
    TIME001: Time axis check for a single file.

    - Primary: FREQ_INC (table_id, frequency)
    - Start: filename start boundary
    - Average data: midpoint convention for AVERAGE_CORRECTION_FREQ
    - Optional policy: calendar / ref_time_units equality checks
    """
    ctx = TestCtx(severity, "[TIME001] Check Time Squareness ")

    if "time" not in ds.variables:
        return []

    time_var = ds.variables["time"]
    units = getattr(time_var, "units", "") or ""
    cal = getattr(time_var, "calendar", "standard") or "standard"

    if calendar and str(cal) != str(calendar):
        ctx.add_failure(f"time.calendar='{cal}' differs from expected '{calendar}'.")
    if ref_time_units and str(units).strip() != str(ref_time_units).strip():
        ctx.add_failure(
            f"time.units='{units}' differs from expected '{ref_time_units}'."
        )

    if not units:
        ctx.add_failure("Missing time.units; cannot rebuild theoretical axis.")
        return [ctx.to_result()]

    freq_id = _resolve_frequency(ds)
    if not freq_id:
        ctx.add_failure(
            "Missing global attribute 'frequency'; cannot resolve expected step."
        )
        return [ctx.to_result()]

    table_id = _resolve_table_id(ds)
    inc = _resolve_increment(table_id, freq_id, frequency or {})
    if not inc:
        ctx.add_failure(
            f"Cannot resolve increment for (table_id={table_id}, frequency={freq_id})."
        )
        return [ctx.to_result()]

    inc_val, inc_unit = int(inc[0]), str(inc[1])

    # Start boundary from filename
    start_tuple = _parse_filename_start(_get_ds_path(ds))
    if not start_tuple:
        ctx.add_failure("Cannot parse filename time range start (_YYYY..-YYYY..nc).")
        return [ctx.to_result()]

    start_boundary = cftime.datetime(*start_tuple, calendar=cal)

    # Instantaneous vs average
    target = _resolve_target_variable(ds)
    instantaneous = _is_instantaneous(ds, target, freq_id)
    use_midpoint = (not instantaneous) and (freq_id in set(AVERAGE_CORRECTION_FREQ))

    # Read actual time axis
    raw = time_var[:]
    if hasattr(raw, "compressed"):
        raw = raw.compressed()
    actual = np.asarray(raw, dtype=float)

    if actual.size == 0:
        ctx.add_failure("Time axis is empty.")
        return [ctx.to_result()]

    # Build theoretical axis in numeric space (file units)
    theo = np.zeros(actual.size, dtype=float)
    variable_step = inc_unit in ("months", "years")

    if not variable_step:
        d0 = start_boundary
        d1 = _add_time_increment(d0, inc_val, inc_unit, cal)
        n0 = float(cftime.date2num(d0, units=units, calendar=cal))
        n1 = float(cftime.date2num(d1, units=units, calendar=cal))
        step_num = n1 - n0
        first = (n0 + n1) / 2.0 if use_midpoint else n0
        theo = first + np.arange(actual.size, dtype=float) * float(step_num)
    else:
        cur = start_boundary
        for i in range(actual.size):
            nxt = _add_time_increment(cur, inc_val, inc_unit, cal)
            theo[i] = (
                _midpoint_num(cur, nxt, units, cal)
                if use_midpoint
                else float(cftime.date2num(cur, units=units, calendar=cal))
            )
            cur = nxt

    # Construct tolerance in file units for comparison
    # use a unit-based tolerance to avoid floating-point
    # drift issues (see https://github.com/ESGF/cc-plugin-wcrp/issues/77)
    # when rounding to an absolute number of decimal places, sufficiently
    # large time values can lose precision and cause false positives.
    tol = _tolerance(start_boundary, units, cal)
    # keep names so the rest of the function reads as before
    a_t, t_t = actual, theo

    # For monthly instantaneous data, allow three valid timestamp conventions,
    # but require consistency across the whole file (single convention per file):
    #  - first day of each month at 00:00
    #  - 15th day of each month at 00:00
    #  - exact calendar-aware midpoint of each month interval
    allow_mon_point_midmonth = (
        instantaneous
        and (freq_id in {"mon", "monPt"})
        and (inc_unit == "months")
        and (inc_val == 1)
    )

    if allow_mon_point_midmonth:
        theo_mid = np.zeros(actual.size, dtype=float)
        theo_center = np.zeros(actual.size, dtype=float)
        cur = start_boundary
        for i in range(actual.size):
            nxt = _add_time_increment(cur, inc_val, inc_unit, cal)
            mid = cftime.datetime(cur.year, cur.month, 15, 0, 0, 0, calendar=cal)
            theo_mid[i] = float(cftime.date2num(mid, units=units, calendar=cal))
            theo_center[i] = _midpoint_num(cur, nxt, units, cal)
            cur = nxt

        # retain the full original precision of the actual axis for comparison
        # to a tolerance
        t_mid, t_center = theo_mid, theo_center
        # create a tolerance mask for each candidate axis,
        # and a full-match check for each candidate
        ok_start = np.abs(a_t - t_t) <= tol
        ok_mid = np.abs(a_t - t_mid) <= tol
        ok_center = np.abs(a_t - t_center) <= tol
        full_match_start, full_match_mid, full_match_center = ok_start.all(), ok_mid.all(), ok_center.all()

        if not (full_match_start or full_match_mid or full_match_center):
            # Concstruct composite mask of points that match at least one candidate axis
            bad = np.where(~ok_start & ~ok_mid & ~ok_center)[0]
            if bad.size:
                i = int(bad[0])
            else:
                # Mixed-convention axis: every point matches at least one candidate,
                # but no single candidate matches the entire file.
                first_start = np.where(~ok_start)[0]
                i = int(first_start[0]) if first_start.size else 0
            off_start = _to_seconds(a_t[i] - t_t[i], tol)
            off_mid = _to_seconds(a_t[i] - t_mid[i], tol)
            off_center = _to_seconds(a_t[i] - t_center[i], tol)
            ctx.add_failure(
                f"Mismatch at index {i}: actual={a_t[i]:.{NDECIMALS}f} is off by "
                f"{off_start:+.{NDECIMALS}f}s (month-start), {off_mid:+.{NDECIMALS}f}s (day-15), "
                f"{off_center:+.{NDECIMALS}f}s (exact center); tolerance is {TOL_SECONDS:.{NDECIMALS}f}s. "
                "The full file must consistently follow one of these conventions. "
                f"(table_id={table_id}, frequency={freq_id}, var={target}, midpoint={use_midpoint})"
            )
    else:
        # For all other cases, check against the single candidate axis
        bad = np.where(np.abs(a_t - t_t) > tol)[0]
        if bad.size:
            i = int(bad[0])
            off = _to_seconds(a_t[i] - t_t[i], tol)
            ctx.add_failure(
                f"Mismatch at index {i}: actual={a_t[i]:.{NDECIMALS}f}, expected={t_t[i]:.{NDECIMALS}f} "
                f"(off by {off:+.{NDECIMALS}f}s; tolerance is {TOL_SECONDS:.{NDECIMALS}f}s). "
                f"(table_id={table_id}, frequency={freq_id}, var={target}, midpoint={use_midpoint})"
            )

    if not ctx.messages:
        ctx.add_pass()

    return [ctx.to_result()]
