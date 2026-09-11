#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import cftime
from compliance_checker.base import BaseCheck, TestCtx
from checks.time_checks.time_constants import FREQ_INC, AVERAGE_CORRECTION_FREQ
from checks.time_checks.reporting import (
    count_phrase,
    format_time_interval,
    format_time_value,
)
from checks.utils import add_time_increment, severity_word

NDECIMALS = 6
_TIME_RANGE_RE = re.compile(r"_(\d{4,14})-(\d{4,14})(?:-clim)?\.nc$", re.IGNORECASE)


class RepresentativeIntervalError(ValueError):
    """Raised when a coarse representative time has no unique interval."""


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
    # Any explicit non-point time cell method describes an interval statistic
    # (mean, minimum, maximum, sum, etc.) whose representative time is centered.
    if "time:" in cm:
        return False
    if freq_id in set(AVERAGE_CORRECTION_FREQ):
        return False
    return True


def _midpoint_num(d0, d1, units: str, calendar: str) -> float:
    n0 = float(cftime.date2num(d0, units=units, calendar=calendar))
    n1 = float(cftime.date2num(d1, units=units, calendar=calendar))
    return 0.5 * (n0 + n1)


def _representative_interval_start(
    representative_value: float,
    increment: int,
    unit: str,
    units: str,
    calendar: str,
):
    """Find the calendar-aligned interval represented by a coarse timestamp."""
    representative = cftime.num2date(
        representative_value,
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=True,
    )
    candidates = []
    if unit == "years":
        for offset in range(-increment, 1):
            try:
                candidates.append(
                    cftime.datetime(
                        representative.year + offset, 1, 1, calendar=calendar
                    )
                )
            except ValueError:
                continue
    elif unit == "months":
        representative_month = representative.year * 12 + representative.month - 1
        for offset in range(-increment, 1):
            ordinal = representative_month + offset
            try:
                candidates.append(
                    cftime.datetime(
                        ordinal // 12, ordinal % 12 + 1, 1, calendar=calendar
                    )
                )
            except ValueError:
                continue
    else:
        raise ValueError(f"Unsupported calendar-dependent increment unit {unit!r}")

    expected = _round(np.asarray([representative_value]), NDECIMALS)[0]
    matches = []
    for candidate in candidates:
        interval_end = add_time_increment(candidate, increment, unit, calendar)
        midpoint = _midpoint_num(candidate, interval_end, units, calendar)
        if _round(np.asarray([midpoint]), NDECIMALS)[0] == expected:
            matches.append(candidate)

    if len(matches) != 1:
        display_unit = unit[:-1] if unit.endswith("s") else unit
        raise RepresentativeIntervalError(
            f"found {len(matches)} calendar-aligned {increment}-{display_unit} intervals "
            "with the first time value as their midpoint"
        )
    return matches[0]


def _check_declared_bounds_midpoints(
    ctx,
    ds,
    time_var,
    report_mismatch=True,
) -> tuple[bool, np.ndarray | None]:
    """Check time values against declared regular or climatological bounds.

    Return whether structurally usable bounds were checked and, when readable,
    their numeric values. Presence, existence, and shape errors remain
    owned by the coordinate and CF bounds checks; TIME001 does not repeat them.
    """
    climatology_name = str(getattr(time_var, "climatology", "") or "")
    bounds_name = climatology_name or str(getattr(time_var, "bounds", "") or "")
    if not bounds_name or bounds_name not in ds.variables:
        return False, None
    bounds_var = ds.variables[bounds_name]
    if (
        bounds_var.ndim != 2
        or bounds_var.shape != (time_var.shape[0], 2)
        or bounds_var.dimensions[0] != time_var.dimensions[0]
    ):
        return False, None
    try:
        actual = np.ma.asarray(time_var[:], dtype="float64")
        bounds = np.ma.asarray(bounds_var[:], dtype="float64")
    except (TypeError, ValueError, IndexError):
        return False, None
    if np.ma.is_masked(bounds):
        affected = np.where(np.any(np.ma.getmaskarray(bounds), axis=1))[0]
        index = int(affected[0])
        units = time_var.units
        calendar = getattr(time_var, "calendar", "standard")
        agreement = "contains" if len(affected) == 1 else "contain"
        ctx.add_failure(
            f"{count_phrase(len(affected), bounds_name + ' interval')} {agreement} missing "
            f"values, so their time midpoints cannot be verified. At the first "
            f"incident, index {index}, the interval is "
            f"{format_time_interval(bounds[index], units=units, calendar=calendar, decimals=NDECIMALS)}."
        )
        return True, None
    actual = np.asarray(actual, dtype="float64").reshape(-1)
    bounds = np.asarray(bounds, dtype="float64")
    expected = 0.5 * (bounds[:, 0] + bounds[:, 1])
    rounded_actual = _round(actual, NDECIMALS)
    rounded_expected = _round(expected, NDECIMALS)
    bad = np.where(rounded_actual != rounded_expected)[0]
    if bad.size and report_mismatch:
        index = int(bad[0])
        kind = "climatology" if climatology_name else "bounds"
        units = time_var.units
        calendar = getattr(time_var, "calendar", "standard")
        if bad.size == 1:
            summary = f"1 time value does not match the midpoint of its {kind} interval"
        else:
            summary = (
                f"{bad.size} time values do not match the midpoints of their "
                f"{kind} intervals"
            )
        ctx.add_failure(
            f"{summary}. First incident at index {index}: "
            f"the file contains {format_time_value(actual[index], units=units, calendar=calendar, decimals=NDECIMALS)}. "
            f"The expected midpoint is {format_time_value(expected[index], units=units, calendar=calendar, decimals=NDECIMALS)}. "
            f"The {kind} interval is {format_time_interval(bounds[index], units=units, calendar=calendar, decimals=NDECIMALS)}."
        )
    return True, bounds


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

    - Declared regular bounds: configured interval spacing and coordinate midpoint
    - Declared climatological bounds: coordinate midpoint
    - Primary: FREQ_INC (table_id, frequency)
    - Start: filename start boundary
    - Average data: midpoint convention for AVERAGE_CORRECTION_FREQ
    - Optional policy: calendar / ref_time_units equality checks
    """
    ctx = TestCtx(severity, "[TIME001] Check Time Squareness ")

    if "time" not in ds.variables:
        return []

    time_var = ds.variables["time"]
    if time_var.ndim != 1:
        return []  # Coordinate identity checks own the invalid time shape.
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

    raw_time = np.ma.asarray(time_var[:])
    if np.ma.is_masked(raw_time):
        missing = np.where(np.ma.getmaskarray(raw_time).reshape(-1))[0]
        index = int(missing[0])
        agreement = "is" if len(missing) == 1 else "are"
        ctx.add_failure(
            f"{count_phrase(len(missing), 'time value')} {agreement} missing, so the time axis "
            f"cannot be verified. The first incident is at index {index}, with "
            f"{format_time_value(raw_time.reshape(-1)[index], units=units, calendar=cal, decimals=NDECIMALS)}."
        )
        return [ctx.to_result()]

    is_climatology = bool(getattr(time_var, "climatology", "") or "")
    midpoint_checked, declared_bounds = _check_declared_bounds_midpoints(
        ctx,
        ds,
        time_var,
        report_mismatch=is_climatology,
    )
    if is_climatology:
        # A climatological coordinate can represent an averaging interval
        # spanning many years. Its correct value is determined by the declared
        # climatology bounds, not by advancing once from the filename start.
        if midpoint_checked and not ctx.messages:
            ctx.add_pass()
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
    # Avoid adding a second midpoint offset when the filename label already
    # represents the sample time, exactly for subdaily data or approximately
    # for multi-unit intervals such as decadal means.
    subdaily_label = inc_unit in {"hours", "minutes", "seconds"}
    coarse_representative_label = inc_val > 1 and not subdaily_label
    label_is_representative = subdaily_label or coarse_representative_label

    # Start boundary from filename
    start_tuple = _parse_filename_start(_get_ds_path(ds))
    if not start_tuple:
        ctx.add_failure("Cannot parse filename time range start (_YYYY..-YYYY..nc).")
        return [ctx.to_result()]

    try:
        start_boundary = cftime.datetime(*start_tuple, calendar=cal)
    except Exception as exc:
        ctx.add_failure(
            f"Cannot interpret filename time-range start {start_tuple} using "
            f"calendar {cal!r}. Technical reason: {type(exc).__name__}: {exc}"
        )
        return [ctx.to_result()]

    # Instantaneous vs average
    target = _resolve_target_variable(ds)
    instantaneous = _is_instantaneous(ds, target, freq_id)
    use_midpoint = not instantaneous

    # Read actual time axis
    actual = np.asarray(raw_time, dtype=float)

    if actual.size == 0:
        ctx.add_failure("Time axis is empty.")
        return [ctx.to_result()]

    # Build theoretical axis in numeric space (file units)
    theo = np.zeros(actual.size, dtype=float)
    theo_bounds = None
    variable_step = inc_unit in ("months", "years")
    bounds_available = use_midpoint and declared_bounds is not None

    try:
        if bounds_available:
            theo_bounds = np.zeros_like(declared_bounds, dtype=float)

        if not variable_step:
            # The size of a fixed increment can be established near the filename
            # date. Keep a representative-time anchor numeric so very large
            # offsets need not pass through cftime's finite datetime range.
            d0 = start_boundary
            d1 = add_time_increment(d0, inc_val, inc_unit, cal)
            reference_n0 = float(cftime.date2num(d0, units=units, calendar=cal))
            reference_n1 = float(cftime.date2num(d1, units=units, calendar=cal))
            step_num = reference_n1 - reference_n0
            n0 = (
                float(actual[0]) - step_num / 2.0
                if use_midpoint and label_is_representative
                else reference_n0
            )
            n1 = n0 + step_num
            if use_midpoint:
                first = (n0 + n1) / 2.0
            else:
                first = n0
            offsets = np.arange(actual.size, dtype=float) * float(step_num)
            theo = first + offsets
            if bounds_available:
                theo_bounds[:, 0] = n0 + offsets
                theo_bounds[:, 1] = n1 + offsets
        else:
            # Advancing a midpoint can drift when adjacent calendar cells contain
            # different numbers of days; rebuild from the first cell boundary.
            if use_midpoint and label_is_representative:
                cur = _representative_interval_start(
                    actual[0],
                    inc_val,
                    inc_unit,
                    units=units,
                    calendar=cal,
                )
            else:
                cur = start_boundary
            for i in range(actual.size):
                nxt = add_time_increment(cur, inc_val, inc_unit, cal)
                cur_num = float(cftime.date2num(cur, units=units, calendar=cal))
                nxt_num = float(cftime.date2num(nxt, units=units, calendar=cal))
                theo[i] = (
                    0.5 * (cur_num + nxt_num)
                    if use_midpoint
                    else cur_num
                )
                if bounds_available:
                    theo_bounds[i] = (cur_num, nxt_num)
                cur = nxt
    except RepresentativeIntervalError as exc:
        display_unit = inc_unit[:-1] if inc_unit.endswith("s") else inc_unit
        ctx.add_failure(
            f"The first time value "
            f"{format_time_value(actual[0], units=units, calendar=cal, decimals=NDECIMALS)} "
            f"does not identify a unique calendar-aligned {inc_val}-{display_unit} "
            f"interval. The search {exc}."
        )
        return [ctx.to_result()]
    except Exception as exc:
        ctx.add_failure(
            f"Cannot reconstruct the expected time axis using units {units!r} "
            f"and calendar {cal!r}. Technical reason: {type(exc).__name__}: {exc}"
        )
        return [ctx.to_result()]

    if bounds_available:
        rounded_bounds = _round(declared_bounds, NDECIMALS)
        rounded_theoretical_bounds = _round(theo_bounds, NDECIMALS)
        bad_bounds = np.where(
            np.any(rounded_bounds != rounded_theoretical_bounds, axis=1)
        )[0]
        if bad_bounds.size:
            index = int(bad_bounds[0])
            agreement = "does" if bad_bounds.size == 1 else "do"
            display_unit = inc_unit[:-1] if inc_unit.endswith("s") else inc_unit
            ctx.add_failure(
                f"{count_phrase(int(bad_bounds.size), 'time-bounds interval')} {agreement} not "
                f"match regular {inc_val}-{display_unit} cells. First incident at index "
                f"{index}: the file contains {format_time_interval(declared_bounds[index], units=units, calendar=cal, decimals=NDECIMALS)}. "
                f"The expected interval is {format_time_interval(theo_bounds[index], units=units, calendar=cal, decimals=NDECIMALS)}."
            )

    # Compare after rounding to NDECIMALS places. See _round docstring
    # for why this is np.round and not np.trunc.
    a_t = _round(actual, NDECIMALS)
    t_t = _round(theo, NDECIMALS)

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
            nxt = add_time_increment(cur, inc_val, inc_unit, cal)
            mid = cftime.datetime(cur.year, cur.month, 15, 0, 0, 0, calendar=cal)
            theo_mid[i] = float(cftime.date2num(mid, units=units, calendar=cal))
            theo_center[i] = _midpoint_num(cur, nxt, units, cal)
            cur = nxt

        t_mid = _round(theo_mid, NDECIMALS)
        t_center = _round(theo_center, NDECIMALS)

        # A file passes only if the full axis matches one convention end-to-end.
        full_match_start = np.array_equal(a_t, t_t)
        full_match_mid = np.array_equal(a_t, t_mid)
        full_match_center = np.array_equal(a_t, t_center)

        if not (full_match_start or full_match_mid or full_match_center):
            # Report deviations from the closest complete convention. This also
            # gives a meaningful count for an axis that mixes valid conventions.
            differences = [
                np.where(a_t != candidate)[0] for candidate in (t_t, t_mid, t_center)
            ]
            closest = min(differences, key=lambda indices: indices.size)
            failure_count = int(closest.size)
            i = int(closest[0])
            agreement = "prevents" if failure_count == 1 else "prevent"
            ctx.add_failure(
                f"{count_phrase(failure_count, 'time value')} {agreement} the full axis "
                f"from following one permitted monthly convention. First incident at "
                f"index {i}: the file contains {format_time_value(actual[i], units=units, calendar=cal, decimals=NDECIMALS)}. "
                f"Accepted values at this index are {format_time_value(theo[i], units=units, calendar=cal, decimals=NDECIMALS)} (month-start), "
                f"{format_time_value(theo_mid[i], units=units, calendar=cal, decimals=NDECIMALS)} (day-15), or "
                f"{format_time_value(theo_center[i], units=units, calendar=cal, decimals=NDECIMALS)} (exact center). "
                f"It is {severity_word(severity)} for the full file to consistently "
                "follow one of these conventions. "
                f"(table_id={table_id}, frequency={freq_id}, var={target}, midpoint={use_midpoint})"
            )
    else:
        bad = np.where(a_t != t_t)[0]
        if bad.size:
            i = int(bad[0])
            agreement = "does" if bad.size == 1 else "do"
            ctx.add_failure(
                f"{count_phrase(int(bad.size), 'time value')} {agreement} not match the expected "
                f"axis. First incident at index {i}: "
                f"the file contains {format_time_value(actual[i], units=units, calendar=cal, decimals=NDECIMALS)}. "
                f"The expected value is {format_time_value(theo[i], units=units, calendar=cal, decimals=NDECIMALS)}. "
                f"(table_id={table_id}, frequency={freq_id}, var={target}, midpoint={use_midpoint})"
            )

    if not ctx.messages:
        ctx.add_pass()

    return [ctx.to_result()]
