#!/usr/bin/env python

import os
from compliance_checker.base import BaseCheck, TestCtx
from netCDF4 import num2date
from ..consistency_checks.check_attributes_match_filename import _parse_filename_components

def check_time_range_vs_filename(ds, severity=BaseCheck.MEDIUM):
    check_id = "VAR009"
    ctx = TestCtx(severity, f"[{check_id}] Check Time Range Filename")

    if "time" not in ds.variables:
        ctx.add_failure("Missing 'time' variable.")
        return [ctx.to_result()]

    time_var = ds.variables["time"]
    time_vals = time_var[:]
    if hasattr(time_vals, "compressed"):
        time_vals = time_vals.compressed()

    if time_vals.size == 0:
        ctx.add_failure("The 'time' variable is empty.")
        return [ctx.to_result()]

    try:
        units = time_var.units
        calendar = getattr(time_var, "calendar", "standard")
        time_dates = num2date(time_vals, units=units, calendar=calendar)
    except Exception as e:
        ctx.add_failure(f"Error converting time values to datetime: {e}")
        return [ctx.to_result()]

    try:
        filename = os.path.basename(ds.filepath())
        components = _parse_filename_components(filename)
        metadata = components[0]
        time_range = metadata.get("time_range", "")
        if not time_range or "-" not in time_range:
            raise ValueError(f"Invalid or missing time_range: {time_range}")
        start_str, end_str = time_range.split("-")

        # Handle YYYYMMDD or YYYYMM format
        if len(start_str) == 8:
            expected_start = (int(start_str[:4]), int(start_str[4:6]), int(start_str[6:8]))
            expected_end = (int(end_str[:4]), int(end_str[4:6]), int(end_str[6:8]))
            use_day = True
        elif len(start_str) == 6:
            expected_start = (int(start_str[:4]), int(start_str[4:6]))
            expected_end = (int(end_str[:4]), int(end_str[4:6]))
            use_day = False
        else:
            raise ValueError(f"Unrecognized time range format: {time_range}")
    except Exception as e:
        ctx.add_failure(f"Error parsing time range from filename: {e}")
        return [ctx.to_result()]

    try:
        first = time_dates[0]
        last = time_dates[-1]

        if use_day:
            start_parts = (first.year, first.month, first.day)
            end_parts = (last.year, last.month, last.day)
        else:
            start_parts = (first.year, first.month)
            end_parts = (last.year, last.month)
    except Exception as e:
        ctx.add_failure(f"Error interpreting datetime objects: {e}")
        return [ctx.to_result()]
    

    if start_parts > expected_start or end_parts < expected_end:
        ctx.add_failure(
            f"Time axis [{start_parts} - {end_parts}] does not fully cover filename range [{start_str}, {end_str}]."
        )
    else:
        ctx.add_pass()

    return [ctx.to_result()]
