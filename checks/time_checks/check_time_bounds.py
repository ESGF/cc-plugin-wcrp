#!/usr/bin/env python
"""


Checks that each *time* value falls inside its declared cell
bounds (*time_bnds*) and that shapes are consistent.

"""

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx

from checks.time_checks.reporting import (
    count_phrase,
    format_time_interval,
    format_time_value,
)
from checks.utils import severity_word


def check_time_bounds(ds, severity=BaseCheck.MEDIUM, coord_name="time"):
    """Check regular bounds shape and containment for one time coordinate."""
    check_id = "TIME002"
    ctx = TestCtx(severity, f"[{check_id}] Check time bounds for '{coord_name}'")

    if coord_name not in ds.variables:
        return [ctx.to_result()]

    time_var = ds.variables[coord_name]

    bnds_name = getattr(time_var, "bounds", None)
    if bnds_name is None or bnds_name not in ds.variables:
        # Presence of bounds is checked elsewhere – we are only interested
        # in *consistency* if they exist.
        return [ctx.to_result()]

    bnds_var = ds.variables[bnds_name]

    # Shape consistency: (n, 2)
    if (
        time_var.ndim != 1
        or bnds_var.ndim != 2
        or bnds_var.shape[1] != 2
        or bnds_var.shape[0] != time_var.shape[0]
    ):
        ctx.add_failure(
            f"It is {severity_word(severity)} for {bnds_name} to have shape (n, 2) with "
            f"n == len({coord_name})"
        )
        return [ctx.to_result()]

    # Numerical consistency
    time_vals = np.ma.asarray(time_var[:]).filled(np.nan)
    pairs = np.ma.asarray(bnds_var[:]).filled(np.nan)
    lower = np.minimum(pairs[:, 0], pairs[:, 1])
    upper = np.maximum(pairs[:, 0], pairs[:, 1])

    outside = (
        np.logical_or(time_vals < lower, time_vals > upper)
        | ~np.isfinite(time_vals)
        | ~np.isfinite(lower)
        | ~np.isfinite(upper)
    )

    if outside.any():
        all_indices = np.where(outside)[0]
        index = int(all_indices[0])
        units = getattr(time_var, "units", "") or ""
        calendar = getattr(time_var, "calendar", "standard") or "standard"
        agreement = "lies" if len(all_indices) == 1 else "lie"
        ctx.add_failure(
            f"{count_phrase(len(all_indices), repr(coord_name) + ' value')} {agreement} outside "
            f"declared bounds. First incident at index {index}: "
            f"the file contains {format_time_value(time_vals[index], units=units, calendar=calendar)}. "
            f"The bounds are {format_time_interval(pairs[index], units=units, calendar=calendar)}."
        )
    else:
        ctx.add_pass()

    return [ctx.to_result()]
