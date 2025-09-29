#!/usr/bin/env python
"""


Checks that each *time* value falls inside its declared cell
bounds (*time_bnds*) and that shapes are consistent.

"""

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx


def check_time_bounds(ds, severity=BaseCheck.MEDIUM):
    """
    
    """
    check_id = "WWWWW"
    ctx = TestCtx(severity, f"[{check_id}] Check Time bounds")

    if "time" not in ds.variables:
        return [ctx.to_result()]

    time_var = ds.variables["time"]

    
    bnds_name = getattr(time_var, "bounds", None)
    if bnds_name is None or bnds_name not in ds.variables:
        # Presence of bounds is checked elsewhere – we are only interested
        # in *consistency* if they exist.
        return [ctx.to_result()]

    bnds_var = ds.variables[bnds_name]

    # Shape consistency: (n, 2)
    if bnds_var.ndim != 2 or bnds_var.shape[1] != 2 \
       or bnds_var.shape[0] != time_var.shape[0]:
        ctx.add_failure(
            f"{bnds_name} must have shape (n, 2) with n == len(time)"
        )
        return [ctx.to_result()]

    # Numerical consistency
    time_vals  = time_var[:].compressed()
    lower, upper = bnds_var[:, 0], bnds_var[:, 1]

    outside = np.logical_or(time_vals < lower, time_vals > upper)

    if outside.any():
        idx = np.where(outside)[0][:5]   # show only first offenders
        ctx.add_failure(
            f"{len(idx)} time value(s) lie outside declared bounds, "
            f"example index/val/bnds: "
            + ", ".join(
                f"{i}/{time_vals[i]}∉[{lower[i]}, {upper[i]}]" for i in idx
            )
        )
    else:
        ctx.add_pass()

    return [ctx.to_result()]
