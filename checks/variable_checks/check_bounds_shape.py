#!/usr/bin/env python

"""
[VAR004] Bounds shape check.

For a coordinate variable that declares a `bounds` attribute, verify that
the referenced bounds variable has a CF-compliant shape:

- 1-D coordinate of length N  -> bounds shape (N, 2)
- Multi-D coordinate (e.g. curvilinear lat/lon) of shape (d1, d2, ...)
  -> bounds shape (d1, d2, ..., nv) with nv >= 3
  (CF §7.1, polygon vertices for cell corners)

This check is purely structural. Bounds *values* are checked separately
by [VAR012] (check_bounds_value_consistency).
"""

from compliance_checker.base import BaseCheck, TestCtx


def check_bounds_shape(ds, var_name, severity=BaseCheck.MEDIUM):
    """
    Validate the shape of the bounds variable declared by `var_name`.

    Parameters
    ----------
    ds : netCDF4.Dataset
    var_name : str
        Name of the coordinate (or auxiliary coordinate) variable to check.
    severity : int
        Severity level used for the emitted Result.

    Returns
    -------
    list[Result]
        - Empty list if the variable does not exist or has no `bounds`
          attribute (nothing to check).
        - One Result describing pass/failure otherwise.
    """
    check_id = "VAR004"

    if var_name not in ds.variables:
        return []

    cvar = ds.variables[var_name]
    bnds_name = getattr(cvar, "bounds", None)
    if not bnds_name:
        return []

    ctx = TestCtx(severity, f"[{check_id}] Bounds for '{var_name}'")

    if bnds_name not in ds.variables:
        ctx.add_failure(
            f"Bounds variable '{bnds_name}' referenced by '{var_name}' not found."
        )
        return [ctx.to_result()]

    bvar = ds.variables[bnds_name]
    bshape = getattr(bvar, "shape", None)
    try:
        cshape = tuple(cvar.shape)
        bshape = tuple(bvar.shape)
        if cvar.ndim == 0:
            # CF §7.1: scalar coordinate (e.g. a single depth level).
            # Bounds have no coordinate dimension to align with: shape (2,).
            ok_shape = bvar.ndim == 1 and bshape[0] == 2
            expected_desc = (
                f"(2,) for scalar coordinate '{var_name}' "
                f"(CF §7.1, bounds of a single-valued coordinate)"
            )
        elif cvar.ndim == 1:
            # CF §7.1: 1-D coord of length N.
            # Two valid layouts depending on grid topology:
            #   - regular grid coord on its own dimension (e.g. lat(lat)):
            #     bounds (N, 2) interval form only.
            #   - unstructured / generic-cell coord (e.g. lat(ncell) and
            #     lon(ncell) sharing the same cell dimension): bounds (N, m)
            #     with m >= 2 (interval) or m >= 3 (polygon vertices).
            #
            # The distinguisher is whether another 1-D coord variable lives
            # on the same dimension and also declares bounds. If yes, this
            # is unstructured topology. If no, treat as a classic 1-D coord
            # on its own dimension and require interval bounds.
            cdim = cvar.dimensions[0]
            sibling_has_bounds = any(
                vname != var_name
                and v.ndim == 1
                and v.dimensions == (cdim,)
                and getattr(v, "bounds", None)
                for vname, v in ds.variables.items()
            )
            if sibling_has_bounds:
                ok_shape = (
                    bvar.ndim == 2
                    and bshape[0] == cshape[0]
                    and bshape[1] >= 2
                )
                expected_desc = (
                    f"(n, 2) or (n, m) with m >= 3 and n == len({var_name})={cshape[0]} "
                    f"(unstructured 1-D coord on shared dimension '{cdim}')"
                )
            else:
                ok_shape = (
                    bvar.ndim == 2
                    and bshape[0] == cshape[0]
                    and bshape[1] == 2
                )
                expected_desc = (
                    f"(n, 2) with n == len({var_name})={cshape[0]} "
                    f"(regular 1-D coord on its own dimension '{cdim}')"
                )
        elif cvar.ndim >= 2:
            # CF §7.1: multi-D coord (curvilinear, unstructured...)
            # bounds shape is (*coord.shape, nv), nv >= 3
            ok_shape = (
                bvar.ndim == cvar.ndim + 1
                and bshape[:-1] == cshape
                and bshape[-1] >= 3
            )
            expected_desc = (
                f"({', '.join(str(x) for x in cshape)}, nv) "
                f"with nv >= 3 (CF §7.1, curvilinear grid)"
            )
        else:
            ok_shape = False
            expected_desc = "(n, 2)"
    except Exception:
        ok_shape = False
        expected_desc = "(n, 2)"

    if ok_shape:
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"'{bnds_name}' must have shape {expected_desc}. Found {bshape}."
        )

    return [ctx.to_result()]