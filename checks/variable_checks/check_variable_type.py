#!/usr/bin/env python

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx


def configured_data_types(value):
    """Return configured archive datatype names as a normalized list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _matches_data_type(dtype, expected):
    """Match netCDF storage precision or a legacy NumPy dtype-kind code."""
    expected = str(expected).strip().lower()
    if expected in {"real", "float32"}:
        return dtype.kind == "f" and dtype.itemsize == 4
    if expected in {"double", "float64"}:
        return dtype.kind == "f" and dtype.itemsize == 8
    if expected == "float":
        return dtype.kind == "f"
    if expected in {"character", "char", "str"}:
        return dtype.kind in {"S", "U"}
    if len(expected) == 1:
        return dtype.kind == expected
    try:
        return dtype == np.dtype(expected)
    except TypeError:
        return False


def check_variable_type(ds, variable_name, allowed_types=None, severity=BaseCheck.HIGH):
    check_id = "VAR005"
    ctx = TestCtx(severity, f"[{check_id}] Variable Type Check: '{variable_name}'")

    if variable_name not in ds.variables:
        return []

    var = ds.variables[variable_name]
    if allowed_types is None:
        allowed_types = ["float"]
    allowed_types = configured_data_types(allowed_types)

    try:
        dtype = np.dtype(var.dtype)
    except AttributeError:
        ctx.add_failure(f"Could not determine dtype for variable '{variable_name}'.")
        return [ctx.to_result()]

    if any(_matches_data_type(dtype, expected) for expected in allowed_types):
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"Variable '{variable_name}' has data type '{dtype.name}'; expected one of "
            f"{allowed_types}."
        )
    return [ctx.to_result()]
