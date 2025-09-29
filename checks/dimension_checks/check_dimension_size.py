#!/usr/bin/env python

"""
check_dimension_size.py

Check if the specified dimension size is valid in a netCDF dataset.

Intended to be included in the WCRP plugins.
"""

from compliance_checker.base import BaseCheck, TestCtx


def check_dimension_size_is_equals_to(
        ds,
        dimension_name,
        expected_size,
        severity=BaseCheck.MEDIUM):
    """
        Verify if the dimension has the expected size.

    """
    check_id = "DIM003"
    ctx = TestCtx(severity, f"[{check_id}] Dimension Size Check: '{dimension_name}'")

    dim_size = ds.dimensions[dimension_name].size  # no check of value existence, it must be done elsewhere
    if dim_size != expected_size:
        ctx.add_failure(
            f"Unexpected dimension size for '{dimension_name}'. Got {dim_size}, expected {expected_size}.")
    else:
        ctx.add_pass()

    return [ctx.to_result()]


def check_dimension_size_is_strictly_greater_than(
        ds,
        dimension_name,
        lower_bound,
        severity=BaseCheck.MEDIUM):
    """
        Verify if the dimension size is greater than the provided size.

        Parameters
        ----------
        ds : netCDF4.Dataset
            An already open netCDF dataset.
        dimension_name : str
            The name of the dimension to check.
        lower_bound :
            Size to be compared.
        severity : int, optional
            The severity level of this check (default: BaseCheck.MEDIUM).

        Returns
        -------
        List[Result]
            A list containing one Result object. The .value is a tuple
            (passed_assertions, total_assertions), and .msgs contains error messages
            if the dimension is missing or cannot be retrieved.
    """
    check_id = "DIM003"
    ctx = TestCtx(severity, f"[{check_id}] Dimension Size is greather than '{lower_bound}': '{dimension_name}'")

    dim_size = ds.dimensions[dimension_name].size  # no check of value existence, it has to be done elsewhere
    if dim_size < lower_bound:
        ctx.add_failure(
            f"Unexpected dimension size for '{dimension_name}'. Got {dim_size}, expected strictly greater than {lower_bound}.")
    else:
        ctx.add_pass()

    return [ctx.to_result()]
