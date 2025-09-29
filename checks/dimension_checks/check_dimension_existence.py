#!/usr/bin/env python

"""
check_dimension_existence.py

Check if a specified dimension exists in a netCDF dataset.

"""

from compliance_checker.base import BaseCheck, TestCtx


def check_dimension_existence(ds, dimension_name, severity): 
    check_id="DIM001"
    """
        Verify if the given dimension exists in the dataset 'ds'.

        Parameters
        ----------
        ds : netCDF4.Dataset
            An already open netCDF dataset.
        dimension_name : str
            The name of the dimension to check.
        severity : int, optional
            The severity level of this check (default: BaseCheck.MEDIUM).

        Returns
        -------
        List[Result]
            A list containing one Result object. The .value is a tuple
            (passed_assertions, total_assertions), and .msgs contains error messages
            if the dimension is missing or cannot be retrieved.
    """
    ctx = TestCtx(
        severity,
        f"[{check_id}] Dimension Existence: '{dimension_name}'"
    )

    if dimension_name not in ds.dimensions:
        ctx.add_failure(f"Dimension '{dimension_name}' is missing.")
    else:
        ctx.add_pass()

    return [ctx.to_result()]
