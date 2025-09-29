"""
#!/usr/bin/env python
This module provides an atomic check that verifies whether a specified variable
exists in a netCDF dataset.
"""

from compliance_checker.base import BaseCheck, TestCtx


def check_variable_existence(
    ds,
    var_name,
    severity=BaseCheck.MEDIUM
):
    check_id="VAR001"
    """
    Verify that an variable named 'var_name' exists in the dataset 'ds'.
    Parameters
    ----------
    ds : netCDF4.Dataset
        An already open netCDF dataset.
    var_name : str
        The name of the variable to check for.
    severity : int, optional
        The severity level of this check (default: BaseCheck.MEDIUM).
    check_id : str, optional
        A check identifier included in results, unless None.
        Default is None.
    Returns
    -------
    List[Result]
        A list containing one Result object. The .value is a tuple
        (passed_assertions, total_assertions), and .msgs contains error messages
        if the variable is missing.
    Usage Example:
        from check_variable_existence import check_variable_existence
        results = check_variable_existence(ds, 'time')
    """
    ctx = TestCtx(severity, f"[{check_id}] Variable Existence: '{var_name}'")

    ctx.assert_true(var_name in ds.variables, f"Variable '{var_name}' is missing.")

    return [ctx.to_result()]

