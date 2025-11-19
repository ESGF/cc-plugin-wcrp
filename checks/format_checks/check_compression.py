#!/usr/bin/env python


from compliance_checker.base import BaseCheck, TestCtx

from checks.utils import severity_word


def check_compression(
    ds,
    variable_name=None,
    expected_complevel=1,
    expected_shuffle=True,
    severity=BaseCheck.MEDIUM,
):
    """
    Checks if the main variable is compressed in the recommended way.

    Parameters
    ----------
    ds : netCDF4.Dataset
        The dataset to check.
    variable_name : str, optional
        Name of the variable to check compression for. If None, the first data variable is used.
    expected_complevel : int, optional
        Recommended compression level (default: 1).
    expected_shuffle : bool, optional
        Whether shuffle filter should be enabled (default: True).
    severity : str
        The severity level (BaseCheck.LOW, MEDIUM, HIGH). Default: BaseCheck.MEDIUM.

    Returns
    -------
    list[compliance_checker.base.Result]
    """
    check_id = "FILE003"
    desc = f"[{check_id}] Compression"
    testctx = TestCtx(severity, desc)

    qualifier = severity_word(severity)  # "required", "recommended", or "suggested"
    # print(variable_name, expected_complevel, expected_shuffle, severity)
    # Select a data variable if not specified
    if variable_name is None:
        vars_non_dim = [v for v in ds.variables if v not in ds.dimensions]
        variable_name = vars_non_dim[0] if vars_non_dim else None

    if variable_name is None:
        testctx.add_pass("No data variable found, skipping compression check.")
        return [testctx.to_result()]

    # Retrieve compression info
    complevel = ds[variable_name].filters()["complevel"]
    shuffle = ds[variable_name].filters()["shuffle"]

    if complevel is None or shuffle is None:
        testctx.add_failure(
            f"It is {qualifier} that data variable be compressed with a 'deflate level' of '{expected_complevel}'"
            f"""{"and with the 'shuffle' option enabled." if expected_shuffle else "."} """
            "The data appears uncompressed."
        )
        return [testctx.to_result()]

    # Original failure logic preserved, only the phrasing adapts
    if complevel != expected_complevel or shuffle is not expected_shuffle:
        msg = (
            f"It is {qualifier} that data be compressed with a 'deflate level' of '1' "
            f"and with the 'shuffle' option enabled."
        )
        if complevel == 0 and expected_complevel > 0:
            msg += " The data is uncompressed."
        elif complevel < expected_complevel:
            msg += (
                f" The data is compressed with a lower 'deflate level': '{complevel}'."
            )
        elif complevel > expected_complevel:
            msg += (
                f" The data is compressed with a higher 'deflate level': '{complevel}'. "
                "This can lead to performance issues when accessing the data."
            )
        if shuffle is False:
            msg += " The 'shuffle' option is disabled."
        testctx.add_failure(msg)
    else:
        testctx.add_pass()

    return [testctx.to_result()]
