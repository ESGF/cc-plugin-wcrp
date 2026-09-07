#!/usr/bin/env python


from compliance_checker.base import BaseCheck, TestCtx


def check_format(ds, expected_format, allowed_data_models, severity=BaseCheck.MEDIUM):
    """
    Checks whether the dataset uses the expected disk format and one of the
    accepted NetCDF data models.

    Parameters
    ----------
    ds : netCDF4.Dataset
        The dataset being checked.
    expected_format : str
        The expected disk format, e.g. "HDF5".
    allowed_data_models : list[str]
        A list of accepted NetCDF data models, e.g.
        ["NETCDF4_CLASSIC", "NETCDF4"].
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    """
    check_id = "FILE002"
    desc = f"[{check_id}] File Format"
    testctx = TestCtx(severity, desc)

    if ds.disk_format != expected_format or ds.data_model not in allowed_data_models:
        testctx.add_failure(
            f"File format differs from expectation "
            f"(allowed data models={allowed_data_models}, expected disk format={expected_format}): "
            f"'{ds.data_model}/{ds.disk_format}'."
        )
    else:
        testctx.add_pass()

    return [testctx.to_result()]
