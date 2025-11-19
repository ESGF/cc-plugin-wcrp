#!/usr/bin/env python


import xarray as xr
from compliance_checker.base import BaseCheck, TestCtx

from checks.utils import severity_word


def check_lon_value_range(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if longitude values are within the range required by the CORDEX-CMIP6 Archive Specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXV003"
    desc = f"[{check_id}] "
    testctx = TestCtx(severity, desc)

    if "longitude" in CheckerObject.xrds.cf.coordinates:
        lon = CheckerObject.xrds[CheckerObject.xrds.cf.coordinates["longitude"][0]]
    elif "lon" in CheckerObject.xrds:
        lon = CheckerObject.xrds["lon"]
    else:
        testctx.add_pass()
        return [testctx.to_result()]

    # Check if longitude coordinates are strictly monotonically increasing
    if lon.ndim != 2:
        testctx.add_failure("The longitude coordinate should have two dimensions.")
    else:
        increasing_0 = ((lon[1:, :].data - lon[:-1, :].data) > 0).all()
        increasing_1 = ((lon[:, 1:].data - lon[:, :-1].data) > 0).all()
        if "X" in CheckerObject.xrds.cf.axes:
            rlon_idx = lon.dims.index(CheckerObject.xrds.cf.axes["X"][0])
            if rlon_idx == 0:
                if increasing_0:
                    testctx.add_pass()
                else:
                    testctx.add_failure(
                        "The longitude coordinate should be strictly monotonically increasing."
                    )
            elif rlon_idx == 1:
                if increasing_1:
                    testctx.add_pass()
                else:
                    testctx.add_failure(
                        "The longitude coordinate should be strictly monotonically increasing."
                    )
        elif increasing_0 or increasing_1:
            testctx.add_pass()
        else:
            testctx.add_failure(
                "The longitude coordinate should be strictly monotonically increasing."
                f"{increasing_0}, {increasing_1}"
            )

    # Check if longitude coordinates are confined to the range -180 to 360
    in_range = (lon >= -180).all() and (lon <= 360).all()
    if in_range:
        testctx.add_pass()
    else:
        testctx.add_failure(
            "Longitude coordinates should be confined to the range -180 to 360."
        )

    # Check if longitude coordinates have absolute values as small as possible
    abs = (lon > 180).any() and (xr.where(lon >= 180, lon - 360, lon) >= -180).all()
    if not abs:
        testctx.add_pass()
    else:
        testctx.add_failure(
            "Longitude values are required to take the smalles absolute value in the range [-180, 360]."
        )

    return [testctx.to_result()]


def check_horizontal_axes_bounds(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if rlat/rlon bounds or x/y bounds are present as recommended in the CORDEX-CMIP6 Archive Specifications.

    Args
    ----
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXV002"
    desc = f"[{check_id}] Existence of horizontal axes bounds"
    testctx = TestCtx(severity, desc)

    if "X" in CheckerObject.xrds.cf.bounds and "Y" in CheckerObject.xrds.cf.bounds:
        testctx.add_pass()
    elif ("rlat_bnds" in CheckerObject.xrds and "rlon_bnds" in CheckerObject.xrds) or (
        "x_bnds" in CheckerObject.xrds and "y_bnds" in CheckerObject.xrds
    ):
        testctx.add_pass()
    else:
        testctx.add_failure(
            f"It is {severity_word(severity)} for the variables 'rlat' and 'rlon' or 'x' and 'y' to have bounds defined."
        )

    return [testctx.to_result()]


def check_lat_lon_bounds(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if lat and lon bounds are present as recommended in the CORDEX-CMIP6 Archive Specifications.

    Args
    ----
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXV001"
    desc = f"[{check_id}] Existence of latitude and longitude bounds"
    testctx = TestCtx(severity, desc)

    if (
        "longitude" in CheckerObject.xrds.cf.bounds
        and "latitude" in CheckerObject.xrds.cf.bounds
    ):
        testctx.add_pass()
    elif ("lat_bnds" in CheckerObject.xrds and "lon_bnds" in CheckerObject.xrds) or (
        "vertices_lat" in CheckerObject.xrds and "vertices_lon" in CheckerObject.xrds
    ):
        testctx.add_pass()
    else:
        testctx.add_failure(
            f"It is {severity_word(severity)} for the variables 'lat' and 'lon' to have bounds defined."
        )

    return [testctx.to_result()]
