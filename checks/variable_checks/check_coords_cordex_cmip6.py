#!/usr/bin/env python


import numpy as np
from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util as cfutil

from checks.utils import crosses_anti_meridian, crosses_zero_meridian, severity_word

# Compliance Checker 6 moved these helpers from compliance_checker.cfutil.
if not hasattr(cfutil, "get_geophysical_variables"):
    from compliance_checker import cfutil


def _cf_names(dataset, function):
    """Return names discovered by a Compliance Checker CF utility."""
    return list(function(dataset) or [])


def _true_horizontal_coordinates(dataset):
    """Return the CF-identified true latitude and longitude variables."""
    latitudes = _cf_names(dataset, cfutil.get_true_latitude_variables)
    longitudes = _cf_names(dataset, cfutil.get_true_longitude_variables)
    return latitudes, longitudes


def _has_1d_lat_lon(dataset):
    """Whether CF discovery finds one-dimensional true lat/lon coordinates."""
    latitudes, longitudes = _true_horizontal_coordinates(dataset)
    return bool(
        latitudes
        and longitudes
        and dataset.variables[latitudes[0]].ndim == 1
        and dataset.variables[longitudes[0]].ndim == 1
    )


def _cell_boundary_map(dataset):
    """Return mappings from coordinate names to their bounds variables."""
    return cfutil.get_cell_boundary_map(dataset)


def _axis_names(CheckerObject, axis):
    """Find coordinate names belonging to an axis of the main variable."""
    names = []
    for variable in CheckerObject.varname:
        if variable not in CheckerObject.ds.variables:
            continue
        names.extend(cfutil.get_axis_map(CheckerObject.ds, variable).get(axis, []))
    for name in _cf_names(CheckerObject.ds, cfutil.get_axis_variables):
        if getattr(CheckerObject.ds.variables[name], "axis", None) == axis:
            names.append(name)
    return list(dict.fromkeys(names))


def _all_true(values):
    """Reduce a possibly masked boolean array to an ordinary bool."""
    result = np.ma.all(values)
    return False if np.ma.is_masked(result) else bool(result)


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

    _, longitudes = _true_horizontal_coordinates(CheckerObject.ds)
    if not longitudes:
        testctx.add_pass()
        return [testctx.to_result()]
    lon = CheckerObject.ds.variables[longitudes[0]]
    lon_values = np.ma.asarray(lon[:])

    grid_mapping_name = False
    if len(CheckerObject.varname) > 0:
        crs = getattr(
            CheckerObject.ds.variables[CheckerObject.varname[0]], "grid_mapping", False
        )
        if crs in CheckerObject.ds.variables:
            grid_mapping_name = getattr(
                CheckerObject.ds.variables[crs], "grid_mapping_name", False
            )
        else:
            # If no grid_mapping is present, but lat and lon are 1D, we assume it's a latitude_longitude grid
            if _has_1d_lat_lon(CheckerObject.ds):
                grid_mapping_name = "latitude_longitude"

    # Get domain_id from global attributes
    domain_id = CheckerObject._get_attr("domain_id", default="")
    if not isinstance(domain_id, str):
        domain_id = ""

    # Check if longitude coordinates are strictly monotonically increasing
    if grid_mapping_name == "latitude_longitude":
        if lon.ndim != 1:
            testctx.add_failure(
                "The longitude coordinate should have one dimension for grid_mapping_name"
                " 'latitude_longitude'."
            )
        elif _all_true((lon_values[1:] - lon_values[:-1]) > 0):
            testctx.add_pass()
        else:
            testctx.add_failure(
                "The longitude coordinate should be strictly monotonically increasing."
            )
    elif lon.ndim != 2:
        testctx.add_failure("The longitude coordinate should have two dimensions.")
    elif (
        domain_id.startswith("ARC")
        or domain_id.startswith("ANT")
        or (crosses_anti_meridian(lon_values) and crosses_zero_meridian(lon_values))
    ):
        # The polar domains are exempt from monotony tests because they cross both the meridian and anti-meridian
        testctx.add_pass()
    else:
        increasing_0 = _all_true((lon_values[1:, :] - lon_values[:-1, :]) > 0)
        increasing_1 = _all_true((lon_values[:, 1:] - lon_values[:, :-1]) > 0)
        x_dimensions = [
            name for name in _axis_names(CheckerObject, "X") if name in lon.dimensions
        ]
        if x_dimensions:
            rlon_idx = lon.dimensions.index(x_dimensions[0])
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
    in_range = _all_true(lon_values >= -180) and _all_true(lon_values <= 360)
    if in_range:
        testctx.add_pass()
    else:
        testctx.add_failure(
            "Longitude coordinates should be confined to the range -180 to 360."
        )

    # Check if longitude coordinates have absolute values as small as possible
    # If values are monotonic increasing, only the case 180 <= lon [< 360] is problematic
    valid_lon_values = lon_values.compressed()
    if valid_lon_values.size and valid_lon_values.min() >= 180:
        testctx.add_failure(
            "Longitude values are required to take the smallest absolute value in the range [-180, 360]."
        )
    else:
        testctx.add_pass()

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

    # Check if we have 1D lat/lon (regular lat/lon grid)
    has_1d_lat_lon = _has_1d_lat_lon(CheckerObject.ds)
    bounds_map = _cell_boundary_map(CheckerObject.ds)

    grid_mapping_name = False
    if len(CheckerObject.varname) > 0:
        crs = getattr(
            CheckerObject.ds.variables[CheckerObject.varname[0]], "grid_mapping", False
        )
        if crs in CheckerObject.ds.variables:
            grid_mapping_name = getattr(
                CheckerObject.ds.variables[crs], "grid_mapping_name", False
            )
        else:
            if has_1d_lat_lon:
                grid_mapping_name = "latitude_longitude"

    if grid_mapping_name == "latitude_longitude" or has_1d_lat_lon:
        # Check if lat/lon bounds are defined (since they are the native horizontal axes)
        latitudes, longitudes = _true_horizontal_coordinates(CheckerObject.ds)
        if (
            latitudes
            and longitudes
            and latitudes[0] in bounds_map
            and longitudes[0] in bounds_map
        ) or (
            (
                "lat_bnds" in CheckerObject.ds.variables
                and "lon_bnds" in CheckerObject.ds.variables
            )
            or (
                "vertices_lat" in CheckerObject.ds.variables
                and "vertices_lon" in CheckerObject.ds.variables
            )
        ):
            testctx.add_pass()
        else:
            testctx.add_failure(
                f"It is {severity_word(severity)} for the horizontal axes variables 'lat' and 'lon' to have bounds defined."
            )
        return [testctx.to_result()]

    x_axes = _axis_names(CheckerObject, "X")
    y_axes = _axis_names(CheckerObject, "Y")
    if any(name in bounds_map for name in x_axes) and any(
        name in bounds_map for name in y_axes
    ):
        testctx.add_pass()
    elif (
        "rlat_bnds" in CheckerObject.ds.variables
        and "rlon_bnds" in CheckerObject.ds.variables
    ) or (
        "x_bnds" in CheckerObject.ds.variables
        and "y_bnds" in CheckerObject.ds.variables
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

    # If lat/lon are 1D, CDXV001 should pass early because there are no 2D lat/lon fields.
    # The 1D lat/lon bounds check will be handled by CDXV002 (horizontal axes bounds).
    has_1d_lat_lon = _has_1d_lat_lon(CheckerObject.ds)

    if has_1d_lat_lon:
        testctx.add_pass()
        return [testctx.to_result()]

    latitudes, longitudes = _true_horizontal_coordinates(CheckerObject.ds)
    bounds_map = _cell_boundary_map(CheckerObject.ds)
    if (
        latitudes
        and longitudes
        and latitudes[0] in bounds_map
        and longitudes[0] in bounds_map
    ):
        testctx.add_pass()
    elif (
        "lat_bnds" in CheckerObject.ds.variables
        and "lon_bnds" in CheckerObject.ds.variables
    ) or (
        "vertices_lat" in CheckerObject.ds.variables
        and "vertices_lon" in CheckerObject.ds.variables
    ):
        testctx.add_pass()
    else:
        testctx.add_failure(
            f"It is {severity_word(severity)} for the variables 'lat' and 'lon' to have bounds defined."
        )

    return [testctx.to_result()]
