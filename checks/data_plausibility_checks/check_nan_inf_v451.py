#!/usr/bin/env python
"""
check_nan_inf.py

Check if the specified netCDF dataset has NaN values in the data along specific dimensions.

Intended to be included in the WCRP plugins.
"""

from compliance_checker.base import BaseCheck, TestCtx
import numpy as np


from checks.data_plausibility_checks.utils.auxiliar import (
    ExtendedTestCtx,
    Coordinate,
    dump_data_file_extended
)

def check_any_nan(data_slice):
    """Check if there are any NaN values in the data slice."""
    return np.any(np.isnan(data_slice))

def check_any_inf(data_slice):
    """Check if there are any infinite values in the data slice."""
    return np.any(np.isinf(data_slice))

def get_nan_coordinates(data_slice):
    """Get the coordinates of all NaN values in the data slice."""
    return np.where(np.isnan(data_slice))

def get_inf_coordinates(data_slice):
    """Get the coordinates of all infinite values in the data slice."""
    return np.where(np.isinf(data_slice))



def check_nan_inf(dataset, variable, parameter="NaN", severity=BaseCheck.MEDIUM):
    """
    Check for NaN or Inf values in a dataset. The function inspects the specified variable
    for the presence of either NaN or Inf values, logs their coordinates, and records
    results using ExtendedTestCtx when the condition checked fails. Special attention is given to _FillValue attributes
    that may be NaN.
    
    Parameters:
    - dataset (netCDF4.Dataset): The dataset containing the variable to be checked.
    - variable (str): The variable to be checked.
    - parameter (str): The type of value to check for; either "NaN" or "Inf".
    - severity : The severity level of the check.

    Returns:
    - TestCtx: An object containing detailed results of the check, including
      pass/failure status, messages, and coordinates of detected outliers.
    - file: A file containing the coordinates and values of detected outliers is written when the check condition fails.
    """
    ctx = ExtendedTestCtx(
        category=severity,
        description=f"Check for {parameter} values in the dataset.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_nan_inf",
        parameters={"parameter": parameter},
        variable=variable,
    )

    var = dataset.variables[variable]
    var.set_auto_mask(False)
    var.set_auto_scale(False)
    data = var[:]
    fill_value = getattr(var, '_FillValue', None)

    if fill_value is not None and np.isnan(fill_value):
        ctx.add_failure("Warning: _FillValue is NaN. See Fill_value check for more information.")
        return ctx
    try:
        if parameter == "NaN":
            check = check_any_nan(data)
        elif parameter == "Inf":
            check = check_any_inf(data)
    except Exception as e:
        ctx.add_failure(f"Error during {parameter} check: {e}")

    if check:
        if parameter == "NaN":
            coords = get_nan_coordinates(data)
        elif parameter == "Inf":
            coords = get_inf_coordinates(data)
        else:
            coords = []

        coord_list = list(zip(*coords))

        for coord in coord_list:
            clean_coord = tuple(map(int, coord))  
            coord_obj = Coordinate(
                name=parameter.lower(),
                indices=[clean_coord],
                values=[True],
                result=True
            )
            ctx.coordinates.append(coord_obj)

        ctx.add_failure(f"{parameter} values detected: {len(coord_list)}")
        dump_data_file_extended(dataset, variable, 'check_nan_inf', ctx)
    else:
        ctx.messages.append(f"No {parameter} detected in the dataset.")
        ctx.add_pass()

    return ctx