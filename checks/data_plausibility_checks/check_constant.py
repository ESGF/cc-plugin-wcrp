#!/usr/bin/env python
"""
check_constants.py

Check if the specified netCDF dataset has constant values in the data along specific dimensions.

Intended to be included in the WCRP plugins.
"""

from compliance_checker.base import BaseCheck, TestCtx
import numpy as np

from checks.data_plausibility_checks.utils.dimensions import get_filtered_dimensions
from checks.data_plausibility_checks.utils.data import check_variable_conditions
from checks.data_plausibility_checks.utils.auxiliar import (
                        ExtendedTestCtx,
                        dump_data_file_extended,
                        Coordinate)

def check_all_constant(data_slice):
    
    if np.isscalar(data_slice):
        return True
    elif data_slice.size == 0:
        return False
    else:
        # If it's an array, check if all values are equal to the first one
        return bool(np.all(data_slice == data_slice.flat[0]))




def check_constants(dataset, variable, severity=BaseCheck.MEDIUM):
    """
    Check for constant values in 2d slices of the specified variable in the netCDF dataset.
    Parameters:
    - dataset (netCDF4.Dataset): The dataset containing the values to be checked.
    - variable (str): The variable to be checked.
    Returns:
    - TestCtx: An object containing detailed results of the check, including
      pass/failure status, messages, and coordinates of detected outliers.
    - file: A file containing the coordinates and values of detected outliers is written when the check condition fails.
    """
    ctx = ExtendedTestCtx(
        category=severity,
        description="Check for constant values in the dataset.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_constants",
        parameters={},
        variable=variable,
    )
    try:
        check_dims = get_filtered_dimensions(dataset, variable)
        values = check_variable_conditions(dataset, variable, check_dims, check_all_constant)
        detected = [coord for coord in values if bool]
    except Exception as e:
        ctx.add_failure(f"Error during constant value check: {e}")
        return ctx
    if len(detected) > 0:
        for coord in detected:
            coord_obj = Coordinate(
                name="constant_values",
                indices=[coord],
                values=np.unique([dataset[variable][coord]]),
                result=True
            )
            ctx.coordinates.append(coord_obj)

        num_constants = len(detected)
        ctx.add_failure(f"Constant values detected: {num_constants}")
        dump_data_file_extended(dataset, variable, 'check_constant', ctx)
    else:
        ctx.add_pass()
        ctx.messages.append("No constant values detected in the dataset.")

    return ctx
