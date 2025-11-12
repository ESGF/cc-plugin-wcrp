#!/usr/bin/env python
"""
check_fill_missing.py

Check if the specified netCDF dataset has FillValue or missing values in the data along specific dimensions.

Intended to be included in the WCRP plugins.
"""

from compliance_checker.base import BaseCheck, TestCtx
import numpy as np
import numpy.ma as ma


from checks.data_plausibility_checks.utils.dimensions import get_filtered_dimensions
from checks.data_plausibility_checks.utils.data import check_variable_conditions,check_variable_conditions_expanded
from checks.data_plausibility_checks.utils.auxiliar import (dump_data_file_extended,
                            ExtendedTestCtx
                            )
def detect_changes_in_values(coordinate_values):
    """
    Detects coordinates where the values change in a list of coordinate-value pairs.

    Parameters:
    coordinate_values (list of tuples): A list of tuples where each tuple is (coordinate, value).

    Returns:
    list: A list of tuples where the values change.
    """
    if not coordinate_values:
        return []

    coordinate_values = [(coords, int(val)) for coords, val in coordinate_values]
    coordinates, values = zip(*coordinate_values)
    values_array = np.array(values)

    # Compute the differences between consecutive values
    diff_values = np.diff(values_array)

    # Find the indices where the differences are not zero
    non_zero_diff_indices = np.nonzero(diff_values)[0]

    # Adjust indices to match the original array
    non_zero_diff_indices = non_zero_diff_indices + 1
        
    # Get the coordinates where the values are not the same
    detected = [coordinate_values[i] for i in non_zero_diff_indices]
    # Include the first element if it's different from the second
    if len(values_array) > 1 and values_array[0] != values_array[1]:
        detected.insert(0, coordinate_values[0])
    return detected
def check_value(data_slice, parameters):
    val = parameters['val']
    val_name = parameters['name']
    if val_name == 'FillValue' or val_name == 'MissingValue':
        if isinstance(data_slice, ma.MaskedArray):
            filled_data = data_slice.filled(fill_value=val)
            return np.sum(filled_data == val)
        else:
            return np.sum(data_slice == val)

def load_value_to_check(var_obj, parameter, ctx):
    # Load the FillValue or MissingValue from the variable attributes
    fill_value = getattr(var_obj, '_FillValue', None)
    missing_value = getattr(var_obj, 'missing_value', None)

    if fill_value is not None and missing_value is not None and fill_value == missing_value:
        ctx.messages.append("Warning: _FillValue and missing_value are the same.")
    if parameter == "FillValue":
        parameters_func = {'val': fill_value, "name": "FillValue"}
        if fill_value is None:
            raise ValueError("FillValue not found in the variable attributes.")  
    elif parameter == "MissingValue":
        parameters_func = {'val': missing_value, "name": "MissingValue"}
    else:
        raise ValueError(f"Invalid parameter {parameter}")
    return parameters_func, ctx


def check_fillvalues_timeseries(
    dataset, variable, parameter="FillValue", severity=BaseCheck.MEDIUM
):
    """
    Check for FillValue or MissingValue in a dataset. First it checks if the value is present, then it checks if the number of the checked value is constant along the time series,
    logs their coordinates, and records results using ExtendedTestCtx when the condition checked fails.
    
    Parameters:
    - dataset (netCDF4.Dataset): The dataset containing the values to be checked.
    - variable (str): The variable to be checked.
    - parameter (str): The parameter to check, either "FillValue" or "MissingValue".
    - severity : The severity level of the check.

    Returns:
    - TestCtx: An object containing detailed results of the check, including
      pass/failure status, messages, and coordinates of detected outliers.
    - file: A file containing the coordinates and values of detected outliers is written when the check condition fails.
    """
    ctx = ExtendedTestCtx(
        category=severity,
        description=f"Check for {parameter} in a dataset.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_fillvalues_timeseries",
        parameters={"parameter": parameter},
        variable=variable,
    )

    check_dims = get_filtered_dimensions(dataset, variable)
    var_obj = dataset.variables[variable]
    parameters_func, ctx = load_value_to_check(var_obj, parameter, ctx)
    if parameters_func["val"] is None:
        ctx.add_pass()
        ctx.messages.append(f"{parameter} not found in the variable attributes.")
        return ctx

    try:
        # Detect failed coordinates
        if len(check_dims) > 1:
            failing_coords = check_variable_conditions_expanded(
                dataset, variable, check_dims, check_value, parameters=parameters_func
            )
        else:
            failing_coords = {
                parameter: {
                    "None": check_variable_conditions(
                        dataset, variable, check_dims, check_value, parameters=parameters_func
                    )
                }
            }

        flattened= [item for d in failing_coords.values() for v in d.values() for item in v]
        #check if any fillvalue/missing value detected
        if len(flattened) > 0:
            check = True
        else:
            check = False
        #check if fillvalue/missing value are constant
        detected_diff=detect_changes_in_values(flattened)
        if len(detected_diff) > 0:
            check_diff_flag = True
        else:
            check_diff_flag = False
        #Preparing output for each case
    except Exception as e:
        ctx.add_failure(f"Error during {parameter} check: {e}")
        return ctx
    
    if check and check_diff_flag==False:
        total_coords = [coord for (coord, _) in flattened]
        vals = [val1 for (_, val1) in flattened]
        for coord,value in zip(total_coords, vals):
            ctx.add_coordinate(
                                name=",".join(check_dims),
                                indices=[coord],
                                values=[value],)
        ctx.add_pass()
        ctx.messages.append(
        f"{parameter} detected in the dataset. "
        f"{parameter} are constant. "
        f"Number of {parameter}: {len(flattened)}.")

    elif check and check_diff_flag:
        total_coords = [coord for (coord, _) in detected_diff]
        vals = [val1 for (_, val1) in detected_diff]
        for coord,value in zip(total_coords, vals):
            ctx.add_coordinate(
                                name=",".join(check_dims),
                                indices=[coord],
                                values=[value],)
        message = (
            f"{parameter} detected in the dataset. "
            f"{parameter} are not constant. "
            f"Number of difference in {parameter}: {len(detected_diff)}. "
        )
        ctx.add_failure(message)
        dump_data_file_extended(dataset, variable, "check_fillvalues", ctx)
    else:
        ctx.add_pass()
        ctx.messages.append(f"No anomalous {parameter} detected in the dataset.")

    return ctx
