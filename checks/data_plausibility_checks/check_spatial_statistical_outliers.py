#!/usr/bin/env python
"""
check_spatial_statistical_outliers.py

Check for outliers in the specified netCDF dataset based on the Z-Score along specific dimensions.
"""

from compliance_checker.base import BaseCheck, TestCtx
import numpy as np
import numpy.ma as ma

from checks.data_plausibility_checks.utils.dimensions import (
    get_filtered_dimensions,
    get_dimension_indices,
    get_var_dimensions,
)
from checks.data_plausibility_checks.utils.data import (
    check_variable_conditions,
    check_variable_conditions_expanded
)
from checks.data_plausibility_checks.utils.auxiliar import(
    ExtendedTestCtx,
    dump_data_file_extended,
    Coordinate,
)

def calculate_iqr(data_slice):
    """Calculate the IQR for a given data slice."""
    if isinstance(data_slice, ma.MaskedArray):
        data_slice = data_slice.filled(np.nan)

    q1 = np.percentile(data_slice, 25)
    q3 = np.percentile(data_slice, 75)
    iqr = q3 - q1
    return iqr, q1, q3

def is_outlier_iqr(data_slice, iqr, q1, q3, threshold=1.5):
    """Determine if a value in the data slice is an outlier based on IQR."""
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data_slice < lower_bound) | (data_slice > upper_bound)


def calculate_zscore(data_slice):
    """Calculate the Z-Score for a given data slice."""
    mean = np.mean(data_slice)
    std = np.std(data_slice)
    if std == 0:
        return np.zeros_like(data_slice)  # Avoid division by zero
    return (data_slice - mean) / std

def is_outlier(zscore, threshold=5):
    """Determine if a Z-Score is an outlier."""
    return np.abs(zscore) > threshold

def calculate_time_series_max_min(dataset, variable):
    """
    Calculate the maximum and minimum time series along the time axis.

    Parameters:
    - dataset (netCDF4.Dataset): The dataset containing the values to be checked.
    - variable (str): The variable to be checked.

    Returns:
    - tuple: A tuple containing the max and min time series.
    """
    i_time = get_dimension_indices(dataset, variable)["time"]
    variable_data = dataset.variables[variable][:]
    max_time_series = np.max(variable_data, axis=i_time)
    min_time_series = np.min(variable_data, axis=i_time)

    return max_time_series, min_time_series


def extract_coordinates_from_indices(flattened,ctx,parameter):
    coords = [coord for (coord, _, _) in flattened]
    vals = [val1 for (_, val1, _) in flattened]
    scores = [score1 for (_, _, score1) in flattened]
    if coords:
        detected=True
    else:
        detected=False
    ctx.coordinates=[]
    for coord,value,score in zip(coords, vals, scores):
        coord_obj = Coordinate(
            name=parameter.lower(),                
            indices=[coord],
            values=[f"{value},{parameter}:{score}"],
            result=True
        )
        ctx.coordinates.append(coord_obj)
    return ctx,detected
def check_spatial_statistical_outliers(dataset, variable, severity=BaseCheck.MEDIUM, threshold=5, parameter="Z-Score"):
    """
    Check for outliers in a dataset based on Z-Score and IQR, logs their coordinates, and records
    results using ExtendedTestCtx when the condition checked fails

    Parameters:
    - dataset (netCDF4.Dataset): The dataset containing the values to be checked.
    - variable (str): The variable to be checked.
    - parameter (str): The method to use for outlier detection; either "Z-Score" or "IQR".
    - threshold (float): The Z-Score threshold to consider a value an outlier.
    - severity: Severity level for the check.


    Returns:
    - TestCtx: An object containing detailed results of the check, including
      pass/failure status, messages, and coordinates of detected outliers.
    - file: A file containing the coordinates and values of detected outliers is written when the check condition fails.
    """

    ctx = ExtendedTestCtx(
        category=severity,
        description=f"Check for outliers in a dataset based on {parameter} with threshold {threshold}.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_spatial_statistical_outliers",
        parameters={"threshold": threshold, "method": parameter},
        variable=variable,
    )
    # Define condition functions
    def zscore_condition(data_slice):
        zscores = calculate_zscore(data_slice)
        data_mask = is_outlier(zscores, threshold)
        return data_mask, data_slice, zscores

    def iqr_condition(data_slice):
        iqr, q1, q3 = calculate_iqr(data_slice)
        data_mask = is_outlier_iqr(data_slice, iqr, q1, q3, threshold)
        return data_mask,data_slice, iqr

    dim_dict = get_var_dimensions(dataset, variable)
    check_dims = get_filtered_dimensions(dataset, variable)

    try:
        max_ts, min_ts = calculate_time_series_max_min(dataset, variable)
        # Select function according to parameter
        if parameter == "Z-Score":
            condition_function = zscore_condition
        elif parameter == "IQR":
            condition_function = iqr_condition
    except Exception as e:
        ctx.add_failure(f"Error during {parameter} condition check: {e}")

    # if time_dim exists, remove it for spatial checking
    if 't' in dim_dict and dim_dict['t'] in check_dims:
        try:
            check_dims = list(check_dims)  # avoid modifying original
            check_dims.remove(dim_dict['t'])
        except Exception:
            pass
    try:
        # Detect outliers
        if len(check_dims) > 0:
            values_max = check_variable_conditions_expanded(dataset, variable, check_dims, condition_function, max_ts)
            values_min = check_variable_conditions_expanded(dataset, variable, check_dims, condition_function, min_ts)
        else:
            values_max = {"timeseries": {"max": check_variable_conditions(dataset, variable, check_dims, condition_function, max_ts)}}
            values_min = {"timeseries": {"min": check_variable_conditions(dataset, variable, check_dims, condition_function, min_ts)}}

        flattened_min = [item for d in values_min.values() for v in d.values() for item in v]
        flattened_max = [item for d in values_max.values() for v in d.values() for item in v]
        flattened = flattened_min + flattened_max
        ctx, detected = extract_coordinates_from_indices(flattened, ctx, parameter)
    except Exception as e:
        ctx.add_failure(f"Error during outlier detection: {e}")
        return ctx

    # Messages / result
    if detected:
        ctx.add_failure(f"Outliers detected: {len(ctx.coordinates)} points.")
        ctx_min, detected_min = extract_coordinates_from_indices(flattened_min, ctx, parameter)
        if detected_min:
            ctx_min.messages.append(f"Outliers detected in minimum time series: {len(ctx_min.coordinates)} points.")
            dump_data_file_extended(dataset, variable, 'check_spatial_statistical_outliers_min_series', ctx_min, parameter)
            ctx.messages.extend(ctx_min.messages)
        ctx_max, detected_max = extract_coordinates_from_indices(flattened_max, ctx, parameter)
        if detected_max:
            ctx.messages.append(f"Outliers detected in maximum time series: {len(ctx_max.coordinates)} points.")
            dump_data_file_extended(dataset, variable, 'check_spatial_statistical_outliers_max_series', ctx_max, parameter)
            ctx.messages.extend(ctx_max.messages)
        ctx.messages.append(f"No outliers detected in the dataset based on {parameter}.")
        ctx.add_pass()

    return ctx
