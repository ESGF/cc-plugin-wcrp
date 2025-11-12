#!/usr/bin/env python
"""
detect_physically_impossible_outlier.py

Check if the specified netCDF dataset has physically impossible outliers.

Intended to be included in the WCRP plugins.
"""

from compliance_checker.base import BaseCheck, TestCtx
import numpy as np
import numpy.ma as ma
import os
import json


from checks.data_plausibility_checks.utils.dimensions import get_dimension_info
from checks.data_plausibility_checks.utils.auxiliar import ExtendedTestCtx, dump_data_file_extended,Coordinate

def get_thresholds_variable(dataset, thresholds_file):
    """
    Get variables in the dataset that have a standard_name matching any field in the JSON file.

    Parameters:
    - dataset (netCDF4.Dataset): The opened NetCDF dataset.
    - thresholds_file (str): The path to the JSON file containing the outlier definitions.

    Returns:
    - thresholds (dict): The thresholds for the matching variable.
    - matching_var (str): The name of the variable that matches the standard name in the JSON file.

    Raises:
    - ValueError: If no matching variables are found.
    - ValueError: If more than one matching variable is found.
    """
    # Load the JSON file
    thresholds_file = os.path.join(os.path.dirname(__file__), thresholds_file)
    with open(thresholds_file, 'r') as f:
        thresholds = json.load(f)

    # Get the list of standard names from the JSON file
    standard_names = thresholds.keys()

    # List to store matching variables
    matching_vars = []

    # Iterate through the variables in the dataset
    for var_name, var in dataset.variables.items():
        if hasattr(var, 'standard_name'):
            if var.standard_name in standard_names:
                if "threshold" in var_name.lower():
                    continue
                matching_vars.append(var_name)
                standard_name = var.standard_name

    if not matching_vars:
        raise ValueError("No matching variables found in the dataset.")
    if len(matching_vars) > 1:
        raise ValueError("More than one matching variable found in the dataset.")

    thresholds_values = thresholds[standard_name]
    thresholds_values = check_units(dataset, matching_vars[0], thresholds_values)
    return thresholds_values, matching_vars[0]

def check_units(dataset, var, thresholds_values):
    """
    Check and convert units if necessary.

    Parameters:
    - dataset (netCDF4.Dataset): The opened NetCDF dataset.
    - var (str): The name of the variable.
    - thresholds_values (dict): The thresholds for the variable.

    Returns:
    - thresholds_values (dict): The updated thresholds with converted units if necessary.
    """
    if dataset.variables[var].units != thresholds_values["unit"]:
        #temperature
        if dataset.variables[var].units in ["C", "celsius", "Celsius", "c", "°C", "degC", "degrees_Celsius", "degreesC", "degrees_C", "degrees_Celsius"] and thresholds_values["unit"] in ["Kelvin", "K", "kel", "degK", "degrees_Kelvin", "degreesK", "degrees_K", "degrees_Kelvin"]:
            thresholds_values["min"] -= 273.15
            thresholds_values["max"] -= 273.15
            thresholds_values["unit"] = "Celsius"
        #clt,ice,hurs
        elif dataset.variables[var].units in ["%", "percent", "Percent", "perc", "percentage", "Percentage"]  and thresholds_values["unit"] in ["l","1", "fraction", "fractional"]:
            thresholds_values["min"] *= 100
            thresholds_values["max"] *= 100
            thresholds_values["unit"] = "percent"
        #hurs
        elif dataset.variables[var].units in ["gr kg-1"] and thresholds_values["unit"] in ["l","1", "fraction", "fractional"]:
            thresholds_values["min"] *= 1000
            thresholds_values["max"] *= 1000
            thresholds_values["unit"] = "gr kg-1"
        #pr
        elif dataset.variables[var].units in ["mm", "millimeter"] and thresholds_values["unit"] in ["m"]:
            thresholds_values["min"] *= 1000
            thresholds_values["max"] *= 1000
            thresholds_values["unit"] = "mm"
        #rsds,rlds
        elif dataset.variables[var].units in ["J m**-2"] and thresholds_values["unit"] in ["W m**-2"]:
            thresholds_values["min"] *= 3600*24
            thresholds_values["max"] *= 3600*24
            thresholds_values["unit"] = "J m**-2 day**-1"

    return thresholds_values

def detect_outliers(data: np.ndarray, min_threshold: float, max_threshold: float):
    """
    Detect outliers in a numpy array based on predefined thresholds.

    Parameters:
    - data (np.ndarray): The input numpy array to be checked.
    - min_threshold (float): The minimum threshold for the variable.
    - max_threshold (float): The maximum threshold for the variable.

    Returns:
    - outliers (dict): A dictionary containing the coordinates and values of detected outliers.
    """
    mask_min = data < min_threshold
    mask_max = data > max_threshold
    outlier_coords_min = np.column_stack(np.where(mask_min))
    outlier_coords_max = np.column_stack(np.where(mask_max))

    outliers = {
        'coordinates_indices': [],
        'values': [],
        'types': []
    }

    if len(outlier_coords_min) > 0:
        outliers['coordinates_indices'].extend(outlier_coords_min)
        outliers['values'].extend(data[mask_min])
        outliers['types'].extend(['min'] * len(outlier_coords_min))

    if len(outlier_coords_max) > 0:
        outliers['coordinates_indices'].extend(outlier_coords_max)
        outliers['values'].extend(data[mask_max])
        outliers['types'].extend(['max'] * len(outlier_coords_max))

    # Calculate the proportion of outliers
    total_data_points = data.size
    num_outliers = len(outliers['values'])
    proportion_outliers = num_outliers / total_data_points
    #print(outliers["values"])
    outliers['values'] = [int(x) for x in outliers['values']]

    outliers['proportion'] = proportion_outliers
    return outliers

def extract_outlier_coordinates(dataset, results, dim_dict):
    """
    Extract the longitude, latitude, and time values for the outlier indices.

    Parameters:
    - dataset (netCDF4.Dataset): The opened NetCDF dataset.
    - results (dict): The results from the detect_outliers function.
    - lon_dim (str): The name of the longitude dimension.
    - lat_dim (str): The name of the latitude dimension.
    - time_dim (str): The name of the time dimension.

    Returns:
    - outlier_coords_values (np.ndarray): The coordinate values for the outliers.
    """
    outlier_coords = np.array(results["outliers"]["coordinates_indices"])

    # Access the variables for longitude, latitude, and time
    lon = dataset.variables[dim_dict['x']['name']][:]
    lat = dataset.variables[dim_dict['y']['name']][:]
    time = dataset.variables[dim_dict['t']['name']][:]

    outlier_lon = lon[outlier_coords[:, dim_dict['x']['i']]]
    outlier_lat = lat[outlier_coords[:, dim_dict['y']['i']]]
    outlier_time = time[outlier_coords[:, dim_dict['t']['i']]]

    
    coords_list = [outlier_time, outlier_lat, outlier_lon]

    # Extract the outlier indices from the results
    if 'member' in dim_dict and dim_dict['member']:
        mem = dataset.variables["member_id"][:]
        outlier_mem = mem[outlier_coords[:, dim_dict['member']['i']]]
        coords_list.append(outlier_mem)
    if 'z' in dim_dict and dim_dict['z']:
        z = dataset.variables[dim_dict['z']['name']][:] 
        outlier_z = z[outlier_coords[:, dim_dict['z']['i']]]
        coords_list.append(outlier_z)

    if 'member' in dim_dict and dim_dict['member']:
        outlier_mem = mem[outlier_coords[:, dim_dict['member']['i']]] if outlier_coords.shape[1] > 3 else None
    if 'z' in dim_dict and dim_dict['z']:
        outlier_z = z[outlier_coords[:, dim_dict['z']['i']]] if outlier_coords.shape[1] > 4 else None
    # Combine the coordinate values into a single array for each set of outliers
    outlier_coords_values = np.column_stack(coords_list)

    return outlier_coords_values

def prepare_results(outliers, thresholds, dataset, variable, number_limit=100000):
    """
    Prepare the results for outliers detection.

    Parameters:
    - outliers (dict): Containing the outliers information.
    - thresholds (dict): Containing the threshold information.
    - dataset (netCDF4.Dataset): The opened NetCDF dataset.
    - variable (str): The name of the variable being checked.
    - number_limit (int): The maximum number of outliers to report.

    Returns:
    - results (dict): The prepared results.
    - check (bool): Indicating whether outliers were detected.
    """
    dim_dict = get_dimension_info(dataset, variable)

    # Prepare the results
    results = {
        'outliers': {
            'coordinates_indices': outliers['coordinates_indices'],
            'values': outliers['values'],
            'types': outliers['types'],
            'proportion': outliers['proportion']
        },
        'num_outliers': len(outliers['coordinates_indices']),
        'min_threshold': thresholds['min'],
        'max_threshold': thresholds['max']
    }

    if len(results["outliers"]["coordinates_indices"]) > 0:
        outlier_coords_values = extract_outlier_coordinates(dataset, results, dim_dict)
        check = True
    else:
        check = False
        outlier_coords_values = []

    results["outliers"]["coordinates_values"] = outlier_coords_values
    results['variable'] = variable
    results['unit'] = thresholds['unit']

    return results, check



def check_outliers(dataset, thresholds_file='outliers_thresholds.json', severity=BaseCheck.MEDIUM):
    """
    Check for physically impossible outliers in a dataset based on predefined thresholds. 
    The function reads thresholds from a JSON file, detects outliers in the dataset variable,
    and logs the coordinates and values of any detected outliers using ExtendedTestCtx.

    Parameters:
    - dataset (netCDF4.Dataset): The dataset containing the variable to be checked.
    - thresholds_file (str): Path to a JSON file containing the min and max thresholds for the variable.
    - severity : The severity level of the check.

    Returns:
    - TestCtx: An object containing detailed results of the check, including
      pass/failure status, messages, and coordinates of detected outliers.
    - file: If outliers are detected, a file is written containing their coordinates and values.
    """
    ctx = ExtendedTestCtx(
        category=severity,
        description="Check for outliers in a dataset based on predefined thresholds.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_outliers",
        parameters={"thresholds_file": thresholds_file},
    )

    try:
        thresholds, variable = get_thresholds_variable(dataset, thresholds_file)
    except ValueError as e:
        ctx.add_failure(f"Error getting thresholds: {str(e)}")
        return ctx

    data = dataset.variables[variable][:]
    if isinstance(data, ma.MaskedArray):
        data = data.filled(np.nan)
    outliers = detect_outliers(data, thresholds['min'], thresholds['max'])
    try:
        results, check = prepare_results(outliers, thresholds, dataset, variable)
    except Exception as e:
        ctx.add_failure(f"Error preparing results: {e}")
        return ctx

    if check:
        coords_values = results["outliers"]["values"]
        coords_indices = results["outliers"]["coordinates_indices"]

        for idx, value in zip(coords_indices, coords_values):
            clean_idx = tuple(map(int, idx))  # Limpieza de np.int64
            coord_obj = Coordinate(
                name="outlier",
                indices=[clean_idx],
                values=[value],
                result=True
            )
            ctx.coordinates.append(coord_obj)

        ctx.add_failure(
            f"Physically impossible outliers detected: {results['num_outliers']} "
            f"Thresholds: {thresholds}"
        )
        dump_data_file_extended(dataset, variable, 'check_physically_impossible_outliers', ctx)
    else:
        ctx.messages.append("No physically impossible outliers detected in the dataset.")
        ctx.add_pass()

    return ctx