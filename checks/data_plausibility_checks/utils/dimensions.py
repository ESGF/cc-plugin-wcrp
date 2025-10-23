
def get_ds_dimensions(dataset):
    """
    Get the names of the longitude, latitude, time, and z dimensions.
    Check for variables with axis attributes or common dimension names.

    Parameters:
    - dataset: netCDF4.Dataset object, the opened NetCDF dataset.

    Returns:
    - dict: A dictionary with keys 'x', 'y', 't', and 'z'.
    """
    # Constants for axis names
    AXIS_NAMES = ("X", "Y", "Z", "T")

    # Initialize dimension names
    dimensions = {
        'x': None,
        'y': None,
        't': None,
        'z': None,
        'member': None,
    }

    # Check for variables with axis attributes or common names
    for var_name, var in dataset.variables.items():
        # Check axis attribute
        if hasattr(var, 'axis'):
            axis = var.axis
            if axis in AXIS_NAMES:
                if axis == 'X':
                    dimensions['x'] = var_name
                elif axis == 'Y':
                    dimensions['y'] = var_name
                elif axis == 'T':
                    dimensions['time'] = var_name
                elif axis == 'Z':
                    dimensions['z'] = var_name

        # Check common dimension names if axis attribute is not present
        if dimensions['x'] is None and var_name.lower() in ['lon', 'longitude',"x"]:
            dimensions['x'] = var_name
        if dimensions['y'] is None and var_name.lower() in ['lat', 'latitude',"y"]:
            dimensions['y'] = var_name
        if dimensions['t'] is None and var_name.lower() in ['time']:
            dimensions['t'] = var_name
        if dimensions['z'] is None and var_name.lower() in ['z', 'depth', 'level']:
            dimensions['z'] = var_name

    if 'member' in dataset.dimensions:
        dimensions['member'] = 'member'
    # Raise an error if any of the essential dimensions are not found
    if None in [dimensions['x'], dimensions['y'], dimensions['time']]:
        raise ValueError("Could not determine longitude, latitude, or time dimensions.")
    # Drop keys with None values before returning
    dimensions = {k: v for k, v in dimensions.items() if v is not None}
    return dimensions


def get_filtered_dimensions(dataset, variable_name):
    """
    Get the filtered dimensions of a variable in a NetCDF file, excluding longitude and latitude dimensions.

    Parameters:
    - dataset: netCDF4.Dataset object, the opened NetCDF dataset.
    - variable_name: str, the name of the variable in the NetCDF file.

    Returns:
    - filtered_dimensions: list, the filtered list of dimension names.
    """

    # Choose a variable
    variable = dataset.variables[variable_name]

    # Get the longitude, latitude, and time dimensions
    dimensions= get_ds_dimensions(dataset)
    x_dim = dimensions['x']
    y_dim = dimensions['y']
 
    # Get the list of dimension names for the variable
    dimension_names = list(variable.dimensions)

    # Filter out the longitude and latitude dimensions
    filtered_dimensions = [dim for dim in dimension_names if dim not in [x_dim, y_dim]]

    return filtered_dimensions



def get_var_dimensions(dataset,variable):
    """
    Compare the dimensions of a specific variable with the dataset dimensions.

    Parameters:
    - dataset: netCDF4.Dataset object, the opened NetCDF dataset.
    - variable: str, the name of the variable to compare dimensions.

    Returns:
    - dict: A dictionary with the same keys as dataset_dims but only with the dimensions that exist in the variable.

    Raises:
    - ValueError: If the variable dimensions are not a subset of the dataset dimensions.
    """

    var_dims=dataset.variables[variable].dimensions
    ds_dims=get_ds_dimensions(dataset)
    # Check if variable dimensions are a subset of dataset dimensions
    if not all(dim in ds_dims.values() for dim in var_dims):
        raise ValueError("The variable dimensions are not a subset of the dataset dimensions.")

    # Create a dictionary with the same keys as dataset_dims but only with the dimensions that exist in the variable
    common_dims = {key: value for key, value in ds_dims.items() if value in var_dims}

    return common_dims

def get_dimension_indices(dataset, variable):
    """
    Get the index positions of the dimensions for a specific variable in a NetCDF dataset.

    Parameters:
    - dataset: netCDF4.Dataset object, the opened NetCDF dataset.
    - variable: str, the name of the variable to get dimension indices for.

    Returns:
    - dict: A dictionary with dimension names as keys and their index positions as values.
    """
    # Get the variable object
    var = dataset.variables[variable]

    # Get the dimension names of the variable
    dim_names = var.dimensions

    # Create a dictionary to store dimension indices
    dim_indices = {dim_name: idx for idx, dim_name in enumerate(dim_names)}

    return dim_indices

def get_dimension_info(dataset, variable):
    """
    Get a dictionary with dimension names and the indices for a specific variable.

    Parameters:
    - dataset: netCDF4.Dataset object, the opened NetCDF dataset.
    - variable: str, the name of the variable to get dimension info for.

    Returns:
    - dict: A dictionary with keys from ds_dims and values as {'name': dimension_name, 'index': dimension_index}.
    """
    ds_dims = get_ds_dimensions(dataset)
    dim_indices = get_dimension_indices(dataset, variable)

    dimension_info = {}

    for key, dim_name in ds_dims.items():
        if dim_name in dim_indices:
            dimension_info[key] = {
                'name': dim_name,
                'i': dim_indices[dim_name]
            }

    return dimension_info


