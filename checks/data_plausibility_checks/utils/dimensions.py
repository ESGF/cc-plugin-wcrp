
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
        var_dims = tuple(getattr(var, 'dimensions', ()))
        x_dim = var_dims[-1] if var_dims else var_name
        y_dim = var_dims[-2] if len(var_dims) > 1 else (
            var_dims[0] if var_dims else var_name
        )
        primary_dim = var_dims[0] if var_dims else var_name
        # Check axis attribute
        if hasattr(var, 'axis'):
            axis = var.axis
            if axis in AXIS_NAMES:
                if axis == 'X':
                    dimensions['x'] = x_dim
                elif axis == 'Y':
                    dimensions['y'] = y_dim
                elif axis == 'T':
                    dimensions['t'] = primary_dim
                elif axis == 'Z':
                    dimensions['z'] = primary_dim

        # Curvilinear ocean grids generally expose auxiliary nav_lon/nav_lat
        # variables. Use their CF identity and return the dimensions beneath
        # them rather than the auxiliary-variable names.
        standard_name = str(getattr(var, 'standard_name', '')).lower()
        if dimensions['x'] is None and standard_name in {
            'longitude', 'grid_longitude', 'projection_x_coordinate'
        }:
            dimensions['x'] = x_dim
        if dimensions['y'] is None and standard_name in {
            'latitude', 'grid_latitude', 'projection_y_coordinate'
        }:
            dimensions['y'] = y_dim
        if dimensions['t'] is None and standard_name == 'time':
            dimensions['t'] = primary_dim
        if dimensions['z'] is None and str(
            getattr(var, 'positive', '')
        ).lower() in {'up', 'down'}:
            dimensions['z'] = primary_dim

        # Check common dimension names if axis attribute is not present
        if dimensions['x'] is None and var_name.lower() in ['lon', 'longitude',"x"]:
            dimensions['x'] = x_dim
        if dimensions['y'] is None and var_name.lower() in ['lat', 'latitude',"y"]:
            dimensions['y'] = y_dim
        if dimensions['t'] is None and var_name.lower() in ['time']:
            dimensions['t'] = primary_dim
        if dimensions['z'] is None and var_name.lower() in [
            'z', 'depth', 'level', 'lev', 'olevel', 'plev'
        ]:
            dimensions['z'] = primary_dim

    if 'member' in dataset.dimensions:
        dimensions['member'] = 'member'
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
    horizontal_dims = {dimensions.get('x'), dimensions.get('y')}
    horizontal_dims.discard(None)
 
    # Get the list of dimension names for the variable
    dimension_names = list(variable.dimensions)

    # Filter out the longitude and latitude dimensions
    filtered_dimensions = [dim for dim in dimension_names if dim not in horizontal_dims]

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


