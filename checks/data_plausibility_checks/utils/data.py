import numpy as np
import itertools
from checks.data_plausibility_checks.utils.dimensions import get_ds_dimensions
def check_variable_conditions_expanded(
    dataset, variable_name, check_dims, check_func, data=None, parameters=None
):
    """
    Check values for all combinations of additional dimensions (beyond basic ones)
    and collect results of a given function.
    """
    if data is None:
        data = dataset.variables[variable_name][:]
    ds_dims = get_ds_dimensions(dataset)
    basic_dimensions = ["x", "y", "t"]
    basic_dims_name = [value for key, value in ds_dims.items() if key in basic_dimensions]
    additional_dimensions = [dim for dim in check_dims if dim not in basic_dims_name]
    check_dims_slice = [dim for dim in check_dims if dim in basic_dims_name]

    # Generate all combinations for additional dimensions
    combinations = list(itertools.product(*[range(len(dataset.dimensions[dim])) for dim in additional_dimensions]))

    combination_dict = {}
    for combo in combinations:
        combination_name = ', '.join(additional_dimensions)
        combo_key = ', '.join(map(str, combo))

        # Prepare a selection tuple for data slicing
        # Build a full slice for all dimensions, defaulting to slice(None)
        full_slice = [slice(None)] * data.ndim
        for idx, dim in enumerate(additional_dimensions):
            dim_index = dataset.variables[variable_name].dimensions.index(dim)
            full_slice[dim_index] = combo[idx]
        data_slice = data[tuple(full_slice)]
        # Run the check on the selected slice
        combination_dict.setdefault(combination_name, {})[combo_key] = check_variable_conditions(
            dataset, variable_name, check_dims_slice, check_func, data=data_slice, parameters=parameters, combo=combo
        )
    return combination_dict

def check_variable_conditions(
    dataset, variable_name, check_dims, check_func, data=None, parameters=None, combo=None
):
    """
    Apply check_func to all index combinations of check_dims for a variable.
    """
    if data is None:
        variable = dataset.variables[variable_name][:]
    else:
        variable = data

    var_dim_names = dataset.variables[variable_name].dimensions
    check_dim_indices = [var_dim_names.index(dim) for dim in check_dims]
    results = []
    # Determine shape for check_dims
    index_shapes = [variable.shape[i] for i in check_dim_indices]
    for indices in np.ndindex(*index_shapes):
        # Prepare full slice tuple for all dims
        full_slice = [slice(None)] * variable.ndim
        for idx, dim_index in enumerate(check_dim_indices):
            full_slice[dim_index] = indices[idx]
        data_slice = variable[tuple(full_slice)]
        # Apply the check function
        output= check_func(data_slice, parameters) if parameters is not None else check_func(data_slice)
        # Handle different types of outputs from check_func
        if isinstance(output, tuple) and len(output) == 3:
            result, values, scores = output
        else:
            result = output
            values = None
            scores = None

        # Store results with indices
        if isinstance(result, bool):
            if result:
                results.append(indices)
        elif isinstance(result, np.ndarray) and result.dtype == bool:
            true_coords = np.argwhere(result)
            for coord in true_coords:
                if combo is not None:
                    coords_tuple = tuple(int(c) for c in combo+indices + tuple(coord))
                else:
                    coords_tuple = tuple(int(c) for c in indices + tuple(coord))

                value = float(values[tuple(coord)] if values is not None else None)
                if scores is None:
                    score = None
                else:
                    # Step 2: Check if scores is an array or list (indexable)
                    if isinstance(scores, (np.ndarray, list)):
                        # Step 3: If scores is an array or list, index it
                        score = float(scores[tuple(coord)])
                    else:
                        # Step 4: If scores is a scalar, use it directly
                        score = float(scores)

                results.append((coords_tuple, value, score))
        elif  isinstance(result, (int, np.integer)):
            if combo is not None:
                coords_tuple = tuple(int(c) for c in combo + indices)
            else:
                coords_tuple = indices
            value = float(result)
            results.append((coords_tuple, value))
    return results


