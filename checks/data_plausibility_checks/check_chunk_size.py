#!/usr/bin/env python
"""
check_chunk_size.py

Check if chunksize of the time dimension for a netCDF dataset is greater than 1.

Intended to be included in the WCRP plugins.
"""

from compliance_checker.base import BaseCheck, TestCtx
from checks.data_plausibility_checks.utils.dimensions import get_filtered_dimensions, get_ds_dimensions

def chunk_sizes_conditions(ds, variable='time'):
    """
    Check if the chunk size of a netCDF dataset is greater than 1.

    Returns:
     - int: 
        - 0 if chunk size equals the length of the time dimension (passes)
        - 1 if chunk size does not equal the length (fails)
        - 2 if chunking is not defined
    """
    time_var = ds.variables[variable]
    time_len = len(time_var)
    time_chunks = time_var.chunking() if hasattr(time_var, "chunking") else None
    if time_chunks is None:
        # print(f"[WARNING] The time variable '{variable}' does not have chunking information.")
        return 2, None, time_len
    else:
        time_chunk_size = time_chunks[0]
        if time_chunk_size == time_len:
            return 0, time_chunk_size, time_len  # Pass
        else:
            return 1, time_chunk_size, time_len  # Fail


def check_chunk_size(dataset, severity=BaseCheck.MEDIUM):
    """
    Check the chunk size of a netCDF dataset.

    Parameters:
    - dataset: netCDF4.Dataset
        The NetCDF dataset to check.
    - severity: BaseCheck severity level (default is MEDIUM)

    Returns:
    - TestCtx: An object containing detailed results of the check, including
      pass/failure status, messages, and coordinates of detected outliers.
    - file: A file containing the coordinates and values of detected outliers is written when the check condition fails.
    """
    ctx = TestCtx(severity)
    variables = []
    time_dim = get_ds_dimensions(dataset)['t']
    if time_dim in dataset.variables:
        variables.append(time_dim)
    if f"{time_dim}_bnds" in dataset.variables:
        variables.append(f"{time_dim}_bnds")
    if len(variables) == 0:
        raise ValueError("No variables time found in the dataset.")
    ctx.variable = variables[0] + " and " + variables[1] if len(variables) > 1 else variables[0]
    for variable in variables:
        ctx.variable = variable
        if variable not in dataset.variables:
            continue

        filtered_dims = [dim for dim in get_filtered_dimensions(dataset, variable) if dim != 'bnds']

        for dim in filtered_dims:
            if dim not in dataset.dimensions:
                continue
            try:
                result, chunk_size, len_time = chunk_sizes_conditions(dataset, dim)
            except Exception as e:
                ctx.is_valid = False
                ctx.add_failure(f"Error checking chunk size for '{dim}': {e}")
                continue

            match result:
                case 0:
                    ctx.is_valid = True
                    # Chunk size of '{variable}' matches its length ({len_time}).
                    ctx.add_pass()
                    ctx.messages.append(f"Chunk size of '{variable}' matches its length ({len_time}).")

                case 1:
                    ctx.is_valid = True
                    ctx.add_failure(f"Chunk size of '{variable}' ({chunk_size}) is different from its length ({len_time}), but not too small.")
                case 2:
                    ctx.is_valid = False
                    ctx.add_failure(f"Chunk size of '{variable}' ({chunk_size}) is not defined.")

    return ctx
