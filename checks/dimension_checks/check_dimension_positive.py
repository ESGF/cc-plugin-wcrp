#!/usr/bin/env python


from compliance_checker.base import BaseCheck, TestCtx
import numpy as np

def check_dimension_positive(
    ds,
    dimension_name,
    severity=BaseCheck.MEDIUM
):
    """
    Check that the size (length) of a dimension is a positive integer.
    """
   
    # Fixed Check Id
    check_id = "DIM002"
    
    description = f"[{check_id}] Positive Integer Size Check for Dimension '{dimension_name}'"
    ctx = TestCtx(severity, description)

    if dimension_name not in ds.dimensions:
        
        ctx.messages.append(f"Dimension '{dimension_name}' not found, check skipped.")
        return [ctx.to_result()]

    try:
        dim_size = ds.dimensions[dimension_name].size
        
        if dim_size > 0:
            ctx.add_pass()
        else:
            # if dimension is equal to 0 , failure
            ctx.add_failure(f"Dimension '{dimension_name}' must have a positive size, but found {dim_size}.")

    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]



    
   
