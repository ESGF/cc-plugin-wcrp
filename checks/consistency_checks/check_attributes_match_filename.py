#!/usr/bin/env python



import os
import re
from compliance_checker.base import TestCtx

def _parse_filename_components(filename):
    """
    Parses a CMIP6-style filename to extract its components.
    Returns a dictionary of the components or an error message.
    """
    # Template for a standard CMIP6 filename
    filename_template_keys = [
        "variable_id", "table_id", "source_id", "experiment_id",
        "variant_label", "grid_label", "time_range"
    ]
    
    # Remove the .nc extension and split by the underscore separator
    filename_parts = filename.replace(".nc", "").split("_")

    if len(filename_parts) != len(filename_template_keys):
        return None, f"Filename '{filename}' does not have the expected 7 components."

    # Create a dictionary from the keys and parts
    facets = dict(zip(filename_template_keys, filename_parts))
    return facets, None


def check_filename_vs_global_attrs(ds, severity):
    """
    [ATTR005] Checks if filename components are consistent with global attributes.
    """
    fixed_check_id = "ATTR005"
    description = f"[{fixed_check_id}] Consistency: Filename vs Global Attributes"
    ctx = TestCtx(severity, description)

    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    filename = os.path.basename(filepath)
    
    # Parse the filename to get its components
    filename_facets, error = _parse_filename_components(filename)

    if error:
        ctx.add_failure(f"Could not perform check. Reason: {error}")
        return [ctx.to_result()]

    failures = []
    # Define which filename components should match which global attributes
    keys_to_compare = [
        "variable_id", "table_id", "source_id", "experiment_id",
        "variant_label", "grid_label"
    ]

    for key in keys_to_compare:
        filename_value = filename_facets.get(key)
        
        if key in ds.ncattrs():
            attr_value = str(ds.getncattr(key))
            if filename_value != attr_value:
                failures.append(f"Inconsistency for '{key}': filename has '{filename_value}', global attribute has '{attr_value}'.")
        else:
            # This is not a failure of this specific check, but a note.
            # The existence of the attribute should be caught by another check.
            ctx.messages.append(f"Global attribute '{key}' not found in file, skipping comparison for this token.")

    if not failures:
        ctx.add_pass()
    else:
        for f in failures:
            ctx.add_failure(f)
            
    return [ctx.to_result()]
