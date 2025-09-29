#!/usr/bin/env python



import os
import re
from compliance_checker.base import TestCtx

def _get_drs_facets(filepath, project_id="CMIP6"):
    """
    parses a full filepath to extract DRS components from both the
    directory path and the filename.
    """
    
    dir_template_keys = ["mip_era", "activity_id", "institution_id", "source_id", 
                         "experiment_id", "variant_label", "table_id", 
                         "variable_id", "grid_label", "version"]
    
    filename_template_keys = ["variable_id", "table_id", "source_id", 
                              "experiment_id", "variant_label", "grid_label", "time_range"]
    try:
        directory_path = os.path.dirname(filepath)
        path_parts = directory_path.split(os.sep)
        
        start_index = path_parts.index(project_id)
        drs_path_parts = path_parts[start_index:]
        
        if len(drs_path_parts) != len(dir_template_keys):
            return None, None, f"Directory path does not match expected DRS depth. Found {len(drs_path_parts)}, expected {len(dir_template_keys)}."

        dir_facets = dict(zip(dir_template_keys, drs_path_parts))

        filename = os.path.basename(filepath)
        filename_parts = filename.replace(".nc", "").split("_")

        if len(filename_parts) != len(filename_template_keys):
            return None, None, f"Filename '{filename}' does not match expected structure (wrong number of '_' separators)."

        filename_facets = dict(zip(filename_template_keys, filename_parts))

        return dir_facets, filename_facets, None
    except ValueError:
        return None, None, f"DRS project root '{project_id}' not found in the file path."
    except Exception as e:
        return None, None, f"An unexpected error occurred during DRS parsing: {e}"

# ==============================================================================
# == CHECK 1 : Check if DRS match Global Attributes
# ==============================================================================
def check_attributes_match_directory_structure(ds, severity, project_id="cmip6"):
    """
    [PATH001] Checks if global attributes match the DRS directory structure.
    """
    fixed_check_id = "PATH001"
    description = f"[{fixed_check_id}] Consistency: Directory Structure vs Global Attributes"
    ctx = TestCtx(severity, description)

    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    dir_facets, _, error = _get_drs_facets(filepath, project_id)
    
    if error:
        ctx.add_failure(f"Could not perform check. Reason: {error}")
        return [ctx.to_result()]

    failures = []
    for drs_key, drs_value in dir_facets.items():
        if drs_key == "version": continue
        if drs_key in ds.ncattrs():
            attr_value = str(ds.getncattr(drs_key))
            if drs_value != attr_value:
                failures.append(f"DRS path component '{drs_key}' ('{drs_value}') does not match global attribute ('{attr_value}').")
        else:
            ctx.messages.append(f"Global attribute '{drs_key}' not found, skipping comparison.")

    for f in failures:
        ctx.add_failure(f)

    if not failures:
        ctx.add_pass()
            
    return [ctx.to_result()]

# ==============================================================================
# == CHECK 2 : Check if Filename match DRS
# ==============================================================================
def check_filename_matches_directory_structure(ds, severity, project_id="CMIP6"):
    """
    [PATH002] Checks if filename tokens match the DRS directory structure tokens.
    """
    fixed_check_id = "PATH002"
    description = f"[{fixed_check_id}] Consistency: Directory Structure vs Filename"
    ctx = TestCtx(severity, description)
    
    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    dir_facets, filename_facets, error = _get_drs_facets(filepath, project_id)

    if error:
        ctx.add_failure(f"Could not perform check. Reason: {error}")
        return [ctx.to_result()]

    keys_to_compare = ["source_id", "experiment_id", "variant_label", "table_id", "variable_id", "grid_label"]
    
    failures = []
    for key in keys_to_compare:
        path_val = dir_facets.get(key)
        filename_val = filename_facets.get(key)
        
        if path_val != filename_val:
            failures.append(f"Token '{key}' is inconsistent: path has '{path_val}', filename has '{filename_val}'.")

    for f in failures:
        ctx.add_failure(f)

    if not failures:
        ctx.add_pass()
            
    return [ctx.to_result()]
