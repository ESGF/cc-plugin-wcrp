#!/usr/bin/env python



import os
from compliance_checker.base import TestCtx

try:
    from esgvoc.apps.drs.validator import DrsValidator
    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False

def _find_drs_directory(filepath, project_id="cmip6"):
    """
    Intelligently finds the DRS directory path by locating the project_id.
    """
    try:
        path_parts = filepath.lower().split(os.sep)
        original_parts = filepath.split(os.sep)
        start_index = path_parts.index(project_id.lower())
        drs_directory = os.path.join(*original_parts[start_index:-1])
        return drs_directory, None
    except (ValueError, TypeError):
        return None, f"DRS project root '{project_id}' not found in the file path '{filepath}'."

# ==============================================================================
# == CHECK 1 Check filename against CMIP6 CV pattern
# ==============================================================================
def check_drs_filename(ds, severity, project_id="cmip6"):
    """
    [FILE001] Validates the filename against the DRS controlled vocabulary.
    """
    fixed_check_id = "FILE001"
    description = f"[{fixed_check_id}] DRS Filename Vocabulary Check"
    ctx = TestCtx(severity, description)

    if not ESG_VOCAB_AVAILABLE:
        ctx.add_failure("The 'esgvoc' library is required but not installed.")
        return [ctx.to_result()]

    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    filename = os.path.basename(filepath)
    
    try:
        validator = DrsValidator(project_id=project_id)
        file_report = validator.validate_file_name(filename)
        
        if file_report.errors:
            error_details = '; '.join(str(e) for e in file_report.errors)
            ctx.add_failure(f"Filename '{filename}' has validation errors: {error_details}")
        else:
            ctx.add_pass()

    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred during filename validation: {e}")

    return [ctx.to_result()]

# ==============================================================================
# == CHECK 2 : Check filename against CMIP6 CV pattern
# ==============================================================================
def check_drs_directory(ds, severity, project_id="cmip6"):
    """
    [FILE001] Validates the directory structure against the DRS controlled vocabulary.
    """
    fixed_check_id = "FILE001"
    description = f"[{fixed_check_id}] DRS Directory Vocabulary Check"
    ctx = TestCtx(severity, description)

    if not ESG_VOCAB_AVAILABLE:
        ctx.add_failure("The 'esgvoc' library is required but not installed.")
        return [ctx.to_result()]

    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    drs_directory, error_msg = _find_drs_directory(filepath, project_id)

    if error_msg:
        ctx.add_failure(error_msg)
        return [ctx.to_result()]
    
    try:
        validator = DrsValidator(project_id=project_id)
        dir_report = validator.validate_directory(drs_directory)
        
        if dir_report.errors:
            error_details = '; '.join(str(e) for e in dir_report.errors)
            ctx.add_failure(f"DRS directory '{drs_directory}' has validation errors: {error_details}")
        else:
            ctx.add_pass()

    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred during directory validation: {e}")

    return [ctx.to_result()]
