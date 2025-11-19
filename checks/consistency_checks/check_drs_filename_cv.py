#!/usr/bin/env python


import os

from compliance_checker.base import BaseCheck, TestCtx

from checks.utils import _compare_CV, _find_drs_directory_and_filename

try:
    from esgvoc.apps.drs.validator import DrsValidator

    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False


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
            error_details = "; ".join(str(e) for e in file_report.errors)
            ctx.add_failure(
                f"Filename '{filename}' has validation errors: {error_details}"
            )
        else:
            ctx.add_pass()

    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred during filename validation: {e}")

    return [ctx.to_result()]


# ==============================================================================
# == CHECK 2 : Check directory structure against CMIP6 CV pattern
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

    drs_directory, _, error_msg = _find_drs_directory_and_filename(filepath, project_id)

    if error_msg:
        ctx.add_failure(error_msg)
        return [ctx.to_result()]

    try:
        validator = DrsValidator(project_id=project_id)
        dir_report = validator.validate_directory(drs_directory)

        if dir_report.errors:
            error_details = "; ".join(str(e) for e in dir_report.errors)
            ctx.add_failure(
                f"DRS directory '{drs_directory}' has validation errors: {error_details}"
            )
        else:
            ctx.add_pass()

    except Exception as e:
        ctx.add_failure(
            f"An unexpected error occurred during directory validation: {e}"
        )

    return [ctx.to_result()]


# ==============================================================================
# == CHECK 1' : Check filename against CV pattern against <project>_CV.json
# ==============================================================================
def check_drs_filename_cv(
    CheckerObject,
    drs_elements_hard_checks=[],
    project_id="CORDEX-CMIP6",
    severity=BaseCheck.HIGH,
):
    """[FILE001] DRS building blocks in filename checked against CV."""
    check_id = "FILE001"
    description = f"[{check_id}] DRS Filename (against {project_id}_CV.json)"
    testctx = TestCtx(severity, description)

    # File suffix
    suffix = ".".join(os.path.basename(CheckerObject.filepath).split(".")[1:])
    if CheckerObject.drs_suffix == suffix:
        testctx.add_pass()
    else:
        testctx.add_failure(
            f"File suffix differs from expectation ('{CheckerObject.drs_suffix}'): '{suffix}'."
        )

    # DRS filename
    drs_fn_checked, drs_fn_messages = _compare_CV(
        CheckerObject, CheckerObject.drs_fn, "DRS filename building block "
    )
    if len(drs_fn_messages) == 0:
        testctx.add_pass()
    else:
        for message in drs_fn_messages:
            testctx.add_failure(message)

    # Unchecked DRS filename building blocks
    unchecked = [
        key
        for key in CheckerObject.drs_fn.keys()
        if not drs_fn_checked[key] and key not in drs_elements_hard_checks
    ]
    if len(unchecked) == 0:
        testctx.add_pass()
    else:
        testctx.add_failure(
            f"""DRS filename building blocks could not be checked: {', '.join(f"'{ukey}'" for ukey in sorted(unchecked))}."""
        )
    return [testctx.to_result()]


# ==============================================================================
# == CHECK 2' : Check directory structure against CV pattern against
#               <project>_CV.json
# ==============================================================================
def check_drs_directory_cv(
    CheckerObject,
    drs_elements_hard_checks,
    project_id="CORDEX-CMIP6",
    severity=BaseCheck.HIGH,
):
    """[FILE001] DRS building blocks in directory path checked against CV."""
    check_id = "FILE001"
    description = f"[{check_id}] DRS Directory Structure (against {project_id}_CV.json)"
    testctx = TestCtx(severity, description)

    # DRS path
    drs_dir_checked, drs_dir_messages = _compare_CV(
        CheckerObject, CheckerObject.drs_dir, "DRS path building block "
    )
    if len(drs_dir_messages) == 0:
        testctx.add_pass()
    else:
        for message in drs_dir_messages:
            testctx.add_failure(message)

    # Unchecked DRS path building blocks
    unchecked = [
        key
        for key in CheckerObject.drs_dir.keys()
        if not drs_dir_checked[key] and key not in drs_elements_hard_checks
    ]
    if len(unchecked) == 0:
        testctx.add_pass()
    else:
        testctx.add_failure(
            f"""DRS path building blocks could not be checked: {', '.join(f"'{ukey}'" for ukey in sorted(unchecked))}."""
        )

    return [testctx.to_result()]
