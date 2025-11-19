#!/usr/bin/env python


from compliance_checker.base import TestCtx

from checks.utils import _get_drs_facets

# Default directory and filename templates as defined for CMIP6
_dir_template_keys = [
    "mip_era",
    "activity_id",
    "institution_id",
    "source_id",
    "experiment_id",
    "variant_label",
    "table_id",
    "variable_id",
    "grid_label",
    "version",
]
_filename_template_keys = [
    "variable_id",
    "table_id",
    "source_id",
    "experiment_id",
    "variant_label",
    "grid_label",
    "time_range",
]


# ==============================================================================
# == CHECK 1 : Check if DRS match Global Attributes
# ==============================================================================
def check_attributes_match_directory_structure(
    ds,
    severity,
    project_id="cmip6",
    dir_template_keys=None,
    filename_template_keys=None,
):
    """
    [PATH001] Checks if global attributes match the DRS directory structure.
    """
    fixed_check_id = "PATH001"
    description = (
        f"[{fixed_check_id}] Consistency: Directory Structure vs Global Attributes"
    )
    ctx = TestCtx(severity, description)

    # If not specified, use default/CMIP6 templates
    if not dir_template_keys:
        dir_template_keys = _dir_template_keys
    if not filename_template_keys:
        filename_template_keys = _filename_template_keys

    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    dir_facets, _, error = _get_drs_facets(
        filepath, project_id, dir_template_keys, filename_template_keys
    )

    if error:
        ctx.add_failure(f"Could not perform check. Reason: {error}")
        return [ctx.to_result()]

    failures = []
    for drs_key, drs_value in dir_facets.items():
        if drs_key == "version":
            continue
        if drs_key in ds.ncattrs():
            attr_value = str(ds.getncattr(drs_key))
            if drs_value != attr_value:
                failures.append(
                    f"DRS path component '{drs_key}' ('{drs_value}') does not match global attribute ('{attr_value}')."
                )
        else:
            ctx.messages.append(
                f"Global attribute '{drs_key}' not found, skipping comparison."
            )

    for f in failures:
        ctx.add_failure(f)

    if not failures:
        ctx.add_pass()

    return [ctx.to_result()]


# ==============================================================================
# == CHECK 2 : Check if Filename match DRS
# ==============================================================================
def check_filename_matches_directory_structure(
    ds,
    severity,
    project_id="cmip6",
    dir_template_keys=None,
    filename_template_keys=None,
):
    """
    [PATH002] Checks if filename tokens match the DRS directory structure tokens.
    """
    fixed_check_id = "PATH002"
    description = f"[{fixed_check_id}] Consistency: Directory Structure vs Filename"
    ctx = TestCtx(severity, description)

    # If not specified, use default/CMIP6 templates
    if not dir_template_keys:
        dir_template_keys = _dir_template_keys
    if not filename_template_keys:
        filename_template_keys = _filename_template_keys

    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    dir_facets, filename_facets, error = _get_drs_facets(
        filepath, project_id, dir_template_keys, filename_template_keys
    )

    if error:
        ctx.add_failure(f"Could not perform check. Reason: {error}")
        return [ctx.to_result()]

    keys_to_compare = [
        key for key in dir_template_keys if key in filename_template_keys
    ]

    failures = []
    for key in keys_to_compare:
        path_val = dir_facets.get(key)
        filename_val = filename_facets.get(key)

        if path_val != filename_val:
            failures.append(
                f"Token '{key}' is inconsistent: path has '{path_val}', filename has '{filename_val}'."
            )

    for f in failures:
        ctx.add_failure(f)

    if not failures:
        ctx.add_pass()

    return [ctx.to_result()]
