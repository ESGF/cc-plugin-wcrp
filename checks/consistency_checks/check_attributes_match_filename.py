#!/usr/bin/env python


import os

from compliance_checker.base import TestCtx

from checks.consistency_checks.check_drs_consistency import _filename_template_keys
from checks.utils import _parse_filename_components


def check_filename_vs_global_attrs(ds, severity, filename_template_keys=None):
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
    if not filename_template_keys:
        filename_template_keys = _filename_template_keys

    # Parse the filename to get its components
    filename_facets, error = _parse_filename_components(
        filename, filename_template_keys
    )

    if error:
        ctx.add_failure(f"Could not perform check. Reason: {error}")
        return [ctx.to_result()]

    failures = []
    # Define which filename components should match which global attributes
    keys_to_compare = [
        key for key in filename_template_keys if key not in ["time_range"]
    ]

    for key in keys_to_compare:
        filename_value = filename_facets.get(key)

        if key in ds.ncattrs():
            attr_value = str(ds.getncattr(key))
            if filename_value != attr_value:
                failures.append(
                    f"Inconsistency for '{key}': filename has '{filename_value}', global attribute has '{attr_value}'."
                )
        else:
            # This is not a failure of this specific check, but a note.
            # The existence of the attribute should be caught by another check.
            ctx.messages.append(
                f"Global attribute '{key}' not found in file, skipping comparison for this token."
            )

    if not failures:
        ctx.add_pass()
    else:
        for f in failures:
            ctx.add_failure(f)

    return [ctx.to_result()]
