#!/usr/bin/env python

import os
from compliance_checker.base import TestCtx

from checks.utils import _get_drs_facets, resolve_member_id

def _unwrap_facets(x):

    if isinstance(x, tuple) and len(x) > 0 and isinstance(x[0], dict):
        return x[0]
    return x

def _normalize_drs_global_attr_for_path_compare(attr_name: str, value) -> str:
    """
    Normalize global attribute values before comparing them with DRS path components.

    activity_id may contain several space-separated activities, e.g.
    "ScenarioMIP AerChemMIP", while the DRS path contains only the first one:
    "ScenarioMIP".
    """
    text = str(value).strip()

    if attr_name == "activity_id":
        parts = text.split()
        if parts:
            return parts[0]

    return text
_dir_template_keys_cmip6 = [
    "mip_era", "activity_id", "institution_id", "source_id", "experiment_id",
    "variant_label", "table_id", "variable_id", "grid_label", "version",
]
_filename_template_keys_cmip6 = [
    "variable_id", "table_id", "source_id", "experiment_id",
    "variant_label", "grid_label", "time_range",
]

# CMIP7 directoryStructure
_dir_template_keys_cmip7 = [
    "drs_specs", "mip_era", "activity_id", "institution_id", "source_id",
    "experiment_id", "variant_label", "region", "frequency", "variable_id",
    "branding_suffix", "grid_label", "directory_date",
]
# CMIP7 fileName
_filename_template_keys_cmip7 = [
    "variable_id", "branding_suffix", "frequency", "region", "grid_label",
    "source_id", "experiment_id", "variant_label", "time_range",
]


def _is_cmip7(ds, project_id: str) -> bool:
    try:
        if str(ds.getncattr("mip_era")).upper() == "CMIP7":
            return True
    except Exception:
        pass
    return isinstance(project_id, str) and project_id.lower() == "cmip7"


def _parse_directory_from_drs_specs(filepath: str, drs_specs: str, template_keys: list[str]):
    parts = os.path.normpath(filepath).split(os.sep)
    if not drs_specs or drs_specs not in parts:
        return {}, f"Directory path does not contain drs_specs '{drs_specs}'."
    i = parts.index(drs_specs)
    terms = parts[i:-1]  # from drs_specs to parent directory
    if len(terms) != len(template_keys):
        return {}, (
            f"Directory path does not match expected DRS depth. "
            f"Found {len(terms)}, expected {len(template_keys)}."
        )
    return dict(zip(template_keys, terms)), None


def _parse_cmip7_filename(filename: str):
    stem = filename[:-3] if filename.endswith(".nc") else filename
    parts = stem.split("_")
    if len(parts) not in (8, 9):
        return {}, "Filename does not match expected CMIP7 token count (8 or 9 with time_range)."
    facets = {
        "variable_id": parts[0],
        "branding_suffix": parts[1],
        "frequency": parts[2],
        "region": parts[3],
        "grid_label": parts[4],
        "source_id": parts[5],
        "experiment_id": parts[6],
        "variant_label": parts[7],
        "time_range": parts[8] if len(parts) == 9 else None,
    }
    return facets, None


def check_attributes_match_directory_structure(
    ds,
    severity,
    project_id="cmip6",
    dir_template_keys=None,
    filename_template_keys=None,
    *,
    report_directory_structure_error=True,
):
    fixed_check_id = "PATH001"
    description = f"[{fixed_check_id}] Consistency: Directory Structure vs Global Attributes"
    ctx = TestCtx(severity, description)

    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    if _is_cmip7(ds, project_id):
        
        try:
            drs_specs = str(ds.getncattr("drs_specs"))
        except Exception:
            ctx.add_failure("Could not perform check. Reason: missing global attribute 'drs_specs'.")
            return [ctx.to_result()]

        dir_facets, error = _parse_directory_from_drs_specs(filepath, drs_specs, _dir_template_keys_cmip7)
        if error:
            if not report_directory_structure_error:
                return []
            ctx.add_failure(f"Could not perform check. Reason: {error}")
            return [ctx.to_result()]

        skip_keys = {"directory_date"}
    else:
        if not dir_template_keys:
            dir_template_keys = _dir_template_keys_cmip6
        if not filename_template_keys:
            filename_template_keys = _filename_template_keys_cmip6

        dir_facets, _, error = _get_drs_facets(filepath, project_id, dir_template_keys, filename_template_keys)
        dir_facets = _unwrap_facets(dir_facets)
        if error:
            if not report_directory_structure_error:
                return []
            ctx.add_failure(f"Could not perform check. Reason: {error}")
            return [ctx.to_result()]

        skip_keys = {"version"}

    # For CMIP6 / CMIP6Plus, the directory segment at the "variant_label"
    # position is the DRS member_id: "<sub_experiment_id>-<variant_label>"
    # when a sub-experiment is present, otherwise just "<variant_label>".
    member_id_projects = ("cmip6", "cmip6plus")
    failures = []
    for key, drs_value in dir_facets.items():
        if key in skip_keys:
            continue

        if key == "variant_label" and project_id in member_id_projects:
            expected = resolve_member_id(ds)
            if str(drs_value) != expected:
                failures.append(
                    f"DRS path component 'member_id' ('{drs_value}') does not match "
                    f"expected member_id ('{expected}') derived from global attributes "
                    f"sub_experiment_id + variant_label."
                )
            continue

        if key in ds.ncattrs():
            raw_attr_value = ds.getncattr(key)
            attr_value = _normalize_drs_global_attr_for_path_compare(key, raw_attr_value)

            if str(drs_value) != attr_value:
              failures.append(
                 f"DRS path component '{key}' ('{drs_value}') does not match "
                 f"global attribute '{key}' ('{raw_attr_value}'; compared as '{attr_value}')."
        )
        else:
             ctx.messages.append(f"Global attribute '{key}' not found, skipping comparison.")

    for f in failures:
        ctx.add_failure(f)
    if not failures:
        ctx.add_pass()

    return [ctx.to_result()]


def check_filename_matches_directory_structure(
    ds,
    severity,
    project_id="cmip6",
    dir_template_keys=None,
    filename_template_keys=None,
    *,
    report_directory_structure_error=True,
):
    fixed_check_id = "PATH002"
    description = f"[{fixed_check_id}] Consistency: Directory Structure vs Filename"
    ctx = TestCtx(severity, description)

    filepath = ds.filepath()
    if not isinstance(filepath, str):
        ctx.add_failure("File path could not be determined.")
        return [ctx.to_result()]

    if _is_cmip7(ds, project_id):
        try:
            drs_specs = str(ds.getncattr("drs_specs"))
        except Exception:
            ctx.add_failure("Could not perform check. Reason: missing global attribute 'drs_specs'.")
            return [ctx.to_result()]

        dir_facets, error = _parse_directory_from_drs_specs(filepath, drs_specs, _dir_template_keys_cmip7)
        if error:
            if not report_directory_structure_error:
                return []
            ctx.add_failure(f"Could not perform check. Reason: {error}")
            return [ctx.to_result()]

        filename = os.path.basename(filepath)
        file_facets, ferr = _parse_cmip7_filename(filename)
        if ferr:
            ctx.add_failure(f"Could not perform check. Reason: {ferr}")
            return [ctx.to_result()]

        # Compare shared facets between directory and filename (exclude time_range)
        failures = []
        for key in ("variable_id", "branding_suffix", "frequency", "region", "grid_label", "source_id", "experiment_id", "variant_label"):
            if str(dir_facets.get(key)) != str(file_facets.get(key)):
                failures.append(
                    f"Token '{key}' is inconsistent: path has '{dir_facets.get(key)}', filename has '{file_facets.get(key)}'."
                )

        for f in failures:
            ctx.add_failure(f)
        if not failures:
            ctx.add_pass()
        return [ctx.to_result()]

    # CMIP6 fallback
    if not dir_template_keys:
        dir_template_keys = _dir_template_keys_cmip6
    if not filename_template_keys:
        filename_template_keys = _filename_template_keys_cmip6

    dir_facets, filename_facets, error = _get_drs_facets(filepath, project_id, dir_template_keys, filename_template_keys)
    dir_facets = _unwrap_facets(dir_facets)
    filename_facets = _unwrap_facets(filename_facets)
    if error:
        if not report_directory_structure_error:
            return []
        ctx.add_failure(f"Could not perform check. Reason: {error}")
        return [ctx.to_result()]

    keys_to_compare = [k for k in dir_template_keys if k in filename_template_keys]

    failures = []
    for key in keys_to_compare:
        if dir_facets.get(key) != filename_facets.get(key):
            failures.append(
                f"Token '{key}' is inconsistent: path has '{dir_facets.get(key)}', filename has '{filename_facets.get(key)}'."
            )

    for f in failures:
        ctx.add_failure(f)
    if not failures:
        ctx.add_pass()

    return [ctx.to_result()]
