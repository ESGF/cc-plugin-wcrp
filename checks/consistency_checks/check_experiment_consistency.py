#!/usr/bin/env python

from compliance_checker.base import TestCtx

try:
    import esgvoc.api as voc

    ESG_VOCAB_AVAILABLE = True
except ImportError:
    voc = None
    ESG_VOCAB_AVAILABLE = False


# ============================================================================
# Mapping visible: attribute name stays the same in the NetCDF file,
# only the ESGVOC collection name changes between CMIP6 and CMIP7.
# ============================================================================
CV_COLLECTION_MAP = {
    "cmip6": {
        "experiment_id": "experiment_id",
    },
    "cmip7": {
        "experiment_id": "experiment",
    },
}


def _get_cv_collection(project_id, attribute_name):
    project_key = str(project_id).strip().lower()
    return CV_COLLECTION_MAP.get(project_key, {}).get(attribute_name, attribute_name)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _lower_str_list(values):
    return [str(v).strip().lower() for v in _as_list(values) if v is not None]


def _get_global_attr(ds, attr_name, missing_attrs):
    """
    Return the stripped global attribute value if present, else None.
    Track missing attributes in missing_attrs without raising.
    """
    if attr_name not in ds.ncattrs():
        missing_attrs.append(attr_name)
        return None

    return str(ds.getncattr(attr_name)).strip()


def _get_experiment_reference_term(ds, project_id, failures, missing_attrs):
    """
    Resolve the ESGVOC experiment term from file global attribute experiment_id.

    Return:
      actual_experiment_id, reference_term
    """
    actual_experiment_id = _get_global_attr(ds, "experiment_id", missing_attrs)

    if actual_experiment_id is None:
        return None, None

    collection_id = _get_cv_collection(project_id, "experiment_id")

    reference_term = voc.get_term_in_collection(
        project_id=project_id,
        collection_id=collection_id,
        term_id=actual_experiment_id,
    )

    # Fallback: if the term is not found by term_id, try to resolve the
    # underlying ESGVOC id from an exact drs_name match.
    if not reference_term:
        candidates = voc.find_terms_in_collection(
            project_id=project_id,
            collection_id=collection_id,
            expression=actual_experiment_id,
            selected_term_fields=["id", "drs_name"],
        )

        resolved_term_id = None

        for item in candidates:
            candidate_drs_name = str(getattr(item, "drs_name", "")).strip()
            candidate_id = str(getattr(item, "id", "")).strip()

            if candidate_drs_name == actual_experiment_id:
                resolved_term_id = candidate_id
                break

        if resolved_term_id:
            reference_term = voc.get_term_in_collection(
                project_id=project_id,
                collection_id=collection_id,
                term_id=resolved_term_id,
            )

    if not reference_term:
        failures.append(
            f"The experiment_id '{actual_experiment_id}' was not found in the ESGF vocabulary."
        )

    return actual_experiment_id, reference_term


def _expected_matches_actual(expected_value, actual_value):
    """
    Compare expected ESGVOC value(s) with actual file value.
    """
    if expected_value is None:
        return True

    expected_norm = _lower_str_list(expected_value)

    if not expected_norm:
        return True

    return str(actual_value).strip().lower() in expected_norm


def _check_experiment_id_vs_global_attr(
    ds,
    severity,
    project_id,
    attr_name,
    expected_field=None,
):
    """
    Generic single-comparison check:
      experiment_id CV term field <expected_field> vs file global attribute <attr_name>
    """
    fixed_check_id = "ATTR007"
    expected_field = expected_field or attr_name
    description = f"[{fixed_check_id}] Consistency: experiment_id vs {attr_name}"
    ctx = TestCtx(severity, description)

    if not ESG_VOCAB_AVAILABLE:
        ctx.add_failure("The 'esgvoc' library is not installed.")
        return [ctx.to_result()]

    try:
        failures = []
        missing_attrs = []

        actual_experiment_id, reference_term = _get_experiment_reference_term(
            ds,
            project_id,
            failures,
            missing_attrs,
        )

        actual_value = _get_global_attr(ds, attr_name, missing_attrs)

        if reference_term is not None and actual_value is not None:
            expected_value = getattr(reference_term, expected_field, None)

            if not _expected_matches_actual(expected_value, actual_value):
                failures.append(
                    f"Inconsistency: experiment_id '{actual_experiment_id}' vs {attr_name}. "
                    f"CV expects one of {list(_as_list(expected_value))}, "
                    f"file has '{actual_value}'."
                )

        for attr in missing_attrs:
            failures.append(f"Missing required global attribute: '{attr}'.")

        if failures:
            for failure in failures:
                ctx.add_failure(failure)
        else:
            ctx.add_pass()

    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]


# ==============================================================================
# Atomic ATTR007 checks
# ==============================================================================

def check_experiment_id_vs_activity_id(ds, severity, project_id="cmip6"):
    return _check_experiment_id_vs_global_attr(
        ds,
        severity,
        project_id=project_id,
        attr_name="activity_id",
        expected_field="activity_id",
    )


def check_experiment_id_vs_experiment(ds, severity, project_id="cmip6"):
    return _check_experiment_id_vs_global_attr(
        ds,
        severity,
        project_id=project_id,
        attr_name="experiment",
        expected_field="experiment",
    )


def check_experiment_id_vs_parent_experiment_id(ds, severity, project_id="cmip6"):
    return _check_experiment_id_vs_global_attr(
        ds,
        severity,
        project_id=project_id,
        attr_name="parent_experiment_id",
        expected_field="parent_experiment_id",
    )


def check_experiment_id_vs_sub_experiment_id(ds, severity, project_id="cmip6"):
    return _check_experiment_id_vs_global_attr(
        ds,
        severity,
        project_id=project_id,
        attr_name="sub_experiment_id",
        expected_field="sub_experiment_id",
    )

