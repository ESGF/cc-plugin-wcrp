#!/usr/bin/env python

from compliance_checker.base import TestCtx

try:
    import esgvoc.api as voc
    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False


# ============================================================================
# Mapping visible: NetCDF attribute names stay the same.
# Only the ESGVOC collection name changes between CMIP6 and CMIP7.
# ============================================================================
CV_COLLECTION_MAP = {
    "cmip6": {
        "institution_id": "institution_id",
        "source_id": "source_id",
    },
    "cmip7": {
        "institution_id": "institution",
        "source_id": "source",
    },
    "cmip6plus": {
        "institution_id": "institution_id",
        "source_id": "source_id",
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


def _resolved_term(project_id, collection_id, term_id):
    """Resolve a full ESGVoc term, accounting for normalized internal IDs."""
    for candidate in (term_id, term_id.lower()):
        term = voc.get_term_in_collection(
            project_id=project_id,
            collection_id=collection_id,
            term_id=candidate,
        )
        if term is not None and (
            getattr(term, "id", None) or getattr(term, "drs_name", None)
        ):
            return term
    return None


def check_id_attribute_consistency(
    ds,
    severity,
    *,
    project_id,
    id_attribute,
    value_attribute,
    term_field=None,
    collection_id=None,
):
    """Compare an attribute with a field on the term selected by its ID."""
    term_field = term_field or value_attribute
    collection_id = collection_id or _get_cv_collection(project_id, id_attribute)
    ctx = TestCtx(
        severity,
        f"[ATTR011] Consistency: {id_attribute} vs {value_attribute}",
    )

    attributes = set(ds.ncattrs())
    if id_attribute not in attributes or value_attribute not in attributes:
        ctx.add_pass()
        return [ctx.to_result()]

    identifier = str(ds.getncattr(id_attribute)).strip()
    actual = str(ds.getncattr(value_attribute)).strip()

    if not ESG_VOCAB_AVAILABLE:
        ctx.add_failure("The 'esgvoc' library is required but not installed.")
        return [ctx.to_result()]

    try:
        term = _resolved_term(project_id, collection_id, identifier)
    except Exception as exc:  # noqa: BLE001 -- report catalogue failures as findings
        ctx.add_failure(
            f"Could not retrieve {collection_id!r} term {identifier!r} from "
            f"ESGVoc: {type(exc).__name__}: {exc}"
        )
        return [ctx.to_result()]

    if term is None:
        ctx.add_failure(
            f"The {id_attribute} value {identifier!r} was not found in ESGVoc "
            f"collection {collection_id!r}."
        )
        return [ctx.to_result()]

    expected = [
        str(value).strip()
        for value in _as_list(getattr(term, term_field, None))
        if value is not None and str(value).strip()
    ]
    if not expected:
        ctx.add_failure(
            f"ESGVoc term {identifier!r} has no value for field {term_field!r}, "
            f"so {value_attribute!r} could not be verified."
        )
    elif actual in expected:
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"The global attribute {value_attribute!r} is inconsistent with "
            f"{id_attribute}={identifier!r}: found {actual!r}; expected "
            f"{expected[0]!r}."
            if len(expected) == 1
            else f"The global attribute {value_attribute!r} is inconsistent with "
            f"{id_attribute}={identifier!r}: found {actual!r}; expected one of "
            f"{expected!r}."
        )
    return [ctx.to_result()]


def check_institution_consistency(ds, severity, project_id="cmip6"):
    """
    [ATTR009] Checks if the global attribute 'institution' is consistent with the
    'description' from the ESGF CV for the given 'institution_id'.
    """
    fixed_check_id = "ATTR009"
    description = f"[{fixed_check_id}] Consistency: institution_id vs institution attribute"
    ctx = TestCtx(severity, description)

    if not ESG_VOCAB_AVAILABLE:
        ctx.add_failure("The 'esgvoc' library is required but not installed.")
        return [ctx.to_result()]

    try:
        # Read attributes from the NetCDF file
        institution_id = str(ds.getncattr("institution_id")).strip()
        actual_institution = str(ds.getncattr("institution")).strip()

        # Resolve project-specific ESGVOC collection
        collection_id = _get_cv_collection(project_id, "institution_id")

        # Query esgvoc with the file value as-is
        reference_term = voc.get_term_in_collection(
            project_id=project_id,
            collection_id=collection_id,
            term_id=institution_id,
        )

        # Fallback on lowercase because some vocabularies may normalize ids
        if not reference_term:
            reference_term = voc.get_term_in_collection(
                project_id=project_id,
                collection_id=collection_id,
                term_id=institution_id.lower(),
            )

        if not reference_term:
            ctx.add_failure(
                f"The institution_id '{institution_id}' was not found in the ESGF vocabulary."
            )
            return [ctx.to_result()]

        # Compare the file's 'institution' attribute with the CV description
        expected_description = getattr(reference_term, "description", None)

        if expected_description and actual_institution == str(expected_description).strip():
            ctx.add_pass()
        else:
            msg = (
                f"Inconsistency for 'institution' attribute. "
                f"CV expects description: '{expected_description}', "
                f"file has: '{actual_institution}'."
            )
            ctx.add_failure(msg)

    except AttributeError as e:
        ctx.add_failure(
            f"Missing a required global attribute for the check (e.g., 'institution_id' or 'institution'): {e}"
        )
    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]


def check_source_consistency(ds, severity, project_id="cmip6"):
    """
    [ATTR010] Checks if the global attribute 'institution_id' is consistent with the
    'organisation_id' from the ESGF CV for the given 'source_id'.
    """
    fixed_check_id = "ATTR010"
    description = f"[{fixed_check_id}] Consistency: source_id vs institution_id"
    ctx = TestCtx(severity, description)

    if not ESG_VOCAB_AVAILABLE:
        ctx.add_failure("The 'esgvoc' library is required but not installed.")
        return [ctx.to_result()]

    try:
        # Read attributes from the NetCDF file
        source_id = str(ds.getncattr("source_id")).strip()
        actual_institution_id = str(ds.getncattr("institution_id")).strip()

        # Resolve project-specific ESGVOC collection
        collection_id = _get_cv_collection(project_id, "source_id")

        # Query esgvoc with the file value as-is
        reference_term = voc.get_term_in_collection(
            project_id=project_id,
            collection_id=collection_id,
            term_id=source_id,
        )

        # Fallback on lowercase because some vocabularies may normalize ids
        if not reference_term:
            reference_term = voc.get_term_in_collection(
                project_id=project_id,
                collection_id=collection_id,
                term_id=source_id.lower(),
            )

        if not reference_term:
            ctx.add_failure(
                f"The source_id '{source_id}' was not found in the ESGF vocabulary."
            )
            return [ctx.to_result()]

        # Compare the file's institution_id with the organisation_id from the CV
        expected_org_ids = getattr(reference_term, "organisation_id", [])

        if actual_institution_id.lower() in _lower_str_list(expected_org_ids):
            ctx.add_pass()
        else:
            msg = (
                f"Inconsistency for 'institution_id'. For the source_id '{source_id}', "
                f"the CV expects one of {list(_as_list(expected_org_ids))}, "
                f"but the file has '{actual_institution_id}'."
            )
            ctx.add_failure(msg)

    except AttributeError as e:
        ctx.add_failure(
            f"Missing a required global attribute for the check (e.g., 'source_id' or 'institution_id'): {e}"
        )
    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]
