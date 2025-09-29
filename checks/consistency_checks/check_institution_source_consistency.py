#!/usr/bin/env python

from compliance_checker.base import TestCtx

# Import the esgvoc library, handling the case where it is not installed
try:
    import esgvoc.api as voc
    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False

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
        institution_id = str(ds.getncattr("institution_id"))
        actual_institution = str(ds.getncattr("institution"))

        # Query esgvoc to get the reference specifications for the institution_id
        reference_term = voc.get_term_in_collection(
            project_id=project_id,
            collection_id="institution_id",
            term_id=institution_id.lower()
        )

        if not reference_term:
            ctx.add_failure(f"The institution_id '{institution_id}' was not found in the ESGF vocabulary.")
            return [ctx.to_result()]

        # Compare the 'institution' attribute with the 'description' from the CV
        expected_description = getattr(reference_term, 'description', None)
        
        if expected_description and actual_institution == expected_description:
            ctx.add_pass()
        else:
            msg = (f"Inconsistency for 'institution' attribute. "
                   f"CV expects description: '{expected_description}', "
                   f"file has: '{actual_institution}'.")
            ctx.add_failure(msg)

    except AttributeError as e:
        ctx.add_failure(f"Missing a required global attribute for the check (e.g., 'institution_id' or 'institution'): {e}")
    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]


def check_source_consistency(ds, severity, project_id="cmip6"):
    """
    [ATTR009] Checks if the global attribute 'institution_id' is consistent with the
    'organisation_id' from the ESGF CV for the given 'source_id'.
    """
    fixed_check_id = "ATTR009"
    description = f"[{fixed_check_id}] Consistency: source_id vs institution_id"
    ctx = TestCtx(severity, description)

    if not ESG_VOCAB_AVAILABLE:
        ctx.add_failure("The 'esgvoc' library is required but not installed.")
        return [ctx.to_result()]

    try:
        #  Read attributes from the NetCDF file
        source_id = str(ds.getncattr("source_id"))
        actual_institution_id = str(ds.getncattr("institution_id"))

        # Query esgvoc to get the reference specifications for the source_id
        reference_term = voc.get_term_in_collection(
            project_id=project_id,
            collection_id="source_id",
            term_id=source_id.lower()
        )

        if not reference_term:
            ctx.add_failure(f"The source_id '{source_id}' was not found in the ESGF vocabulary.")
            return [ctx.to_result()]

        #  Compare the file's 'institution_id' with the 'organisation_id' from the CV
        
        expected_org_ids = getattr(reference_term, 'organisation_id', [])
        
        if actual_institution_id.lower() in [org_id.lower() for org_id in expected_org_ids]:
            ctx.add_pass()
            
        else:
            msg = (f"Inconsistency for 'institution_id'. For the source_id '{source_id}', "
                   f"the CV expects one of {expected_org_ids}, "
                   f"but the file has '{actual_institution_id}'.")
            ctx.add_failure(msg)

    except AttributeError as e:
        ctx.add_failure(f"Missing a required global attribute for the check (e.g., 'source_id' or 'institution_id'): {e}")
    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]
