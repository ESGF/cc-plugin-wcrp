#!/usr/bin/env python



from compliance_checker.base import TestCtx

try:
    import esgvoc.api as voc
    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False

def check_experiment_consistency(ds, severity, project_id="cmip6"):
    """
    [ATTR007] Checks if attributes are consistent with the 'experiment_id' from the ESGF CV.
    """
    fixed_check_id = "ATTR007"
    description = f"[{fixed_check_id}] Consistency: experiment_id vs other global attributes"
    ctx = TestCtx(severity, description)

    if not ESG_VOCAB_AVAILABLE:
        ctx.add_failure("The 'esgvoc' library is not installed.")
        return [ctx.to_result()]

    try:
        # Read experiment_id from NetCDF file
        experiment_id_from_file = ds.getncattr("experiment_id")


        # Force conversion to standard Python string to be sure
        experiment_id_str = str(experiment_id_from_file).strip()

        #  Query esgvoc with cleaned value
        reference_term = voc.get_term_in_collection(
            project_id=project_id,
            collection_id="experiment_id",
            term_id=experiment_id_str
        )

        if not reference_term:
            ctx.add_failure(f"The experiment_id '{experiment_id_str}' was not found in the ESGF vocabulary.")
            return [ctx.to_result()]

        # Perform comparisons
        failures = []
        
        # Comparison of ‘activity_id’ (case insensitive)
        expected_activity_id = getattr(reference_term, 'activity_id', [None])[0]
        actual_activity_id = str(ds.getncattr("activity_id"))
        if expected_activity_id and actual_activity_id.lower() != expected_activity_id.lower():
            failures.append(f"Inconsistency for 'activity_id': CV expects '{expected_activity_id}', file has '{actual_activity_id}'.")
            
        # 'experiment' comparison

        expected_experiment = getattr(reference_term, 'experiment', None)
        actual_experiment = str(ds.getncattr("experiment"))
        if expected_experiment and actual_experiment != expected_experiment:
            failures.append(f"Inconsistency for 'experiment': CV expects '{expected_experiment}', file has '{actual_experiment}'.")
        
        # Comparison of ‘parent_experiment_id’ (case insensitive)
        expected_parent_id = getattr(reference_term, 'parent_experiment_id', [None])[0]
        actual_parent_id = str(ds.getncattr("parent_experiment_id"))
        if expected_parent_id and actual_parent_id.lower() != expected_parent_id.lower():
            failures.append(f"Inconsistency for 'parent_experiment_id': CV expects '{expected_parent_id}', file has '{actual_parent_id}'.")
        # Comparison of ‘sub_experiment_id’ (case insensitive)
        expected_sub_experiment_id = getattr(reference_term, 'sub_experiment_id', [None])[0]
        actual_expected_sub_experiment_id = str(ds.getncattr("sub_experiment_id"))
        if expected_sub_experiment_id and actual_expected_sub_experiment_id.lower() != expected_sub_experiment_id.lower():
            failures.append(f"Inconsistency for 'sub_experiment_id': CV expects '{expected_sub_experiment_id}', file has '{actual_expected_sub_experiment_id}'.")
        # Report results
        if not failures:
            ctx.add_pass()
        else:   
            for f in failures:
                ctx.add_failure(f)

    except AttributeError as e:
        ctx.add_failure(f"Missing a required global attribute for the check (e.g., 'experiment_id'): {e}")
    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]
