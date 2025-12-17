import compliance_checker.cf.util as cfutil
from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util

from checks.utils import severity_word


def check_calendar_cmip7(ds, severity=BaseCheck.MEDIUM):
    """
    Checks if the "standard" calendar is used, and if so, promotes the use of "proleptic_gregorian" instead.

    1. The "standard" calendar includes "missing" dates in the transition from Gregorian to Julian, causing confusion.
    2. In the period following the Julian period (i.e., after October 1582) the "proleptic_gregorian" calendar is no different from the "standard" calendar.
    3. CF conventions 3.14 recommends that models avoid adopting the "standard" calendar.
    This check can likely be removed once a comparable check is implemented in the CF checker.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "C7OR001"
    desc = f"[{check_id}] CMIP7 Output Requirements - calendar attribute"
    testctx = TestCtx(severity, desc)
    failure_registered = False

    # This will only fetch variables with time units defined
    # (adapted from CF checker's `check_calendar`)
    for time_var_name in cfutil.get_time_variables(ds):
        if time_var_name not in {var.name for var in util.find_coord_vars(ds)}:
            continue
        time_var = ds.variables[time_var_name]
        # If has a calendar, raise an issue if the 'standard' calendar is used
        if not hasattr(time_var, "calendar"):
            continue
        if time_var.calendar.lower() == "standard":
            err_msg = (
                f"Variable '{time_var.name}' has a calendar attribute with the value 'standard'. "
                f"It is {severity_word(severity)} to use 'proleptic_gregorian' instead."
            )
            testctx.add_failure(err_msg)
            failure_registered = True
    if not failure_registered:
        testctx.add_pass()
    return [testctx.to_result()]
