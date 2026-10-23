import compliance_checker.cf.util as cfutil
from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util

from checks.utils import severity_word


def check_calendar_recommendation(
    ds,
    severity=BaseCheck.MEDIUM,
    *,
    discouraged_values=("standard",),
    preferred_value="proleptic_gregorian",
):
    """
    Checks for configured discouraged calendars and promotes the preferred one.
    1. The "standard" calendar includes "missing" dates in the transition from Julian to Gregorian, causing confusion.
    2. In the period following the Julian period (i.e., after October 1582) the "proleptic_gregorian" calendar is no different from the "standard" calendar.
    3. CF conventions 3.14 recommends that models avoid adopting the "standard" calendar.
    This check can likely be removed once a comparable check is implemented in the CF checker.

    Note: this is a recommendation only. The validity of the calendar value
    itself is checked separately by the ATTR enum check on the calendar
    attribute (which legitimately accepts "standard").
    """
    check_id = "TIME003a"
    desc = f"[{check_id}] Calendar for time coordinate"
    testctx = TestCtx(severity, desc)
    failure_registered = False
    discouraged = {
        str(value).strip().lower()
        for value in discouraged_values
        if str(value).strip()
    }
    preferred = str(preferred_value).strip()

    # This will only fetch variables with time units defined

    for time_var_name in cfutil.get_time_variables(ds):
        if time_var_name not in {var.name for var in util.find_coord_vars(ds)}:
            continue
        time_var = ds.variables[time_var_name]
        # Allowed-value validation is owned separately by ATTR004. This check
        # only reports values configured as valid but discouraged.
        if not hasattr(time_var, "calendar"):
            continue
        calendar = str(time_var.calendar).strip()
        if calendar.lower() in discouraged:
            err_msg = (
                f"Variable '{time_var.name}' has a calendar attribute with the "
                f"value {calendar!r}. It is {severity_word(severity)} to use "
                f"{preferred!r} instead."
            )
            testctx.add_failure(err_msg)
            failure_registered = True
    if not failure_registered:
        testctx.add_pass()
    return [testctx.to_result()]
