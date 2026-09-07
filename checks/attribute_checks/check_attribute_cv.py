#!/usr/bin/env python

from compliance_checker.base import BaseCheck, TestCtx

from checks.utils import _compare_CV


def check_required_global_attributes_existence_cv(
    CheckerObject, severity=BaseCheck.HIGH, use_esgvoc=False
):
    """[ATTR001] Checks existence of required global attributes against <project>_CV.json."""
    check_id = "ATTR001"
    desc = f"[{check_id}] Global Attribute Existence"
    testctx = TestCtx(severity, desc)

    # Do not run comparison against CV if esgvoc is used
    if use_esgvoc:
        testctx.add_pass()
        return [testctx.to_result()]

    required_attributes = CheckerObject.CV.get("required_global_attributes", {})

    for attr in required_attributes:
        test = attr in list(CheckerObject.dataset.ncattrs())
        if not test:
            testctx.add_failure(f"Required global attribute '{attr}' is missing.")
        else:
            testctx.add_pass()

    return [testctx.to_result()]


def check_required_global_attributes_value_cv(
    CheckerObject,
    global_attrs_hard_checks=[],
    severity=BaseCheck.HIGH,
    use_esgvoc=False,
):
    """[ATTR004] Checks value of required global attributes against <project>_CV.json."""
    check_id = "ATTR004"
    desc = f"[{check_id}] Global Attribute Value"
    testctx = TestCtx(severity, desc)

    # Do not run comparison against CV if esgvoc is used
    if use_esgvoc:
        testctx.add_pass()
        return [testctx.to_result()]

    required_attributes = CheckerObject.CV.get("required_global_attributes", {})
    file_attrs = {
        key: CheckerObject.dataset.getncattr(key)
        for key in CheckerObject.dataset.ncattrs()
        if key in required_attributes
    }
    for k in required_attributes:
        if k not in file_attrs:
            file_attrs[k] = "unset"

    # Global attributes
    ga_checked, ga_messages = _compare_CV(
        CheckerObject, file_attrs, "Global attribute "
    )
    if len(ga_messages) == 0:
        testctx.add_pass()
    else:
        for message in ga_messages:
            testctx.add_failure(message)

    # Unchecked global attributes
    unchecked = [
        key
        for key in required_attributes
        if not ga_checked[key] and key not in global_attrs_hard_checks
    ]
    if len(unchecked) == 0:
        testctx.add_pass()
    else:
        testctx.add_failure(
            f"""Required global attributes could not be checked against CV: {', '.join(f"'{ukey}'" for ukey in sorted(unchecked))}."""
        )
    return [testctx.to_result()]
