"""Data Request frequency association check."""

from compliance_checker.base import BaseCheck, TestCtx


def check_frequency_branded_variable_consistency(
    ds,
    expected_frequencies,
    severity,
    *,
    project_label,
    branded_variable_id=None,
    missing_attributes_delegated=(),
    invalid_frequency_delegated=False,
    frequency_is_registered=True,
    frequency_error=None,
    metadata_error=None,
    metadata_error_delegated=False,
):
    """Compare the file frequency with the branded-variable frequency list."""
    ctx = TestCtx(
        severity,
        "[ATTR008] Consistency: frequency vs branded_variable",
    )

    delegated = set(missing_attributes_delegated)
    missing = [
        name for name in ("frequency", "branded_variable") if name not in ds.ncattrs()
    ]
    unowned = [name for name in missing if name not in delegated]
    if missing:
        for name in unowned:
            ctx.add_failure(
                f"Global attribute {name!r} is missing, so its association "
                f"with the {project_label} Data Request could not be checked."
            )
        return [ctx.to_result()] if unowned else []

    frequency = str(ds.getncattr("frequency")).strip()
    branded = str(
        branded_variable_id or ds.getncattr("branded_variable")
    ).strip()

    if frequency_error:
        if invalid_frequency_delegated:
            return []
        ctx.add_failure(
            f"The {project_label} frequency vocabulary could not be read, so the "
            f"association between frequency {frequency!r} and branded variable "
            f"{branded!r} could not be checked. Technical reason: "
            f"{frequency_error}"
        )
        return [ctx.to_result()]

    if frequency_is_registered is False:
        if invalid_frequency_delegated:
            return []
        ctx.add_failure(
            f"Frequency {frequency!r} is not a registered {project_label} "
            "frequency, so "
            f"its association with branded variable {branded!r} could not be checked."
        )
        return [ctx.to_result()]

    if metadata_error:
        if metadata_error_delegated:
            return []
        ctx.add_failure(
            f"Branded-variable metadata for {branded!r} could not be resolved, "
            "so its registered frequencies could not be checked. Technical "
            f"reason: {metadata_error}"
        )
        return [ctx.to_result()]

    # An absent frequency field is not interpreted as an empty list. It may be
    # an older or incomplete record, and another catalogue-integrity process is
    # better placed to diagnose it.
    if not expected_frequencies:
        return []

    allowed = [
        str(value).strip()
        for value in expected_frequencies
        if str(value).strip()
    ]
    if frequency in allowed:
        ctx.add_pass()
    else:
        registered = ", ".join(repr(value) for value in allowed)
        permission = (
            " Submission at additional frequencies is allowed."
            if severity == BaseCheck.LOW
            else ""
        )
        ctx.add_failure(
            f"Frequency {frequency!r} is not listed for branded variable "
            f"{branded!r} in the {project_label} Data Request. Registered "
            f"frequencies are [{registered}].{permission}"
        )
    return [ctx.to_result()]
