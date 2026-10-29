"""Data Request region association check."""

from compliance_checker.base import BaseCheck, TestCtx


def infer_regions_from_compound_names(compound_names):
    """Return regions encoded by project compound names, when all are usable."""
    if not compound_names:
        return None

    if isinstance(compound_names, str):
        compound_names = [compound_names]

    regions = []
    for value in compound_names:
        compound_name = str(value).strip()
        prefix, separator, region = compound_name.rpartition(".")
        region = region.strip()
        if not separator or not prefix.strip() or not region:
            return None
        if region not in regions:
            regions.append(region)
    return regions or None


def check_region_branded_variable_consistency(
    ds,
    expected_regions,
    severity,
    *,
    project_label,
    branded_variable_id=None,
    missing_attributes_delegated=(),
    invalid_region_delegated=False,
    region_is_registered=True,
    region_error=None,
    metadata_error=None,
    metadata_error_delegated=False,
):
    """Compare the file region with regions inferred from compound names."""
    ctx = TestCtx(
        severity,
        "[ATTR012] Consistency: region vs branded_variable",
    )

    delegated = set(missing_attributes_delegated)
    missing = [
        name for name in ("region", "branded_variable") if name not in ds.ncattrs()
    ]
    unowned = [name for name in missing if name not in delegated]
    if missing:
        for name in unowned:
            ctx.add_failure(
                f"Global attribute {name!r} is missing, so its association "
                f"with the {project_label} Data Request could not be checked."
            )
        return [ctx.to_result()] if unowned else []

    region = str(ds.getncattr("region")).strip()
    branded = str(
        branded_variable_id or ds.getncattr("branded_variable")
    ).strip()

    if region_error:
        if invalid_region_delegated:
            return []
        ctx.add_failure(
            f"The {project_label} region vocabulary could not be read, so the "
            f"association between region {region!r} and branded variable "
            f"{branded!r} could not be checked. Technical reason: {region_error}"
        )
        return [ctx.to_result()]

    if region_is_registered is False:
        if invalid_region_delegated:
            return []
        ctx.add_failure(
            f"Region {region!r} is not a registered {project_label} region, so "
            f"its association with branded variable {branded!r} could not be checked."
        )
        return [ctx.to_result()]

    if metadata_error:
        if metadata_error_delegated:
            return []
        ctx.add_failure(
            f"Branded-variable metadata for {branded!r} could not be resolved, "
            "so its registered regions could not be checked. Technical reason: "
            f"{metadata_error}"
        )
        return [ctx.to_result()]

    # An absent or unparseable compound-name list is catalogue metadata we
    # cannot safely interpret. Do not turn that into a finding against the file.
    if not expected_regions:
        return []

    allowed = [
        str(value).strip() for value in expected_regions if str(value).strip()
    ]
    if region in allowed:
        ctx.add_pass()
    else:
        registered = ", ".join(repr(value) for value in allowed)
        permission = (
            " Submission for additional regions is allowed."
            if severity == BaseCheck.LOW
            else ""
        )
        ctx.add_failure(
            f"Region {region!r} is not listed for branded variable {branded!r} "
            f"in the {project_label} Data Request. Regions inferred from its "
            f"registered compound names are [{registered}].{permission}"
        )
    return [ctx.to_result()]
