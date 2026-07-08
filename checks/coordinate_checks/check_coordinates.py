#!/usr/bin/env python
"""
Validate a file's coordinates against the CMOR-table standard.

This is the compliance-checker glue. It builds the standard from the already
loaded CMOR tables (passed in -- no re-reading), classifies the file with the
``plugins.coordinate_standard`` engine, matches each coordinate to its standard
entry, and emits one Result per coordinate. Soft findings (matched with
warnings, or unresolvable through no fault of the file) pass the main check
and are surfaced as LOW-severity advisories.

Facets with an existing atomic check are delegated instead of re-implemented:

* direction: once a coordinate is pinned to its table entry, the entry's
  ``stored_direction`` is fed to [VAR005] ``check_coordinate_monotonicity``.
  Coordinates whose direction is already handled by a plugin's TOML rules can
  be excluded via ``skip_direction_for`` (the TOML rule wins; no double check).
* missing coordinate variables: resolved by the engine's
  ``missing_coordinates`` (a data variable uses a dimension matching a table
  ``out_name``, but no such variable exists), then reported through [VAR001]
  ``check_variable_existence``.
* bounds correctness ([VAR004]/[VAR012]) is NOT invoked here: the plugins
  already run both for every coordinate that has a ``bounds`` attribute. The
  only gap -- bounds required by the table but absent -- is caught by the
  engine's ``diff`` ("missing bounds").

Wiring (in a plugin's check method)::

    from checks.coordinate_checks.check_coordinates import check_coordinates

    def check_coordinates_vs_standard(self, ds):
        return check_coordinates(
            ds, self.CTcoords, self.CTgrids, self.CTformulas,
            severity=BaseCheck.HIGH,
            skip_direction_for=set(self.coords_cfg_monotonicity_names),
        )

``ds`` is the netCDF4 dataset compliance-checker passes to NC checks.
"""
from compliance_checker.base import BaseCheck, TestCtx

from checks.variable_checks.check_coordinate_monotonicity import (
    check_coordinate_monotonicity,
)
from checks.variable_checks.check_variable_existence import (
    check_variable_existence,
)
from plugins.coordinate_standard import (
    GOOD_OUTCOMES,
    WARN_OUTCOMES,
    CmorTableProvider,
    classify_dataset,
    match,
    missing_coordinates,
    read_1d_values,
)

CHECK_ID = "COORD001"

# Outcomes whose detail is the resolved standard-entry name; only then can
# table-driven parameters (stored_direction) be handed to other checks.
_RESOLVED = ("MATCH", "PINNED", "MATCH_WITH_WARNINGS", "MISMATCH")


def check_coordinates(ds, ctcoords, ctgrids, ctformulas,
                      severity=BaseCheck.HIGH, project="CMOR",
                      skip_direction_for=None, ct_dimensions=None):
    """Check every coordinate in ``ds`` against the standard.

    Parameters
    ----------
    ds : netCDF4.Dataset
        The dataset being checked.
    ctcoords, ctgrids, ctformulas : dict
        The already-loaded CMOR coordinate / grids / formula_terms tables
        (e.g. ``self.CTcoords`` etc.).
    severity : str
        Check severity (default: ``BaseCheck.HIGH``).
    project : str
        Project label for provenance, e.g. "CMIP6" / "CORDEX-CMIP6".
    skip_direction_for : iterable of str, optional
        Coordinate names whose monotonicity direction is already checked
        elsewhere (e.g. via a plugin's TOML ``monotonicity`` rule); [VAR005]
        is not invoked for these.
    ct_dimensions : iterable of str, optional
        Token list from the data variable's CMOR *variable table* entry
        ``dimensions`` string (e.g. ``self.CT[table_id]["variable_entry"]
        [var]["dimensions"].split()``). Enables detection of missing scalar
        and auxiliary coordinates (``height2m``, 2-D lat/lon) that are never
        netCDF dimensions.

    Returns
    -------
    list[Result]
        One Result per coordinate variable found in the file, [VAR005]
        Results for table-directed coordinates, [VAR001] Results for
        expected-but-missing coordinate variables, and a LOW-severity
        advisory Result per soft finding.
    """
    skip_direction = set(skip_direction_for or ())

    std = CmorTableProvider(prefix=project, tables={
        "coordinate": ctcoords,
        "grids": ctgrids,
        "formula_terms": ctformulas,
    }).build()

    kinds, candidates = classify_dataset(ds)
    if not candidates:
        ctx = TestCtx(severity, f"[{CHECK_ID}] Coordinates")
        ctx.add_pass()                       # nothing to check in this file
        return [ctx.to_result()]

    results = []
    for name, candidate in candidates.items():
        ctx = TestCtx(severity, f"[{CHECK_ID}] Coordinate '{name}'")
        try:
            values = read_1d_values(ds.variables[name])
            outcome, detail = match(candidate, values, std)
        except Exception as e:   # a malformed entry must not kill the suite
            ctx.add_failure(
                f"internal error while checking coordinate '{name}': {e}")
            results.append(ctx.to_result())
            continue
        kind = outcome.split(":", 1)[0]      # strip the ":<why>" off MISMATCH

        if kind in GOOD_OUTCOMES:
            ctx.add_pass()
        elif kind in WARN_OUTCOMES:
            # Matched with soft issues, or cannot resolve which entry (not
            # the file's fault, e.g. time/lat/lon): pass the main check and
            # surface the finding as a LOW-severity advisory.
            ctx.add_pass()
            warn_ctx = TestCtx(BaseCheck.LOW,
                               f"[{CHECK_ID}] Coordinate '{name}' (advisory)")
            warn_ctx.add_failure(
                f"coordinate '{name}': {outcome}"
                + (f" [candidates: {detail}]" if detail else ""))
            results.append(warn_ctx.to_result())
        else:
            ctx.add_failure(
                f"coordinate '{name}' ({candidate.role.value}, "
                f"{candidate.representation.value}) does not match the standard: "
                f"{outcome}" + (f" [candidates: {detail}]" if detail else "")
            )
        results.append(ctx.to_result())

        # Direction: delegate to [VAR005] with the pinned entry's
        # stored_direction (TOML-managed coordinates are skipped -- the
        # plugin's own VAR005 call wins).
        if kind in _RESOLVED and isinstance(detail, str) and name not in skip_direction:
            entry = next(
                (c for c in std.coordinates
                 if c.name == detail
                 and c.representation == candidate.representation), None)
            if (entry is not None and entry.stored_direction
                    and name in getattr(ds, "dimensions", {})):
                results.extend(check_coordinate_monotonicity(
                    ds, coord_name=name,
                    direction=entry.stored_direction, severity=severity))

    # Expected coordinate variables that don't exist (a data variable uses a
    # dimension that matches a table out_name, but no such variable exists):
    # resolved by the engine, reported through [VAR001].
    try:
        missing = missing_coordinates(ds, kinds, std,
                                      ct_dimensions=ct_dimensions)
    except Exception as e:
        ctx = TestCtx(severity, f"[{CHECK_ID}] Missing coordinates")
        ctx.add_failure(f"internal error while checking for missing "
                        f"coordinates: {e}")
        results.append(ctx.to_result())
        missing = {}
    for name in missing:
        results.extend(check_variable_existence(ds, name, severity))

    return results
