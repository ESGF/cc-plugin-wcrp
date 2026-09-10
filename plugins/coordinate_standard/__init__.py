"""Coordinate validation engine built on compliance-checker and cf-xarray.

Builds a standard from the CMOR tables (later esgvoc), classifies a netCDF
file the same way, then matches the two and reports differences. Structural
detection, role detectors and units logic come from compliance_checker.cf.util
(and cf.appendix_d for parametric vertical coordinates). Name vocabularies
come from cf_xarray.criteria. Result objects are created in
checks/coordinate_checks/.

Modules: model (enums and containers), rules (classification rules),
classify (file-side classifier), providers (CMOR tables / esgvoc seam),
matching (match and diff), report (plain-text report).

Example:
    from plugins.coordinate_standard import (
        CmorTableProvider, classify_dataset, match, read_1d_values,
    )

    std = CmorTableProvider(tables={
        "coordinate": CTcoords, "grids": CTgrids, "formula_terms": CTformulas,
    }).build()
    kinds, candidates = classify_dataset(ds)
    outcome, detail = match(candidate, read_1d_values(ds.variables[name]), std)
"""
from plugins.coordinate_standard.classify import (
    attr,
    classify_dataset,
    neutral_dtype,
    read_1d_values,
    read_scalar,
)
from plugins.coordinate_standard.matching import (
    GOOD_OUTCOMES,
    WARN_OUTCOMES,
    diff,
    match,
    missing_coordinates,
)
from plugins.coordinate_standard.model import (
    Coordinate,
    Representation,
    Role,
    Standard,
    VariableKind,
)
from plugins.coordinate_standard.providers import (
    CmorTableProvider,
    EsgvocProvider,
    StandardProvider,
    esgvoc_name_findings,
    esgvoc_readiness,
)
from plugins.coordinate_standard.report import report
from plugins.coordinate_standard.rules import (
    PARAMETRIC_STANDARD_NAMES,
    ROLE_STANDARD_NAMES,
    VERTICAL_STANDARD_NAMES,
    _role_from_weak_signals,
    classify_representation,
    classify_role,
)

__all__ = [
    # model
    "Role", "Representation", "VariableKind", "Coordinate", "Standard",
    # provider
    "StandardProvider", "CmorTableProvider", "EsgvocProvider",
    # esgvoc helpers (optional)
    "esgvoc_readiness", "esgvoc_name_findings",
    # classification
    "classify_dataset", "classify_role", "classify_representation", "read_1d_values",
    # match
    "match", "diff", "missing_coordinates", "GOOD_OUTCOMES", "WARN_OUTCOMES",
    # report
    "report",
]
