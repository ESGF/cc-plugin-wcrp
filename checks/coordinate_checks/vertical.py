"""Generic model-level coordinate and formula-term validation."""

from __future__ import annotations

from checks.coordinate_checks.formulas import formula_vertical_profile
from checks.coordinate_checks.model import reference_id, reference_ids
from checks.coordinate_checks.utils import (
    expected_formula_terms,
    formatted,
    ncattr,
    parse_formula_terms,
)
from checks.coordinate_checks.validation import (
    Findings,
    check_attributes,
    check_bounds,
    check_direct_vertical_values,
    check_direction,
    check_dtype,
    check_profile_direction,
    check_valid_range,
    report_allowed_when_unset,
)


def select_model_level(findings: Findings, ds, catalog, generic_id: str):
    generic = catalog.data_coordinates[generic_id]
    out_name = str(generic.get("out_name") or "").strip()
    expectation = (
        f"Coordinate {generic_id!r} could not be verified. Expected is a generic "
        f"vertical coordinate with out_name {out_name!r}."
        if out_name
        else f"Coordinate {generic_id!r} could not be verified."
    )
    if not out_name:
        findings.add(
            "formula",
            f"{expectation} Its generic data-coordinate definition does not define "
            "out_name, so no file variable can be selected without making an "
            "assumption.",
        )
        return None
    candidates = [
        (identifier, entry)
        for identifier, entry in catalog.model_levels.items()
        if reference_id(entry.get("generic_level_name")) == generic_id
    ]
    if out_name not in ds.variables:
        findings.add(
            "identity",
            f"{expectation} Variable {out_name!r} does not exist in the file.",
        )
        return None
    var = ds.variables[out_name]
    standard_name = ncattr(var, "standard_name")
    if not standard_name:
        findings.add(
            "formula",
            f"{expectation} Variable {out_name!r} has no standard_name, which is "
            f"{findings.severity_word('formula')} to select a "
            "model_level_coordinate.",
        )
        return None
    matches = [
        item for item in candidates if item[1].get("cf_standard_name") == standard_name
    ]
    if not matches:
        available = sorted(
            {str(entry.get("cf_standard_name") or "<unset>") for _, entry in candidates}
        )
        findings.add(
            "formula",
            f"{expectation} No model_level_coordinate marked for "
            f"generic_level_name={generic_id!r} has standard_name="
            f"{standard_name!r}; available standard names are {available}.",
        )
        return None
    if len(matches) > 1:
        actual_formula = ncattr(var, "formula")
        choices = {
            identifier: str(entry.get("formula") or "") for identifier, entry in matches
        }
        if not actual_formula:
            findings.add(
                "formula",
                f"{expectation} standard_name={standard_name!r} matches multiple "
                "model_level_coordinate records, but the file variable has "
                f"no formula with which to select one. Candidate formulas are {choices}.",
            )
            return None
        formula_matches = [
            item
            for item in matches
            if str(item[1].get("formula") or "") == actual_formula
        ]
        if len(formula_matches) == 1:
            matches = formula_matches
        elif not formula_matches:
            findings.add(
                "formula",
                f"{expectation} standard_name={standard_name!r} matches multiple "
                "model_level_coordinate records, but formula="
                f"{actual_formula!r} does not match any candidate formula {choices}.",
            )
            return None
        else:
            identifiers = [identifier for identifier, _ in formula_matches]
            findings.add(
                "formula",
                f"{expectation} standard_name={standard_name!r} and formula="
                f"{actual_formula!r} still match multiple records "
                f"{identifiers}; a unique model_level_coordinate cannot be selected.",
            )
            return None
    if len(matches) != 1:
        findings.add(
            "formula",
            f"{expectation} The coordinate catalogue contains {len(matches)} matching "
            "model_level_coordinate records, so a unique record cannot be selected.",
        )
        return None
    return matches[0][0], matches[0][1], var


def validate_formula_term(
    findings,
    ds,
    name,
    entry,
    generic_id,
    catalog,
    horizontal_dimensions,
    *,
    bounds=False,
):
    if name not in ds.variables:
        findings.add(
            "formula",
            f"The {findings.severity_word('formula')} formula term variable "
            f"'{name}' for the selected model-level coordinate is absent.",
        )
        return
    var = ds.variables[name]
    if not entry.get("out_name"):
        findings.add(
            "formula",
            f"Formula term '{name}' has no usable coordinate-catalogue descriptor.",
        )
        return
    check_dtype(findings, var, name, entry, family="formula")
    check_attributes(findings, var, name, entry, fallback_long_name=False)
    expected_dimensions = reference_ids(entry.get("dimensions"))
    if not expected_dimensions:
        if var.ndim:
            findings.add(
                "formula",
                f"Formula term '{name}' has dimensions {list(var.dimensions)}; "
                f"the {findings.severity_word('formula')} representation "
                "is dimensionless.",
            )
        return
    if expected_dimensions:
        expected = []
        horizontal_added = False
        level_name = str(
            catalog.data_coordinates.get(generic_id, {}).get("out_name") or ""
        )
        for dimension_id in reversed(expected_dimensions):
            coordinate_entry = catalog.data_coordinates.get(dimension_id, {})
            coordinate_kind = reference_id(coordinate_entry.get("coordinate_type"))
            if coordinate_kind == "generic_horizontal":
                if horizontal_dimensions is None:
                    # Grid topology resolution already emitted one decisive
                    # failure. Exact factor dimensions cannot be derived safely.
                    return
                if not horizontal_added:
                    expected.extend(horizontal_dimensions)
                    horizontal_added = True
                continue
            if dimension_id == generic_id:
                expected.append(level_name)
            else:
                expected.append(str(coordinate_entry.get("out_name") or dimension_id))
        actual = list(var.dimensions)
        # Bounds factor entries add a size-two dimension to the selected level.
        if bounds and level_name in expected:
            level_index = expected.index(level_name)
            valid = (
                var.ndim == len(expected) + 1
                and actual[level_index] == level_name
                and var.shape[-1] == 2
                and actual[:-1] == expected
            )
            if not valid:
                findings.add(
                    "formula",
                    f"Bounds formula term '{name}' has dimensions {actual}; expected "
                    "the level dimension followed by a size-2 bounds dimension "
                    "(plus any horizontal dimensions).",
                )
        elif actual != expected:
            findings.add(
                "formula",
                f"Formula term '{name}' has dimensions {actual}; its coordinate "
                f"definition describes {expected_dimensions}; the "
                f"{findings.severity_word('formula')} netCDF/C dimensions are "
                f"{expected}.",
            )


def validate_model_level(
    findings: Findings,
    ds,
    catalog,
    generic_id: str,
    horizontal_dimensions,
    *,
    check_direct_physical_values: bool = True,
    check_formula_derived_profile: bool = True,
    attributes_allowed_when_unset=(),
):
    selected = select_model_level(findings, ds, catalog, generic_id)
    if selected is None:
        return None
    level_id, level, var = selected
    name = var.name
    model_out_name = str(level.get("out_name") or "").strip()
    if model_out_name and model_out_name != name:
        findings.add(
            "identity",
            f"Selected model-level entry {level_id!r} has out_name "
            f"{model_out_name!r}, but generic coordinate {generic_id!r} selected "
            f"file variable {name!r}.",
        )
    if var.dimensions != (name,):
        findings.add(
            "identity",
            f"The {findings.severity_word('identity')} generic vertical coordinate "
            f"is '{name}({name})'; found "
            f"'{var.name}({', '.join(var.dimensions)})'.",
        )
    check_dtype(findings, var, name, level)
    check_attributes(
        findings,
        var,
        name,
        level,
        empty_values_must_be_absent=True,
        allowed_when_unset=attributes_allowed_when_unset,
    )
    expected_computed_standard_name = str(level.get("computed_standard_name") or "")
    actual_computed_standard_name = ncattr(var, "computed_standard_name")
    if (
        expected_computed_standard_name
        and actual_computed_standard_name != expected_computed_standard_name
    ):
        findings.add(
            "attributes",
            f"'{name}' computed_standard_name="
            f"{formatted(actual_computed_standard_name)}; the "
            f"{findings.severity_word('attributes')} value is "
            f"{expected_computed_standard_name!r}.",
        )
    elif (
        not expected_computed_standard_name
        and "computed_standard_name" in var.ncattrs()
    ):
        if "computed_standard_name" in attributes_allowed_when_unset:
            report_allowed_when_unset(
                findings,
                var,
                name,
                "computed_standard_name",
                actual_computed_standard_name,
            )
        else:
            findings.add(
                "attributes",
                f"'{name}' defines computed_standard_name="
                f"{formatted(actual_computed_standard_name)}; it is "
                f"{findings.severity_word('attributes')} for the attribute to be "
                "absent when its coordinate-definition value is empty.",
            )
    check_direction(findings, var, name, level)
    check_valid_range(findings, var, name, level)

    expected_formula = str(level.get("formula") or "")
    actual_formula = ncattr(var, "formula")
    if expected_formula and actual_formula != expected_formula:
        findings.add(
            "formula",
            f"'{name}' formula={formatted(actual_formula)}; selected model-level entry "
            f"{level_id!r} has {expected_formula!r} as its "
            f"{findings.severity_word('formula')} value.",
        )
    elif not expected_formula and "formula" in var.ncattrs():
        if "formula" in attributes_allowed_when_unset:
            report_allowed_when_unset(
                findings,
                var,
                name,
                "formula",
                actual_formula,
            )
        else:
            findings.add(
                "formula",
                f"'{name}' defines formula={formatted(actual_formula)}; it is "
                f"{findings.severity_word('formula')} for the attribute to be absent "
                f"because selected model-level entry {level_id!r} defines an empty "
                "value.",
            )

    term_entries = {}
    expected_terms = expected_formula_terms(
        level,
        catalog.formula_terms,
        generic_id=generic_id,
        resolved=term_entries,
    )
    actual_terms = parse_formula_terms(ncattr(var, "formula_terms"))
    if not expected_terms and "formula_terms" in var.ncattrs():
        if "formula_terms" in attributes_allowed_when_unset:
            report_allowed_when_unset(
                findings,
                var,
                name,
                "formula_terms",
                ncattr(var, "formula_terms"),
            )
        else:
            findings.add(
                "formula",
                f"'{name}' defines formula_terms={actual_terms}; it is "
                f"{findings.severity_word('formula')} for the attribute to be absent "
                f"because selected model-level entry {level_id!r} defines no formula "
                "terms.",
            )
    elif expected_terms != actual_terms:
        findings.add(
            "formula",
            f"'{name}' formula_terms={actual_terms}; selected model-level entry "
            f"{level_id!r} has {expected_terms} as its "
            f"{findings.severity_word('formula')} value.",
        )
    for symbol, term_name in expected_terms.items():
        if term_name == name:
            continue
        term_entry = term_entries.get(symbol, {})
        validate_formula_term(
            findings,
            ds,
            term_name,
            term_entry,
            generic_id,
            catalog,
            horizontal_dimensions,
        )

    stored_direction = str(level.get("stored_direction") or "")
    formula_inputs_are_valid = (
        bool(expected_formula)
        and bool(expected_terms)
        and actual_terms == expected_terms
        and all(term_name in ds.variables for term_name in expected_terms.values())
    )
    if (
        check_formula_derived_profile
        and formula_inputs_are_valid
        and stored_direction in {"increasing", "decreasing"}
    ):
        try:
            profile, sample = formula_vertical_profile(
                ds,
                var,
                level,
                expected_terms,
            )
            check_profile_direction(
                findings,
                profile,
                f"Formula-derived physical profile for '{name}' at {sample}",
                stored_direction,
            )
        except Exception as exc:
            # This optional physical check must not stop the catalogue suite.
            findings.add(
                "direction",
                f"Could not evaluate the physical vertical profile for '{name}': "
                f"{type(exc).__name__}: {exc}.",
            )
    elif check_direct_physical_values and not expected_formula:
        check_direct_vertical_values(findings, var, name, level)

    allowed_attributes = tuple(
        attribute
        for attribute in ("formula", "formula_terms")
        if attribute in var.ncattrs()
    )
    bounds_var = check_bounds(
        findings, ds, var, name, level, allowed_attributes=allowed_attributes
    )
    if bounds_var is not None and "formula" in bounds_var.ncattrs():
        if ncattr(bounds_var, "formula") != actual_formula:
            findings.add(
                "formula",
                f"'{bounds_var.name}' formula={formatted(ncattr(bounds_var, 'formula'))}; "
                f"the {findings.severity_word('formula')} value is the same formula "
                f"as its parent '{name}': {formatted(actual_formula)}.",
            )
    bounds_term_entries = {}
    expected_bounds_terms = expected_formula_terms(
        level,
        catalog.formula_terms,
        generic_id=generic_id,
        bounds=True,
        resolved=bounds_term_entries,
    )
    if expected_bounds_terms:
        if bounds_var is None:
            # Missing bounds is already reported; avoid a cascade per factor.
            return var
        actual_bounds_terms = parse_formula_terms(ncattr(bounds_var, "formula_terms"))
        if actual_bounds_terms != expected_bounds_terms:
            findings.add(
                "formula",
                f"'{bounds_var.name}' formula_terms={actual_bounds_terms}; selected "
                f"model-level entry {level_id!r} has {expected_bounds_terms} as its "
                f"{findings.severity_word('formula')} value.",
            )
        for symbol, term_name in expected_bounds_terms.items():
            if term_name == bounds_var.name:
                continue
            term_entry = bounds_term_entries.get(symbol, {})
            validate_formula_term(
                findings,
                ds,
                term_name,
                term_entry,
                generic_id,
                catalog,
                horizontal_dimensions,
                bounds=True,
            )
    return var
