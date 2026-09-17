"""Adapt ESGVoc global-attribute reports to Compliance Checker results.

This module deliberately contains no project policy.  ESGVoc supplies the
project attribute specification, while callers supply per-attribute severity
from their TOML configuration.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx
from esgvoc.apps.ncattvalid import GAValidator

from checks.attribute_checks.check_attribute_suite import check_attribute_suite


@cache
def get_global_attribute_validator(project_id: str) -> GAValidator:
    """Build and cache one stateless validator for each ESGVoc project."""
    return GAValidator(project_id=project_id)


def _python_scalar(value: Any) -> Any:
    """Convert NumPy scalars and bytes to values accepted by ESGVoc."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _string_array_attribute_names(validator: GAValidator) -> set[str]:
    """Return attributes declared as string arrays by the active project.

    ESGVoc 6.1 does not expose the loaded attribute specification through a
    public property.  Keep this compatibility access in one place so it can be
    removed when the library exposes that information publicly.
    """
    names: set[str] = set()
    for spec in getattr(validator, "_specs", ()):
        if getattr(spec, "attr_field_value_type", None) != "string_array":
            continue
        name = getattr(spec, "attr_field_name", None) or getattr(
            spec, "source_collection", None
        )
        if name:
            names.add(str(name))
    return names


def _attribute_specifications(validator: GAValidator) -> dict[str, Any]:
    """Index the active ESGVoc attribute specifications by NetCDF name."""
    specifications: dict[str, Any] = {}
    for spec in getattr(validator, "_specs", ()):
        name = getattr(spec, "attr_field_name", None) or getattr(
            spec, "source_collection", None
        )
        if name:
            specifications[str(name)] = spec
    return specifications


def _rule_value(rule: Any, name: str, default: Any = None) -> Any:
    if isinstance(rule, Mapping):
        return rule.get(name, default)
    return getattr(rule, name, default)


def _has_local_value_rule(rule: Any) -> bool:
    """Return whether TOML defines an ATTR004 rule other than CV membership."""
    return any(
        (
            _rule_value(rule, "pattern") is not None,
            _rule_value(rule, "constant") is not None,
            _rule_value(rule, "threshold") is not None,
            _rule_value(rule, "enum") is not None,
            bool(_rule_value(rule, "as_variable")),
            bool(_rule_value(rule, "is_positive")),
            _rule_value(rule, "cv_source_term_key") is not None,
        )
    )


def normalize_global_attributes(ds, validator: GAValidator) -> dict[str, Any]:
    """Read and normalize NetCDF global attributes for ``GAValidator``.

    The current ESGVoc string-array implementation expects one whitespace-
    separated string.  NetCDF libraries can also expose NC_STRING arrays as a
    list or NumPy array, so normalize both representations here.
    """
    string_arrays = _string_array_attribute_names(validator)
    normalized: dict[str, Any] = {}

    for name in ds.ncattrs():
        value = ds.getncattr(name)
        if isinstance(value, np.ndarray):
            value = value.reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            value = " ".join(str(_python_scalar(item)).strip() for item in value)
        else:
            value = _python_scalar(value)

        if name in string_arrays and isinstance(value, str):
            # ``split(" ")`` in ESGVoc 6.1 produces empty terms for repeated
            # whitespace.  Rejoin with one separator without changing scalar
            # or free-text attributes.
            value = " ".join(value.split())
        normalized[str(name)] = value

    return normalized


def check_global_attributes_esgvoc(
    ds,
    project_id: str,
    *,
    severity_by_attribute: Mapping[str, int] | None = None,
    default_severity: int = BaseCheck.HIGH,
    validator: GAValidator | None = None,
    suppress_existence_for: set[str] | None = None,
    suppress_vocabulary_for: set[str] | None = None,
):
    """Validate global attributes with ESGVoc and return checker results.

    Extra attributes remain informational in ESGVoc and are intentionally not
    converted into failures.  Project-specific checks may still validate them
    through their TOML rules.
    """
    severities = severity_by_attribute or {}
    suppressed_existence = suppress_existence_for or set()
    suppressed_vocabulary = suppress_vocabulary_for or set()

    try:
        active_validator = validator or get_global_attribute_validator(project_id)
        attributes = normalize_global_attributes(ds, active_validator)
        filename = ds.filepath() if hasattr(ds, "filepath") else None
        report = active_validator.validate(attributes, filename=filename)
    # This is the integration boundary for database, vocabulary, NetCDF and
    # ESGVoc errors.  Report one setup failure instead of aborting the checker.
    except Exception as exc:  # noqa: BLE001
        ctx = TestCtx(
            default_severity,
            "[ATTR004] ESGVoc global attribute validation setup",
        )
        ctx.add_failure(
            f"Global attributes could not be validated for project "
            f"'{project_id}'. Technical reason: {type(exc).__name__}: {exc}"
        )
        return [ctx.to_result()]

    results = []

    for name in report.missing:
        if name in suppressed_existence:
            continue
        severity = severities.get(name, default_severity)
        ctx = TestCtx(severity, f"[ATTR001] Global attribute '{name}' existence")
        ctx.add_failure(f"Required global attribute '{name}' is missing.")
        results.append(ctx.to_result())

    grouped = defaultdict(list)
    for attribute_result in report.results:
        grouped[attribute_result.name].append(attribute_result)

    for name, attribute_results in grouped.items():
        severity = severities.get(name, default_severity)

        if name not in suppressed_existence:
            existence = TestCtx(
                severity,
                f"[ATTR001] Global attribute '{name}' existence",
            )
            existence.add_pass()
            results.append(existence.to_result())

        # Free-text attributes have no collection and therefore no ATTR004
        # vocabulary rule.
        vocabulary_results = [
            item for item in attribute_results if item.collection is not None
        ]
        if not vocabulary_results or name in suppressed_vocabulary:
            continue

        vocabulary = TestCtx(
            severity,
            f"[ATTR004] Global attribute '{name}' vocabulary check",
        )
        failures = [item for item in vocabulary_results if not item.is_valid]
        if failures:
            # One attribute is one ATTR004 assertion, including string arrays.
            # Combining token failures preserves the existing check score and
            # identity instead of increasing the denominator per invalid token.
            vocabulary.add_failure("; ".join(item.message for item in failures))
        else:
            vocabulary.add_pass()
        results.append(vocabulary.to_result())

    return results


def check_global_attributes_hybrid(
    ds,
    project_id: str,
    attribute_rules: Mapping[str, Any],
    severity_resolver: Callable[[Any], int],
    *,
    default_severity: int = BaseCheck.HIGH,
    validator: GAValidator | None = None,
):
    """Combine project ESGVoc specifications with local TOML rules.

    Existing TOML requiredness and non-vocabulary ATTR004 rules take precedence
    so adopting ESGVoc does not silently change established project policy.
    """
    try:
        active_validator = validator or get_global_attribute_validator(project_id)
        specifications = _attribute_specifications(active_validator)
    except Exception:  # noqa: BLE001
        active_validator = None
        specifications = {}

    severities: dict[str, int] = {}
    local_names: dict[str, tuple[str, Any]] = {}
    suppress_existence: set[str] = set()
    suppress_vocabulary: set[str] = set()

    for key, rule in attribute_rules.items():
        name = str(_rule_value(rule, "attribute_name") or key)
        local_names[name] = (str(key), rule)
        severities[name] = severity_resolver(_rule_value(rule, "severity"))

        spec = specifications.get(name)
        if spec is None:
            continue

        local_required = _rule_value(rule, "is_required", True)
        esgvoc_required = bool(getattr(spec, "is_required", False))
        if not isinstance(local_required, bool) or local_required != esgvoc_required:
            suppress_existence.add(name)

        if _has_local_value_rule(rule):
            suppress_vocabulary.add(name)

    results = check_global_attributes_esgvoc(
        ds,
        project_id,
        severity_by_attribute=severities,
        default_severity=default_severity,
        validator=active_validator,
        suppress_existence_for=suppress_existence,
        suppress_vocabulary_for=suppress_vocabulary,
    )

    for name, (_, rule) in local_names.items():
        spec = specifications.get(name)
        local_required = _rule_value(rule, "is_required", True)
        esgvoc_owns_existence = (
            spec is not None
            and isinstance(local_required, bool)
            and local_required == bool(getattr(spec, "is_required", False))
        )
        esgvoc_owns_vocabulary = (
            spec is not None
            and getattr(spec, "source_collection", None) is not None
            and not _has_local_value_rule(rule)
        )

        results.extend(
            check_attribute_suite(
                ds=ds,
                var_name=None,
                attribute_name=name,
                severity=severities[name],
                value_type=_rule_value(rule, "value_type"),
                is_required=local_required,
                na_value=_rule_value(rule, "na_value"),
                pattern=_rule_value(rule, "pattern"),
                constant=_rule_value(rule, "constant"),
                threshold=_rule_value(rule, "threshold"),
                is_above_threshold=_rule_value(rule, "is_above_threshold"),
                enum=_rule_value(rule, "enum"),
                as_variable=_rule_value(rule, "as_variable"),
                is_positive=_rule_value(rule, "is_positive"),
                cv_source_collection=_rule_value(rule, "cv_source_collection"),
                cv_source_collection_key=_rule_value(rule, "cv_source_collection_key"),
                project_name=project_id,
                cv_source_term_key=_rule_value(rule, "cv_source_term_key"),
                report_existence=not esgvoc_owns_existence,
                validate_vocabulary=not esgvoc_owns_vocabulary,
            )
        )

    return results
