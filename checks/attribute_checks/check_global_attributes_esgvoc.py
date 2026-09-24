"""Adapt ESGVoc global-attribute reports to Compliance Checker results.

This module deliberately contains no project policy.  ESGVoc supplies the
project attribute specification, while callers supply per-attribute severity
from their TOML configuration.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx
from esgvoc.apps.ncattvalid import GAValidator

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


def get_global_attribute_names(validator: GAValidator) -> set[str]:
    """Return the NetCDF attribute names defined by an ESGVoc validator."""
    names: set[str] = set()
    for spec in getattr(validator, "_specs", ()):
        name = getattr(spec, "attr_field_name", None) or getattr(
            spec, "source_collection", None
        )
        if name:
            names.add(str(name))
    return names


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
):
    """Validate global attributes with ESGVoc and return checker results.

    Extra attributes remain informational in ESGVoc and are intentionally not
    converted into failures.  Project-specific checks may still validate them
    through their TOML rules.
    """
    severities = severity_by_attribute or {}

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
        severity = severities.get(name, default_severity)
        ctx = TestCtx(severity, f"[ATTR001] Global attribute '{name}' existence")
        ctx.add_failure(f"Required global attribute '{name}' is missing.")
        results.append(ctx.to_result())

    grouped = defaultdict(list)
    for attribute_result in report.results:
        grouped[attribute_result.name].append(attribute_result)

    for name, attribute_results in grouped.items():
        severity = severities.get(name, default_severity)

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
        if not vocabulary_results:
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
