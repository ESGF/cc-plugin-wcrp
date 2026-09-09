"""Compatibility adapter for ESGVoc known-branded-variable metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class KnownBrandedVariableLookupError(RuntimeError):
    """Raised when required known-branded-variable metadata cannot be read."""


@dataclass(frozen=True)
class ExpectedVariableMetadata:
    """Canonical metadata consumed by geophysical-variable checks."""

    id: str | None = None
    cf_standard_name: str | None = None
    units: str | None = None
    dimensions: list[str] | None = None
    cell_methods: str | None = None
    cell_measures: str | None = None
    description: str | None = None
    long_name: str | None = None
    out_name: str | None = None
    flag_values: Any = None
    flag_meanings: Any = None
    variable_root_name: str | None = None

    @property
    def cf_units(self) -> str | None:
        """Compatibility alias for external configurations using the old key."""
        return self.units


@dataclass(frozen=True)
class VariableMetadataLookup:
    """Normalized metadata plus a non-fatal legacy long-name warning."""

    expected: ExpectedVariableMetadata
    warning: str | None = None


def _value(term: Any, field: str, default=None):
    if isinstance(term, dict):
        return term.get(field, default)
    return getattr(term, field, default)


def _nonempty(value):
    return value if value is not None and str(value).strip() else None


def _reference_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    identifier = _value(value, "id")
    return str(identifier).strip() if identifier else None


def _reference_ids(values: Any) -> list[str] | None:
    if values is None:
        return None
    return [identifier for value in values if (identifier := _reference_id(value))]


def normalize_known_branded_variable(
    term: Any,
    *,
    fallback_long_name: str | None = None,
) -> ExpectedVariableMetadata:
    """Normalize current and legacy ESGVoc models to canonical field names."""
    units = _nonempty(_value(term, "units"))
    if units is None:
        units = _nonempty(_value(term, "cf_units"))
    long_name = _nonempty(_value(term, "long_name")) or _nonempty(fallback_long_name)
    return ExpectedVariableMetadata(
        id=_nonempty(_value(term, "id")),
        cf_standard_name=_nonempty(_value(term, "cf_standard_name")),
        units=units,
        dimensions=_reference_ids(_value(term, "dimensions")),
        cell_methods=_nonempty(_value(term, "cell_methods")),
        cell_measures=_nonempty(_value(term, "cell_measures")),
        description=_nonempty(_value(term, "description")),
        long_name=long_name,
        out_name=_nonempty(_value(term, "out_name")),
        flag_values=_value(term, "flag_values"),
        flag_meanings=_value(term, "flag_meanings"),
        variable_root_name=_reference_id(_value(term, "variable_root_name")),
    )


def _exact_term(terms: list[Any], identifier: str):
    return next(
        (term for term in terms if str(_value(term, "id", "")) == identifier),
        None,
    )


def lookup_expected_variable_metadata(
    find_terms: Callable[..., list[Any]],
    branded_variable_id: str,
    *,
    fallback_variable_id: str | None = None,
) -> VariableMetadataLookup:
    """Read and normalize one known branded variable from ESGVoc.

    Current records use ``units`` and may provide ``long_name`` directly. Legacy
    records use ``cf_units`` and never provide ``long_name``. Whenever long_name
    is absent, the root ``variable`` supplies it through a fallback lookup. That
    lookup is optional and therefore produces a warning, while failure to read
    the known-branded-variable record is fatal.
    """
    try:
        terms = find_terms(
            expression=str(branded_variable_id),
            data_descriptor_id="known_branded_variable",
            only_id=True,
        )
    except Exception as exc:
        raise KnownBrandedVariableLookupError(
            "Registry lookup error for known_branded_variable "
            f"{branded_variable_id!r}: {type(exc).__name__}: {exc}"
        ) from exc

    known_branded_variable = _exact_term(terms or [], str(branded_variable_id))
    if known_branded_variable is None:
        raise KnownBrandedVariableLookupError(
            f"Known branded variable {branded_variable_id!r} was not found in "
            "the registry."
        )

    expected = normalize_known_branded_variable(known_branded_variable)
    if expected.long_name:
        return VariableMetadataLookup(expected)

    variable_id = expected.variable_root_name or _nonempty(fallback_variable_id)
    if not variable_id:
        return VariableMetadataLookup(
            expected,
            "The known_branded_variable record has no long_name and no "
            "variable root could be selected for the fallback lookup. Only "
            "'long_name' is unavailable.",
        )

    try:
        variable_terms = find_terms(
            expression=variable_id,
            data_descriptor_id="variable",
            only_id=True,
            selected_term_fields=["long_name"],
        )
    except Exception as exc:
        return VariableMetadataLookup(
            expected,
            f"Registry lookup error for variable {variable_id!r}: "
            f"{type(exc).__name__}: {exc}. Only 'long_name' is unavailable.",
        )

    variable = _exact_term(variable_terms or [], variable_id)
    if variable is None:
        return VariableMetadataLookup(
            expected,
            f"Variable {variable_id!r} was not found in the registry data "
            "descriptor 'variable'. Only 'long_name' is unavailable.",
        )

    return VariableMetadataLookup(
        normalize_known_branded_variable(
            known_branded_variable,
            fallback_long_name=_value(variable, "long_name"),
        )
    )
