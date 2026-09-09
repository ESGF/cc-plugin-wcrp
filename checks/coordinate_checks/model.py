"""Source-neutral models used by the coordinate checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def as_dict(record: Any) -> dict:
    if record is None:
        return {}
    if isinstance(record, dict):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="python")
    return dict(record)


def reference_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    source = as_dict(value)
    return str(source.get("id") or source.get("drs_name") or "")


def reference_ids(values: Any) -> list[str]:
    if not values:
        return []
    return [identifier for value in values if (identifier := reference_id(value))]


@dataclass(frozen=True)
class Catalog:
    """All coordinate records read once for one dataset."""

    project_id: str
    branded_variable_id: str
    branded_variable: dict
    coordinate_ids: tuple[str, ...]
    data_coordinates: dict[str, dict] = field(default_factory=dict)
    model_levels: dict[str, dict] = field(default_factory=dict)
    formula_terms: dict[str, dict] = field(default_factory=dict)
    grid_variables: dict[str, dict] = field(default_factory=dict)
    grid_axes: dict[str, dict] = field(default_factory=dict)

    @property
    def data_variable_name(self) -> str:
        return str(self.branded_variable.get("out_name") or "")


def records_by_id(records: Any) -> dict[str, dict]:
    result = {}
    for record in records or []:
        source = as_dict(record)
        identifier = reference_id(source)
        if identifier:
            result[identifier] = source
    return result
