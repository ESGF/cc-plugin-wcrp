"""Configuration-backed horizontal-grid topology resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import toml


TOPOLOGIES = frozenset({"rectilinear", "curvilinear", "unstructured"})


class GridTopologyConfigError(ValueError):
    """Raised when the topology mapping cannot be interpreted safely."""


def _topology(value, source: str) -> str:
    result = str(value or "").strip().lower()
    if result not in TOPOLOGIES:
        raise GridTopologyConfigError(
            f"{source} maps to {value!r}; expected one of {sorted(TOPOLOGIES)}."
        )
    return result


def _identifier(value) -> str:
    """Return an ESGVoc identifier from a string or resolved reference."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return str(value.get("id") or value.get("drs_name") or "").strip()
    if hasattr(value, "model_dump"):
        return _identifier(value.model_dump(mode="python"))
    return ""


@dataclass(frozen=True)
class GridTopologyConfig:
    """Mappings used to reduce grid metadata to a checkable topology.

    ``grid_type_and_mapping`` keys are ``(grid_type, grid_mapping)`` pairs.
    They take precedence over the less-specific grid-type mapping.  Grid-label
    mappings are a temporary fallback until the EMD grid relationship is
    available through ESGVoc.
    """

    grid_labels: dict[str, str] = field(default_factory=dict)
    grid_types: dict[str, str] = field(default_factory=dict)
    grid_type_and_mapping: dict[tuple[str, str], str] = field(default_factory=dict)
    allow_standard_name_fallback: bool = True


def _mapping_table(source: Mapping, key: str) -> Mapping:
    value = source.get(key, {}) or {}
    if not isinstance(value, Mapping):
        raise GridTopologyConfigError(f"[{key}] must be a TOML table.")
    return value


def _parse_config(source: Mapping) -> GridTopologyConfig:
    identification = _mapping_table(source, "coordinate_identification")
    allow_standard_name_fallback = identification.get(
        "allow_standard_name_fallback", True
    )
    if not isinstance(allow_standard_name_fallback, bool):
        raise GridTopologyConfigError(
            "coordinate_identification.allow_standard_name_fallback must be a "
            "TOML boolean."
        )
    labels = {
        str(key): _topology(value, f"grid_label {key!r}")
        for key, value in _mapping_table(source, "grid_label").items()
    }
    groups = _mapping_table(source, "grid_label_groups")
    grouped_labels = {}
    for topology, identifiers in groups.items():
        resolved_topology = _topology(topology, f"grid-label group {topology!r}")
        if not isinstance(identifiers, list):
            raise GridTopologyConfigError(
                f"grid_label_groups.{topology} must be a TOML array."
            )
        for identifier in identifiers:
            identifier = str(identifier)
            previous = grouped_labels.get(identifier)
            if previous and previous != resolved_topology:
                raise GridTopologyConfigError(
                    f"grid_label {identifier!r} occurs in both {previous!r} and "
                    f"{resolved_topology!r} groups."
                )
            grouped_labels[identifier] = resolved_topology
            # Explicit [grid_label] entries deliberately override groups.
            labels.setdefault(identifier, resolved_topology)

    grid_types = {
        str(key): _topology(value, f"grid_type {key!r}")
        for key, value in _mapping_table(source, "grid_type").items()
    }
    combinations = {}
    for key, value in _mapping_table(source, "grid_type_and_mapping").items():
        parts = str(key).split("|", 1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise GridTopologyConfigError(
                "[grid_type_and_mapping] keys must have the form "
                "'grid_type|grid_mapping'."
            )
        pair = tuple(part.strip() for part in parts)
        combinations[pair] = _topology(value, f"grid_type/grid_mapping {key!r}")
    return GridTopologyConfig(
        labels,
        grid_types,
        combinations,
        allow_standard_name_fallback,
    )


def load_grid_topology_config(path: str | Path) -> GridTopologyConfig:
    """Read and validate a project topology mapping once during setup."""
    path = Path(path)
    if not path.is_file():
        raise GridTopologyConfigError(
            f"Grid-topology mapping file does not exist: '{path}'."
        )
    try:
        return _parse_config(toml.load(path))
    except GridTopologyConfigError:
        raise
    except Exception as exc:
        raise GridTopologyConfigError(
            f"Could not read grid-topology mapping '{path}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def resolve_grid_topology(
    config: GridTopologyConfig,
    *,
    grid_label="",
    grid_type="",
    grid_mapping="",
) -> tuple[str | None, str | None]:
    """Resolve topology from EMD metadata, falling back to ``grid_label``.

    ESGVoc references may be strings, dictionaries, or resolved Pydantic
    models.  No topology is inferred from shapes in the netCDF file: an
    unmapped value returns a diagnostic suitable for one HIGH check result.
    """
    label = _identifier(grid_label)
    type_id = _identifier(grid_type)
    mapping_id = _identifier(grid_mapping)

    if type_id and mapping_id:
        topology = config.grid_type_and_mapping.get((type_id, mapping_id))
        if topology:
            return topology, None
    if type_id:
        topology = config.grid_types.get(type_id)
        if topology:
            return topology, None
    if label:
        topology = config.grid_labels.get(label)
        if topology:
            return topology, None
    if type_id:
        detail = f"grid_type={type_id!r}"
        if mapping_id:
            detail += f" and grid_mapping={mapping_id!r}"
        if label:
            detail += f", or fallback grid_label={label!r}"
        return None, f"No horizontal topology is configured for {detail}."
    if not label:
        return None, (
            "The file has no non-empty global 'grid_label' attribute and ESGVoc "
            "does not yet expose the EMD grid_type/grid_mapping relationship."
        )
    return None, f"No horizontal topology is configured for grid_label={label!r}."
