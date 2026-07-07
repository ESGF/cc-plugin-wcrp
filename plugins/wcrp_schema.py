from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    AliasChoices,
)


# =============================================================================
# Shared Attribute Rule
# =============================================================================
class AttributeRule(BaseModel):
    """
    Unified attribute rule used by:
      - global attributes
      - geophysical variable attributes
      - coordinate variable attributes

    Notes:
      - attribute_name: allows mapping from logical config key to actual netCDF attribute name
      - is_positive: accepts a legacy typo "is_postive" via validation_alias
      - cv_source_term_key: means "compare to expected_term.<field>" from Variable Registry
    """

    model_config = ConfigDict(extra="forbid")

    severity: Optional[str] = None
    value_type: Optional[str] = None
    is_required: bool = True
    na_value: Optional[Any] = None
    # Optional alias of attribute name as stored in netCDF (case sensitive in practice).
    attribute_name: Optional[str] = None

    # Mutually exclusive "value rules"
    pattern: Optional[str] = None
    constant: Optional[Any] = None
    enum: Optional[List[Any]] = None
    as_variable: Optional[bool] = None

    # Accept "is_postive" typo from old TOML without exploding Pydantic.
    is_positive: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("is_positive", "is_postive"),
    )

    # ESGVOC vocabulary membership
    cv_source_collection: Optional[str] = None
    cv_source_collection_key: Optional[str] = None  # optional

    # Variable Registry "expected-term" key (must be used alone)
    cv_source_term_key: Optional[str] = None

    @model_validator(mode="after")
    def exclusivity(self):
        # If a collection key is used, the collection must exist.
        if self.cv_source_collection_key and not self.cv_source_collection:
            raise ValueError("cv_source_collection_key requires cv_source_collection")

        # Registry mode must be exclusive with all other rules.
        if self.cv_source_term_key is not None:
            other = any(
                [
                    self.pattern is not None,
                    self.constant is not None,
                    self.enum is not None,
                    bool(self.as_variable),
                    bool(self.is_positive),
                    self.cv_source_collection is not None,
                    self.cv_source_collection_key is not None,
                ]
            )
            if other:
                raise ValueError(
                    "cv_source_term_key is mutually exclusive with other rules"
                )
            return self

        # Otherwise, only ONE rule among value rules + vocab rule.
        active = [
            self.pattern is not None,
            self.constant is not None,
            self.enum is not None,
            bool(self.as_variable),
            bool(self.is_positive),
            self.cv_source_collection is not None,
        ]
        if sum(active) > 1:
            raise ValueError("Multiple mutually exclusive rules defined.")
        return self


# =============================================================================
# Top-level project keys
# =============================================================================
class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str
    project_version: str


# =============================================================================
# file.toml
# =============================================================================
class FileFormatRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None
    expected_format: Optional[str] = None
    allowed_data_models: Optional[List[str]] = None


class FileCompressionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None
    expected_complevel: Optional[int] = None
    expected_shuffle: Optional[bool] = None


class FileSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Optional[FileFormatRule] = None
    compression: Optional[FileCompressionRule] = None
    internal_packing: Optional[FileInternalPackingRule] = None

class FileInternalPackingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


# =============================================================================
# drs.toml
# =============================================================================
class DrsRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


class DrsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: Optional[DrsRule] = None
    directory: Optional[DrsRule] = None
    attributes_vs_directory: Optional[DrsRule] = None
    filename_vs_directory: Optional[DrsRule] = None
    time_range_label_precision: Optional[Dict[str, str]] = None


# =============================================================================
# global_attributes.toml
# =============================================================================
class ConsistencyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


class GlobalConsistency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename_vs_attributes: Optional[ConsistencyRule] = None
    experiment_properties: Optional[ConsistencyRule] = None
    institution_properties: Optional[ConsistencyRule] = None
    source_properties: Optional[ConsistencyRule] = None
    frequency_properties: Optional[ConsistencyRule] = None
    variant_properties: Optional[ConsistencyRule] = None


class GlobalSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attributes: Dict[str, AttributeRule] = Field(default_factory=dict)
    consistency: Optional[GlobalConsistency] = None


# =============================================================================
# variable (geophysical_variable.toml): [variable.*]
# =============================================================================
class VarExistenceRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


class VarTypeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None
    data_type: Optional[str] = None  # e.g. "float", "double", "int"


class VarDimensionsRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


class GeophysicalVariableSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    existence: Optional[VarExistenceRule] = None
    type: Optional[VarTypeRule] = None
    dimensions: Optional[VarDimensionsRule] = None
    attributes: Dict[str, AttributeRule] = Field(default_factory=dict)


# =============================================================================
# coordinates (coordinate_variables.toml)
# =============================================================================
class CoordinateGlobalRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


class CoordinateNameRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None
    variable_name: str


class MonotonicityRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None
    direction: Literal["increasing", "decreasing"]


class TimeSquarenessRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None
    ref_calendar: Optional[str] = None
    ref_time_units: Optional[str] = None
    ref_increment: Optional[str] = None


class TimeCoverageRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None

class CalendarRecommendationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None

class CoordinateVariableConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[CoordinateNameRule] = None
    type: Optional[VarTypeRule] = None
    dimensions: Optional[VarDimensionsRule] = None
    monotonicity: Optional[MonotonicityRule] = None
    squareness: Optional[TimeSquarenessRule] = None
    coverage: Optional[TimeCoverageRule] = None
    calendar_recommendation: Optional[CalendarRecommendationRule] = None
    attributes: Dict[str, AttributeRule] = Field(default_factory=dict)


class CoordinatesSection(BaseModel):
    """
    Accepts TOML like:
      [coordinates.bounds]
      severity="H"

      [coordinates.dimensions]
      severity="H"

      [coordinates.lev.monotonicity]
      ...

      [coordinates.lat.attributes.standard_name]
      ...

    Internally we normalize to:
      coordinates.bounds
      coordinates.dimensions
      coordinates.variables = { "lev": CoordinateVariableConfig, "lat": ... }
    """

    model_config = ConfigDict(extra="forbid")

    bounds: Optional[CoordinateGlobalRule] = None
    dimensions: Optional[CoordinateGlobalRule] = None
    variables: Dict[str, CoordinateVariableConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values):
        if values is None or not isinstance(values, dict):
            return values

        out: dict = {}
        out["bounds"] = values.get("bounds")
        out["dimensions"] = values.get("dimensions")

        # Everything else under [coordinates.*] is interpreted as a coordinate variable rule set.
        var_keys = {
            k: v for k, v in values.items() if k not in {"bounds", "dimensions"}
        }
        out["variables"] = var_keys
        return out


# =============================================================================
# Full merged config
# =============================================================================
class WCRPConfig(ProjectConfig):
    model_config = ConfigDict(extra="forbid")

    file: Optional[FileSection] = None
    drs: Optional[DrsSection] = None
    global_: Optional[GlobalSection] = Field(default=None, alias="global")

    variable: Optional[GeophysicalVariableSection] = None
    coordinates: Optional[CoordinatesSection] = None
