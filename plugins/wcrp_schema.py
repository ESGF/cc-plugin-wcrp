from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
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
    # is_required is normally a bool. It may also be a dynamic keyword string
    # resolved at runtime against esgvoc:
    #   "has_parent_experiment" -> required only if the CV declares a parent experiment
    #   "has_sub_experiment" -> required only if the CV declares a sub-experiment
    is_required: Union[
        bool,
        Literal[
            "has_parent_experiment",
            "has_sub_experiment",
            "has_parent_activity",
            "has_parent_mip_era",
        ],
    ] = True
    na_value: Optional[Any] = None
    # Optional alias of attribute name as stored in netCDF (case sensitive in practice).
    attribute_name: Optional[str] = None

    # Mutually exclusive "value rules"
    pattern: Optional[str] = None
    constant: Optional[Any] = None
    threshold: Optional[Any] = None
    is_above_threshold: Optional[bool] = None
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
    expected_term_comparison: Literal["equal", "contains"] = "equal"
    report_missing_expected_term: bool = False

    @model_validator(mode="after")
    def exclusivity(self):
        if self.is_above_threshold is not None and self.threshold is None:
            raise ValueError("is_above_threshold requires threshold")

        if self.threshold is not None and self.is_above_threshold is None:
            raise ValueError("threshold requires is_above_threshold")

        # If a collection key is used, the collection must exist.
        if self.cv_source_collection_key and not self.cv_source_collection:
            raise ValueError("cv_source_collection_key requires cv_source_collection")

        # Registry mode must be exclusive with all other rules.
        if self.cv_source_term_key is not None:
            other = any(
                [
                    self.pattern is not None,
                    self.constant is not None,
                    self.threshold is not None,
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

        if self.report_missing_expected_term:
            raise ValueError(
                "report_missing_expected_term requires cv_source_term_key"
            )

        if self.expected_term_comparison != "equal":
            raise ValueError("expected_term_comparison requires cv_source_term_key")

        # Otherwise, only ONE rule among value rules + vocab rule.
        active = [
            self.pattern is not None,
            self.constant is not None,
            self.threshold is not None,
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


class FileInternalPackingSectionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


class FileInternalPackingDataRule(FileInternalPackingSectionRule):
    min_chunk_size_bytes: Optional[int] = Field(default=None, ge=1)
    frequency_min_timesteps: Optional[Dict[str, int]] = None

    @field_validator("frequency_min_timesteps", mode="before")
    @classmethod
    def _validate_frequency_min_timesteps(cls, value):
        if value is None:
            return value
        if not isinstance(value, dict):
            raise ValueError("frequency_min_timesteps must be a table")
        for frequency, steps in value.items():
            if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
                raise ValueError(
                    f"frequency_min_timesteps[{frequency!r}] must be a positive integer"
                )
        return value


class FileInternalPackingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metadata: Optional[FileInternalPackingSectionRule] = None
    time: Optional[FileInternalPackingSectionRule] = None
    data: Optional[FileInternalPackingDataRule] = None


class FileSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Optional[FileFormatRule] = None
    compression: Optional[FileCompressionRule] = None
    internal_packing: Optional[FileInternalPackingRule] = None


# =============================================================================
# drs.toml
# =============================================================================
class DrsRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


class TimeRangeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None
    climatology_suffix: str = Field(
        default="",
        pattern=r"^$|^-[A-Za-z][A-Za-z0-9-]*$",
    )
    label_precision: Dict[str, str] = Field(default_factory=dict)


class DrsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: Optional[DrsRule] = None
    directory: Optional[DrsRule] = None
    attributes_vs_directory: Optional[DrsRule] = None
    filename_vs_directory: Optional[DrsRule] = None
    time_range: Optional[TimeRangeRule] = None
    directory_template_keys: List[str] = Field(default_factory=list)
    filename_template_keys: List[str] = Field(default_factory=list)


# =============================================================================
# global_attributes.toml
# =============================================================================
class ConsistencyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Optional[str] = None


class GlobalConsistency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Filename vs global attributes (ATTR005a)
    filename_vs_attributes: Optional[ConsistencyRule] = None

    # Experiment consistency, atomic (ATTR007a-d)
    experiment_id_vs_activity_id: Optional[ConsistencyRule] = None
    experiment_id_vs_experiment: Optional[ConsistencyRule] = None
    experiment_id_vs_parent_experiment_id: Optional[ConsistencyRule] = None
    experiment_id_vs_sub_experiment_id: Optional[ConsistencyRule] = (
        None  # CMIP6/plus only
    )

    # Institution / source (ATTR009, ATTR010)
    institution_id_vs_institution: Optional[ConsistencyRule] = None
    source_id_vs_source: Optional[ConsistencyRule] = None
    driving_source_id_vs_driving_source: Optional[ConsistencyRule] = None
    source_id_vs_institution_id: Optional[ConsistencyRule] = None  # CMIP6/plus only

    # Frequency vs table (ATTR008)
    frequency_vs_table_id: Optional[ConsistencyRule] = None  # CMIP6/plus only
    frequency_vs_branded_variable: Optional[ConsistencyRule] = None  # CMIP7 only

    # Variant label consistency, atomic (ATTR006a-d)
    variant_label_vs_realization_index: Optional[ConsistencyRule] = None
    variant_label_vs_initialization_index: Optional[ConsistencyRule] = None
    variant_label_vs_physics_index: Optional[ConsistencyRule] = None
    variant_label_vs_forcing_index: Optional[ConsistencyRule] = None


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
    data_type: Optional[str | list[str]] = None  # e.g. "real" or ["real", "double"]


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


class CoordinateGridRule(CoordinateGlobalRule):
    """Controls project-specific horizontal-grid structural requirements."""

    require_explicit_grid_axes: bool = False


class CoordinateDirectionRule(CoordinateGlobalRule):
    """Controls the optional physical parts of coordinate direction checks."""

    check_direct_physical_values: bool = True
    check_formula_derived_profile: bool = True


class CoordinateNominalResolutionRule(CoordinateGlobalRule):
    """Restrict the not-applicable resolution value by horizontal layout."""

    none_value: str = "None"
    none_forbidden_horizontal_labels: list[str] = Field(default_factory=list)

    @field_validator("none_value")
    @classmethod
    def validate_none_value(cls, value):
        if not value:
            raise ValueError("none_value must be a non-empty string")
        return value

    @field_validator("none_forbidden_horizontal_labels")
    @classmethod
    def validate_horizontal_labels(cls, values):
        if any(not value for value in values):
            raise ValueError(
                "none_forbidden_horizontal_labels may not contain empty values"
            )
        if len(set(values)) != len(values):
            raise ValueError(
                "none_forbidden_horizontal_labels may not contain duplicates"
            )
        return values


class CoordinateAttributesRule(CoordinateGlobalRule):
    """Controls exceptions to strict coordinate-attribute absence rules."""

    allowed_when_unset: list[str] = Field(default_factory=list)
    allowed_when_unset_severity: Optional[str] = "L"


class CoordinateAssociationRule(CoordinateGlobalRule):
    """Controls auxiliary-coordinate associations required by coordinate types."""

    unstructured_latitude_longitude_for: list[str] = Field(default_factory=list)

    @field_validator("unstructured_latitude_longitude_for")
    @classmethod
    def validate_coordinate_types(cls, values):
        if any(not value for value in values):
            raise ValueError(
                "unstructured_latitude_longitude_for may not contain empty values"
            )
        if len(set(values)) != len(values):
            raise ValueError(
                "unstructured_latitude_longitude_for may not contain duplicates"
            )
        return values


class CoordinateMissingValueAttributesRule(BaseModel):
    """Policy for missing-value attributes on verified coordinates."""

    model_config = ConfigDict(extra="forbid")

    # Presence/value policy for numeric auxiliary coordinates.
    severity: Optional[str] = None
    # NaN values and disagreement between the two attributes.
    integrity_severity: Optional[str] = None
    # Missing-value attributes prohibited for scalar/character coordinates.
    prohibited_severity: Optional[str] = None
    required_for_auxiliary: bool = False
    expected_values_by_data_type: Dict[str, float] = Field(default_factory=dict)

    @field_validator("expected_values_by_data_type")
    @classmethod
    def validate_expected_values(cls, values):
        allowed = {"real", "double"}
        unexpected = set(values) - allowed
        if unexpected:
            raise ValueError(
                "expected_values_by_data_type supports only "
                f"{sorted(allowed)}; found {sorted(unexpected)}"
            )
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("configured missing values must be finite")
        return values

    @model_validator(mode="after")
    def at_least_one_result_family(self):
        if (
            self.severity is None
            and self.integrity_severity is None
            and self.prohibited_severity is None
        ):
            raise ValueError(
                "missing-value attribute checking requires severity or "
                "integrity_severity or prohibited_severity"
            )
        if self.required_for_auxiliary and self.severity is None:
            raise ValueError(
                "required_for_auxiliary requires the configured-value severity"
            )
        return self


class CoordinateBoundsNameRule(CoordinateGlobalRule):
    """Configured names for bounds-related variables and dimensions."""

    bounds_dimension_name: str = "bnds"
    vertices_dimension_name: str = "vertices"
    climatology_bounds_name: str = "climatology_bnds"


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
    discouraged_values: List[str] = Field(default_factory=lambda: ["standard"])
    preferred_value: str = "proleptic_gregorian"


class CoordinateRegistrySection(BaseModel):
    """Severity and enablement controls for vocabulary-driven coordinates.

    Each configured rule is an independent result family.  Omitting a rule
    disables that family, which lets projects adopt the shared implementation
    incrementally without changing the Python checker.
    """

    model_config = ConfigDict(extra="forbid")

    setup: Optional[CoordinateGlobalRule] = None
    identity: Optional[CoordinateGlobalRule] = None
    dimension_order: Optional[CoordinateGlobalRule] = None
    attributes: Optional[CoordinateAttributesRule] = None
    recommendations: Optional[CoordinateGlobalRule] = None
    direction: Optional[CoordinateDirectionRule] = None
    valid_range: Optional[CoordinateGlobalRule] = None
    missing_value_attributes: Optional[CoordinateMissingValueAttributesRule] = None
    grid_geometry_missing_values: Optional[CoordinateGlobalRule] = None
    grid_latitude_valid_range: Optional[CoordinateGlobalRule] = None
    grid_longitude_valid_range: Optional[CoordinateGlobalRule] = None
    grid_longitude_single_cycle: Optional[CoordinateGlobalRule] = None
    grid_mapping_consistency: Optional[CoordinateGlobalRule] = None
    grid_region_consistency: Optional[CoordinateGlobalRule] = None
    grid_label_recommendation: Optional[CoordinateGlobalRule] = None
    grid_cell_count_availability: Optional[CoordinateGlobalRule] = None
    grid_cell_count_consistency: Optional[CoordinateGlobalRule] = None
    nominal_resolution: Optional[CoordinateNominalResolutionRule] = None
    requested_values: Optional[CoordinateGlobalRule] = None
    bounds: Optional[CoordinateGlobalRule] = None
    bounds_name: Optional[CoordinateBoundsNameRule] = None
    bounds_name_recommendations: Optional[CoordinateGlobalRule] = None
    associations: Optional[CoordinateAssociationRule] = None
    grid: Optional[CoordinateGridRule] = None
    formula: Optional[CoordinateGlobalRule] = None


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
    registry: Optional[CoordinateRegistrySection] = None
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
        out["registry"] = values.get("registry")
        var_keys = {
            k: v
            for k, v in values.items()
            if k not in {"bounds", "dimensions", "registry"}
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
