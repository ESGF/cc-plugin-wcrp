#!/usr/bin/env python
"""WCRP CORDEX-CMIP6 compliance checker."""

from __future__ import annotations

import os
from typing import Any, Optional

import toml
from compliance_checker.base import BaseCheck, TestCtx
from netCDF4 import Dataset

from checks.attribute_checks.check_attribute_suite import check_attribute_suite
from checks.attribute_checks.check_attrs_cordex_cmip6 import (
    check_domain_id,
    check_driving_attributes,
    check_grid,
    check_grid_mapping,
    check_institution,
    check_references,
    check_version_realization,
    check_version_realization_info,
)
from checks.consistency_checks.check_attributes_match_filename import (
    check_filename_vs_global_attrs,
)
from checks.consistency_checks.check_drs_consistency import (
    check_attributes_match_directory_structure,
    check_filename_matches_directory_structure,
)
from checks.consistency_checks.check_drs_filename_cv import (
    check_drs_directory,
    check_drs_filename,
)
from checks.consistency_checks.check_institution_source_consistency import (
    check_id_attribute_consistency,
)
from checks.coordinate_checks import (
    CoordinateMetadataError,
    GridTopologyConfigError,
    check_coordinate_catalog,
    coordinate_catalog_required,
    coordinate_metadata_setup_result,
    missing_configured_coordinate_result,
    load_catalog,
    load_grid_topology_config,
    resolve_grid_topology,
)
from checks.coordinate_checks.cmor import (
    catalog_from_cmor,
    expected_variable_from_cmor,
)
from checks.coordinate_checks.utils import coordinate_type, ncattr
from checks.dimension_checks.check_dimension_existence import check_dimension_existence
from checks.dimension_checks.check_dimension_positive import check_dimension_positive
from checks.format_checks.check_compression import check_compression
from checks.format_checks.check_format import check_format
from checks.format_checks.check_internal_packing import (
    check_internal_packing,
    finalize_internal_packing_session,
)
from checks.time_checks.check_time_bounds import check_time_bounds
from checks.time_checks.check_time_calendar import check_calendar_recommendation
from checks.time_checks.check_time_cordex_cmip6 import (
    check_calendar,
    check_time_chunking,
    check_time_units,
)
from checks.time_checks.check_time_range_vs_filename import (
    check_time_range_vs_filename,
)
from checks.time_checks.check_time_squareness import check_time_squareness
from checks.utils import retrieve
from checks.variable_checks.check_coordinate_monotonicity import (
    check_coordinate_monotonicity,
)
from checks.variable_checks.check_coords_cordex_cmip6 import (
    check_horizontal_axes_bounds,
    check_lat_lon_bounds,
    check_lon_value_range,
    infer_horizontal_topology,
)
from checks.variable_checks.check_variable_existence import check_variable_existence
from checks.variable_checks.check_variable_type import (
    check_variable_type,
    configured_data_types,
)
from checks.variable_checks.known_branded_variable import (
    KnownBrandedVariableLookupError,
    lookup_expected_variable_metadata_in_collection,
)
from plugins.wcrp_base import WCRPBaseCheck
from plugins.wcrp_schema import WCRPConfig

try:
    from compliance_checker.cf.util import get_geophysical_variables
except ImportError as exc:
    raise ImportError("Unable to import compliance-checker CF utilities.") from exc

try:
    import esgvoc.api as esgvoc_api
except Exception:
    esgvoc_api = None


DEFAULT_CORDEX_CMIP6_CMOR_TABLES_URL = (
    "https://raw.githubusercontent.com/WCRP-CORDEX/"
    "cordex-cmip6-cmor-tables/main/Tables/"
)


def _deep_merge(left: dict, right: dict) -> dict:
    merged = dict(left)
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return toml.load(stream)


def _option_enabled(options: dict, name: str) -> bool:
    if name not in options:
        return False
    value = options[name]
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


class CordexCmip6ProjectCheck(WCRPBaseCheck):
    """CORDEX-CMIP6 checks with ESGVoc as the default metadata source."""

    _cc_spec = "wcrp_cordex_cmip6"
    _cc_spec_version = "1.0"
    _cc_description = "WCRP CORDEX-CMIP6 Project Checks"
    _uses_esgvoc_project_specs = True
    _cc__url = "https://doi.org/10.5281/zenodo.15047096"
    _cc_display_headers = {3: "Required", 2: "Recommended", 1: "Suggested"}
    _defer_consistency_output = True
    supported_ds = [Dataset]

    def __init__(self, options=None):
        super().__init__(options)
        self.project_name = "cordex-cmip6"
        this_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_config_dir = (
            options.get("project_config_dir")
            if options and options.get("project_config_dir")
            else os.path.join(this_dir, "config", "wcrp")
        )
        self.config: Optional[WCRPConfig] = None
        self.cfg: dict = {}
        self.cordex_config: dict = {}
        self.variable_mapping: dict[str, str] = {}
        self.table_id_to_time_increment: dict[str, Any] = {}
        self.verification_against_tables = False
        self._geo_var_cache: Optional[str] = None
        self._expected_term_cache = None
        self._coordinate_catalog = None
        self._coordinate_setup_error: Optional[str] = None
        self._grid_topology_config = None
        self._grid_topology_config_error: Optional[str] = None
        self._coordinate_grid_topology: Optional[str] = None
        self._coordinate_grid_error: Optional[str] = None

    def _load_split_config(self):
        merged = {}
        for filename in (
            "project.toml",
            "file.toml",
            "drs.toml",
            "global_attributes.toml",
            "geophysical_variable.toml",
            "coordinate_variables.toml",
        ):
            path = os.path.join(self.project_config_dir, filename)
            if not os.path.isfile(path):
                self._record_setup_warning(
                    f"Project configuration file not found at '{path}'"
                )
                continue
            merged = _deep_merge(merged, _load_toml(path))
        self.cfg = merged
        self.config = WCRPConfig.model_validate(merged)

        cordex_path = os.path.join(self.project_config_dir, "cordex.toml")
        self.cordex_config = (
            _load_toml(cordex_path) if os.path.isfile(cordex_path) else {}
        )

    def _load_mappings(self):
        mapping_dir = os.path.join(self.project_config_dir, "mappings")
        files = {
            "variable_mapping": (
                "frequency_and_variable_id_to_branded_variable.toml",
                "mapping_variables",
            ),
            "table_id_to_time_increment": (
                "table_id_to_time_increment.toml",
                "time_increment_mapping",
            ),
        }
        for attribute, (filename, section) in files.items():
            path = os.path.join(mapping_dir, filename)
            if not os.path.isfile(path):
                self._record_setup_warning(
                    f"Project mapping file not found at '{path}'"
                )
                setattr(self, attribute, {})
                continue
            setattr(self, attribute, _load_toml(path).get(section, {}) or {})

        topology_path = os.path.join(mapping_dir, "grid_topology.toml")
        try:
            self._grid_topology_config = load_grid_topology_config(topology_path)
        except GridTopologyConfigError as exc:
            self._grid_topology_config_error = str(exc)

    def _install_time_increment_mapping(self):
        mapping = {}
        for key, value in self.table_id_to_time_increment.items():
            if (
                not isinstance(key, str)
                or "." not in key
                or not isinstance(value, (list, tuple))
                or len(value) != 2
            ):
                self._record_setup_warning(
                    f"Ignored invalid time-increment mapping {key!r}: {value!r}"
                )
                continue
            table_id, frequency = key.split(".", 1)
            try:
                mapping[(table_id, frequency)] = (
                    int(str(value[0]).strip()),
                    str(value[1]).strip(),
                )
            except (TypeError, ValueError) as exc:
                self._record_setup_warning(
                    f"Could not interpret time-increment mapping for {key!r}", exc
                )
        self._time_increment_mapping = mapping

    def _load_cmor_tables(self):
        if getattr(self, "CT", None):
            return
        cmor_config = self.cordex_config.get("cmor_tables", {})
        base_url = (
            str(
                cmor_config.get("base_url") or DEFAULT_CORDEX_CMIP6_CMOR_TABLES_URL
            ).rstrip("/")
            + "/"
        )
        tables_path = self.options.get("tables") or self.options.get(
            "tables_dir", "~/.wcrp_metadata/cordex-cmip6-cmor-tables"
        )
        for table in (
            "coordinate",
            "grids",
            "formula_terms",
            "CV",
            "1hr",
            "3hr",
            "6hr",
            "day",
            "mon",
            "fx",
        ):
            filename = f"CORDEX-CMIP6_{table}.json"
            retrieve(
                base_url + filename,
                filename,
                tables_path,
                force=_option_enabled(self.options, "force_table_download"),
            )
        self._initialize_CV_info(tables_path)

    def _mapping_key(self, ds):
        variable_id = str(self._get_attr("variable_id", "") or "")
        frequency = str(self._get_attr("frequency", "") or "")
        if not frequency:
            frequency = str(self._get_attr("table_id", "") or "")
        return f"{frequency}.{variable_id}", variable_id

    def _mapped_branded_variable(self, ds):
        key, _ = self._mapping_key(ds)
        return self.variable_mapping.get(key)

    def _cmor_variable_entry(self, ds):
        key, variable_id = self._mapping_key(ds)
        frequency = key.split(".", 1)[0]
        table = (getattr(self, "CT", {}) or {}).get(frequency)
        if table is None:
            table_id = str(self._get_attr("table_id", "") or "")
            table = (getattr(self, "CT", {}) or {}).get(table_id)
        entries = (table or {}).get("variable_entry", {})
        if variable_id in entries:
            return variable_id, entries[variable_id]
        for identifier, entry in entries.items():
            if entry.get("out_name") == variable_id:
                return identifier, entry
        raise LookupError(
            f"No CMOR variable entry was found for frequency/variable_id {key!r}."
        )

    def _grid_mapping_name(self, ds):
        geo = self._geo_var_cache
        if not geo or geo not in ds.variables:
            _, variable_id = self._mapping_key(ds)
            geo = variable_id if variable_id in ds.variables else None
        if not geo:
            return ""
        mapping = ncattr(ds.variables[geo], "grid_mapping")
        mapping_variable = (
            mapping.split()[0].rstrip(":")
            if isinstance(mapping, str) and mapping
            else ""
        )
        if mapping_variable in ds.variables:
            return ncattr(ds.variables[mapping_variable], "grid_mapping_name")
        return ""

    def _initialize_coordinate_catalog(self, ds):
        branded = self._mapped_branded_variable(ds)
        key, variable_id = self._mapping_key(ds)
        if not branded:
            raise CoordinateMetadataError(
                f"No known-branded-variable mapping exists for {key!r}."
            )
        if self.verification_against_tables:
            identifier, entry = self._cmor_variable_entry(ds)
            self._coordinate_catalog = catalog_from_cmor(
                str(branded),
                variable_id or entry.get("out_name") or identifier,
                entry,
                self.CTcoords,
                self.CTgrids,
                self.CTformulas,
            )
        else:
            self._coordinate_catalog = load_catalog(
                str(branded).lower(),
                project_id=self.project_name,
                branded_collection="known_branded_variable",
                file_variable_name=variable_id,
                allow_universe_coordinate_fallback=True,
            )

        requires_horizontal = any(
            coordinate_type(entry) == "generic_horizontal"
            for identifier, entry in self._coordinate_catalog.data_coordinates.items()
            if identifier in self._coordinate_catalog.coordinate_ids
        )
        if not requires_horizontal:
            return
        grid_mapping = self._grid_mapping_name(ds)
        if grid_mapping in {"", "latitude_longitude"}:
            geometry_topology, geometry_error = infer_horizontal_topology(ds)
            self._coordinate_grid_topology = geometry_topology
            self._coordinate_grid_error = None if geometry_topology else geometry_error
            return
        if self._grid_topology_config_error:
            self._coordinate_grid_error = (
                "The CORDEX-CMIP6 grid-topology mapping could not be loaded. "
                f"Technical reason: {self._grid_topology_config_error}"
            )
            return
        if self._grid_topology_config is None:
            self._coordinate_grid_error = (
                "The CORDEX-CMIP6 grid-topology mapping is unavailable."
            )
            return
        (
            self._coordinate_grid_topology,
            self._coordinate_grid_error,
        ) = resolve_grid_topology(
            self._grid_topology_config,
            grid_mapping=grid_mapping,
        )
        if self._coordinate_grid_topology == "unstructured":
            self._coordinate_grid_topology = None
            self._coordinate_grid_error = (
                "The plugin does not currently support unstructured horizontal "
                "grids for CORDEX-CMIP6. Please open a GitHub issue and provide "
                "test data so that support can be discussed and, if appropriate, "
                "implemented."
            )

    def setup(self, ds):
        super().setup(ds)
        self._run_setup_step(
            "load the CORDEX-CMIP6 project configuration",
            self._load_split_config,
        )
        self._run_setup_step("load the CORDEX-CMIP6 mappings", self._load_mappings)
        self._run_setup_step(
            "install the CORDEX-CMIP6 time-increment mapping",
            self._install_time_increment_mapping,
        )
        self.verification_against_tables = _option_enabled(
            self.options, "verification_against_tables"
        ) or bool(self.options.get("tables"))
        self._geo_var_cache = None
        self._expected_term_cache = None
        self._coordinate_catalog = None
        self._coordinate_setup_error = None
        self._coordinate_grid_topology = None
        self._coordinate_grid_error = None

        if self.verification_against_tables:
            self._run_setup_step(
                "load the CORDEX-CMIP6 CMOR tables", self._load_cmor_tables
            )
        if coordinate_catalog_required(self.config):
            try:
                self._initialize_coordinate_catalog(ds)
            except Exception as exc:
                self._coordinate_setup_error = (
                    str(exc)
                    if isinstance(exc, CoordinateMetadataError)
                    else f"Unexpected {type(exc).__name__}: {exc}"
                )
        if self.consistency_output:
            self._write_consistency_output()

    def _get_geo_var(self, ds, severity):
        if self._geo_var_cache and self._geo_var_cache in ds.variables:
            return self._geo_var_cache, []
        variable_id = str(self._get_attr("variable_id", "") or "")
        if variable_id in ds.variables:
            self._geo_var_cache = variable_id
            return variable_id, []
        candidates = list(get_geophysical_variables(ds) or [])
        if len(candidates) == 1:
            self._geo_var_cache = candidates[0]
            return candidates[0], []
        ctx = TestCtx(severity, "Geophysical Variable Detection")
        if not candidates:
            ctx.add_failure("No geophysical variable detected in the file.")
        else:
            ctx.add_failure(
                f"Expected exactly 1 geophysical variable, found "
                f"{len(candidates)}: {candidates}."
            )
        return None, [ctx.to_result()]

    def _get_expected_variable_metadata(self, ds, severity):
        if self._expected_term_cache is not None:
            return self._expected_term_cache, []
        results = []
        branded = self._mapped_branded_variable(ds)
        key, variable_id = self._mapping_key(ds)
        if not branded:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(f"No known-branded-variable mapping found for {key!r}.")
            return None, [ctx.to_result()]
        try:
            if self.verification_against_tables:
                identifier, entry = self._cmor_variable_entry(ds)
                expected = expected_variable_from_cmor(identifier, entry)
            else:
                if esgvoc_api is None:
                    raise KnownBrandedVariableLookupError(
                        "ESGVoc is not installed or could not be imported."
                    )
                lookup = lookup_expected_variable_metadata_in_collection(
                    esgvoc_api.get_term_in_collection,
                    self.project_name,
                    str(branded),
                    fallback_variable_id=variable_id.lower(),
                    branded_collection="known_branded_variable",
                    variable_collection="variable_id",
                )
                expected = lookup.expected
                if lookup.warning:
                    ctx = TestCtx(severity, "Variable Registry")
                    ctx.add_failure(lookup.warning)
                    results.append(ctx.to_result())
        except (KnownBrandedVariableLookupError, LookupError) as exc:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(str(exc))
            return None, [ctx.to_result()]
        self._expected_term_cache = expected
        return expected, results

    def check_File_Format(self, ds):
        if not self.config or not self.config.file or not self.config.file.format:
            return []
        rule = self.config.file.format
        return check_format(
            ds,
            rule.expected_format,
            rule.allowed_data_models,
            self.get_severity(rule.severity),
        )

    def check_File_Compression(self, ds):
        if not self.config or not self.config.file or not self.config.file.compression:
            return []
        rule = self.config.file.compression
        geo, _ = self._get_geo_var(ds, BaseCheck.HIGH)
        return check_compression(
            ds,
            variable_name=geo,
            expected_complevel=rule.expected_complevel,
            expected_shuffle=rule.expected_shuffle,
            severity=self.get_severity(rule.severity),
        )

    def check_File_Internal_Packing_Metadata(self, ds):
        """
        [FILE004a] Internal packing consolidated metadata check.
        """
        try:
            rule = (
                self.config.file.internal_packing
                if self.config and self.config.file
                else None
            )
            if not rule or not rule.metadata:
                return []
            return check_internal_packing(
                ds,
                severity=self.get_severity(rule.metadata.severity),
                run_metadata=True,
                run_time=False,
                run_data=False,
            )
        finally:
            finalize_internal_packing_session(ds)

    def check_File_Internal_Packing_Time(self, ds):
        """
        [FILE004b-c] Internal packing time and time-bounds checks.
        """
        try:
            rule = (
                self.config.file.internal_packing
                if self.config and self.config.file
                else None
            )
            if not rule or not rule.time:
                return []
            return check_internal_packing(
                ds,
                severity=self.get_severity(rule.time.severity),
                run_metadata=False,
                run_time=True,
                run_data=False,
            )
        finally:
            finalize_internal_packing_session(ds)

    def check_File_Internal_Packing_Data(self, ds):
        """
        [FILE004d] Internal packing data-variable chunking check.
        """
        try:
            rule = (
                self.config.file.internal_packing
                if self.config and self.config.file
                else None
            )
            if not rule or not rule.data:
                return []
            return check_internal_packing(
                ds,
                severity=self.get_severity(rule.data.severity),
                min_chunk_size_bytes=(rule.data.min_chunk_size_bytes or 4 * (2**20)),
                frequency=self.frequency,
                frequency_min_timesteps=rule.data.frequency_min_timesteps,
                run_metadata=False,
                run_time=False,
                run_data=True,
            )
        finally:
            finalize_internal_packing_session(ds)

    def check_Global_Attributes(self, ds):
        return self._check_global_attributes(ds)

    def check_DRS(self, ds):
        results = []
        if self._esgvoc_project_setup_error:
            return results
        if not self.config or not self.config.drs:
            return results
        drs = self.config.drs
        directory_structure_owner = self._drs_directory_structure_owner(drs)
        if drs.filename:
            results.extend(
                check_drs_filename(
                    ds,
                    self.get_severity(drs.filename.severity),
                    project_id=self.project_name,
                )
            )
        if drs.directory:
            results.extend(
                check_drs_directory(
                    ds,
                    self.get_severity(drs.directory.severity),
                    project_id=self.project_name,
                    report_directory_structure_error=(
                        directory_structure_owner == "directory"
                    ),
                )
            )
        if drs.attributes_vs_directory:
            results.extend(
                check_attributes_match_directory_structure(
                    ds,
                    self.get_severity(drs.attributes_vs_directory.severity),
                    project_id=self.project_name,
                    dir_template_keys=drs.directory_template_keys or None,
                    filename_template_keys=drs.filename_template_keys or None,
                    report_directory_structure_error=(
                        directory_structure_owner == "attributes_vs_directory"
                    ),
                )
            )
        if drs.filename_vs_directory:
            results.extend(
                check_filename_matches_directory_structure(
                    ds,
                    self.get_severity(drs.filename_vs_directory.severity),
                    project_id=self.project_name,
                    dir_template_keys=drs.directory_template_keys or None,
                    filename_template_keys=drs.filename_template_keys or None,
                    report_directory_structure_error=(
                        directory_structure_owner == "filename_vs_directory"
                    ),
                )
            )
        return results

    def check_Geophysical_Variable(self, ds):
        results = []
        if not self.config or not self.config.variable:
            return results
        variable_name, detection = self._get_geo_var(ds, BaseCheck.HIGH)
        results.extend(detection)
        if not variable_name:
            return results
        config = self.config.variable
        if config.existence:
            results.extend(
                check_variable_existence(
                    ds,
                    variable_name,
                    self.get_severity(config.existence.severity),
                )
            )
        if config.type:
            allowed = configured_data_types(config.type.data_type)
            if allowed:
                results.extend(
                    check_variable_type(
                        ds,
                        variable_name,
                        allowed_types=allowed,
                        severity=self.get_severity(config.type.severity),
                    )
                )
        if config.dimensions:
            severity = self.get_severity(config.dimensions.severity)
            for dimension in ds.variables[variable_name].dimensions:
                results.extend(check_dimension_existence(ds, dimension, severity))
                results.extend(check_dimension_positive(ds, dimension, severity))

        expected_term = None
        if any(rule.cv_source_term_key for rule in config.attributes.values()):
            expected_term, lookup_results = self._get_expected_variable_metadata(
                ds, BaseCheck.HIGH
            )
            results.extend(lookup_results)
        for key, rule in config.attributes.items():
            if rule.cv_source_term_key and expected_term is None:
                continue
            results.extend(
                check_attribute_suite(
                    ds=ds,
                    var_name=variable_name,
                    attribute_name=rule.attribute_name or key,
                    severity=self.get_severity(rule.severity),
                    value_type=rule.value_type,
                    is_required=rule.is_required,
                    na_value=rule.na_value,
                    pattern=rule.pattern,
                    constant=rule.constant,
                    threshold=rule.threshold,
                    is_above_threshold=rule.is_above_threshold,
                    enum=rule.enum,
                    as_variable=rule.as_variable,
                    is_positive=rule.is_positive,
                    cv_source_collection=rule.cv_source_collection,
                    cv_source_collection_key=rule.cv_source_collection_key,
                    project_name=self.project_name,
                    expected_term=expected_term,
                    cv_source_term_key=rule.cv_source_term_key,
                    expected_term_comparison=rule.expected_term_comparison,
                    report_missing_expected_term=rule.report_missing_expected_term,
                )
            )
        return results

    def check_Global_Consistency(self, ds):
        if (
            not self.config
            or not self.config.global_
            or not self.config.global_.consistency
        ):
            return []

        consistency = self.config.global_.consistency
        results = []
        if consistency.filename_vs_attributes:
            rule = consistency.filename_vs_attributes
            results.extend(
                check_filename_vs_global_attrs(
                    ds,
                    self.get_severity(rule.severity),
                    project_id=self.project_name,
                    filename_structure_delegated=bool(
                        self.config.drs and self.config.drs.filename
                    ),
                )
            )

        pairs = (
            ("institution_id_vs_institution", "institution_id", "institution"),
            ("source_id_vs_source", "source_id", "source"),
            (
                "driving_source_id_vs_driving_source",
                "driving_source_id",
                "driving_source",
            ),
        )
        for rule_name, id_attribute, value_attribute in pairs:
            rule = getattr(consistency, rule_name)
            if rule is None:
                continue
            results.extend(
                check_id_attribute_consistency(
                    ds,
                    self.get_severity(rule.severity),
                    project_id=self.project_name,
                    id_attribute=id_attribute,
                    value_attribute=value_attribute,
                )
            )
        return results

    def _coordinate_entries_for_axis(self, axis):
        if self._coordinate_catalog is None:
            return []
        return [
            (identifier, self._coordinate_catalog.data_coordinates[identifier])
            for identifier in self._coordinate_catalog.coordinate_ids
            if self._coordinate_catalog.data_coordinates[identifier].get("axis") == axis
        ]

    def check_Coordinate_Metadata_Setup(self, ds):
        if not self.config:
            return []
        source = "CMOR-table" if self.verification_against_tables else "ESGVoc"
        return coordinate_metadata_setup_result(
            config=self.config,
            catalog=self._coordinate_catalog,
            error=self._coordinate_setup_error,
            project_label=f"CORDEX-CMIP6 {source}",
            get_severity=self.get_severity,
        )

    def check_Coordinate_Standard(self, ds):
        registry = (
            self.config.coordinates.registry
            if self.config and self.config.coordinates
            else None
        )
        if registry is None or self._coordinate_catalog is None:
            return []
        families = (
            "identity",
            "dimension_order",
            "attributes",
            "recommendations",
            "direction",
            "valid_range",
            "grid_latitude_valid_range",
            "grid_longitude_valid_range",
            "grid_longitude_single_cycle",
            "grid_cell_count_availability",
            "grid_cell_count_consistency",
            "requested_values",
            "bounds",
            "bounds_name",
            "associations",
            "grid",
            "formula",
        )
        severities = {
            family: self.get_severity(rule.severity, "HIGH")
            for family in families
            if (rule := getattr(registry, family)) is not None
        }
        naming = registry.bounds_name
        direction = registry.direction
        attributes = registry.attributes
        if attributes is not None and attributes.allowed_when_unset:
            severities["allowed_when_unset"] = self.get_severity(
                attributes.allowed_when_unset_severity, "LOW"
            )
        coverage_rule = next(
            (
                rule.coverage
                for rule in self.config.coordinates.variables.values()
                if rule.coverage is not None
            ),
            None,
        )
        results = check_coordinate_catalog(
            ds,
            self._coordinate_catalog,
            severities=severities,
            grid_topology=self._coordinate_grid_topology,
            grid_resolution_error=self._coordinate_grid_error,
            allow_standard_name_fallback=(
                self._grid_topology_config.allow_standard_name_fallback
                if self._grid_topology_config
                else True
            ),
            require_explicit_grid_axes=(
                registry.grid.require_explicit_grid_axes if registry.grid else False
            ),
            bounds_dimension_name=(naming.bounds_dimension_name if naming else "bnds"),
            vertices_dimension_name=(
                naming.vertices_dimension_name if naming else "vertices"
            ),
            climatology_bounds_name=(
                naming.climatology_bounds_name if naming else "climatology_bnds"
            ),
            time_bounds_delegated=coverage_rule is not None,
            data_variable_presence_delegated=bool(
                self.config.variable and self.config.variable.existence
            ),
            check_direct_physical_values=(
                direction.check_direct_physical_values if direction else False
            ),
            check_formula_derived_profile=(
                direction.check_formula_derived_profile if direction else False
            ),
            attributes_allowed_when_unset=(
                attributes.allowed_when_unset if attributes else ()
            ),
        )
        if coverage_rule is not None:
            severity = self.get_severity(coverage_rule.severity, "HIGH")
            for identifier, entry in self._coordinate_entries_for_axis("T"):
                if not entry.get("is_climatology"):
                    results.extend(
                        check_time_bounds(
                            ds,
                            severity=severity,
                            coord_name=str(entry.get("out_name") or identifier),
                        )
                    )
        return results

    def check_Coordinates(self, ds):
        results = []
        if not self.config or not self.config.coordinates:
            return results
        coordinate_bounds_enabled = bool(
            self.config.coordinates.registry and self.config.coordinates.registry.bounds
        )
        coordinate_identity_enabled = bool(
            self.config.coordinates.registry
            and self.config.coordinates.registry.identity
        )
        coordinate_attributes_enabled = bool(
            self.config.coordinates.registry
            and self.config.coordinates.registry.attributes
        )
        time_squareness_enabled = any(
            rule.squareness for rule in self.config.coordinates.variables.values()
        )
        for key, rule in self.config.coordinates.variables.items():
            name = str(rule.name.variable_name if rule.name else key)
            if name not in ds.variables:
                if not coordinate_identity_enabled:
                    results.extend(
                        missing_configured_coordinate_result(
                            name, rule, self.get_severity
                        )
                    )
                continue
            if rule.monotonicity and name in ds.dimensions:
                results.extend(
                    check_coordinate_monotonicity(
                        ds,
                        coord_name=name,
                        direction=rule.monotonicity.direction,
                        severity=self.get_severity(rule.monotonicity.severity),
                    )
                )
            if rule.squareness:
                if self._coordinate_catalog is not None:
                    results.extend(
                        check_time_squareness(
                            ds,
                            severity=self.get_severity(rule.squareness.severity),
                            calendar=rule.squareness.ref_calendar or "",
                            ref_time_units=rule.squareness.ref_time_units or "",
                            frequency=None,
                            expected_cell_methods=self._coordinate_catalog.branded_variable.get(
                                "cell_methods"
                            ),
                            report_structural_prerequisites=(
                                not coordinate_identity_enabled
                            ),
                            increment_mapping=self._time_increment_mapping,
                            report_units_prerequisite=(
                                not coordinate_attributes_enabled
                            ),
                            report_bounds_prerequisites=(not coordinate_bounds_enabled),
                            filename_structure_delegated=bool(
                                self.config.drs and self.config.drs.filename
                            ),
                        )
                    )
            if rule.calendar_recommendation:
                results.extend(
                    check_calendar_recommendation(
                        ds,
                        severity=self.get_severity(
                            rule.calendar_recommendation.severity
                        ),
                    )
                )
            for attr_key, attr_rule in rule.attributes.items():
                if (
                    attr_rule.attribute_name or attr_key
                ) != "calendar" or not attr_rule.enum:
                    continue
                checked = check_attribute_suite(
                    ds=ds,
                    var_name=name,
                    attribute_name=attr_rule.attribute_name or attr_key,
                    severity=self.get_severity(attr_rule.severity),
                    is_required=False,
                    enum=attr_rule.enum,
                    project_name=self.project_name,
                    context="Coordinate",
                )
                results.extend(
                    result for result in checked if "[ATTR004]" in result.name
                )
        time_entries = self._coordinate_entries_for_axis("T")
        is_fixed = str(getattr(ds, "frequency", "")).strip() == "fx"
        if self._coordinate_catalog is not None and (
            is_fixed or ("time" in ds.variables and len(time_entries) == 1)
        ):
            time_range = self.config.drs.time_range if self.config.drs else None
            results.extend(
                check_time_range_vs_filename(
                    ds,
                    self.get_severity(
                        time_range.severity if time_range else None, "HIGH"
                    ),
                    precision_by_frequency=(
                        time_range.label_precision if time_range else None
                    ),
                    climatology_suffix=(
                        time_range.climatology_suffix if time_range else ""
                    ),
                    expected_is_climatology=(
                        bool(time_entries[0][1].get("is_climatology"))
                        if time_entries
                        else False
                    ),
                    report_climatology_mismatch=not coordinate_bounds_enabled,
                    report_time_structure_prerequisite=(
                        not coordinate_identity_enabled
                    ),
                    report_time_units_prerequisite=not (
                        coordinate_attributes_enabled or time_squareness_enabled
                    ),
                    report_time_values_prerequisite=(not time_squareness_enabled),
                    report_climatology_bounds_prerequisite=not (
                        coordinate_bounds_enabled or time_squareness_enabled
                    ),
                    report_frequency_prerequisite=(not time_squareness_enabled),
                    filename_structure_delegated=bool(
                        self.config.drs and self.config.drs.filename
                    ),
                )
            )
        return results

    def _specific(self, section, name):
        return self.cordex_config.get(section, {}).get(name)

    def check_time_chunking(self, ds):
        rule = self._specific("time_checks", "check_time_chunking_cordex")
        return (
            check_time_chunking(self, severity=self.get_severity(rule.get("severity")))
            if rule
            else []
        )

    def check_calendar(self, ds):
        rule = self._specific("time_checks", "check_calendar_cordex")
        return (
            check_calendar(self, severity=self.get_severity(rule.get("severity")))
            if rule
            else []
        )

    def check_time_units(self, ds):
        rule = self._specific("time_checks", "check_time_units_cordex")
        return (
            check_time_units(self, severity=self.get_severity(rule.get("severity")))
            if rule
            else []
        )

    def check_attributes_cordex(self, ds):
        results = []
        checks = {
            "check_grid_mapping": check_grid_mapping,
            "check_domain_id": check_domain_id,
            "check_institution": check_institution,
            "check_references": check_references,
            "check_version_realization": check_version_realization,
            "check_version_realization_info": check_version_realization_info,
            "check_grid": check_grid,
            "check_driving_attributes": check_driving_attributes,
        }
        for name, function in checks.items():
            rule = self._specific("attribute_checks", name)
            if not rule:
                continue
            kwargs = {"severity": self.get_severity(rule.get("severity"))}
            if name == "check_grid_mapping":
                kwargs["missing_severity"] = self.get_severity(
                    rule.get("missing_severity"), "MEDIUM"
                )
                kwargs["horizontal_topology"] = self._coordinate_grid_topology
                kwargs["topology_error"] = self._coordinate_grid_error
            if name in {
                "check_domain_id",
                "check_institution",
                "check_version_realization",
                "check_driving_attributes",
            }:
                kwargs["use_esgvoc"] = not self.verification_against_tables
            results.extend(function(self, **kwargs))
        return results

    def check_lat_lon_bounds(self, ds):
        rule = self._specific("variable_checks", "check_lat_lon_bounds")
        return (
            check_lat_lon_bounds(self, severity=self.get_severity(rule.get("severity")))
            if rule
            else []
        )

    def check_horizontal_axes_bounds(self, ds):
        rule = self._specific("variable_checks", "check_horizontal_axes_bounds")
        return (
            check_horizontal_axes_bounds(
                self, severity=self.get_severity(rule.get("severity"))
            )
            if rule
            else []
        )

    def check_lon_value_range(self, ds):
        rule = self._specific("variable_checks", "check_lon_value_range")
        return (
            check_lon_value_range(
                self, severity=self.get_severity(rule.get("severity"))
            )
            if rule
            else []
        )
