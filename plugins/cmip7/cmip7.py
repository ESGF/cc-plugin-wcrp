#!/usr/bin/env python
# =============================================================================
# WCRP CMIP7 plugin
# =============================================================================

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, List

import toml
from netCDF4 import Dataset
from compliance_checker.base import BaseCheck, TestCtx

from plugins.wcrp_base import WCRPBaseCheck
from plugins.wcrp_schema import WCRPConfig

from checks.attribute_checks.check_attribute_suite import check_attribute_suite

from checks.format_checks.check_format import check_format
from checks.format_checks.check_compression import check_compression
from checks.format_checks.check_internal_packing import  check_cmip7_packing
from checks.consistency_checks.check_drs_filename_cv import (
    check_drs_filename,
    check_drs_directory,
)
from checks.consistency_checks.check_drs_consistency import (
    check_attributes_match_directory_structure,
    check_filename_matches_directory_structure,
)
from checks.consistency_checks.check_attributes_match_filename import (
    check_filename_vs_global_attrs,
)
from checks.consistency_checks.check_experiment_consistency import (
    check_experiment_id_vs_activity_id,
    check_experiment_id_vs_experiment,
    check_experiment_id_vs_parent_experiment_id,
    check_experiment_id_vs_sub_experiment_id,
)
from checks.consistency_checks.check_institution_source_consistency import (
    check_institution_consistency,
    check_source_consistency,
)
from checks.consistency_checks.check_variant_label_consistency import (
    check_variant_vs_realization_index,
    check_variant_vs_initialization_index,
    check_variant_vs_physics_index,
    check_variant_vs_forcing_index,
)

try:
    from checks.time_checks.check_time_range_vs_filename import (
        check_time_range_vs_filename,
    )
except Exception:
    check_time_range_vs_filename = None

from checks.variable_checks.check_variable_existence import check_variable_existence
from checks.variable_checks.check_variable_type import check_variable_type

from checks.dimension_checks.check_dimension_existence import check_dimension_existence
from checks.dimension_checks.check_dimension_positive import check_dimension_positive

from checks.time_checks.check_time_squareness import check_time_squareness
import checks.time_checks.check_time_squareness as time_squareness_mod
from checks.time_checks.check_time_bounds import check_time_bounds
from checks.time_checks.check_time_calendar import check_calendar_recommendation
from checks.variable_checks.check_coordinate_monotonicity import (
    check_coordinate_monotonicity,
)

from checks.variable_checks.check_variable_shape_vs_dimensions import (
    check_variable_shape,
)
from checks.variable_checks.known_branded_variable import (
    KnownBrandedVariableLookupError,
    lookup_expected_variable_metadata,
)
from checks.coordinate_checks import (
    CoordinateMetadataError,
    GridTopologyConfigError,
    check_coordinate_catalog,
    load_catalog,
    load_grid_topology_config,
    resolve_grid_topology,
)
from checks.coordinate_checks.utils import coordinate_type


# --- CF Checker helpers ---
try:
    from compliance_checker.cf.util import get_geophysical_variables
except ImportError as e:
    raise ImportError("Unable to import utils from compliance_checker.cf.util.") from e


# --- ESGVOC Variable Registry lookup ---
try:
    from esgvoc.api.universe import find_terms_in_data_descriptor
except Exception:
    find_terms_in_data_descriptor = None


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_toml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return toml.load(f)


def _is_flag_variable(ds, var_name):
    """Return True iff ``var_name`` carries CF flag semantics
    (``flag_values`` or ``flag_meanings``). CMIP7 region-selector
    variables like ``basin`` and ``siline`` are integer flag variables
    per CF §7.5 and CMIP7 data descriptor ``type: integer``. The default
    geophysical_variable.toml rules assume continuous float fields
    (float ``_FillValue = 1e20`` and float type check) and cannot
    describe the flag case.
    """
    try:
        attrs = ds.variables[var_name].ncattrs()
    except Exception:
        return False
    return "flag_values" in attrs or "flag_meanings" in attrs


class Cmip7ProjectCheck(WCRPBaseCheck):
    _cc_spec = "wcrp_cmip7"
    _cc_spec_version = "1.0"
    _cc_description = "WCRP CMIP7 Project Checks"
    supported_ds = [Dataset]

    def __init__(self, options=None):
        super().__init__(options)

        self.project_name = "cmip7"
        self.config: Optional[WCRPConfig] = None
        self.cfg: dict = {}

        # Mapping kept: time increment only
        self.table_id_to_time_increment: Dict[str, Any] = {}

        # Cache
        self._geo_var_cache: Optional[str] = None
        self._expected_term_cache: Any = None
        self._coordinate_catalog = None
        self._coordinate_setup_error: Optional[str] = None
        self._grid_topology_config = None
        self._grid_topology_config_error: Optional[str] = None
        self._coordinate_grid_topology: Optional[str] = None
        self._coordinate_grid_error: Optional[str] = None

        # Config directory
        if options and "project_config_dir" in options:
            self.project_config_dir = options["project_config_dir"]
        else:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_config_dir = os.path.join(this_dir, "config", "wcrp")

    # -------------------------------------------------------------------------
    # Loading split TOML config + mappings
    # -------------------------------------------------------------------------
    def _load_split_config(self) -> None:
        base = self.project_config_dir
        files = [
            "project.toml",
            "file.toml",
            "drs.toml",
            "global_attributes.toml",
            "geophysical_variable.toml",
            "coordinate_variables.toml",
        ]

        merged: dict = {}
        for fn in files:
            p = os.path.join(base, fn)
            if not os.path.isfile(p):
                self._record_setup_warning(
                    f"Project configuration file not found at '{p}'"
                )
                continue
            merged = _deep_merge(merged, _load_toml(p))

        self.cfg = merged
        self.config = WCRPConfig.model_validate(merged)

    def _load_mappings(self) -> None:
        mdir = os.path.join(self.project_config_dir, "mappings")
        self._grid_topology_config = None
        self._grid_topology_config_error = None
        if not os.path.isdir(mdir):
            self._grid_topology_config_error = (
                f"Project mapping directory does not exist: '{mdir}'."
            )
            self._record_setup_warning(
                f"Project mapping directory not found at '{mdir}'"
            )
            return

        p = os.path.join(mdir, "table_id_to_time_increment.toml")
        if os.path.isfile(p):
            d = _load_toml(p)
            self.table_id_to_time_increment = d.get("time_increment_mapping", {}) or {}
        else:
            self._record_setup_warning(f"Project mapping file not found at '{p}'")

        topology_path = os.path.join(mdir, "grid_topology.toml")
        try:
            self._grid_topology_config = load_grid_topology_config(topology_path)
        except GridTopologyConfigError as exc:
            # This is reported by the dedicated HIGH grid result only when the
            # branded variable actually requires generic horizontal coordinates.
            self._grid_topology_config_error = str(exc)

    def _install_time_increment_mapping(self, ds: Dataset) -> None:
        """
        Inject TOML mapping into check_time_squareness.FREQ_INC.


        """
        mapping: Dict[tuple, tuple] = {}

        for k, v in (self.table_id_to_time_increment or {}).items():
            if not isinstance(k, str) or "." not in k:
                self._record_setup_warning(
                    f"Ignored invalid time-increment mapping key {k!r}"
                )
                continue
            table_id, freq = k.split(".", 1)
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                self._record_setup_warning(
                    f"Ignored invalid time-increment mapping for '{k}': {v!r}"
                )
                continue
            try:
                inc_val = int(str(v[0]).strip())
                inc_unit = str(v[1]).strip()
            except Exception as exc:
                self._record_setup_warning(
                    f"Could not interpret time-increment mapping for '{k}'",
                    exc,
                )
                continue
            mapping[(table_id, freq)] = (inc_val, inc_unit)

        # Safe per-dataset fallback: (table_id, freq) -> ("None", freq)
        try:
            ds_table_id = ds.getncattr("table_id")
            ds_freq = ds.getncattr("frequency")
        except Exception:
            ds_table_id = None
            ds_freq = None

        if ds_table_id and ds_freq:
            key = (str(ds_table_id), str(ds_freq))
            if key not in mapping:
                none_key = ("None", str(ds_freq))
                if none_key in mapping:
                    mapping[key] = mapping[none_key]

        if mapping:
            time_squareness_mod.FREQ_INC = mapping

    def setup(self, ds):
        super().setup(ds)
        self._run_setup_step(
            "load the CMIP7 project configuration",
            self._load_split_config,
        )
        self._run_setup_step("load the CMIP7 mappings", self._load_mappings)
        self._run_setup_step(
            "install the CMIP7 time-increment mapping",
            self._install_time_increment_mapping,
            ds,
        )
        self._geo_var_cache = None
        self._expected_term_cache = None
        self._coordinate_catalog = None
        self._coordinate_setup_error = None
        self._coordinate_grid_topology = None
        self._coordinate_grid_error = None
        registry = (
            self.config.coordinates.registry
            if self.config and self.config.coordinates
            else None
        )
        if registry is not None:
            try:
                branded = self._get_attr("branded_variable", "")
                self._coordinate_catalog = load_catalog(
                    str(branded or ""), project_id=self.project_name
                )
                if any(
                    coordinate_type(entry) == "generic_horizontal"
                    for entry in self._coordinate_catalog.data_coordinates.values()
                    if entry.get("id") in self._coordinate_catalog.coordinate_ids
                ):
                    if self._grid_topology_config_error:
                        self._coordinate_grid_error = (
                            "The CMIP7 grid-topology mapping could not be loaded. "
                            f"Technical reason: {self._grid_topology_config_error}"
                        )
                    elif self._grid_topology_config is None:
                        self._coordinate_grid_error = (
                            "The CMIP7 grid-topology mapping is unavailable for an "
                            "unknown reason."
                        )
                    else:
                        self._coordinate_grid_topology, self._coordinate_grid_error = (
                            resolve_grid_topology(
                                self._grid_topology_config,
                                grid_label=self._get_attr("grid_label", ""),
                            )
                        )
            except Exception as exc:
                # Report separately from setup_warnings: coordinate metadata
                # has one dedicated HIGH result only to be reported once
                # and not repeated for every coordinate-family result or by TOOL001.
                self._coordinate_setup_error = (
                    str(exc)
                    if isinstance(exc, CoordinateMetadataError)
                    else f"Unexpected {type(exc).__name__}: {exc}"
                )

    # -------------------------------------------------------------------------
    # CF-based geophysical variable identification
    # -------------------------------------------------------------------------
    def _get_geo_var(
        self, ds: Dataset, severity: int
    ) -> Tuple[Optional[str], List[Any]]:
        if self._geo_var_cache and self._geo_var_cache in ds.variables:
            return self._geo_var_cache, []

        res: list = []
        try:
            geo_vars = list(get_geophysical_variables(ds) or [])
        except Exception as e:
            ctx = TestCtx(severity, "Geophysical Variable Detection")
            ctx.add_failure(f"Error detecting geophysical variables: {e}")
            return None, [ctx.to_result()]

        if len(geo_vars) == 1:
            self._geo_var_cache = geo_vars[0]
            return self._geo_var_cache, res

        ctx = TestCtx(severity, "Geophysical Variable Detection")

        if len(geo_vars) == 0:
            # CF detection returns zero candidates for files whose data
            # variable carries flag_meanings (compliance-checker's
            # is_geophysical heuristic treats those as QC flags). CMIP7
            # region-selector fx files (basin, siline, ...) are flag-valued
            # by spec but ARE the geophysical variable of the file. Fall
            # back to the global variable_id attribute, which CMIP7 defines
            # as the single geophysical variable of the file, when the
            # named variable exists in the dataset.
            vid = getattr(ds, "variable_id", None)
            vid = str(vid) if vid else None
            if vid and vid in ds.variables:
                self._geo_var_cache = vid
                return vid, res
            ctx.add_failure("No geophysical variable detected in the file.")
            res.append(ctx.to_result())
            return None, res

        # CF detection is ambiguous (multiple candidates). Disambiguate
        # using the global attribute variable_id, which by CMIP definition
        # designates the single geophysical variable of the file. This
        # resolves false positives where cell measures (area, areacella,
        # areacello, volcello...) are also flagged as geophysical.
        vid = getattr(ds, "variable_id", None)
        vid = str(vid) if vid else None

        if vid and vid in geo_vars:
            self._geo_var_cache = vid
            return vid, res

        if vid:
            ctx.add_failure(
                f"Expected exactly 1 geophysical variable, found {len(geo_vars)}: "
                f"{geo_vars}. Global attribute variable_id='{vid}' is not among "
                f"the detected candidates."
            )
        else:
            ctx.add_failure(
                f"Expected exactly 1 geophysical variable, found {len(geo_vars)}: "
                f"{geo_vars}. No variable_id global attribute available to "
                f"disambiguate."
            )
        res.append(ctx.to_result())
        return None, res

    # -------------------------------------------------------------------------
    # Registry expected_term lookup (CMIP7: use global attribute branded_variable)
    # -------------------------------------------------------------------------
    def _get_expected_from_registry(self, ds: Dataset, severity: int):
        if self._expected_term_cache is not None:
            return self._expected_term_cache, []

        results = []
        if find_terms_in_data_descriptor is None:
            return None, results

        branded = None
        variable_id = None

        try:
            branded = ds.getncattr("branded_variable")
        except Exception:
            branded = None

        try:
            variable_id = ds.getncattr("variable_id")
        except Exception:
            variable_id = None

        if not branded:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure("Missing global attribute 'branded_variable'.")
            results.append(ctx.to_result())
            return None, results

        if not variable_id:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure("Missing global attribute 'variable_id'.")
            results.append(ctx.to_result())
            return None, results

        try:
            lookup = lookup_expected_variable_metadata(
                find_terms_in_data_descriptor,
                str(branded),
                fallback_variable_id=str(variable_id).lower(),
            )
        except KnownBrandedVariableLookupError as e:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(str(e))
            results.append(ctx.to_result())
            return None, results

        if lookup.warning:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(lookup.warning)
            results.append(ctx.to_result())

        self._expected_term_cache = lookup.expected
        return lookup.expected, results

    # -------------------------------------------------------------------------
    # 1) File checks
    # -------------------------------------------------------------------------
    def check_File_Format(self, ds):
        if not self.config or not self.config.file or not self.config.file.format:
            return []
        r = self.config.file.format
        sev = self.get_severity(r.severity)
        return check_format(ds, r.expected_format, r.allowed_data_models, sev)

    def check_File_Compression(self, ds):
        if not self.config or not self.config.file or not self.config.file.compression:
            return []
        r = self.config.file.compression
        sev = self.get_severity(r.severity)

        try:
            return check_compression(ds, severity=sev)
        except TypeError:
            return check_compression(ds, sev)

    def check_File_Internal_Packing(self, ds):
        if (
            not self.config
            or not self.config.file
            or not self.config.file.internal_packing
        ):
            return []

        r = self.config.file.internal_packing
        sev = self.get_severity(r.severity)

        return check_cmip7_packing(ds, severity=sev)

    # -------------------------------------------------------------------------
    # 2) Global attributes
    # -------------------------------------------------------------------------
    def check_Global_Attributes(self, ds):
        res = []
        if not self.config or not self.config.global_:
            return res

        for attr_key, rule in self.config.global_.attributes.items():
            sev = self.get_severity(rule.severity)
            name_in_file = rule.attribute_name or attr_key
            res.extend(
                check_attribute_suite(
                    ds=ds,
                    var_name=None,
                    attribute_name=name_in_file,
                    severity=sev,
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
                    expected_term=None,
                    cv_source_term_key=rule.cv_source_term_key,
                )
            )
        return res

    # -------------------------------------------------------------------------
    # 3) DRS checks
    # -------------------------------------------------------------------------
    def check_DRS(self, ds):
        res = []
        if not self.config or not self.config.drs:
            return res

        drs = self.config.drs
        if drs.filename:
            sev = self.get_severity(drs.filename.severity)
            res.extend(check_drs_filename(ds, sev, project_id=self.project_name))

        if drs.directory:
            sev = self.get_severity(drs.directory.severity)
            res.extend(check_drs_directory(ds, sev, project_id=self.project_name))

        if drs.attributes_vs_directory:
            sev = self.get_severity(drs.attributes_vs_directory.severity)
            res.extend(
                check_attributes_match_directory_structure(
                    ds, sev, project_id=self.project_name
                )
            )

        if drs.filename_vs_directory:
            sev = self.get_severity(drs.filename_vs_directory.severity)
            res.extend(
                check_filename_matches_directory_structure(
                    ds, sev, project_id=self.project_name
                )
            )

        return res

    # -------------------------------------------------------------------------
    # 4) Geophysical variable checks
    # -------------------------------------------------------------------------
    def check_Geophysical_Variable(self, ds):
        res = []
        if not self.config or not self.config.variable:
            return res

        sev_default = BaseCheck.HIGH
        geo, geo_r = self._get_geo_var(ds, sev_default)
        res.extend(geo_r)
        if not geo:
            return res

        vcfg = self.config.variable
        # CF flag-valued variables (basin, siline, similar CMIP7 region
        # selectors) are integer by construction. The default TOML rules
        # (float type, _FillValue = 1e20, missing_value = 1e20) do not
        # fit them. Gate the float type check and the two fill/missing
        # attribute rules on ``is_flag``; everything else still applies.
        # See #59.
        is_flag = _is_flag_variable(ds, geo)

        # existence
        if vcfg.existence:
            sev = self.get_severity(vcfg.existence.severity)
            res.extend(check_variable_existence(ds, geo, sev))

        # type
        if vcfg.type and not is_flag:
            sev = self.get_severity(vcfg.type.severity)
            dt = (vcfg.type.data_type or "").lower()
            allowed = ["f"] if dt in {"float", "double", "real"} else None
            if allowed:
                res.extend(
                    check_variable_type(ds, geo, allowed_types=allowed, severity=sev)
                )

        # dimensions
        if vcfg.dimensions:
            sev = self.get_severity(vcfg.dimensions.severity)
            for d in list(ds.variables[geo].dimensions):
                res.extend(check_dimension_existence(ds, d, sev))
                res.extend(check_dimension_positive(ds, d, sev))

        # shape 
        shape_rule = getattr(vcfg, "shape", None)
        if shape_rule:
            sev = self.get_severity(shape_rule.severity)
            res.extend(check_variable_shape(ds, geo, severity=sev))

        # attributes (registry if needed)
        expected_term = None
        if vcfg.attributes and any(
            r.cv_source_term_key for r in vcfg.attributes.values()
        ):
            expected_term, vr_r = self._get_expected_from_registry(ds, sev_default)
            res.extend(vr_r)

        for attr_key, rule in vcfg.attributes.items():
            sev = self.get_severity(rule.severity)
            name_in_file = rule.attribute_name or attr_key
            if is_flag and name_in_file in ("_FillValue", "missing_value"):
                continue
            res.extend(
                check_attribute_suite(
                    ds=ds,
                    var_name=geo,
                    attribute_name=name_in_file,
                    severity=sev,
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
                )
            )

        return res

    # -------------------------------------------------------------------------
    # 5) Global consistency 
    # -------------------------------------------------------------------------
    def check_Global_Consistency(self, ds):
        res = []
        if (
            not self.config
            or not self.config.global_
            or not self.config.global_.consistency
        ):
            return res

        c = self.config.global_.consistency

        if c.filename_vs_attributes:
            sev = self.get_severity(c.filename_vs_attributes.severity)
            res.extend(
                check_filename_vs_global_attrs(ds, sev, project_id=self.project_name)
            )

        # --- Experiment consistency (atomic ATTR007a-c ; no sub in CMIP7) ---
        if c.experiment_id_vs_activity_id:
            sev = self.get_severity(c.experiment_id_vs_activity_id.severity)
            res.extend(
                check_experiment_id_vs_activity_id(
                    ds, sev, project_id=self.project_name
                )
            )

        if c.experiment_id_vs_experiment:
            sev = self.get_severity(c.experiment_id_vs_experiment.severity)
            res.extend(
                check_experiment_id_vs_experiment(ds, sev, project_id=self.project_name)
            )

        if c.experiment_id_vs_parent_experiment_id:
            sev = self.get_severity(c.experiment_id_vs_parent_experiment_id.severity)
            res.extend(
                check_experiment_id_vs_parent_experiment_id(
                    ds, sev, project_id=self.project_name
                )
            )

        # sub_experiment_id has no meaning in CMIP7; the key is absent from its
        # TOML, so this simply never runs. Kept for symmetry / robustness.
        if c.experiment_id_vs_sub_experiment_id:
            sev = self.get_severity(c.experiment_id_vs_sub_experiment_id.severity)
            res.extend(
                check_experiment_id_vs_sub_experiment_id(
                    ds, sev, project_id=self.project_name
                )
            )

        # --- Institution / source ---
        if c.institution_id_vs_institution:
            sev = self.get_severity(c.institution_id_vs_institution.severity)
            res.extend(
                check_institution_consistency(ds, sev, project_id=self.project_name)
            )

        if c.source_id_vs_institution_id:
            sev = self.get_severity(c.source_id_vs_institution_id.severity)
            res.extend(check_source_consistency(ds, sev, project_id=self.project_name))

        # --- Frequency vs table: not part of CMIP7 (no such key in its TOML) ---

        # --- Variant label consistency (atomic ATTR006a-d) ---
        if c.variant_label_vs_realization_index:
            sev = self.get_severity(c.variant_label_vs_realization_index.severity)
            res.extend(check_variant_vs_realization_index(ds, sev))

        if c.variant_label_vs_initialization_index:
            sev = self.get_severity(c.variant_label_vs_initialization_index.severity)
            res.extend(check_variant_vs_initialization_index(ds, sev))

        if c.variant_label_vs_physics_index:
            sev = self.get_severity(c.variant_label_vs_physics_index.severity)
            res.extend(check_variant_vs_physics_index(ds, sev))

        if c.variant_label_vs_forcing_index:
            sev = self.get_severity(c.variant_label_vs_forcing_index.severity)
            res.extend(check_variant_vs_forcing_index(ds, sev))

        return res

    # -------------------------------------------------------------------------
    # 6) Coordinates checks
    # -------------------------------------------------------------------------
    def _coordinate_entries_for_axis(self, axis):
        if self._coordinate_catalog is None:
            return []
        return [
            (identifier, self._coordinate_catalog.data_coordinates[identifier])
            for identifier in self._coordinate_catalog.coordinate_ids
            if self._coordinate_catalog.data_coordinates[identifier].get("axis") == axis
        ]

    def check_Coordinate_Metadata_Setup(self, ds):
        registry = (
            self.config.coordinates.registry
            if self.config and self.config.coordinates
            else None
        )
        if registry is None or registry.setup is None:
            return []
        severity = self.get_severity(registry.setup.severity, "HIGH")
        ctx = TestCtx(
            severity,
            "[COORD000] ESGVoc coordinate metadata initialization",
        )
        if self._coordinate_setup_error:
            ctx.add_failure(
                "The CMIP7 coordinate checks could not be initialized, so all "
                "vocabulary-driven coordinate checks were skipped for this file. "
                f"Technical reason: {self._coordinate_setup_error}"
            )
        elif self._coordinate_catalog is None:
            ctx.add_failure(
                "The CMIP7 coordinate catalog is unavailable for an unknown reason; "
                "all vocabulary-driven coordinate checks were skipped."
            )
        else:
            ctx.add_pass()
        return [ctx.to_result()]

    def check_Coordinate_Standard(self, ds):
        registry = (
            self.config.coordinates.registry
            if self.config and self.config.coordinates
            else None
        )
        if registry is None or self._coordinate_catalog is None:
            return []
        severities = {
            family: self.get_severity(rule.severity, "HIGH")
            for family in (
                "identity",
                "dimension_order",
                "attributes",
                "recommendations",
                "direction",
                "valid_range",
                "requested_values",
                "bounds",
                "bounds_name",
                "associations",
                "grid",
                "formula",
            )
            if (rule := getattr(registry, family)) is not None
        }
        naming = registry.bounds_name
        direction = registry.direction
        attributes = registry.attributes
        if attributes is not None and attributes.allowed_when_unset:
            severities["allowed_when_unset"] = self.get_severity(
                attributes.allowed_when_unset_severity,
                "LOW",
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
                if self._grid_topology_config is not None
                else True
            ),
            bounds_dimension_name=(
                naming.bounds_dimension_name if naming is not None else "bnds"
            ),
            vertices_dimension_name=(
                naming.vertices_dimension_name if naming is not None else "vertices"
            ),
            climatology_bounds_name=(
                naming.climatology_bounds_name
                if naming is not None
                else "climatology_bnds"
            ),
            time_bounds_delegated=coverage_rule is not None,
            check_direct_physical_values=(
                direction.check_direct_physical_values
                if direction is not None
                else False
            ),
            check_formula_derived_profile=(
                direction.check_formula_derived_profile
                if direction is not None
                else False
            ),
            attributes_allowed_when_unset=(
                attributes.allowed_when_unset if attributes is not None else ()
            ),
        )

        if coverage_rule is not None:
            severity = self.get_severity(coverage_rule.severity, "HIGH")
            for identifier, entry in self._coordinate_entries_for_axis("T"):
                if entry.get("is_climatology"):
                    continue
                coord_name = str(entry.get("out_name") or identifier)
                results.extend(
                    check_time_bounds(
                        ds,
                        severity=severity,
                        coord_name=coord_name,
                    )
                )
        return results

    def check_Coordinates(self, ds):
        res: list = []

        if not self.config or not self.config.coordinates:
            return res

        coords_cfg = self.config.coordinates
        time_coordinate_names = {
            str(entry.get("out_name") or identifier)
            for identifier, entry in self._coordinate_entries_for_axis("T")
        }
        # Established time checks must remain runnable when the ESGVoc
        # coordinate catalogue cannot be loaded. Their configured rules are
        # independent of the catalogue and already identify the time variable.
        time_coordinate_names.update(
            str(rule.name.variable_name if rule.name else key)
            for key, rule in (coords_cfg.variables or {}).items()
            if rule.squareness
            or rule.coverage
            or getattr(rule, "calendar_recommendation", None)
        )

        # The ESGVoc catalogue owns general coordinate validation. Retain only
        # the established time checks that are not replaced by that suite.
        for key, rule in (coords_cfg.variables or {}).items():
            cname = str(rule.name.variable_name if rule.name else key)
            if cname not in time_coordinate_names or cname not in ds.variables:
                continue

            if rule.monotonicity and cname in ds.dimensions:
                sev = self.get_severity(rule.monotonicity.severity)
                res.extend(
                    check_coordinate_monotonicity(
                        ds,
                        coord_name=cname,
                        direction=rule.monotonicity.direction,
                        severity=sev,
                    )
                )

            if rule.squareness:
                sev = self.get_severity(rule.squareness.severity)
                res.extend(
                    check_time_squareness(
                        ds,
                        severity=sev,
                        calendar=rule.squareness.ref_calendar or "",
                        ref_time_units=rule.squareness.ref_time_units or "",
                        frequency=None,  # increments injected from TOML in setup()
                    )
                )

            if getattr(rule, "calendar_recommendation", None):
                sev = self.get_severity(rule.calendar_recommendation.severity)
                res.extend(check_calendar_recommendation(ds, severity=sev))

            for attr_key, arule in (rule.attributes or {}).items():
                if (arule.attribute_name or attr_key) != "calendar" or not arule.enum:
                    continue
                sev = self.get_severity(arule.severity)
                name_in_file = arule.attribute_name or attr_key
                attribute_results = check_attribute_suite(
                    ds=ds,
                    var_name=cname,
                    attribute_name=name_in_file,
                    severity=sev,
                    is_required=False,
                    enum=arule.enum,
                    project_name=self.project_name,
                    context="Coordinate",
                )
                res.extend(
                    result for result in attribute_results if "[ATTR004]" in result.name
                )

        if check_time_range_vs_filename is not None:
            precision_map = None
            climatology_suffix = ""
            severity = BaseCheck.HIGH
            if self.config and self.config.drs:
                time_range = self.config.drs.time_range
                precision_map = time_range.label_precision
                climatology_suffix = time_range.climatology_suffix
                severity = self.get_severity(time_range.severity, "HIGH")
            res.extend(
                check_time_range_vs_filename(
                    ds,
                    severity,
                    precision_by_frequency=precision_map,
                    climatology_suffix=climatology_suffix,
                )
            )

        return res
