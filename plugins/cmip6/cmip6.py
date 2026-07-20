#!/usr/bin/env python
# =============================================================================
# WCRP CMIP6 plugin
# =============================================================================

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, List

import toml
import traceback
from netCDF4 import Dataset
from compliance_checker.base import BaseCheck, TestCtx
from types import SimpleNamespace
from plugins.wcrp_base import WCRPBaseCheck
from plugins.wcrp_schema import WCRPConfig

from checks.attribute_checks.check_attribute_suite import check_attribute_suite

from checks.format_checks.check_format import check_format
from checks.format_checks.check_compression import check_compression

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
from checks.consistency_checks.check_frequency_table_consistency import (
    check_frequency_table_id_consistency,
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

try:
    from checks.variable_checks.check_bounds_value_consistency import (
        check_bounds_value_consistency,
    )
except Exception:
    check_bounds_value_consistency = None

try:
    from checks.variable_checks.check_bounds_shape import (
        check_bounds_shape,
    )
except Exception:
    check_bounds_shape = None


# --- CF Checker helpers ---
try:
    from compliance_checker.cf.util import (
        get_geophysical_variables,
        get_coordinate_variables,
        get_auxiliary_coordinate_variables,
    )
except ImportError as e:
    raise ImportError("Unable to import utils from compliance_checker.cf.util.") from e


# --- ESGVOC Variable Registry lookup ---
try:
    from esgvoc.api.universe import find_terms_in_data_descriptor
except Exception:
    find_terms_in_data_descriptor = None


def _deep_merge(a: dict, b: dict) -> dict:
    """Deep merge dictionaries (b overrides a)."""
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


class Cmip6ProjectCheck(WCRPBaseCheck):
    _cc_spec = "wcrp_cmip6"
    _cc_spec_version = "1.0"
    _cc_description = "WCRP CMIP6 Project PLugin"
    supported_ds = [Dataset]

    def __init__(self, options=None):
        super().__init__(options)

        self.project_name = "cmip6"
        self.config: Optional[WCRPConfig] = None
        self.cfg: dict = {}

        # Mappings
        self.table_id_to_frequency: Dict[str, Any] = {}
        self.table_id_to_time_increment: Dict[str, Any] = {}
        self.variable_id_to_branded_variable: Dict[str, str] = {}

        # Cache
        self._geo_var_cache: Optional[str] = None
        self._expected_term_cache: Any = None

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
                continue
            merged = _deep_merge(merged, _load_toml(p))

        self.cfg = merged
        self.config = WCRPConfig.model_validate(merged)

    def _load_mappings(self) -> None:
        mdir = os.path.join(self.project_config_dir, "mappings")
        if not os.path.isdir(mdir):
            return

        p = os.path.join(mdir, "table_id_to_frequency.toml")
        if os.path.isfile(p):
            d = _load_toml(p)
            self.table_id_to_frequency = d.get("table_id_to_frequency", {}) or {}

        p = os.path.join(mdir, "table_id_to_time_increment.toml")
        if os.path.isfile(p):
            d = _load_toml(p)
            self.table_id_to_time_increment = d.get("time_increment_mapping", {}) or {}

        p = os.path.join(mdir, "variable_id_to_branded_variable.toml")
        if os.path.isfile(p):
            d = _load_toml(p)
            self.variable_id_to_branded_variable = (
                d.get("variable_id_to_branded_variable", {}) or {}
            )


    def _install_time_increment_mapping(self) -> None:
        """
        mapping TOML: "<table_id>.<frequency>" -> ["<int>", "<unit>"]
        used by check_time_squareness: FREQ_INC[(table_id, frequency)] -> (int, unit)
        """
        mapping: Dict[tuple, tuple] = {}

        for k, v in (self.table_id_to_time_increment or {}).items():
            if not isinstance(k, str) or "." not in k:
                continue
            table_id, freq = k.split(".", 1)

            if not isinstance(v, (list, tuple)) or len(v) != 2:
                continue

            try:
                inc_val = int(str(v[0]).strip())
                inc_unit = str(v[1]).strip()
            except Exception:
                continue

            mapping[(table_id, freq)] = (inc_val, inc_unit)

        if mapping:
            time_squareness_mod.FREQ_INC = mapping

    def setup(self, ds):
        super().setup(ds)
        self._load_split_config()
        self._load_mappings()
        self._install_time_increment_mapping()
        self._geo_var_cache = None
        self._expected_term_cache = None

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
    # Variable Registry expected_term lookup (CMIP6 mapping: table_id.variable_id)
    # -------------------------------------------------------------------------
    def _get_expected_from_registry(self, ds: Dataset, severity: int):
        if self._expected_term_cache is not None:
            return self._expected_term_cache, []

        results = []
        if find_terms_in_data_descriptor is None:
            return None, results

        try:
            variable_id = ds.getncattr("variable_id")
            table_id = ds.getncattr("table_id")
        except Exception as e:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(f"Missing required attributes for registry lookup: {e}")
            results.append(ctx.to_result())
            return None, results

        map_key = f"{table_id}.{variable_id}"
        branded = self.variable_id_to_branded_variable.get(map_key)
        if not branded:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(
                f"No branded variable mapping found for '{map_key}' in variable_id_to_branded_variable.toml."
            )
            results.append(ctx.to_result())
            return None, results

        expected_kbv = None
        expected_var = None

        # 1) known_branded_variable lookup
        try:
            kbv_terms = find_terms_in_data_descriptor(
                expression=str(branded),
                data_descriptor_id="known_branded_variable",
                only_id=True,
                selected_term_fields=[
                    "cf_standard_name",
                    "cf_units",
                    "dimensions",
                    "cell_methods",
                    "cell_measures",
                    "description",
                ],
            )
            if kbv_terms:
                expected_kbv = kbv_terms[0]
        except Exception as e:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(
                f"Registry lookup error for known_branded_variable '{branded}': {e}"
            )
            results.append(ctx.to_result())
            return None, results

        if not expected_kbv:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(
                f"Known branded variable '{branded}' was not found in the registry."
            )
            results.append(ctx.to_result())
            return None, results

        # 2) variable lookup for long_name only
        try:
            var_id_lower = str(variable_id).lower()
            var_terms = find_terms_in_data_descriptor(
                expression=var_id_lower,
                data_descriptor_id="variable",
                selected_term_fields=["long_name"],
            )
            
            if var_terms:
                for term in var_terms:
                    if getattr(term, "id", None) == var_id_lower :
                        expected_var = term
                        break
        except Exception as e:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(
                f"Registry lookup error for variable '{variable_id}': {e}"
            )
            results.append(ctx.to_result())
            return None, results

        if not expected_var:
            ctx = TestCtx(severity, "Variable Registry")
            ctx.add_failure(
                f"Variable '{variable_id}' was not found in the registry data descriptor 'variable'. "
                "Only 'long_name' may be unavailable."
            )
            results.append(ctx.to_result())

        merged = {
            "cf_standard_name": getattr(expected_kbv, "cf_standard_name", None),
            "cf_units": getattr(expected_kbv, "cf_units", None),
            "dimensions": getattr(expected_kbv, "dimensions", None),
            "cell_methods": getattr(expected_kbv, "cell_methods", None),
            "cell_measures": getattr(expected_kbv, "cell_measures", None),
            "description": getattr(expected_kbv, "description", None),
            "long_name": getattr(expected_var, "long_name", None) if expected_var else None,
        }

        expected = SimpleNamespace(**merged)

        self._expected_term_cache = expected
        return expected, results
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
            # fallback for older signature (ds, severity)
            return check_compression(ds, sev)

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

        # existence
        if vcfg.existence:
            sev = self.get_severity(vcfg.existence.severity)
            res.extend(check_variable_existence(ds, geo, sev))

        # type
        if vcfg.type:
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
            res.extend(check_filename_vs_global_attrs(ds, sev, project_id=self.project_name))

        # --- Experiment consistency (atomic ATTR007a-d) ---
        if c.experiment_id_vs_activity_id:
            sev = self.get_severity(c.experiment_id_vs_activity_id.severity)
            res.extend(check_experiment_id_vs_activity_id(ds, sev, project_id=self.project_name))

        if c.experiment_id_vs_experiment:
            sev = self.get_severity(c.experiment_id_vs_experiment.severity)
            res.extend(check_experiment_id_vs_experiment(ds, sev, project_id=self.project_name))

        if c.experiment_id_vs_parent_experiment_id:
            sev = self.get_severity(c.experiment_id_vs_parent_experiment_id.severity)
            res.extend(check_experiment_id_vs_parent_experiment_id(ds, sev, project_id=self.project_name))

        if c.experiment_id_vs_sub_experiment_id:
            sev = self.get_severity(c.experiment_id_vs_sub_experiment_id.severity)
            res.extend(check_experiment_id_vs_sub_experiment_id(ds, sev, project_id=self.project_name))

        # --- Institution / source (ATTR009, ATTR010) ---
        if c.institution_id_vs_institution:
            sev = self.get_severity(c.institution_id_vs_institution.severity)
            res.extend(check_institution_consistency(ds, sev, project_id=self.project_name))

        if c.source_id_vs_institution_id:
            sev = self.get_severity(c.source_id_vs_institution_id.severity)
            res.extend(check_source_consistency(ds, sev, project_id=self.project_name))

        # --- Frequency vs table (ATTR008) ---
        if c.frequency_vs_table_id:
            sev = self.get_severity(c.frequency_vs_table_id.severity)
            res.extend(
                check_frequency_table_id_consistency(ds, self.table_id_to_frequency, sev)
            )

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
    def check_Coordinates(self, ds):
        """
        Coordinate checks:
        - Apply only to coords PRESENT in dataset
        - TOML-declared coords that exist
        """
        res: list = []

        if not self.config or not self.config.coordinates:
            return res

        coords_cfg = self.config.coordinates

        def _sev(x, default=BaseCheck.HIGH) -> int:
            """
            Get numeric severity for compliance-checker.
            
            """
            try:
                v = self.get_severity(x)
                return int(v)
            except Exception:
                return int(default)

        # Build mapping: netCDF coord name -> rule
        rule_by_nc: dict = {}
        for key, rule in (coords_cfg.variables or {}).items():
            nc_name = rule.name.variable_name if rule.name else key
            rule_by_nc[str(nc_name)] = rule

        # CF detected coords 
        try:
            cf_coords = set(get_coordinate_variables(ds) or [])
        except Exception:
            cf_coords = set()
        try:
            cf_aux = set(get_auxiliary_coordinate_variables(ds) or [])
        except Exception:
            cf_aux = set()

        # TOML-listed coords present in dataset
        toml_present = {n for n in rule_by_nc.keys() if n in ds.variables}

        coord_set = sorted(cf_coords | cf_aux | toml_present)

        # ---------------- Global dimension checks ----------------
        if getattr(coords_cfg, "dimensions", None):
            sev = _sev(coords_cfg.dimensions.severity, default=BaseCheck.MEDIUM)
            for cname in coord_set:
                if cname not in ds.variables:
                    continue
                var = ds.variables[cname]
                for dim in getattr(var, "dimensions", ()):
                    res.extend(check_dimension_existence(ds, dim, sev))
                    res.extend(check_dimension_positive(ds, dim, sev))

        # ---------------- Bounds checks ----------------
        if getattr(coords_cfg, "bounds", None):
            sev = _sev(coords_cfg.bounds.severity, default=BaseCheck.MEDIUM)

            for cname in coord_set:
                if cname not in ds.variables:
                    continue
                cvar = ds.variables[cname]
                if not getattr(cvar, "bounds", None):
                    continue

                if check_bounds_shape is not None:
                    res.extend(check_bounds_shape(ds, cname, severity=sev))

                if check_bounds_value_consistency is not None:
                    res.extend(check_bounds_value_consistency(ds, cname, severity=sev))

        # ---------------- Per-coordinate TOML rules ----------------
        for cname in coord_set:
            if cname not in ds.variables:
                continue
            rule = rule_by_nc.get(cname)
            if not rule:
                continue

            var = ds.variables[cname]

            # type
            if getattr(rule, "type", None):
                sev = _sev(rule.type.severity, default=BaseCheck.MEDIUM)
                dt = (rule.type.data_type or "").lower()
                allowed = ["f"] if dt in {"float", "double", "real"} else None
                if allowed:
                    res.extend(
                        check_variable_type(
                            ds, cname, allowed_types=allowed, severity=sev
                        )
                    )

            # dimensions
            if getattr(rule, "dimensions", None):
                sev = _sev(rule.dimensions.severity, default=BaseCheck.MEDIUM)
                for dim in getattr(var, "dimensions", ()):
                    res.extend(check_dimension_existence(ds, dim, sev))
                    res.extend(check_dimension_positive(ds, dim, sev))

            # monotonicity
            if getattr(rule, "monotonicity", None):
                if cname not in ds.dimensions:
                    continue
                sev = _sev(rule.monotonicity.severity, default=BaseCheck.MEDIUM)
                res.extend(
                    check_coordinate_monotonicity(
                        ds,
                        coord_name=cname,
                        direction=rule.monotonicity.direction,
                        severity=sev,
                    )
                )

            # TIME001
            if getattr(rule, "squareness", None):
                sev = _sev(rule.squareness.severity, default=BaseCheck.MEDIUM)
                res.extend(
                    check_time_squareness(
                        ds,
                        severity=sev,
                        calendar=rule.squareness.ref_calendar or "",
                        ref_time_units=rule.squareness.ref_time_units or "",
                        frequency=None,
                    )
                )

            # TIME002
            if getattr(rule, "coverage", None):
                sev = _sev(rule.coverage.severity, default=BaseCheck.MEDIUM)
                res.extend(check_time_bounds(ds, severity=sev))

            #time calendar recommendation (TIME003a)
            if getattr(rule, "calendar_recommendation", None):
                sev = _sev(rule.calendar_recommendation.severity, default=BaseCheck.MEDIUM)
                res.extend(check_calendar_recommendation(ds, severity=sev))

            # coordinate variable attributes
            for attr_key, arule in (getattr(rule, "attributes", None) or {}).items():
                sev = _sev(arule.severity, default=BaseCheck.MEDIUM)
                name_in_file = arule.attribute_name or attr_key
                res.extend(
                    check_attribute_suite(
                        ds=ds,
                        var_name=cname,
                        attribute_name=name_in_file,
                        severity=sev,
                        value_type=arule.value_type,
                        is_required=arule.is_required,
                        na_value=arule.na_value,
                        pattern=arule.pattern,
                        constant=arule.constant,
                        threshold=arule.threshold,
                        is_above_threshold=arule.is_above_threshold,
                        enum=arule.enum,
                        as_variable=arule.as_variable,
                        is_positive=arule.is_positive,
                        cv_source_collection=arule.cv_source_collection,
                        cv_source_collection_key=arule.cv_source_collection_key,
                        cv_source_term_key=arule.cv_source_term_key,
                        project_name=self.project_name,
                        expected_term=None,
                        context="Coordinate",
                    )
                )

        if check_time_range_vs_filename is not None:
            precision_map = None
            if self.config and self.config.drs:
                precision_map = self.config.drs.time_range_label_precision
            res.extend(
                check_time_range_vs_filename(
                    ds,
                    BaseCheck.HIGH,
                    precision_by_frequency=precision_map,
                )
            )

        return res
    
