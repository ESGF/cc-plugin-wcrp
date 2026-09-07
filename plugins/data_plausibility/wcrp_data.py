
#!/usr/bin/env python

# =============================================================================
""" 
WCRP Data Plausibility
This module implements the Data Plausibility suite as a standalone compliance plugin separate from 
the core `wcrp_cmip6` and `wcrp_cmip7` plugins.
"""
# =============================================================================
from netCDF4 import Dataset
import os
import toml
from compliance_checker.base import BaseCheck, Result, TestCtx
from plugins.wcrp_base import WCRPBaseCheck
from checks.data_plausibility_checks.check_nan_inf import check_nan_inf
from checks.data_plausibility_checks.check_fill_missing import check_fillvalues_timeseries
from checks.data_plausibility_checks.check_constant import check_constants
from checks.data_plausibility_checks.detect_physically_impossible_outlier import check_outliers
from checks.data_plausibility_checks.check_spatial_statistical_outliers import check_spatial_statistical_outliers
from checks.data_plausibility_checks.check_chunk_size import check_chunk_size


# --- Esgvoc universe import---
try:
    from esgvoc.api.universe import find_terms_in_data_descriptor
    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False



class DatapluginProjectCheck(WCRPBaseCheck):
    """
    Class for WCRP CMIP6 project-specific compliance checks.
    """

    _cc_spec = "wcrp_data"
    _cc_spec_version = "1.0"
    _cc_description = "WCRP Project Checks"
    supported_ds = [Dataset]

    def __init__(self, options=None):
        super().__init__(options)
        if options and 'project_config_path' in options:
            self.project_config_path = options['project_config_path']
        else:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_config_path = os.path.join(this_dir, "resources", "wcrp_config.toml")

        # Define project name here for vocabulary checks 
        self.project_name = "cmip6"
       
    def _load_project_config(self):
    
        """Loads the TOML configuration file."""
        if not self.project_config_path or not os.path.exists(self.project_config_path):
            self.config = {}
            self._record_setup_warning(
                f"Project configuration file not found at '{self.project_config_path}'"
            )
            return
        try:
            with open(self.project_config_path, 'r', encoding="utf-8") as f:
                self.config = toml.load(f)
        except Exception as e:
            self.config = {}
            self._record_setup_warning(
                f"Could not parse project configuration file "
                f"'{self.project_config_path}'",
                e,
            )

    def setup(self, ds):
        
        """Loads the main configuration and the variable mapping file before running checks."""
        super().setup(ds)
        self._run_setup_step(
            "load the data-plausibility project configuration",
            self._load_project_config,
        )

        # Load variable mapping directly from 'mapping_variables.toml' located in the same folder as the config
        base_dir = os.path.dirname(self.project_config_path)
        mapping_filepath = os.path.join(base_dir, "mapping_variables.toml")

        
        try:
            with open(mapping_filepath, 'r') as f:
                self.variable_mapping = toml.load(f).get('mapping_variables', {})
                
        except FileNotFoundError:
            self.variable_mapping = {}
            self._record_setup_warning(
                f"Variable mapping file not found at '{mapping_filepath}'"
            )
        except Exception as e:
            self.variable_mapping = {}
            self._record_setup_warning(
                f"Could not load variable mapping file '{mapping_filepath}'",
                e,
            )
        
        
    def check_Data_Plausibility(self, ds):
        """
        Runs all DATAxxx plausibility checks on CMIP6 variables.
        Each check produces a Result with its own DATA00X identifier.
        """
        results = []

        # Load configuration section
        if "data_plausibility_checks" not in self.config:
            return results

        config = self.config["data_plausibility_checks"]
        variable_id = getattr(ds, "variable_id", None)

        # Retrieve the project name defined in TOML (default “CMIP”)
        project = self.config.get("data_plausibility_checks", {}).get("project", "CMIP")


        # === DATA001: NaN check ===
        if config.get("check_nan", {}).get("enabled", False):
            ctx = check_nan_inf(
                dataset=ds,
                variable=variable_id,
                parameter=config["check_nan"].get("parameter", "NaN"),
                severity=self.get_severity(config["check_nan"].get("severity"))
            )
            ctx.description = f"[DATA001] Check for NaN values in variable '{variable_id}'"
            results.append(ctx.to_result())

        # === DATA002: Inf check ===
        if config.get("check_inf", {}).get("enabled", False):
            ctx = check_nan_inf(
                dataset=ds,
                variable=variable_id,
                parameter=config["check_inf"].get("parameter", "Inf"),
                severity=self.get_severity(config["check_inf"].get("severity"))
            )
            ctx.description = f"[DATA002] Check for Inf values in variable '{variable_id}'"
            results.append(ctx.to_result())

        # === DATA003: Fill value check ===
        if config.get("check_fill", {}).get("enabled", False):
            ctx = check_fillvalues_timeseries(
                dataset=ds,
                variable=variable_id,
                parameter=config["check_fill"].get("parameter", "FillValue"),
                severity=self.get_severity(config["check_fill"].get("severity"))
            )
            ctx.description = f"[DATA003] FillValue plausibility for '{variable_id}'"
            results.append(ctx.to_result())

        # === DATA004: Missing value check ===
        if config.get("check_missing", {}).get("enabled", False):
            ctx = check_fillvalues_timeseries(
                dataset=ds,
                variable=variable_id,
                parameter=config["check_missing"].get("parameter", "MissingValue"),
                severity=self.get_severity(config["check_missing"].get("severity"))
            )
            ctx.description = f"[DATA004] MissingValue plausibility for '{variable_id}'"
            results.append(ctx.to_result())

        # === DATA005: Constant value check ===
        if config.get("check_constant", {}).get("enabled", False):
            ctx = check_constants(
                dataset=ds,
                variable=variable_id,
                severity=self.get_severity(config["check_constant"].get("severity"))
            )
            ctx.description = f"[DATA005] Constant field detection for '{variable_id}'"
            results.append(ctx.to_result())

        # === DATA006: Physically impossible outlier check ===
        if config.get("check_physically_impossible_outlier", {}).get("enabled", False):
            ctx = check_outliers(
                dataset=ds,
                thresholds_file='outliers_thresholds.json',
                severity=self.get_severity(config["check_physically_impossible_outlier"].get("severity"))
            )
            ctx.description = f"[DATA006] Physically impossible outlier detection for '{variable_id}'"
            results.append(ctx.to_result())

        # === DATA007: Spatial statistical outlier check Z-Score===
        if config.get("check_spatial_statistical_outliers_Z-Score", {}).get("enabled", False):
            ctx = check_spatial_statistical_outliers(
                dataset=ds,
                variable=variable_id,
                severity=self.get_severity(config["check_spatial_statistical_outliers_Z-Score"].get("severity")),
                threshold=config["check_spatial_statistical_outliers_Z-Score"].get("threshold", 5),
                parameter=config["check_spatial_statistical_outliers_Z-Score"].get("method", "Z-Score")
            )
            ctx.description = f"[DATA007] Spatial statistical outlier ({ctx.description}) check for '{variable_id}'"
            results.append(ctx.to_result())

        # === DATA008: Spatial statistical outlier check IQR===
        if config.get("check_spatial_statistical_outliers_IQR", {}).get("enabled", False):
            ctx = check_spatial_statistical_outliers(
                dataset=ds,
                variable=variable_id,
                severity=self.get_severity(config["check_spatial_statistical_outliers_IQR"].get("severity")),
                threshold=config["check_spatial_statistical_outliers_IQR"].get("threshold", 4.5),
                parameter=config["check_spatial_statistical_outliers_IQR"].get("method", "IQR")
            )
            ctx.description = f"[DATA008] Spatial statistical outlier ({ctx.description}) check for '{variable_id}'"
            results.append(ctx.to_result())


        # === DATA009: Chunk size check ===
        if config.get("check_chunk_size", {}).get("enabled", False):
            ctx = check_chunk_size(
                dataset=ds,
                severity=self.get_severity(config["check_chunk_size"].get("severity"))
            )
            ctx.description = "[DATA009] Chunk size compliance check for 'time' and 'time_bnds'"
            results.append(ctx.to_result())

        return results
