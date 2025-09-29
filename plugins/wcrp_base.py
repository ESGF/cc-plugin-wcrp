
# Simplified base class for WCRP plugins.

import toml
import os
from compliance_checker.base import BaseCheck, Result, TestCtx
from netCDF4 import Dataset

class WCRPBaseCheck(BaseCheck):
    """
    Base class for WCRP project-specific compliance checks.
    Provides common utilities for loading TOML configurations and mapping severities.
    """
    supported_ds = [Dataset]

    SEVERITY_MAP = {
        "HIGH": BaseCheck.HIGH,
        "H":    BaseCheck.HIGH,
        "MEDIUM": BaseCheck.MEDIUM,
        "M":      BaseCheck.MEDIUM,
        "LOW": BaseCheck.LOW,
        "L":      BaseCheck.LOW,
    }

    def __init__(self, options=None):
        super().__init__(options)
        self.config = None
        self.project_config_path = None # To be set by the specific WCRP plugin class

    def _load_project_config(self):
        """Loads the project-specific TOML configuration file using self.project_config_path."""
        if not self.project_config_path or not os.path.exists(self.project_config_path):
            print(f"Warning: Configuration file path not set or file not found at {self.project_config_path}")
            self.config = {}
            return
        try:
            with open(self.project_config_path, 'r', encoding="utf-8") as f:
                self.config = toml.load(f)
        except Exception as e:
            self.config = {}
            print(f"Error parsing TOML configuration from {self.project_config_path}: {e}")

    def get_severity(self, severity_str, default_severity_str="MEDIUM"):
        """Converts a severity string (from TOML) to a BaseCheck constant."""
        default_severity_const = self.SEVERITY_MAP.get(default_severity_str.upper(), BaseCheck.MEDIUM)
        if severity_str is None:
            return default_severity_const
        return self.SEVERITY_MAP.get(str(severity_str).upper(), default_severity_const)
