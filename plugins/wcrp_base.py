# Simplified base class for WCRP plugins.

import json
import os
import re
from hashlib import md5
from pathlib import Path

import numpy as np
import toml
from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util as cfutil
from netCDF4 import Dataset

from checks.utils import infer_frequency, sanitize

# Compliance Checker 6 moved these helpers from compliance_checker.cfutil.
if not hasattr(cfutil, "get_geophysical_variables"):
    from compliance_checker import cfutil


FORMULA_TERM_PATTERN = re.compile(r"(\w+)\s*:\s*([^\s]+)")

# --- Esgvoc universe import ---
try:
    # from esgvoc.api.universe import find_terms_in_data_descriptor
    import esgvoc.api as ev

    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False


class WCRPBaseCheck(BaseCheck):
    """
    Base class for WCRP project-specific compliance checks.
    Provides common utilities for loading TOML configurations and mapping severities.
    """

    # Supported data types
    supported_ds = [Dataset]

    # Severity Mapping
    SEVERITY_MAP = {
        "HIGH": BaseCheck.HIGH,
        "H": BaseCheck.HIGH,
        "MEDIUM": BaseCheck.MEDIUM,
        "M": BaseCheck.MEDIUM,
        "LOW": BaseCheck.LOW,
        "L": BaseCheck.LOW,
    }

    # cc_plugin attributes
    _cc_spec = "wcrp_base"
    _cc_display_headers = {3: "Mandatory", 2: "Warning", 1: "Optional"}
    # A subclass may defer writing until it has initialized additional metadata.
    _defer_consistency_output = False

    def __init__(self, options=None):
        super().__init__(options)
        self.config = None
        self.project_config_path = None  # To be set by the specific WCRP plugin class

    def setup(self, dataset):
        """
        Base checker setup

        Parameters
        ----------
        dataset : netCDF4.Dataset
            An open netCDF4 dataset

        Options
        -------
        consistency_output : bool or str or Path
            Output path for consistency checks
        tables : str or Path
            Path to CV and CMOR tables in case of using CMOR tables instead of ESGVOC
        """
        # === Input Dataset ===
        # netCDF4.Dataset
        self.dataset = dataset
        self.ds = dataset
        # Get path to the dataset
        self.filepath = os.path.realpath(
            os.path.normpath(os.path.expanduser(self.dataset.filepath()))
        )
        # === Options ===
        # Input options
        # - Output for consistency checks across files
        self.consistency_output = self.options.get("consistency_output", False)
        self.consistency_output_error = None
        self.setup_warnings = []
        self.frequency_inferred = False
        self.varname = []
        self.time = None
        self.calendar = None
        self.timeunits = None
        self.timebnds = None
        self.coords = []
        self.bounds = set()
        self.formula_terms = {}
        self.external_variables = []
        self.frequency = "unknown"
        self.cell_methods = "unknown"
        self.drs_fn = {}
        self._run_setup_step(
            "identify the main climate data variable(s)",
            self._initialize_data_variables,
        )
        # - If tables are specified, get path to the tables and initialize
        if self.options.get("tables", False):
            tables_path = self.options["tables"]
            self._run_setup_step(
                "load the requested CV and CMOR tables",
                self._initialize_CV_info,
                tables_path,
            )
            self._run_setup_step(
                "identify the time coordinate",
                self._initialize_time_info,
            )
            self._run_setup_step(
                "identify coordinate and bounds variables",
                self._initialize_coords_info,
            )
        # if only the time checks should be run (so no verification against CV / MIP tables)
        else:
            self._run_setup_step(
                "identify the time coordinate",
                self._initialize_time_info,
            )
            self._run_setup_step(
                "identify coordinate and bounds variables",
                self._initialize_coords_info,
            )
            self.frequency = self._get_attr("frequency", None)
            if self.varname != []:
                self.cell_methods = getattr(
                    self.dataset.variables[self.varname[0]],
                    "cell_methods",
                    "unknown",
                )
            else:
                self.cell_methods = "unknown"
            self.drs_fn = {}
            if self.frequency is None:
                time_bounds = self.dataset.variables.get(self.timebnds)
                self.frequency = infer_frequency(
                    self.time,
                    time_bounds=time_bounds,
                    units=self.timeunits,
                    calendar=self.calendar,
                )
                self.frequency_inferred = self.frequency != "unknown"
        if self.consistency_output and not self._defer_consistency_output:
            self._write_consistency_output()

    def _record_setup_warning(self, message, exc=None):
        """Record a recoverable setup problem without aborting the checker."""
        detail = f"{message}: {exc}" if exc is not None else message
        if detail not in self.setup_warnings:
            self.setup_warnings.append(detail)

    def _record_consistency_output_error(self, message, exc=None):
        """Record a unique problem affecting the cross-file summary."""
        detail = f"{message}: {exc}" if exc is not None else message
        if self.consistency_output_error is None:
            self.consistency_output_error = detail
            return
        recorded = self.consistency_output_error.split("; ")
        if detail not in recorded:
            self.consistency_output_error += f"; {detail}"

    def _run_setup_step(self, description, function, *args, **kwargs):
        """Run one initialization step without allowing it to abort setup."""
        try:
            function(*args, **kwargs)
        except Exception as exc:
            self._record_setup_warning(f"Could not {description}", exc)
            return False
        return True

    def _cf_names(self, function, label):
        """Call a Compliance Checker CF discovery function defensively."""
        try:
            return list(function(self.dataset) or [])
        except Exception as exc:
            self._record_setup_warning(f"Could not identify {label}", exc)
            return []

    def _initialize_data_variables(self):
        """Identify geophysical variables without requiring standard_name."""
        self.varname = self._cf_names(
            cfutil.get_geophysical_variables,
            "geophysical variables",
        )

        # Project metadata can disambiguate files containing cell measures or
        # other geophysical-looking support variables. It is only a preference:
        # CF discovery remains the source of the candidate set.
        preferred = []
        for attr in ("variable_id", "branded_variable"):
            value = self._get_attr(attr, "")
            if isinstance(value, str) and value in self.varname:
                preferred.append(value)
        self.varname = preferred + [v for v in self.varname if v not in preferred]

    def _load_project_config(self):
        """Loads the project-specific TOML configuration file using self.project_config_path."""
        if not self.project_config_path or not os.path.exists(self.project_config_path):
            self._record_setup_warning(
                f"Project configuration file not found at '{self.project_config_path}'"
            )
            self.config = {}
            return
        try:
            with open(self.project_config_path, encoding="utf-8") as f:
                self.config = toml.load(f)
        except Exception as e:
            self.config = {}
            self._record_setup_warning(
                f"Could not parse project configuration file "
                f"'{self.project_config_path}'",
                e,
            )

    def get_severity(self, severity_str, default_severity_str="MEDIUM"):
        """Converts a severity string (from TOML) to a BaseCheck constant."""
        default_severity_const = self.SEVERITY_MAP.get(
            default_severity_str.upper(), BaseCheck.MEDIUM
        )
        if severity_str is None:
            return default_severity_const
        return self.SEVERITY_MAP.get(str(severity_str).upper(), default_severity_const)

    def _initialize_CV_info(self, tables_path):
        """Find and read CV and CMOR tables and extract basic information."""
        # Identify table prefix and table names
        tables_path = os.path.normpath(
            os.path.realpath(os.path.expanduser(tables_path))
        )
        tables = [
            t
            for t in os.listdir(tables_path)
            if os.path.isfile(os.path.join(tables_path, t))
            and t.endswith(".json")
            and "example" not in t
        ]
        table_prefix = tables[0].split("_")[0]
        table_names = ["_".join(t.split("_")[1:]).split(".")[0] for t in tables]
        if not all([table_prefix + "_" + t + ".json" in tables for t in table_names]):
            raise ValueError(
                "CMOR tables do not follow the naming convention '<project_id>_<table_id>.json'."
            )
        # Read CV and coordinate tables
        self.CV = self._read_CV(tables_path, table_prefix, "CV")["CV"]
        self.CTcoords = self._read_CV(tables_path, table_prefix, "coordinate")
        self.CTgrids = self._read_CV(tables_path, table_prefix, "grids")
        self.CTformulas = self._read_CV(tables_path, table_prefix, "formula_terms")
        # Read variable tables (variable tables)
        self.CT = {}
        for table in table_names:
            if table in ["CV", "grids", "coordinate", "formula_terms"]:
                continue
            self.CT[table] = self._read_CV(tables_path, table_prefix, table)
            if "variable_entry" not in self.CT[table]:
                raise KeyError(
                    f"CMOR table '{table}' does not contain the key 'variable_entry'."
                )
            if "Header" not in self.CT[table]:
                raise KeyError(
                    f"CMOR table '{table}' does not contain the key 'Header'."
                )
            for key in ["table_id"]:
                if key not in self.CT[table]["Header"]:
                    raise KeyError(
                        f"CMOR table '{table}' misses the key '{key}' in the header information."
                    )
        # Compile varlist for quick reference
        varlist = list()
        for table in table_names:
            if table in ["CV", "grids", "coordinate", "formula_terms"]:
                continue
            varlist = varlist + [
                v["out_name"] for v in self.CT[table]["variable_entry"].values()
            ]
        varlist = set(varlist)
        # Map DRS building blocks to the filename, filepath and global attributes
        self._map_drs_blocks()
        # Identify variable name(s)
        var_ids = [v for v in varlist if v in list(self.dataset.variables.keys())]
        self.varname = var_ids
        # Identify table_id, requested frequency and cell_methods
        self.table_id_raw = self._get_attr("table_id")
        if self.table_id_raw in self.CT:
            self.table_id = self.table_id_raw
        else:
            self.table_id = "unknown"
        self.frequency = self._get_var_attr("frequency", False)
        if not self.frequency:
            self.frequency = self._get_attr("frequency")
        # In case of unset table_id -
        #  in some projects (eg. CORDEX), the table_id is not required,
        #  since there is one table per frequency, so table_id = frequency.
        if self.table_id == "unknown":
            possible_ids = list()
            if len(self.varname) > 0:
                for table in table_names:
                    if table in ["CV", "grids", "coordinate", "formula_terms"]:
                        continue
                    if (
                        self.varname[0] in self.CT[table]["variable_entry"]
                        and self.frequency
                        == self.CT[table]["variable_entry"][self.varname[0]][
                            "frequency"
                        ]
                    ):
                        possible_ids.append(table)
            if len(possible_ids) == 0:
                possible_ids = [key for key in self.CT.keys() if self.frequency in key]
            if len(possible_ids) == 1:
                self.table_id = possible_ids[0]

        self.cell_methods = self._get_var_attr("cell_methods", "unknown")
        # Get missing_value
        if self.table_id == "unknown":
            self.missing_value = None
        else:
            self.missing_value = float(
                self.CT[self.table_id]["Header"]["missing_value"]
            )

    def _initialize_time_info(self):
        """Get information about the infile time axis."""
        try:
            time_name = cfutil.get_time_variable(self.dataset)
        except Exception as exc:
            self._record_setup_warning("Could not identify the time coordinate", exc)
            time_name = None

        if time_name in self.dataset.variables:
            self.time = self.dataset.variables[time_name]
        else:
            self.time = None

        if self.time is not None:
            self.calendar = getattr(self.time, "calendar", None)
            self.timeunits = getattr(self.time, "units", None)
            self.timebnds = getattr(self.time, "bounds", None)
        else:
            self.calendar = None
            self.timeunits = None
            self.timebnds = None

    def _get_time_invariant_vars(self):
        """Return variables whose values are compared across files."""
        time_dim = None
        if self.time is not None and self.time.dimensions:
            time_dim = self.time.dimensions[0]
        return [
            var
            for var, ncvar in self.dataset.variables.items()
            if time_dim not in ncvar.dimensions and var not in self.varname
        ]

    def _initialize_coords_info(self):
        """Get information about the infile coordinates."""
        self.coords_redundant = {}
        self.bounds_redundant = {}

        try:
            bounds_map = cfutil.get_cell_boundary_map(self.dataset)
        except Exception as exc:
            self._record_setup_warning("Could not identify coordinate bounds", exc)
            bounds_map = {}
        self.bounds = set(bounds_map.values())

        coordinate_groups = {
            "longitude": self._cf_names(
                cfutil.get_longitude_variables,
                "longitude coordinates",
            ),
            "latitude": self._cf_names(
                cfutil.get_latitude_variables,
                "latitude coordinates",
            ),
            "vertical": self._cf_names(
                cfutil.get_z_variables,
                "vertical coordinates",
            ),
            "time": self._cf_names(
                cfutil.get_time_variables,
                "time coordinates",
            ),
        }
        for key, candidates in coordinate_groups.items():
            candidates = list(dict.fromkeys(candidates))
            if len(candidates) > 1:
                self.coords_redundant[key] = candidates
            group_bounds = list(
                dict.fromkeys(
                    bounds_map[name] for name in candidates if name in bounds_map
                )
            )
            if len(group_bounds) > 1:
                self.bounds_redundant[key] = group_bounds

        coord_names = set(
            self._cf_names(
                cfutil.get_coordinate_variables,
                "coordinate variables",
            )
        )
        coord_names.update(
            self._cf_names(
                cfutil.get_auxiliary_coordinate_variables,
                "auxiliary coordinate variables",
            )
        )
        coord_names.update(self._cf_names(cfutil.get_axis_variables, "axis variables"))
        for candidates in coordinate_groups.values():
            coord_names.update(candidates)

        self.formula_terms = {}
        for name, var in self.dataset.variables.items():
            value = getattr(var, "formula_terms", None)
            if not isinstance(value, str):
                continue
            terms = {
                match.group(1): match.group(2)
                for match in FORMULA_TERM_PATTERN.finditer(value)
                if match.group(2) in self.dataset.variables
            }
            if terms:
                self.formula_terms[name] = terms
                coord_names.update(terms.values())

        coord_names.difference_update(self.bounds)
        self.coords = [name for name in self.dataset.variables if name in coord_names]

        # Get the external variables
        external_variables = self._get_attr("external_variables", "")
        self.external_variables = (
            external_variables.split() if isinstance(external_variables, str) else []
        )

        # Update list of variables
        self.varname = [
            v for v in self.varname if v not in self.coords and v not in self.bounds
        ]

    def _get_attr(self, attr, default="unknown"):
        """Get nc attribute."""
        try:
            return self.dataset.getncattr(attr)
        except AttributeError:
            return default

    def _get_var_attr(self, attr, default="unknown"):
        """Get CMOR table variable entry attribute."""
        if self.table_id != "unknown":
            if len(self.varname) > 0:
                try:
                    return self.CT[self.table_id]["variable_entry"][self.varname[0]][
                        attr
                    ]
                except KeyError:
                    return default
        return default

    def _read_CV(self, path, table_prefix, table_name):
        """Reads the specified CV table."""
        table_path = Path(path, f"{table_prefix}_{table_name}.json")
        try:
            with open(table_path) as f:
                return json.load(f)
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
            raise Exception(
                f"Could not find or open table '{table_prefix}_{table_name}.json' under path '{path}'."
            ) from e

    def _write_consistency_output(self):
        """Write consistency output without allowing failures to abort setup."""
        try:
            payload = self._build_consistency_output()
        except Exception as exc:
            self._record_consistency_output_error(
                "Could not build the requested consistency summary",
                exc,
            )
            return False

        try:
            serialized = json.dumps(sanitize(payload), indent=4)
        except Exception as exc:
            self._record_consistency_output_error(
                "Could not serialize the requested consistency summary as JSON",
                exc,
            )
            return False

        output_path = Path(self.consistency_output).expanduser()
        try:
            output_path.write_text(serialized, encoding="utf-8")
        except Exception as exc:
            self._record_consistency_output_error(
                f"Could not write the requested consistency summary to "
                f"'{output_path}'",
                exc,
            )
            return False
        return True

    def check_setup_warnings(self, ds):
        """Report recoverable setup problems as low-severity findings."""
        testctx = TestCtx(
            BaseCheck.LOW,
            "[TOOL001] Plugin initialization and file metadata discovery",
        )
        if self.setup_warnings:
            for warning in self.setup_warnings:
                testctx.add_failure(
                    "The plugin encountered a problem while examining the file "
                    "structure and metadata. Some compliance checks may have "
                    f"been skipped or may be incomplete. Technical details: {warning}"
                )
        else:
            testctx.add_pass()
        return [testctx.to_result()]

    def check_consistency_output(self, ds):
        """Report a recoverable consistency-output error as low severity."""
        testctx = TestCtx(BaseCheck.LOW, "[TOOL002] Cross-file consistency summary")
        if self.consistency_output_error:
            testctx.add_failure(
                "The plugin could not fully prepare or write the JSON summary "
                "used to compare metadata across files. The consistency summary "
                "may therefore be missing or incomplete. Technical details: "
                f"{self.consistency_output_error}"
            )
        else:
            testctx.add_pass()
        return [testctx.to_result()]

    def _build_consistency_output(self):
        """Build the cross-file consistency payload from the NetCDF4 dataset."""
        # Dictionaries of global attributes and their data types
        required_attributes = []
        # Read from CV
        if required_attributes == []:
            required_attributes = list(
                getattr(self, "CV", {}).get("required_global_attributes", [])
            )
        # required_attributes = []
        # Retrieve via esgvoc
        if required_attributes == [] and ESG_VOCAB_AVAILABLE:
            print("Retrieving required attributes from ESGVOC")
            eproj = ev.get_project(self.project_name)
            if eproj:
                for eatt in eproj.attr_specs:
                    if eatt.is_required:
                        if eatt.field_name:
                            required_attributes.append(eatt.field_name)
                        else:
                            required_attributes.append(eatt.source_collection)
        required_attributes.sort(key=lambda x: x.lower())
        # print("Required attributes:", required_attributes)

        global_attrs = {
            name: self.dataset.getncattr(name) for name in self.dataset.ncattrs()
        }
        file_attrs_req = {
            k: str(v) for k, v in global_attrs.items() if k in required_attributes
        }
        file_attrs_nreq = {
            k: str(v)
            for k, v in global_attrs.items()
            if k not in required_attributes
            if k not in ["history"]
        }
        file_attrs_dtypes = {k: type(v).__qualname__ for k, v in global_attrs.items()}
        for k in required_attributes:
            if k not in file_attrs_req:
                file_attrs_req[k] = "unset"
            if k not in file_attrs_dtypes:
                file_attrs_dtypes[k] = "unset"
        # Dictionaries of variable attributes and their data types
        var_attrs = {}
        var_attrs_dtypes = {}
        for var_name, var in self.dataset.variables.items():
            attrs = {name: var.getncattr(name) for name in var.ncattrs()}
            var_attrs[var_name] = {
                key: str(value)
                for key, value in attrs.items()
                if key not in ["history"]
            }
            var_attrs_dtypes[var_name] = {
                key: type(value).__qualname__
                for key, value in attrs.items()
                if key not in ["history"]
            }
        # Dictionary of time information
        time_info = {}
        if self.time is not None:
            # Selecting first and last time_bnds value
            #  (ignoring possible flaws in its definition)
            bound0 = None
            boundn = None
            if self.timebnds in self.dataset.variables:
                try:
                    bounds_var = self.dataset.variables[self.timebnds]
                    bound0 = bounds_var[0, 0]
                    boundn = bounds_var[-1, -1]
                except (IndexError, TypeError):
                    pass
            if self.time.size:
                time_info = {
                    "frequency": self.frequency,
                    "units": self.timeunits,
                    "calendar": self.calendar,
                    "bound0": bound0,
                    "boundn": boundn,
                    "time0": self.time[0],
                    "timen": self.time[-1],
                }
        # Dictionary of time_invariant variable checksums
        coord_checksums = {}
        for coord_var in self._get_time_invariant_vars():
            values = np.ma.asarray(self.dataset.variables[coord_var][:])
            coord_checksums[coord_var] = md5(
                str(values.tobytes()).encode("utf-8")
            ).hexdigest()
        # Dictionary of dimension sizes
        dims = {name: len(dim) for name, dim in self.dataset.dimensions.items()}
        # Do not compare time dimension size, only name
        if self.time is not None and self.time.dimensions:
            dimt = self.time.dimensions[0]
            dims[dimt] = "n"
        # Dictionary of variable data types
        var_dtypes = {
            name: str(var.dtype) for name, var in self.dataset.variables.items()
        }
        return {
            "global_attributes": file_attrs_req,
            "global_attributes_non_required": file_attrs_nreq,
            "global_attributes_dtypes": file_attrs_dtypes,
            "variable_attributes": var_attrs,
            "variable_attributes_dtypes": var_attrs_dtypes,
            "variable_dtypes": var_dtypes,
            "dimensions": dims,
            "coordinates": coord_checksums,
            "time_info": time_info,
        }

    def _map_drs_blocks(self):
        """Maps the file metadata, name and location to the DRS building blocks and required attributes."""
        try:
            drs_path_template = re.findall(
                r"<([^<>]*)\>", self.CV["DRS"]["directory_path_template"]
            )
            drs_filename_template = re.findall(
                r"<([^<>]*)\>", self.CV["DRS"]["filename_template"]
            )
            if "time_range" not in drs_filename_template:
                drs_filename_template.append("time_range")
            self.drs_suffix = (
                ".".join(self.CV["DRS"]["filename_template"].split(".")[1:]) or "nc"
            )
        except KeyError:
            raise KeyError("The CV does not contain DRS information.")

        # Map DRS path elements
        self.drs_dir = {}
        fps = os.path.dirname(self.filepath).split(os.sep)
        for i in range(-1, -len(drs_path_template) - 1, -1):
            try:
                self.drs_dir[drs_path_template[i]] = fps[i]
            except IndexError:
                self.drs_dir[drs_path_template[i]] = False

        # Map DRS filename elements
        self.drs_fn = {}
        fns = os.path.basename(self.filepath).split(".")[0].split("_")
        for i in range(len(drs_filename_template)):
            try:
                self.drs_fn[drs_filename_template[i]] = fns[i]
            except IndexError:
                self.drs_fn[drs_filename_template[i]] = False

        # Map DRS global attributes
        self.drs_gatts = {}
        for gatt in self.CV["required_global_attributes"]:
            if gatt in drs_path_template or gatt in drs_filename_template:
                try:
                    self.drs_gatts[gatt] = self.dataset.getncattr(gatt)
                except AttributeError:
                    self.drs_gatts[gatt] = False
