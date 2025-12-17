# Simplified base class for WCRP plugins.

import json
import os
import re
from collections import ChainMap
from hashlib import md5
from pathlib import Path

import cf_xarray  # noqa
import cftime
import numpy as np
import toml
import xarray as xr
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.utils import deltdic, flatten, sanitize

# --- Esgvoc universe import ---
try:
    # from esgvoc.api.universe import find_terms_in_data_descriptor
    import esgvoc.api as ev

    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False


# Define helper functions for serializing across time axis
get_tseconds = lambda t: t.total_seconds()  # noqa
get_tseconds_vector = np.vectorize(get_tseconds)
get_abs_tseconds = lambda t: abs(t.total_seconds())  # noqa
get_abs_tseconds_vector = np.vectorize(get_abs_tseconds)


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

    def __init__(self, options=None):
        super().__init__(options)
        self.config = None
        self.project_config_path = None  # To be set by the specific WCRP plugin class

    def __del__(self):
        xrds = getattr(self, "xrds", None)
        if xrds is not None and hasattr(xrds, "close"):
            xrds.close()

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
        # xarray.Dataset
        self.xrds = xr.open_dataset(
            self.filepath, decode_coords=True, decode_times=False
        )

        # === Options ===
        # Input options
        # - Output for consistency checks across files
        self.consistency_output = self.options.get("consistency_output", False)
        # - If tables are specified, get path to the tables and initialize
        if self.options.get("tables", False):
            tables_path = self.options["tables"]
            self._initialize_CV_info(tables_path)
            self._initialize_time_info()
            self._initialize_coords_info()
            if self.consistency_output and self._cc_spec == "wcrp_base":
                self._write_consistency_output()
        # if only the time checks should be run (so no verification against CV / MIP tables)
        else:
            self.varname = [
                var
                for var in flatten(list(self.xrds.cf.standard_names.values()))
                if var
                not in flatten(
                    list(self.xrds.cf.coordinates.values())
                    + list(self.xrds.cf.axes.values())
                    + list(self.xrds.cf.bounds.values())
                    + list(self.xrds.cf.formula_terms.values())
                )
            ]
            self._initialize_time_info()
            self._initialize_coords_info()
            self.frequency = self._get_attr("frequency")
            if self.varname != []:
                self.cell_methods = self.xrds[self.varname[0]].attrs.get(
                    "cell_methods", "unknown"
                )
            else:
                self.cell_methods = "unknown"
            self.drs_fn = {}
            if self.frequency == "unknown" and self.time is not None:
                if self.time.sizes[self.time.dims[0]] > 1 and 1 == 2:
                    for ifreq in [
                        fkey
                        for fkey in deltdic.keys()
                        if "max" not in fkey and "min" not in fkey
                    ]:
                        try:
                            intv = abs(
                                get_tseconds(
                                    cftime.num2date(
                                        self.time.values[1],
                                        units=self.timeunits,
                                        calendar=self.calendar,
                                    )
                                    - cftime.num2date(
                                        self.time.values[0],
                                        units=self.timeunits,
                                        calendar=self.calendar,
                                    )
                                )
                            )
                            if (
                                intv <= deltdic[ifreq + "max"]
                                and intv >= deltdic[ifreq + "min"]
                            ):
                                self.frequency = ifreq
                                break
                        except (AttributeError, ValueError):
                            continue
                elif self.timebnds and len(self.xrds[self.timebnds].dims) == 2:
                    for ifreq in [
                        fkey
                        for fkey in deltdic.keys()
                        if "max" not in fkey and "min" not in fkey
                    ]:
                        try:
                            intv = abs(
                                get_tseconds(
                                    cftime.num2date(
                                        self.xrds[self.timebnds].values[0, 1],
                                        units=self.timeunits,
                                        calendar=self.calendar,
                                    )
                                    - cftime.num2date(
                                        self.xrds[self.timebnds].values[0, 0],
                                        units=self.timeunits,
                                        calendar=self.calendar,
                                    )
                                )
                            )
                            if (
                                intv <= deltdic[ifreq + "max"]
                                and intv >= deltdic[ifreq + "min"]
                            ):
                                self.frequency = ifreq
                                break
                        except (AttributeError, ValueError):
                            continue
            if self.consistency_output:
                self._write_consistency_output()

    def _load_project_config(self):
        """Loads the project-specific TOML configuration file using self.project_config_path."""
        if not self.project_config_path or not os.path.exists(self.project_config_path):
            print(
                f"Warning: Configuration file path not set or file not found at {self.project_config_path}"
            )
            self.config = {}
            return
        try:
            with open(self.project_config_path, encoding="utf-8") as f:
                self.config = toml.load(f)
        except Exception as e:
            self.config = {}
            print(
                f"Error parsing TOML configuration from {self.project_config_path}: {e}"
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
            self.time = self.xrds.cf["time"]
        except KeyError:
            self.time = None
        if self.time is not None:
            time_attrs = ChainMap(self.time.attrs, self.time.encoding)
            self.calendar = time_attrs.get("calendar", None)
            self.timeunits = time_attrs.get("units", None)
            self.timebnds = time_attrs.get("bounds", None)
            self.time_invariant_vars = [
                var
                for var in list(self.xrds.data_vars.keys())
                + list(self.xrds.coords.keys())
                if self.time.name not in self.xrds[var].dims and var not in self.varname
            ]
        else:
            self.calendar = None
            self.timeunits = None
            self.timebnds = None
            self.time_invariant_vars = [
                var
                for var in list(self.xrds.data_vars.keys())
                + list(self.xrds.coords.keys())
                if var not in self.varname
            ]

    def _initialize_coords_info(self):
        """Get information about the infile coordinates."""
        # Compile list of coordinates from coords, axes and formula_terms
        #  also check for redundant bounds / coordinates
        self.coords = []
        self.bounds = set()
        self.coords_redundant = dict()
        self.bounds_redundant = dict()
        for bkey, bval in self.xrds.cf.bounds.items():
            if len(bval) > 1:
                self.bounds_redundant[bkey] = bval
            self.bounds.update(bval)
        # ds.cf.coordinates
        # {'longitude': ['lon'], 'latitude': ['lat'], 'vertical': ['height'], 'time': ['time']}
        for ckey, clist in self.xrds.cf.coordinates.items():
            _clist = [c for c in clist if c not in self.bounds]
            if len(_clist) > 1:
                self.coords_redundant[ckey] = _clist
            if _clist[0] not in self.coords:
                self.coords.append(_clist[0])
        # ds.cf.axes
        # {'X': ['rlon'], 'Y': ['rlat'], 'Z': ['height'], 'T': ['time']}
        for ckey, clist in self.xrds.cf.axes.items():
            if len(clist) > 1:
                if ckey not in self.coords_redundant:
                    self.coords_redundant[ckey] = clist
            if clist[0] not in self.coords:
                self.coords.append(clist[0])
        # ds.cf.formula_terms
        # {"lev": {"a":"ab", "ps": "ps",...}}
        for akey in self.xrds.cf.formula_terms.keys():
            for ckey, cval in self.xrds.cf.formula_terms[akey].items():
                if cval not in self.coords:
                    self.coords.append(cval)

        # Get the external variables
        self.external_variables = self._get_attr("external_variables", "").split()

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
        """Write output for consistency checks across files."""
        # Dictionaries of global attributes and their data types
        required_attributes = []
        # Read from CV
        if required_attributes == []:
            required_attributes = getattr(self, "CV", {}).get(
                "required_global_attributes", []
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

        file_attrs_req = {
            k: str(v) for k, v in self.xrds.attrs.items() if k in required_attributes
        }
        file_attrs_nreq = {
            k: str(v)
            for k, v in self.xrds.attrs.items()
            if k not in required_attributes
            if k not in ["history"]
        }
        file_attrs_dtypes = {
            k: type(v).__qualname__ for k, v in self.xrds.attrs.items()
        }
        for k in required_attributes:
            if k not in file_attrs_req:
                file_attrs_req[k] = "unset"
            if k not in file_attrs_dtypes:
                file_attrs_dtypes[k] = "unset"
        # Dictionaries of variable attributes and their data types
        var_attrs = {}
        var_attrs_dtypes = {}
        for var in list(self.xrds.data_vars.keys()) + list(self.xrds.coords.keys()):
            var_attrs[var] = {
                key: str(value)
                for key, value in self.xrds[var].attrs.items()
                if key not in ["history"]
            }
            var_attrs_dtypes[var] = {
                key: type(value).__qualname__
                for key, value in self.xrds[var].attrs.items()
                if key not in ["history"]
            }
        # Dictionary of time information
        time_info = {}
        if self.time is not None:
            # Selecting first and last time_bnds value
            #  (ignoring possible flaws in its definition)
            bound0 = None
            boundn = None
            if self.timebnds is not None:
                try:
                    bound0 = self.xrds[self.timebnds].values[0, 0]
                    boundn = self.xrds[self.timebnds].values[-1, -1]
                except IndexError:
                    pass
            time_info = {
                "frequency": self.frequency,
                "units": self.timeunits,
                "calendar": self.calendar,
                "bound0": bound0,
                "boundn": boundn,
                "time0": self.time.values[0],
                "timen": self.time.values[-1],
            }
        # Dictionary of time_invariant variable checksums
        coord_checksums = {}
        for coord_var in self.time_invariant_vars:
            coord_checksums[coord_var] = md5(
                str(self.xrds[coord_var].values.tobytes()).encode("utf-8")
            ).hexdigest()
        # Dictionary of dimension sizes
        dims = dict(self.xrds.sizes)
        # Do not compare time dimension size, only name
        if self.time is not None:
            dimt = self.time.dims[0]
            dims[dimt] = "n"
        # Dictionary of variable data types
        var_dtypes = {}
        for var in list(self.xrds.data_vars.keys()) + list(self.xrds.coords.keys()):
            var_dtypes[var] = str(self.xrds[var].dtype)
        # Write combined dictionary
        with open(self.consistency_output, "w") as f:
            json.dump(
                sanitize(
                    {
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
                ),
                f,
                indent=4,
            )

    def _map_drs_blocks(self):
        """Maps the file metadata, name and location to the DRS building blocks and required attributes."""
        try:
            drs_path_template = re.findall(
                r"<([^<>]*)\>", self.CV["DRS"]["directory_path_template"]
            )
            drs_filename_template = re.findall(
                r"<([^<>]*)\>", self.CV["DRS"]["filename_template"]
            )
            self.drs_suffix = ".".join(
                self.CV["DRS"]["filename_template"].split(".")[1:]
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
