
#!/usr/bin/env python

# =============================================================================
# WCRP CMIP6 project
#
# This module defines a WCRP example compliance checker, which serves as 
# the main entry point for executing a series of validation checks on 
# climate data. It relies on configuration defined in a TOML file and 
# calls a suite of checks developed for quality control

# =============================================================================
from netCDF4 import Dataset
import os
import toml
from compliance_checker.base import BaseCheck, Result, TestCtx
from plugins.wcrp_base import WCRPBaseCheck
from checks.consistency_checks.check_experiment_consistency import *
from checks.variable_checks.check_variable_existence import check_variable_existence
from checks.variable_checks.check_variable_shape_vs_dimensions import check_variable_shape
from checks.variable_checks.check_bounds_value_consistency import check_bounds_value_consistency
from checks.consistency_checks.check_drs_filename_cv import *
from checks.consistency_checks.check_institution_source_consistency import *
from checks.attribute_checks.check_attribute_suite import check_attribute_suite
from checks.dimension_checks.check_dimension_positive import check_dimension_positive
from checks.dimension_checks.check_dimension_existence import check_dimension_existence
from checks.dimension_checks.check_dimension_size import *
from checks.consistency_checks.check_variant_label_consistency import check_variant_label_consistency
from checks.consistency_checks.check_frequency_table_consistency import check_frequency_table_id_consistency
from checks.consistency_checks.check_drs_consistency import check_attributes_match_directory_structure, check_filename_matches_directory_structure
from checks.consistency_checks.check_attributes_match_filename import check_filename_vs_global_attrs, _parse_filename_components
from checks.time_checks.check_time_bounds import check_time_bounds
from checks.time_checks.check_time_range_vs_filename import *

# --- Esgvoc universe import---
try:
    from esgvoc.api.universe import find_terms_in_data_descriptor
    ESG_VOCAB_AVAILABLE = True
except ImportError:
    ESG_VOCAB_AVAILABLE = False



class Cmip6ProjectCheck(WCRPBaseCheck):
    
    _cc_spec = "plugin_cmip6"
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
            print(f"Warning: Configuration file path not set or file not found at {self.project_config_path}")
            self.config = {}
            return
        try:
            with open(self.project_config_path, 'r', encoding="utf-8") as f:
                self.config = toml.load(f)
        except Exception as e:
            self.config = {}
            print(f"Error parsing TOML configuration from {self.project_config_path}: {e}")

    def setup(self, ds):
        
        """Loads the main configuration and the variable mapping file before running checks."""
        super().setup(ds)
        self._load_project_config()

        # Load variable mapping directly from 'mapping_variables.toml' located in the same folder as the config
        base_dir = os.path.dirname(self.project_config_path)
        mapping_filepath = os.path.join(base_dir, "mapping_variables.toml")

        
        try:
            with open(mapping_filepath, 'r') as f:
                self.variable_mapping = toml.load(f).get('mapping_variables', {})
                
        except FileNotFoundError:
            print(f"Mapping file '{mapping_filepath}' not found.")
            self.variable_mapping = {}
        except Exception as e:
            print(f"Error while loading variable mapping: {e}")
            self.variable_mapping = {}
    
    def check_Drs_Vocabulary(self, ds):
        
        """
        Runs the DRS filename and directory path checks separately against CV pattern
        """
        results = []
        if "drs_checks" not in self.config:
            return results
        
        config = self.config["drs_checks"]
        severity = self.get_severity(config.get("severity"))
        project_id = "cmip6"

        # Call filename CV check
        results.extend(check_drs_filename(
            ds=ds,
            severity=severity,
            project_id=project_id
        ))

        # Call Drs CV check
        results.extend(check_drs_directory(
            ds=ds,
            severity=severity,
            project_id=project_id
        ))
        
        return results
  

    def check_Global_Variable_Attributes(self, ds):
       
        #Orchestrates checks for both global and variable attributes from the TOML config#
   
        results = []
        if not self.config:
            return results

        global_attrs_config = self.config.get('global_attributes', {})
        for attr_name, attr_config in global_attrs_config.items():
            results.extend(check_attribute_suite(
                ds=ds,
                attribute_name=attr_name,
                severity=self.get_severity(attr_config.get('severity')),
                expected_type=attr_config.get('expected_type'),
                constraint=attr_config.get('constraint'),
                var_name=None,
                project_name=self.project_name  
            ))

        variable_attrs_config = self.config.get('variable_attributes', {})
        for var_name, attributes_to_check in variable_attrs_config.items():
            for attr_name, attr_config in attributes_to_check.items():
                results.extend(check_attribute_suite(
                    ds=ds,
                    attribute_name=attr_name,
                    severity=self.get_severity(attr_config.get('severity')),
                    expected_type=attr_config.get('expected_type'),
                    constraint=attr_config.get('constraint'),
                    var_name=var_name,
                    project_name=self.project_name  
                ))

        return results



    def check_Variable_Registry(self, ds):

        results = []
        if "variable_registry_checks" not in self.config:
            return results

        if not ESG_VOCAB_AVAILABLE:
            ctx = TestCtx(BaseCheck.HIGH, "Variable Registry Discovery")
            ctx.add_failure("The 'esgvoc' library is required but not installed.")
            return [ctx.to_result()]

        try:
            variable_id = ds.getncattr("variable_id")
            table_id = ds.getncattr("table_id")
        except AttributeError as e:
            ctx = TestCtx(BaseCheck.HIGH, "Variable Registry Discovery")
            ctx.add_failure(f"Missing required global attribute: {e}.")
            return [ctx.to_result()]

        mapping_key = f"{table_id}.{variable_id}"
        
        known_branded_variable = self.variable_mapping.get(mapping_key)
        
        if not known_branded_variable:
            ctx = TestCtx(BaseCheck.HIGH, "Variable Registry Discovery")
            ctx.add_failure(f"No mapping found for '{mapping_key}'.")
            return [ctx.to_result()]

        fields_to_get = [
            "cf_standard_name",
            "cf_units",
            "dimensions",
            "cell_methods",
            "cell_measures",
            "description"
        ]

        terms = find_terms_in_data_descriptor(
            expression=known_branded_variable,
            data_descriptor_id="known_branded_variable",
            only_id=True,
            selected_term_fields=fields_to_get
        )

        if not terms:
            ctx = TestCtx(BaseCheck.HIGH, "Variable Registry Discovery")
            ctx.add_failure(f"Could not retrieve vocabulary details for '{known_branded_variable}'.")
            return [ctx.to_result()]

        expected = terms[0]

        # === Step 2: Launch checks using discovered info ===
        expected_dims = getattr(expected, 'dimensions', [])
       

        if not isinstance(expected_dims, list) or not all(isinstance(d, str) for d in expected_dims):
            print(f" Unexpected format for expected_dims: {expected_dims}")
            expected_dims = []

        if expected_dims:
            
            actual_dims = set(ds.dimensions.keys())
            actual_vars = set(ds.variables.keys())

            for expected_dim in sorted(expected_dims):
                matched_dim = None

                # Try to find a matching real dimension
                for actual_dim in actual_dims:
                    if actual_dim in expected_dim or expected_dim in actual_dim:
                        matched_dim = actual_dim
                        break

                if matched_dim:
                    results.extend(check_dimension_existence(
                        ds=ds,
                        dimension_name=matched_dim,
                        severity=self.get_severity("H")
                    ))
                    results.extend(check_dimension_positive(
                        ds=ds,
                        dimension_name=matched_dim,
                        severity=self.get_severity("H")
                    )) 
                    results.extend(check_variable_existence(
                        ds=ds,
                        var_name=matched_dim,
                        severity=self.get_severity("H")
                    ))
                        
                else:
                    # Special case: expected 'height2m' but actual variable is 'height'
                    if expected_dim.lower().startswith("height") and "height" in actual_vars:
                        
                        results.extend(check_variable_existence(
                            ds=ds,
                            var_name='height',
                            severity=self.get_severity("H")
                        ))
                    else:
                        # Final fallback: try the expected_dim as-is
                        print(f"WARNING: No match found for '{expected_dim}', trying as-is")
                        results.extend(check_dimension_existence(
                            ds=ds,
                            dimension_name=expected_dim,
                            severity=self.get_severity("H")
                        ))
       # === Step 3: Check geophysical variable existence ===
            results.extend(check_variable_existence(
                ds=ds,
                var_name=variable_id,
                severity=self.get_severity("H")
            ))
           

            # === Step 4: Check variable attributes ===
            attribute_mapping = {
                "cf_standard_name": "standard_name",
                "cf_units": "units",
                "cell_methods": "cell_methods",
                "cell_measures": "cell_measures",
                "description": "description"
            }
            for esg_key, netcdf_attr_name in attribute_mapping.items():
                expected_val = getattr(expected, esg_key, None)
                if expected_val:
                    
                    results.extend(check_attribute_suite(
                        ds=ds,
                        attribute_name=netcdf_attr_name,
                        severity=self.get_severity("H"),
                        expected_type="str",
                        constraint=expected_val,  
                        var_name=variable_id,
                        project_name=self.project_name
                    ))
           
            
            
                
        # === Step 5 : Check remaining dimensions and variables that were not expected ===
        "Check remaining dimensions that were not expected (bnds/vertices/axis_nbounds..."
        already_checked_dims = set(expected_dims)
        all_dims = set(ds.dimensions.keys())
        remaining_dims = all_dims - already_checked_dims
        already_checked_vars  = set(expected_dims + [variable_id])
        remaining_vars = set(ds.variables.keys()) - already_checked_vars
        for dim in sorted(remaining_dims):
            results.extend(check_dimension_existence(
                ds=ds,
                dimension_name=dim,
                severity=self.get_severity("H")  
            ))

        for var in sorted(remaining_vars):
            results.extend(check_variable_existence(
                ds=ds,
                var_name=var,
                severity=self.get_severity("H")  
            ))
    # === Step 6: Shape verification for variables ===
        
        all_expected_vars = expected_dims + [variable_id]
        all_vars_checked = set()

        # Universe Variable checks 
        for var_name in sorted(set(all_expected_vars)):
            if var_name in ds.variables:
                
                results.extend(check_variable_shape(var_name, ds, severity=self.get_severity("H")))
                results.extend(check_bounds_value_consistency(ds, var_name, severity=self.get_severity("H")))
                all_vars_checked.add(var_name)

        # Remaining Variables size checks
        for var_name in sorted(remaining_vars):
            if var_name not in all_vars_checked:
                results.extend(check_variable_shape(var_name, ds, severity=self.get_severity("H")))
                results.extend(check_bounds_value_consistency(ds, var_name, severity=self.get_severity("H")))
                all_vars_checked.add(var_name)
        
    # === Step 7 : Size Checks for bounds and vertices ====
            
        for dim in remaining_dims:
            if dim in ["bnds","axis_nbounds"]:
                results.extend(check_dimension_size_is_equals_to(ds, dimension_name=dim, expected_size=2, severity=self.get_severity("H")))
        
        for dim in remaining_dims:
            if dim in ["vertices","nv4"]:
                results.extend(check_dimension_size_is_equals_to(ds, dimension_name=dim, expected_size=4, severity=self.get_severity("H")))
    
    # === Step 8 : Time Checks if time exists ====
        for var in already_checked_vars:
            if var in ["time"]:
                results.extend(check_time_range_vs_filename(ds, severity=self.get_severity("H"))) 
                results.extend(check_time_bounds(ds,severity=self.get_severity("H")))
                  
        return results


    def check_Drs_Consistency(self, ds):
        
        """
        Runs the DRS consistency checks with Filename and Global Attributes
        """
        results = []
        if "drs_consistency_checks" not in self.config:
            return results
        
        config = self.config["drs_consistency_checks"]
        severity = self.get_severity(config.get("severity"))
        project_id = "CMIP6"  

        # Call check PATH001
        results.extend(check_attributes_match_directory_structure(
            ds=ds,
            severity=severity,
            project_id=project_id
        ))

        # Call check PATH002
        results.extend(check_filename_matches_directory_structure(
            ds=ds,
            severity=severity,
            project_id=project_id
        ))
        
        return results
    
    def check_consistency_filename_from_config(self, ds):
        
        """
        Runs Filename/Global_Attributes consistency checks defined in the TOML.
        """
        results = []
        if "consistency_checks" not in self.config:
            return results
        
        config = self.config.get('consistency_checks', {})

        
        if 'filename_vs_attributes' in config:
            check_config = config['filename_vs_attributes']
            results.extend(check_filename_vs_global_attrs(
                ds=ds,
                severity=self.get_severity(check_config.get('severity'))
            ))

        return results
    
    def check_frequency_consistency_from_config(self, ds):
        """
        Runs the frequency vs table_id consistency check.
        """
        results = []
        
        if "freq_table_id_consistency_checks" not in self.config:
            return results
        
        check_config = self.config['freq_table_id_consistency_checks']
        severity = self.get_severity(check_config.get('severity'))
        
        # Retrieves mapping table from config toml
        mapping = self.config.get('frequency_table_id_mapping', {})
        if not mapping:
        
            return results

        # Calls the atomic check by passing it the mapping
        results.extend(check_frequency_table_id_consistency(
            ds=ds,
            mapping=mapping,
            severity=severity
        ))
        
        return results
    
    

    def check_experiment_consistency(self, ds):
        
        """
        Runs experiment consistency checks defined in the TOML.
        """
        results = []
        if "consistency_checks" not in self.config:
            return results
        
        config = self.config.get('consistency_checks', {})
        project_id = "cmip6"

        

        # --- Call for experiment_id consistency check ---
        if 'experiment_details' in config:
            check_config = config['experiment_details']
            results.extend(check_experiment_consistency(
                ds=ds,
                severity=self.get_severity(check_config.get('severity')),
                project_id=project_id
            ))

        return results
    
    def check_variantlabel_consistency(self, ds):
        
        """
        Runs variant_label consistency checks defined in the TOML.
        """
        results = []
        if "consistency_checks" not in self.config:
            return results

        config = self.config.get('consistency_checks', {})

        # --- Call for variant_label consistency check ---
        if 'variant_label' in config:
            check_config = config['variant_label']
            results.extend(check_variant_label_consistency(
                ds=ds,
                severity=self.get_severity(check_config.get('severity'))
            ))
            
        return results
    
    def check_consistency_institution_source(self, ds):
        """
        Runs institution/source consistency checks defined in the TOML.
        """
        
        results = []
        if "consistency_checks" not in self.config:
            return results
        
        config = self.config.get('consistency_checks', {})
        project_id = "cmip6"
        # --- Call for institution coherence check ---
        if 'institution_details' in config:
            check_config = config['institution_details']
            results.extend(check_institution_consistency(
                ds=ds,
                severity=self.get_severity(check_config.get('severity')),
                project_id=project_id
            ))

        # --- Call for source coherence check ---
        if 'source_details' in config:
            check_config = config['source_details']
            results.extend(check_source_consistency(
                ds=ds,
                severity=self.get_severity(check_config.get('severity')),
                project_id=project_id
            ))

        return results
