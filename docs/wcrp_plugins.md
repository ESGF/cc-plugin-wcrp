# The `WCRP` Plugins

The **WCRP plugins** provide project-specific compliance checks for WCRP datasets such as **CMIP6** and **CORDEX-CMIP6**, built on top of the [IOOS Compliance Checker](https://ioos.github.io/compliance-checker/).

Each plugin has its own orchestration layer (`wcrp_project.py`), which acts as the **controller** for WCRP validation logic — deciding:
- Which checks to run  
- With what severity  
- According to each project’s configuration  

---

## Available Plugins

| Plugin Name | Checker Tag | Description |
|--------------|--------------|-------------|
| **wcrp_cmip6** | `wcrp_cmip6:1.0` | Validation of CMIP6 NetCDF files |
| **wcrp_cordex_cmip6** | `wcrp_cordex_cmip6:1.0` | Validation of CORDEX-CMIP6 NetCDF files |

Each plugin has its own **TOML configuration file** defining specific rules, severities, and variable mappings.

---

## What Each Plugin Does

Each WCRP plugin:

1. **Loads its configuration file** (`wcrp_cmip6.toml` or `wcrp_cordex_cmip6.toml`), which defines:
   - Which checks to run (DRS, attributes, time, metadata, etc.)
   - The severity of each check (High = Mandatory, Medium = Recommended, Low = Warning)

2. **Runs the atomic checks** implemented in `checks/…`:
   - Examples: dimension existence/size, variable shape, time bounds, filename/DRS consistency, physical plausibility, missing values, etc.

3. **Aggregates results** and returns them to the Compliance Checker core,  
   which formats them into human-readable or machine-readable outputs (text, HTML, JSON).

---

## TOML Configuration Files

Each plugin is driven by its own TOML file:

- **`wcrp_cmip6.toml`**  
  Main configuration for CMIP6 checks, mapping variables, severities, and project rules.

- **`wcrp_cordex_cmip6.toml`**  
  Adapted version for CORDEX-CMIP6)context.

- **`mapping_variables.toml`**  
  Helper file mapping `<table_id>.<variable_id>` to standard variable definitions  
  (dimensions, expected attributes, cell methods, etc.).  
  This is a **temporary bridge** until the official ESGVOC vocabulary exposes all required metadata.

---