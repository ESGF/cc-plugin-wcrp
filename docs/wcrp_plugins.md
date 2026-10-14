# WCRP Plugins

The **WCRP plugins** provide project-specific compliance checks for WCRP datasets such as **CMIP6** and **CORDEX-CMIP6**, built on top of the [IOOS Compliance Checker](https://ioos.github.io/compliance-checker/).

Each plugin has its own orchestration layer (`wcrp_project.py`), which acts as the **controller** for WCRP validation logic deciding:

- Which checks to run  
- With what severity  
- According to each project’s configuration  

---

## Available Plugins

| Plugin Name           | Checker Tag             | Description                                 |
| --------------------- | ----------------------- | ------------------------------------------- |
| **wcrp_cmip6**        | `wcrp_cmip6:1.0`        | Validation of CMIP6 NetCDF files            |
| **wcrp_cordex_cmip6** | `wcrp_cordex_cmip6:1.0` | Validation of CORDEX-CMIP6 NetCDF files     |
| **wcrp_cmip7**        | `wcrp_cmip7:1.0`        | Validation of CMIP7 NetCDF files            |
| **wcrp_data**         | `wcrp_data:1.0`         | Data Validation of CMIP6/CMIP7 NetCDF files |



Each plugin has its own **TOML configuration file** defining specific rules, severities, and variable mappings.

---

## What Each Plugin Does

Each WCRP plugin:

**Loads its configuration file(s)**, which defines:

 - Which checks to run (DRS, attributes, time, metadata, etc.)
 - The severity of each check (High = Mandatory, Medium = Recommended, Low = Warning)

**Runs the atomic checks** implemented in `checks/…`:
  
 - Examples: dimension existence/size, variable shape, time bounds, filename/DRS consistency, physical plausibility, missing values, etc.

**Aggregates results** and returns them to the Compliance Checker core,  
   which formats them into human-readable or machine-readable outputs (text, HTML, JSON).

---

## Configuration

WCRP plugins are driven by TOML configuration files. CMIP6, CMIP7, and CORDEX-CMIP6 use a **split-TOML** layout, while the Data plugin retains its single-file layout.

For a full description of the TOML tree, naming conventions, and the underlying configuration model, see: **Configuration** (`configuration.md`).

### Current status by plugin

- **CMIP6 (`wcrp_cmip6`)**: uses the split-TOML layout under `plugins/cmip6/config/wcrp/` (plus `mappings/`).
- **CMIP7 (`wcrp_cmip7`)**: uses the split-TOML layout under `plugins/cmip7/config/wcrp/`.
- **CORDEX-CMIP6 (`wcrp_cordex_cmip6`)**: uses the split-TOML layout under `plugins/cordex_cmip6/config/wcrp/`; ESGVoc is the default variable/coordinate metadata source and CMOR tables remain optional.
- **Data plugin (`wcrp_data`)**: kept as a **single TOML** configuration for now (data plausibility focused), to remain lightweight and independent from project-specific split trees.
