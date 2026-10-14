# Configuration

This page explains how WCRP plugin configuration works, how the TOML files are organised, and how to create a custom configuration.

---

## 1. Configuration layout

CMIP6, CMIP7, and CORDEX-CMIP6 load their configuration from a **split TOML tree** (the Data plugin retains a legacy single-file layout):

```text
plugins/<project>/
└── config/
    └── wcrp/
        ├── project.toml
        ├── file.toml
        ├── drs.toml
        ├── global_attributes.toml
        ├── geophysical_variable.toml
        ├── coordinate_variables.toml
        └── mappings/
            ├── table_id_to_frequency.toml
            ├── table_id_to_time_increment.toml
            └── variable_id_to_branded_variable.toml
```

**Important:**

- The configuration is loaded by the project plugin (e.g. `wcrp_cmip6`), then validated with the Pydantic model (`wcrp_schema.py`).
- If a section is missing in TOML, the corresponding checks are typically **skipped** (the plugin only runs checks for configured sections).

---


## 1.1 Project configurations 

This repository ships reference configurations for each WCRP plugin:

- **CMIP6 (`wcrp_cmip6`)**: split-TOML configuration under `plugins/cmip6/config/wcrp/`.
- **CMIP7 (`wcrp_cmip7`)**: split-TOML configuration under `plugins/cmip7/config/wcrp/`.
- **CORDEX-CMIP6 (`wcrp_cordex_cmip6`)**: split-TOML configuration under `plugins/cordex_cmip6/config/wcrp/`; its `cordex.toml` contains only project-specific checks.
- **Data plugin (`wcrp_data`)**: currently kept as a single TOML configuration (data plausibility focused).

When creating a custom configuration, it is recommended to start from the closest existing project config and adapt it (see Section 6).

## 2. File naming and purpose

### `project.toml`

Project metadata used by the plugin (name/version) and for reporting.

Typical fields:
- `project_name`
- `project_version`

---

### `file.toml`

Rules about the NetCDF container itself:
- format / data model
- compression expectations (if used)

---

### `drs.toml`

DRS-related checks:
- filename vocabulary check
- directory structure vocabulary check
- filename vs directory consistency
- directory vs global attributes consistency

---

### `global_attributes.toml`

Rules for **global attributes** (attributes attached to the NetCDF file).

This file typically contains:
- `[global.attributes.<attr_name>]`: existence/type/value rules (ATTR001–ATTR004)
- `[global.consistency.*]`: higher-level consistency rules (experiment / institution / source / variant / frequency…)

---

### `geophysical_variable.toml`

Rules for the **main geophysical variable** (e.g. `pr`, `tas`), detected using CF utilities.

This file typically contains:
- `variable.existence` 
- `variable.type` 
- `variable.dimensions` 
- `[variable.attributes.<attr_name>]`: variable-level attributes (`units`, `standard_name`, `cell_methods`, etc.)

---

### `coordinate_variables.toml`

Rules for **coordinate variables** such as `time`, `lat`, `lon`, and optional vertical or additional coordinates.

This file usually contains two layers of configuration:

1) **Global coordinate rules**
- `[coordinates.dimensions]`: apply dimension checks on coordinate variables’ dimensions
- `[coordinates.bounds]`: apply checks on bounds variables (e.g. `time_bnds`, `lat_bnds`, `lon_bnds`)

2) **Per-coordinate rules**
- `[coordinates.time.attributes.*]`
- `[coordinates.lat.attributes.*]`
- `[coordinates.lon.attributes.*]`
- optional: `[coordinates.<coord>.monotonicity]` (strict increasing/decreasing)

---

### `mappings/*.toml`

Mappings are used to interpret file metadata (e.g. `table_id` → frequency) or map `variable_id/table_id` to a branded-variable identifier for registry lookups.

Common mappings:
- `table_id_to_frequency.toml`
- `table_id_to_time_increment.toml`
- `variable_id_to_branded_variable.toml`

---

## 3. Rule structure (Pydantic model overview)

The Pydantic schema (`wcrp_schema.py`) defines what keys are accepted and how rules are interpreted. At a high level:

- **Global attributes** apply to the dataset.
- **Variable attributes** apply to a netCDF variables.
- **Coordinate attributes** apply to coordinate variables (`time`, `lat`, `lon`, …).

A typical attribute rule looks like:

```toml
[global.attributes.Conventions]
severity = "H"
value_type = "str"
is_required = true

# ATTR004: choose exactly one rule (see next section)
cv_source_collection = "conventions"
```

---

## 4. ATTR004 rule exclusivity

For ATTR004, exactly **one** rule must be active, with one exception:

Allowed ATTR004 rules (choose one):
- `pattern`
- `constant`
- `enum`
- `as_variable`
- `is_positive`
- `cv_source_collection` (may optionally include `cv_source_collection_key`)
- `cv_source_term_key` (registry expected-term)

**Exception:**  
`cv_source_collection` and `cv_source_collection_key` can be used together because the key is a parameter of the vocabulary rule.

---

## 5. Attribute rule parameters 

Most checks in `global_attributes.toml`, `geophysical_variable.toml`, and `coordinate_variables.toml` are driven by **attribute rules**. A rule is a small TOML table that describes:
- which attribute to check
- how strict the check is (severity)
- whether the attribute must exist
- and (optionally) a single ATTR004 validation rule

### Common fields

- `is_required` (bool)
  - If `true`: the attribute **must exist**. Missing attribute produces an **ATTR001 failure**.
  - If `false`: missing attribute is **ignored** (no failure). If present, it can still be validated by the other fields.

- `value_type` (str)
  Type expected for the attribute value.
  Typical values used in this project:
  	- `str` — string
  	- `int` — integer
  	- `float` — float
  	- `str_array` — list of strings (space-separated string or actual array depending on the file)

### ATTR004: value validation rules (exclusive)

For ATTR004, **exactly one rule** must be configured among the list below.
The only allowed combination is `cv_source_collection` **with** `cv_source_collection_key` (the key is a parameter of the same vocabulary rule).

- `pattern` (str) :
  Regular expression applied to the attribute value (full match).
  Example:
  
      pattern = "^days since [0-9]{4}-[0-9]{2}-[0-9]{2}.*"

- `constant`:
  The attribute value must match exactly (string comparison after trimming).
  Example:
  
      constant = "longitude"

- `enum` (array) :
  Allowed values list.
  Example:
  
      enum = ["noleap", "gregorian", "proleptic_gregorian"]

- `as_variable` (bool) :
  The attribute value is interpreted as one (or several) variable name(s) and must refer to existing variables in the dataset.
  Typical use: `bounds` attributes pointing to `time_bnds`, `lat_bnds`, …

- `is_positive` (bool) :
  The attribute value must be numeric and strictly greater than 0.

- `cv_source_collection` (str) :
  ESGVOC vocabulary collection identifier. The attribute value must be a valid term inside that collection.

- `cv_source_collection_key` (str, optional) :
  Optional key used for strict collection lookup (project+collection+term-id pattern). This is **not** a separate rule; it is an optional parameter of the vocabulary rule.

- `cv_source_term_key` (str) :
  Variable Registry expected-term comparison.
  The plugin resolves an `expected_term` from the variable registry (based on file metadata + mappings for `cmip6` and branded_variable global attribute for `cmip7`) and compares the attribute value to `expected_term.<cv_source_term_key>`.
  Example (geophysical variable only):

      `cv_source_term_key = "cf_units"`

### Quick examples

**Required attribute + constant**

    [coordinates.lat.attributes.standard_name]
    severity = "H"
    value_type = "str"
    is_required = true
    constant = "latitude"

**Optional attribute + enum**

    [coordinates.time.attributes.calendar]
    severity = "M"
    value_type = "str"
    is_required = false
    enum = ["noleap", "gregorian"]

**Bounds pointer check**

    [coordinates.time.attributes.bounds]
    severity = "H"
    value_type = "str"
    is_required = true
    as_variable = true

**Vocabulary check**

    [global.attributes.variant_label]
    severity = "H"
    value_type = "str"
    is_required = true
    cv_source_collection = "variant_label"

**Registry expected-term check**

    [variable.attributes.units]
    severity = "H"
    value_type = "str"
    is_required = true
    cv_source_term_key = "cf_units"


## 6. How to create your own configuration

### Fork from cmip6 or cmip7 project configs

1) Copy an existing configuration directory:

```text
plugins/cmip6/config/wcrp  →  plugins/<your_project>/config/wcrp
```

2) Modify only what you need :

- change vocabulary collections if you use esgvoc.
- change required attributes
- adapt your severities  
- adjust patterns/constants/enums
- adjust coordinate rules (bounds/dimensions/monotonicity) etc..

3) Keep mappings consistent

This approach guarantees you keep the same structure expected by the schema.

---




