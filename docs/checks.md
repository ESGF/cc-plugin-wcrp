# Checks Overview

cc_plugin-wcrp is built from many **atomic checks**, grouped by category. Eah plugin (e.g., `wcrp_cmip6`, `wcrp_cordex_cmip6`) just decides which ones to run; each category focuses on a specific aspect of file quality.

---

## 📏 Dimension Checks
Ensure that all **dimensions** exist and match the variable shapes.  
These checks confirm that:
- Dimensions (e.g. `time`, `lat`, `lon`, `bnds`) are present.
- Their sizes are **positive** and coherent with variable shapes.
- Bounds variables (e.g. `time_bnds`) are consistent in size and order.

## 📊 Variable Checks
Ensure variables are present, their shape matches their dimensions, and (when bounds are declared) each value lies **within its bounds**.

##  ⏱️ Time Checks
Focus on the `time` axis: presence of bounds with the right shape, and consistency between the time axis and the filename’s time range.

## 🏷️ Attribute Checks
Check that mandatory global and variable attributes are there, correctly typed/encoded, and their values are valid against regex patterns or controlled vocabularies.

## 📁 DRS & Path Consistency Checks
Validate filenames and directory paths and compare with the netcdf file's attributes; verify consistency between attributes like **frequency vs table_id**, `experiment_id`, `institution`, `variant_label`, etc., against project rules/CVs.

## 🧬 Data Plausibility Checks
Perform **data-level plausibility tests** on numeric variables:
- Check NaN and infinite .  
- Check Constant .  
- Check Fill Value or Missing values filling.  
- Detect Physically impossible or statistically extreme outliers.  
- Check Chunk size.
- 
## 📦 File Format & Compression Checks
These checks ensure that the **NetCDF file format** and **compression options** comply with project and ESGF recommendations.

## Detailed Inventory
For the full list of checks (IDs, severities), see the spreadsheet :

**🔗[Checks_QC_Table](https://docs.google.com/spreadsheets/d/15LytNx3qE7mvuCpyFYAsGFFKqzmm1MH_BoApoqbmLQk/edit?gid=1447223205#gid=1447223205)**
