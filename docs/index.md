# cc-plugin-wcrp

A **WCRP-specific plugins** for the [IOOS Compliance Checker](https://ioos.github.io/compliance-checker/).  
It provides automated quality control (QC) and compliance validation for WCRP-related datasets, including **CMIP6**, **CORDEX**, and related projects.

---

## What is it?

`cc-plugin-wcrp` extends the **IOOS Compliance Checker** with a comprehensive suite of checks derived from ESGF and WCRP data standards. 



## Key Features

- 🧩 **Plugin-based architecture** — integrates seamlessly into the Compliance Checker framework  
- ⚙️ **Configurable via TOML files** — defines severity levels, project-specific checks, and thresholds  
- 🔍 **Covers all validation layers:**
  - Global and variable attributes  
  - Dimensions and coordinates  
  - File naming and DRS path structure  
  - Time continuity and consistency  
  - Data plausibility
- 📊 **Flexible outputs** — JSON, text, and HTML reports  
- 🧠 **Extensible design** — easily add new checks or adapt existing ones per project or dataset type  



