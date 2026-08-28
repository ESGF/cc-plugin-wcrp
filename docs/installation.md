# Overview & Installation

## 🌍 Scientific Context
The **WCRP projects** (such as CMIP6, CORDEX, CORDEX-CMIP6) produce large volumes of NetCDF files.  
The **cc-plugin-wcrp**  provides plugins that call automated **quality and consistency checks** to verify compliance with WCRP Projects standards , including DRS structure, controlled vocabulary (CV), attributes, and temporal continuity.

## ⚙️ Architecture Overview

| Layer | Component | Description |
|-------|------------|-------------|
| **Base Framework** | [IOOS Compliance Checker](https://github.com/ioos/compliance-checker) | Provides the core plugin mechanism and test runner |
| **Plugin Package** | `cc-plugin-wcrp` | Adds project-specific checks for CMIP6, CORDEX, and CORDEX-CMIP6 |
| **Controlled Vocabulary** | [esgvoc](https://github.com/ESGF/esgvoc) | Supplies the CV (activity_id, source_id, etc.) |
| **Configuration** | `wcrp_config.toml` | Defines active checks, thresholds, and DRS rules |

## 🧩 Requirements
**Python** ≥ 3.10  
**Dependencies** (installed automatically):

  - `netCDF4`
  - `cfchecker`
  - `compliance-checker>=5.1.2`
  - `esgvoc`
  - `cftime`
  - `pooch`

## 🛠️ Installation
**Pip Installation**

```bash
pip install cc-plugin-wcrp
```
**Pip Installation from source**

Clone the repository and cd into the repository folder, then:

```bash
pip install -e .
```
**Esgvoc Installation**

If you have an old version of `esgvoc`, you should upgrade it:
```bash
pip install esgvoc --upgrade
```
Then, use the commands below to activate the project you want:
```bash
esgvoc use project@latest universe@latest
```
for example for CMIP6 :
```bash
esgvoc use cmip6@latest universe@latest
```
The projects currently available  in both 'cc_plugin_wcrp' and 'esgvoc' are:
```bash
cmip6, cmip6plus, cmip7 and cordex-cmip6 

```

## Usage


## Verify the installation:
For **cc-plugin-wcrp** :
```bash
compliance-checker -l
```
Normally, you should have a list of all available plugins with the compliance checker, in addition to the wcrp_cmip6 and wcrp_cordex_cmip6 plugins
```bash
IOOS compliance checker available checker suites:
 - acdd:1.1
 - acdd:1.3
 - cc6:0.4.0
 - cf:1.10
 - cf:1.11
 - cf:1.6
 - cf:1.7
 - cf:1.8
 - cf:1.9
 - ioos:0.1
 - ioos:1.1
 - ioos:1.2
 - ioos_sos:0.1
 - mip:0.4.0
 - wcrp_cmip6:1.0
 - wcrp_cordex_cmip6:1.0
```

For **esgvoc** :
```bash
esgvoc --help
# or
pip show esgvoc
```
