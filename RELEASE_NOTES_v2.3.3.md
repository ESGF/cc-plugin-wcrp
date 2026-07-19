# cc-plugin-wcrp v2.3.3

## What's Changed

### New C3S CMIP6 checker

* Added the new `wcrp_c3scmip6:1.0` Compliance Checker suite and its dedicated configuration.
* Registered the C3S CMIP6 checker as a package entry point.
* Kept C3S CMIP6 validation connected to the CMIP6 ESGVOC project and CMIP6 DRS specification.
* Added configurable numeric threshold validation for attributes such as `_FillValue` and `missing_value`, while preserving exact-value validation through `constant`.

### Attribute and consistency checks

* Added ESGVOC-backed dynamic requirements for parent experiment, sub-experiment, parent activity, and parent MIP-era attributes.
* Improved handling of not-applicable sentinel values for conditionally required attributes.
* Split variant-label consistency into independent `ATTR006a`–`ATTR006d` checks.
* Split experiment consistency into independent `ATTR007a`–`ATTR007d` checks.
* Improved ESGVOC experiment resolution, including case-sensitive DRS names and CMIP6/CMIP7 vocabulary differences.
* Updated global-attribute consistency configuration for CMIP6, CMIP6Plus, and CMIP7.

### Time checks

* Fixed sub-daily filename time-range validation to use time-coordinate values instead of cell-bound starts.
* Changed `TIME001` comparisons from truncation to rounding to avoid floating-point precision mismatches by @JanStreffing in [#55](https://github.com/ESGF/cc-plugin-wcrp/pull/55).
* Added support for valid monthly instantaneous timestamp conventions by @sol1105 in [#63](https://github.com/ESGF/cc-plugin-wcrp/pull/63).

### Variable and coordinate checks

* Added support for scalar-coordinate bounds.
* Fixed bounds-value consistency for decreasing coordinate axes.
* Added support for one-dimensional latitude and longitude coordinates in CORDEX-CMIP6 by @jesusff in [#54](https://github.com/ESGF/cc-plugin-wcrp/pull/54).
* Improved CMIP7 geophysical-variable detection by falling back to `variable_id` when necessary by @JanStreffing in [#49](https://github.com/ESGF/cc-plugin-wcrp/pull/49).
* Skipped inappropriate fill-value and type checks for CMIP7 CF flag-valued geophysical variables by @JanStreffing in [#60](https://github.com/ESGF/cc-plugin-wcrp/pull/60).
* Temporarily disabled the CMIP7 `lev` monotonicity rule by @glevava in [#57](https://github.com/ESGF/cc-plugin-wcrp/pull/57).

## Contributors

Thanks to @anachite, @JanStreffing, @sol1105, @jesusff, and @glevava for their contributions to this release.

**Full Changelog**: https://github.com/ESGF/cc-plugin-wcrp/compare/v2.3.2...v2.3.3
