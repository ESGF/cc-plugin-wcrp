## 🚀 Running the WCRP Plugins
To run the WCRP plugins , use the following command:
```bash  
compliance-checker -t ''check''  ''filepath'' 
``` 
Example for WCRP CMIP6 plugin : 
```bash
compliance-checker  -t wcrp_cmip6:1.0  path/to/data/CMIP6/CMIP/IPSL/IPSL-CM5A2-INCA/historical/r1i1p1f1/Amon/pr/gr/v20240619/pr_Amon_IPSL-CM5A2-INCA_historical_r1i1p1f1_gr_185001-201412.nc
```
- If you don’t pass -f, the default output is text.
- **Output Formats** : text ( default) / html / json (single input file only) / json_new (handles multiple files).

⚠️ **CF plugin**: Always run the CF checks as well when running the WCRP project checks, to ensure that CF-only checks (not duplicated in WCRP projects) are also verified.


## Run several checks
```bash
compliance-checker   -t cf:1.11 -t wcrp_cmip6:1.0 path/to/data/CMIP6/CMIP/IPSL/IPSL-CM5A2-INCA/historical/r1i1p1f1/Amon/pr/gr/v20240619/pr_Amon_IPSL-CM5A2-INCA_historical_r1i1p1f1_gr_185001-201412.nc
```

## Save output to a file
```bash
compliance-checker -t wcrp_cmip6:1.0 file.nc -f text(json/html) -o qc.txt(json/html)   
```

## Skip / Include Specific WCRP_Project Checks

- Only use check names from the wcrp_project (wcrp_cmip6 for example) suite here.

To **skip checks** :
```bash
# Here we skip check_Global_Variable_Attributes from WCRP CMIP6 Plugin for example
compliance-checker -t wcrp_cmip6:1.0 file.nc -s  check_Global_Variable_Attributes
```
To **include checks** ( we do only checks belonging to this specific checks)
```bash
# Here we do only check_Drs_Vocabulary from WCRP CMIP6 Plugin for example
compliance-checker -t wcrp_cmip6:1.0 file.nc -i check_Drs_Vocabulary
```
To **list all available checks**
```bash
compliance-checker -l 
```

## Multiple files

- Use shell wildcards/globs, and use **json_new** if you want one detailed JSON for all files:
```bash
# Here we skip check_Global_Variable_Attributes from WCRP CMIP6 Plugin for example
compliance-checker -t wcrp_cmip6 /data/CMIP6/**/*.nc -f json_new 
```