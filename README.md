# WCRP  Compliance Checker Plugins


This [**IOOS/compliance-checker**](https://github.com/ioos/compliance-checker) plugins checks compliance with WCRP Projects specifications:


## Installation

### Pip
To install IOOS compliance-checker and the wcrp plugins :
```shell
$ pip install cc-plugin-wcrp
```
See the [**IOOS/compliance-checker**](https://github.com/ioos/compliance-checker#installation) for additional Installation notes.

And then install Esgvoc and universe to get the Controlled Vocabulary :

```shell
$ esgvoc config set universe:branch=esgvoc_dev
$ esgvoc config add cordex-cmip6
$ esgvoc install
```

## Usage

```shell
$ compliance-checker -l
```
This command displays the checkers already present on the iOS compliance checker in addition to the recently installed WCRP plugins :
 ```shell
  ...
  - wcrp_cmip6 (x.x.x)
  - wcrp_cordex_cmip6 (x.x.x)
  ...
``` 
To run the plugins on IOOS CC, use the following command:
```shell
$ compliance-checker -t ''plugin'' path/to/data/file.nc
```
Example for WCRP CMIP6 plugin :
```shell
$ compliance-checker -t wcrp_cmip6:1.0  path/to/data/CMIP6/CMIP/IPSL/IPSL-CM5A2-INCA/historical/r1i1p1f1/Amon/pr/gr/v20240619/pr_Amon_IPSL-CM5A2-INCA_historical_r1i1p1f1_gr_185001-201412.nc
```

By default, the output is in plain text, but you can specify other formats with the -f option :
```shell
$ compliance-checker -t ''plugin'' path/to/data/file.nc -f json
$ compliance-checker -t ''plugin'' path/to/data/file.nc -f html
``` 

