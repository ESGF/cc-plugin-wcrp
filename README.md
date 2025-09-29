# WCRP  Compliance Checker Plugins


This [**IOOS/compliance-checker**](https://github.com/ioos/compliance-checker) plugins checks compliance with WCRP Projects specifications:


## Installation

### Pip
To install IOOS compliance-checker and the wcrp plugins :
```shell
$ pip install compliance-checker 
$ pip install wcrp-plugins
```
See the [**IOOS/compliance-checker**](https://github.com/ioos/compliance-checker#installation) for additional Installation notes.

And then install Esgvoc and universe to get the Controlled Vocabulary :

```shell
$ esgvoc config set universe:branch=esgvoc_dev
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
  - wcrp_cordex (x.x.x)
  ...
``` 
To run the plugins on IOOS CC, use the following command:
```shell
$ compliance-checker -t path/to/data/file.nc
```

By default, the output is in plain text, but you can specify other formats with the -f option :
```shell
$ compliance-checker -t path/to/data/file.nc -f json
$ compliance-checker -t path/to/data/file.nc -f html
``` 

