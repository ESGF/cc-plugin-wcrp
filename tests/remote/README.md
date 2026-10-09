Remote plugin regression tests
==============================

The generic runner is tests/test_remote_plugin_checks.py. Project-specific
files in this directory define:

- the public Compliance Checker name;
- remote NetCDF files, normally pinned by repository revision and optionally
  verified by SHA-256;
- every exposed checker method;
- the expected number of Result objects and any expected failure messages.

To add another plugin, create a project module like cordex_cmip6.py and register
its RemoteProject in PROJECT_CASES in __init__.py. The inventory test fails if a
checker method is added or removed without updating every dataset baseline.

Run the network-backed suite with:

    pytest -m remote_data tests/test_remote_plugin_checks.py tests/test_remote_cmip7.py

Files are cached by pooch below its platform-specific
`pooch.os_cache("cc-plugin-wcrp")` directory. On Linux the default layout is:

    ~/.cache/cc-plugin-wcrp/remote-tests/<repository>/<revision>/<relative-path>

For example, the default repository/revision roots are currently:

    ~/.cache/cc-plugin-wcrp/remote-tests/py-cordex-data/<commit>/
    ~/.cache/cc-plugin-wcrp/remote-tests/mini-esgf-data/master/

`WCRP_REMOTE_TEST_DATA_CACHE` selects another cache root while retaining the
`remote-tests/<repository>/<revision>/...` structure.
`WCRP_REMOTE_TEST_DATA_DIR` can instead point to an existing data checkout; in
that case each configured relative path is resolved directly below that
directory. Files with a configured checksum are still verified.

CORDEX-CMIP6 REMO files are hosted by euro-cordex/py-cordex-data. The CMIP7
curvilinear-ocean file and the CMIP6 reference used by the historical
atomic-check suite are hosted by roocs/mini-esgf-data.

The project modules are the authoritative source for repository URLs,
revisions, relative paths, and checksums:

- `tests/remote/cordex_cmip6.py` for the REMO CORDEX-CMIP6 files;
- `tests/remote/cmip7.py` for the CMIP7 curvilinear-ocean file;
- `tests/remote/cmip6.py` for the CMIP6 reference file.
