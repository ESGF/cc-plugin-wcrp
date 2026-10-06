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

    pytest -m remote_data tests/test_remote_plugin_checks.py

Files are cached by pooch. WCRP_REMOTE_TEST_DATA_CACHE selects another cache
root. WCRP_REMOTE_TEST_DATA_DIR can instead point to an existing checkout of
the data repository; files with a configured checksum are still verified.

CORDEX-CMIP6 REMO files are hosted by euro-cordex/py-cordex-data. The CMIP6
reference used by the historical atomic-check suite is configured in cmip6.py
and hosted by roocs/mini-esgf-data.
