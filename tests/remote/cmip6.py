"""CMIP6 reference files hosted by roocs/mini-esgf-data."""

from pathlib import PurePosixPath

from tests.remote.model import RemoteFile


CMIP6_REFERENCE_FILE = RemoteFile(
    repository="mini-esgf-data",
    base_url="https://raw.githubusercontent.com/roocs/mini-esgf-data",
    # Replace master with the commit SHA after this file has been uploaded.
    revision="master",
    relative_path=PurePosixPath(
        "test_data/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM5A2-INCA/"
        "historical/r1i1p1f1/Amon/pr/gr/v20240619/"
        "pr_Amon_IPSL-CM5A2-INCA_historical_r1i1p1f1_gr_185001-201412.nc"
    ),
)
