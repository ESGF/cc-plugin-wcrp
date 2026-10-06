"""Fixtures shared by the historical atomic-check regression suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from netCDF4 import Dataset

from tests.remote.cmip6 import CMIP6_REFERENCE_FILE
from tests.remote.fetch import resolve_remote_file


@pytest.fixture(scope="session")
def cmip6_reference_path() -> Path:
    """Return the CMIP6 fixture, preferring the historical local copy."""
    local = (
        Path(__file__).parents[1]
        / "data"
        / "CMIP6/CMIP/IPSL/IPSL-CM5A2-INCA/historical/r1i1p1f1/Amon/pr/gr/"
        / "v20240619/"
        / CMIP6_REFERENCE_FILE.relative_path.name
    )
    if local.is_file():
        return local
    return resolve_remote_file(CMIP6_REFERENCE_FILE)


@pytest.fixture
def cmip6_reference_dataset(cmip6_reference_path):
    """Open the CMIP6 reference file for one test and always close it."""
    with Dataset(cmip6_reference_path) as dataset:
        yield dataset


def result_passed(result) -> bool:
    """Interpret Compliance Checker scalar and scored result values."""
    value = result.value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0] == value[1]
    return bool(value)
