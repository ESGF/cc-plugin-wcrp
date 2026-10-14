"""Integration coverage for path, filename, and attribute consistency."""

import pytest
from compliance_checker.base import BaseCheck

from checks.consistency_checks.check_drs_consistency import (
    check_attributes_match_directory_structure,
    check_filename_matches_directory_structure,
)
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


def test_check_attributes_match_directory_structure(cmip6_reference_dataset):
    results = check_attributes_match_directory_structure(
        cmip6_reference_dataset,
        severity=BaseCheck.MEDIUM,
        project_id="cmip6",
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].name.startswith("[PATH001]")


def test_check_filename_matches_directory_structure(cmip6_reference_dataset):
    results = check_filename_matches_directory_structure(
        cmip6_reference_dataset,
        severity=BaseCheck.MEDIUM,
        project_id="cmip6",
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].name.startswith("[PATH002]")
