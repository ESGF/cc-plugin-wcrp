"""Integration coverage for filename/global-attribute consistency."""

import pytest
from compliance_checker.base import BaseCheck

from checks.consistency_checks.check_attributes_match_filename import (
    check_filename_vs_global_attrs,
)
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


def test_check_filename_vs_global_attrs(cmip6_reference_dataset):
    results = check_filename_vs_global_attrs(
        cmip6_reference_dataset,
        severity=BaseCheck.HIGH,
        project_id="cmip6",
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].name.startswith("[ATTR005]")
