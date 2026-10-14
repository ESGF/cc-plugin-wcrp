"""Integration coverage for regular time bounds."""

import pytest
from compliance_checker.base import BaseCheck

from checks.time_checks.check_time_bounds import check_time_bounds
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


def test_check_time_bounds(cmip6_reference_dataset):
    results = check_time_bounds(
        cmip6_reference_dataset,
        severity=BaseCheck.MEDIUM,
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].name.startswith("[TIME002]")
