"""Integration coverage for coordinate/bounds value consistency."""

import pytest
from compliance_checker.base import BaseCheck

from checks.variable_checks.check_bounds_value_consistency import (
    check_bounds_value_consistency,
)
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


def test_check_bounds_value_consistency(cmip6_reference_dataset):
    results = check_bounds_value_consistency(
        cmip6_reference_dataset,
        "time",
        severity=BaseCheck.HIGH,
    )

    assert len(results) == 1
    assert result_passed(results[0])
