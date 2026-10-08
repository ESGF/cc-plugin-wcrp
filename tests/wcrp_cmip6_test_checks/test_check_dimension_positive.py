"""Integration coverage for positive dimension sizes."""

import pytest
from compliance_checker.base import BaseCheck

from checks.dimension_checks.check_dimension_positive import check_dimension_positive
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


def test_check_dimension_positive(cmip6_reference_dataset):
    results = check_dimension_positive(
        cmip6_reference_dataset,
        "time",
        severity=BaseCheck.MEDIUM,
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].weight == BaseCheck.MEDIUM
