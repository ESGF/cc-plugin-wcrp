"""Integration coverage for variable shape/dimension agreement."""

import pytest
from compliance_checker.base import BaseCheck

from checks.variable_checks.check_variable_shape_vs_dimensions import check_variable_shape
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


def test_check_variable_shape(cmip6_reference_dataset):
    results = check_variable_shape(
        "lat",
        cmip6_reference_dataset,
        severity=BaseCheck.MEDIUM,
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].name.startswith("[VAR010]")
