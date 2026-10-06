"""Integration coverage for atomic variant-label consistency checks."""

import pytest
from compliance_checker.base import BaseCheck

from checks.consistency_checks import check_variant_label_consistency as checker
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


def test_atomic_variant_label_consistency_checks(cmip6_reference_dataset):
    results = [
        *checker.check_variant_vs_realization_index(
            cmip6_reference_dataset, BaseCheck.MEDIUM
        ),
        *checker.check_variant_vs_initialization_index(
            cmip6_reference_dataset, BaseCheck.MEDIUM
        ),
        *checker.check_variant_vs_physics_index(
            cmip6_reference_dataset, BaseCheck.MEDIUM
        ),
        *checker.check_variant_vs_forcing_index(
            cmip6_reference_dataset, BaseCheck.MEDIUM
        ),
    ]

    assert len(results) == 4
    assert all(result_passed(result) for result in results)
    assert [result.name.split("]", 1)[0] + "]" for result in results] == [
        "[ATTR006a]",
        "[ATTR006b]",
        "[ATTR006c]",
        "[ATTR006d]",
    ]
