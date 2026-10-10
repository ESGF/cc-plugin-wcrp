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


def test_variant_group_reports_malformed_selector_once(tmp_path):
    from netCDF4 import Dataset

    with Dataset(tmp_path / "variant.nc", "w") as dataset:
        dataset.variant_label = "not-a-variant"
        results = checker.check_variant_label_consistency_group(
            dataset,
            [
                (checker.check_variant_vs_realization_index, BaseCheck.HIGH),
                (checker.check_variant_vs_initialization_index, BaseCheck.HIGH),
                (checker.check_variant_vs_physics_index, BaseCheck.HIGH),
                (checker.check_variant_vs_forcing_index, BaseCheck.HIGH),
            ],
        )

    assert len(results) == 1
    assert results[0].name.startswith("[ATTR006a]")
    assert "format of 'variant_label'" in results[0].msgs[0]


def test_variant_group_uses_fixed_priority_between_enabled_checks(tmp_path):
    from netCDF4 import Dataset

    with Dataset(tmp_path / "variant.nc", "w") as dataset:
        results = checker.check_variant_label_consistency_group(
            dataset,
            [
                (checker.check_variant_vs_physics_index, BaseCheck.HIGH),
                (checker.check_variant_vs_forcing_index, BaseCheck.HIGH),
            ],
        )

    assert len(results) == 1
    assert results[0].name.startswith("[ATTR006c]")
    assert "variant_label" in results[0].msgs[0]
