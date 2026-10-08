"""Integration coverage for frequency/table consistency."""

from pathlib import Path

import pytest
import toml
from compliance_checker.base import BaseCheck

from checks.consistency_checks.check_frequency_table_consistency import (
    check_frequency_table_id_consistency,
)
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


def test_check_frequency_table_id_consistency(cmip6_reference_dataset):
    mapping_path = Path(__file__).with_name("mapping.toml")
    mapping = toml.load(mapping_path)["frequency_table_id_mapping"]

    results = check_frequency_table_id_consistency(
        cmip6_reference_dataset,
        mapping,
        severity=BaseCheck.MEDIUM,
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].name.startswith("[ATTR008]")
