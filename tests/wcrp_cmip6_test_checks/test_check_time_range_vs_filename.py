"""Integration coverage for filename time ranges without a live ESGVoc DB."""

from types import SimpleNamespace

import pytest
from compliance_checker.base import BaseCheck

from checks.time_checks import check_time_range_vs_filename as checker
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


class _DrsValidator:
    def validate_file_name(self, _filename):
        return SimpleNamespace(errors=[])


def test_check_time_range_vs_filename(cmip6_reference_dataset, monkeypatch):
    monkeypatch.setattr(checker, "_get_validator", lambda _project_id: _DrsValidator())

    results = checker.check_time_range_vs_filename(
        cmip6_reference_dataset,
        severity=BaseCheck.MEDIUM,
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].name.startswith("[TIME003]")
