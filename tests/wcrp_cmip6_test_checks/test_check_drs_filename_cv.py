"""Unit coverage for ESGVoc DRS validation adapters."""

from types import SimpleNamespace

import pytest
from compliance_checker.base import BaseCheck

from checks.consistency_checks import check_drs_filename_cv as checker
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.remote_data


class _DrsValidator:
    def __init__(self, *, project_id):
        assert project_id == "cmip6"

    def validate_file_name(self, _filename):
        return SimpleNamespace(errors=[])

    def validate_directory(self, _directory):
        return SimpleNamespace(errors=[])


def _mock_esgvoc(monkeypatch):
    monkeypatch.setattr(checker, "ESG_VOCAB_AVAILABLE", True)
    monkeypatch.setattr(checker, "DrsValidator", _DrsValidator)


def test_check_drs_filename(cmip6_reference_dataset, monkeypatch):
    _mock_esgvoc(monkeypatch)
    results = checker.check_drs_filename(
        cmip6_reference_dataset,
        severity=BaseCheck.HIGH,
        project_id="cmip6",
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert "Filename Vocabulary" in results[0].name


def test_check_drs_directory(cmip6_reference_dataset, monkeypatch):
    _mock_esgvoc(monkeypatch)
    results = checker.check_drs_directory(
        cmip6_reference_dataset,
        severity=BaseCheck.HIGH,
        project_id="cmip6",
    )

    assert len(results) == 1
    assert result_passed(results[0])
    assert results[0].name.startswith("[PATH003]")
    assert "Directory Vocabulary" in results[0].name
