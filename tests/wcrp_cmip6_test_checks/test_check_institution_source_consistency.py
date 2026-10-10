"""Unit coverage for institution/source consistency with mocked ESGVoc."""

from types import SimpleNamespace

from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.consistency_checks import check_institution_source_consistency as checker
from tests.wcrp_cmip6_test_checks.conftest import result_passed


def test_institution_and_source_consistency(tmp_path, monkeypatch):
    terms = {
        ("institution_id", "ipsl"): SimpleNamespace(
            id="ipsl",
            description="Institut Pierre Simon Laplace",
        ),
        ("source_id", "ipsl-cm5a2-inca"): SimpleNamespace(
            id="ipsl-cm5a2-inca",
            organisation_id=["IPSL"],
        ),
    }

    def get_term_in_collection(*, project_id, collection_id, term_id):
        assert project_id == "cmip6"
        return terms.get((collection_id, term_id))

    monkeypatch.setattr(checker, "ESG_VOCAB_AVAILABLE", True)
    monkeypatch.setattr(
        checker.voc,
        "get_term_in_collection",
        get_term_in_collection,
    )

    with Dataset(tmp_path / "source.nc", "w") as dataset:
        dataset.institution_id = "IPSL"
        dataset.institution = "Institut Pierre Simon Laplace"
        dataset.source_id = "IPSL-CM5A2-INCA"
        results = [
            *checker.check_institution_consistency(
                dataset,
                severity=BaseCheck.HIGH,
                project_id="cmip6",
            ),
            *checker.check_source_consistency(
                dataset,
                severity=BaseCheck.HIGH,
                project_id="cmip6",
            ),
        ]

    assert len(results) == 2
    assert all(result_passed(result) for result in results)
    assert results[0].name.startswith("[ATTR009]")
    assert results[1].name.startswith("[ATTR010]")


def test_missing_attributes_are_left_to_attr001_when_delegated(tmp_path):
    with Dataset(tmp_path / "source.nc", "w") as dataset:
        assert (
            checker.check_institution_consistency(
                dataset,
                severity=BaseCheck.HIGH,
                missing_attributes_delegated={"institution_id", "institution"},
            )
            == []
        )
        assert (
            checker.check_source_consistency(
                dataset,
                severity=BaseCheck.HIGH,
                missing_attributes_delegated={"source_id", "institution_id"},
            )
            == []
        )


def test_consistency_check_reports_only_unowned_missing_attribute(tmp_path):
    with Dataset(tmp_path / "source.nc", "w") as dataset:
        result = checker.check_institution_consistency(
            dataset,
            severity=BaseCheck.HIGH,
            missing_attributes_delegated={"institution_id"},
        )[0]

    assert "'institution'" in result.msgs[0]
    assert "'institution_id'" not in result.msgs[0]
