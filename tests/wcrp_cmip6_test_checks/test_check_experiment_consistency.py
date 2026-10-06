"""Unit coverage for atomic experiment consistency checks."""

from types import SimpleNamespace

from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.consistency_checks import check_experiment_consistency as checker
from tests.wcrp_cmip6_test_checks.conftest import result_passed


def _mock_experiment(monkeypatch):
    term = SimpleNamespace(
        activity_id=["CMIP"],
        experiment="all-forcing simulation of the recent past",
        parent_experiment_id=["piControl"],
        sub_experiment_id=["forecast"],
    )
    monkeypatch.setattr(checker, "ESG_VOCAB_AVAILABLE", True)
    monkeypatch.setattr(checker, "resolve_experiment_term", lambda *_args: term)


def test_atomic_experiment_consistency_checks(tmp_path, monkeypatch):
    _mock_experiment(monkeypatch)
    with Dataset(tmp_path / "experiment.nc", "w") as dataset:
        dataset.experiment_id = "historical"
        dataset.activity_id = "CMIP"
        dataset.experiment = "all-forcing simulation of the recent past"
        dataset.parent_experiment_id = "piControl"
        dataset.sub_experiment_id = "forecast"

        results = [
            checker.check_experiment_id_vs_activity_id(dataset, BaseCheck.HIGH),
            checker.check_experiment_id_vs_experiment(dataset, BaseCheck.HIGH),
            checker.check_experiment_id_vs_parent_experiment_id(
                dataset, BaseCheck.HIGH
            ),
            checker.check_experiment_id_vs_sub_experiment_id(dataset, BaseCheck.HIGH),
        ]

    flattened = [result for group in results for result in group]
    assert len(flattened) == 4
    assert all(result_passed(result) for result in flattened)
    assert [result.name.split("]", 1)[0] + "]" for result in flattened] == [
        "[ATTR007a]",
        "[ATTR007b]",
        "[ATTR007c]",
        "[ATTR007d]",
    ]


def test_experiment_consistency_reports_mismatch(tmp_path, monkeypatch):
    _mock_experiment(monkeypatch)
    with Dataset(tmp_path / "experiment.nc", "w") as dataset:
        dataset.experiment_id = "historical"
        dataset.experiment = "incorrect description"
        results = checker.check_experiment_id_vs_experiment(
            dataset,
            BaseCheck.HIGH,
        )

    assert len(results) == 1
    assert not result_passed(results[0])
    assert "CV expects" in results[0].msgs[0]
