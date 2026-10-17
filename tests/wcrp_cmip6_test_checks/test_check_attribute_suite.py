"""Unit coverage for the current atomic attribute suite."""

from types import SimpleNamespace

import numpy as np
import pytest
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.attribute_checks import check_attribute_suite as checker
from tests.wcrp_cmip6_test_checks.conftest import result_passed


def test_check_attribute_suite_passes_all_configured_rules(tmp_path):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        dataset.table_id = "Amon"
        results = checker.check_attribute_suite(
            dataset,
            "table_id",
            severity=BaseCheck.MEDIUM,
            value_type="str",
            enum=("Amon", "Omon"),
            project_name="cmip6",
        )

    assert len(results) == 4
    assert all(result_passed(result) for result in results)
    assert [result.name.split("]", 1)[0] + "]" for result in results] == [
        "[ATTR001]",
        "[ATTR002]",
        "[ATTR003]",
        "[ATTR004]",
    ]


def test_check_attribute_suite_reports_invalid_enum(tmp_path):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        dataset.table_id = "invalid"
        results = checker.check_attribute_suite(
            dataset,
            "table_id",
            severity=BaseCheck.HIGH,
            enum=("Amon",),
        )

    assert len(results) == 3
    assert result_passed(results[0])
    assert result_passed(results[1])
    assert not result_passed(results[2])
    assert "not in allowed values" in results[2].msgs[0]


def test_check_attribute_suite_uses_mocked_esgvoc_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "_ESGVOC_AVAILABLE", True)
    monkeypatch.setattr(
        checker.voc,
        "get_all_terms_in_collection",
        lambda **_kwargs: [SimpleNamespace(description="Expected institution")],
    )

    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        dataset.institution = "Expected institution"
        results = checker.check_attribute_suite(
            dataset,
            "institution",
            severity=BaseCheck.HIGH,
            cv_source_collection="institution_id",
            cv_source_collection_key="description",
            project_name="cmip6",
        )

    assert len(results) == 3
    assert all(result_passed(result) for result in results)


def test_missing_registry_attribute_is_optional_when_expected_value_is_empty(
    tmp_path,
):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        dataset.createVariable("tas", "f4")
        results = checker.check_attribute_suite(
            dataset,
            "cell_measures",
            var_name="tas",
            severity=BaseCheck.MEDIUM,
            value_type="str",
            is_required=True,
            cv_source_term_key="cell_measures",
            expected_term=SimpleNamespace(cell_measures=None),
        )

    assert results == []


def test_defined_registry_attribute_reports_that_cv_value_is_unset(tmp_path):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        variable = dataset.createVariable("tas", "f4")
        variable.comment = "Model-specific information"
        results = checker.check_attribute_suite(
            dataset,
            "comment",
            var_name="tas",
            severity=BaseCheck.LOW,
            is_required=False,
            cv_source_term_key="comment",
            expected_term_comparison="contains",
            expected_term=SimpleNamespace(comment=None),
        )

    failure = results[-1]
    assert failure.name == (
        "[ATTR004] Variable 'tas' attribute 'comment' registry expected-term check"
    )
    assert not result_passed(failure)
    assert failure.msgs == [
        "Variable 'tas' attribute 'comment' is defined as "
        "'Model-specific information', but the selected CV entry does not define "
        "a value for 'comment'."
    ]


def test_missing_registry_attribute_remains_required_when_value_is_expected(
    tmp_path,
):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        dataset.createVariable("tas", "f4")
        results = checker.check_attribute_suite(
            dataset,
            "cell_measures",
            var_name="tas",
            severity=BaseCheck.MEDIUM,
            value_type="str",
            is_required=True,
            cv_source_term_key="cell_measures",
            expected_term=SimpleNamespace(cell_measures="area: areacella"),
        )

    assert len(results) == 1
    assert results[0].name == (
        "[ATTR001] Variable 'tas' attribute 'cell_measures' existence"
    )
    assert not result_passed(results[0])
    assert results[0].msgs == [
        "Required variable 'tas' attribute 'cell_measures' is missing."
    ]


@pytest.mark.parametrize("marker", ["--MODEL", "--OPT", "--UGRID"])
def test_special_cell_measures_marker_accepts_producer_value(tmp_path, marker):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        variable = dataset.createVariable("sidmasstrany", "f4")
        variable.cell_measures = "area: areacello"
        results = checker.check_attribute_suite(
            dataset,
            "cell_measures",
            var_name="sidmasstrany",
            severity=BaseCheck.MEDIUM,
            value_type="str",
            is_required=True,
            cv_source_term_key="cell_measures",
            expected_term=SimpleNamespace(cell_measures=marker),
        )

    assert all(result_passed(result) for result in results)


@pytest.mark.parametrize("marker", ["--MODEL", "--OPT", "--UGRID"])
def test_special_cell_measures_marker_does_not_require_attribute(tmp_path, marker):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        dataset.createVariable("sidmasstrany", "f4")
        results = checker.check_attribute_suite(
            dataset,
            "cell_measures",
            var_name="sidmasstrany",
            severity=BaseCheck.MEDIUM,
            value_type="str",
            is_required=True,
            cv_source_term_key="cell_measures",
            expected_term=SimpleNamespace(cell_measures=[marker]),
        )

    assert results == []


def test_missing_optional_registry_attribute_can_be_reported_as_advisory(tmp_path):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        dataset.createVariable("tas", "f4")
        results = checker.check_attribute_suite(
            dataset,
            "comment",
            var_name="tas",
            severity=BaseCheck.LOW,
            is_required=False,
            cv_source_term_key="comment",
            expected_term_comparison="contains",
            report_missing_expected_term=True,
            expected_term=SimpleNamespace(
                comment=["Air temperature", "Alternative comment"]
            ),
        )

    assert len(results) == 1
    assert results[0].weight == BaseCheck.LOW
    assert results[0].name == (
        "[ATTR004] Variable 'tas' attribute 'comment' registry expected-term check"
    )
    assert not result_passed(results[0])
    assert results[0].msgs == [
        "No variable 'tas' attribute 'comment' is defined, so none of the "
        "registered values ['Air temperature', 'Alternative comment'] is included."
    ]


def test_registry_attribute_accepts_one_value_from_expected_list(tmp_path):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        variable = dataset.createVariable("tas", "f4")
        variable.cell_methods = "area: time: mean"
        results = checker.check_attribute_suite(
            dataset,
            "cell_methods",
            var_name="tas",
            severity=BaseCheck.HIGH,
            value_type="str",
            cv_source_term_key="cell_methods",
            expected_term=SimpleNamespace(
                cell_methods=["area: point", "area: time: mean"]
            ),
        )

    assert all(result_passed(result) for result in results)


def test_registry_attribute_reports_allowed_values_from_expected_list(tmp_path):
    with Dataset(tmp_path / "attributes.nc", "w") as dataset:
        variable = dataset.createVariable("tas", "f4")
        variable.long_name = "Wrong name"
        results = checker.check_attribute_suite(
            dataset,
            "long_name",
            var_name="tas",
            severity=BaseCheck.HIGH,
            value_type="str",
            cv_source_term_key="long_name",
            expected_term=SimpleNamespace(
                long_name=["Air Temperature", "Atmospheric Temperature"]
            ),
        )

    assert not result_passed(results[-1])
    assert (
        "one of ['Air Temperature', 'Atmospheric Temperature']" in (results[-1].msgs[0])
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"pattern": "["}, "Invalid regex"),
        ({"pattern": r"Amon", "constant": "Amon"}, "Multiple mutually exclusive"),
        (
            {"cv_source_collection_key": "description"},
            "provided without cv_source_collection",
        ),
        ({"threshold": 1}, "requires 'is_above_threshold'"),
        ({"threshold": 1, "is_above_threshold": True}, "is not numeric"),
    ],
)
def test_attribute_suite_reports_invalid_rules_and_values(
    tmp_path, kwargs, expected_fragment
):
    with Dataset(tmp_path / "invalid-rule.nc", "w") as dataset:
        dataset.test_attribute = "not numeric"
        results = checker.check_attribute_suite(
            dataset,
            "test_attribute",
            severity=BaseCheck.HIGH,
            **kwargs,
        )

    assert expected_fragment in results[-1].msgs[0]


def test_attribute_suite_covers_constant_threshold_variable_and_positive_rules(
    tmp_path,
):
    with Dataset(tmp_path / "rules.nc", "w") as dataset:
        dataset.constant = "expected"
        dataset.threshold = np.float32(2.0)
        dataset.references = "lat missing"
        dataset.positive = -1
        dataset.createVariable("lat", "f8")

        constant = checker.check_attribute_suite(
            dataset, "constant", BaseCheck.HIGH, constant="expected"
        )
        threshold = checker.check_attribute_suite(
            dataset,
            "threshold",
            BaseCheck.HIGH,
            threshold=1,
            is_above_threshold=True,
        )
        references = checker.check_attribute_suite(
            dataset, "references", BaseCheck.HIGH, as_variable=True
        )
        positive = checker.check_attribute_suite(
            dataset, "positive", BaseCheck.HIGH, is_positive=True
        )

    assert result_passed(constant[-1])
    assert result_passed(threshold[-1])
    assert "Referenced variables not found: ['missing']" in references[-1].msgs[0]
    assert "Expected positive value" in positive[-1].msgs[0]


def test_attribute_suite_handles_required_and_optional_not_applicable_values(tmp_path):
    with Dataset(tmp_path / "na-value.nc", "w") as dataset:
        dataset.parent_experiment_id = "no parent"
        optional = checker.check_attribute_suite(
            dataset,
            "parent_experiment_id",
            BaseCheck.HIGH,
            is_required=False,
            na_value="no parent",
        )
        required = checker.check_attribute_suite(
            dataset,
            "parent_experiment_id",
            BaseCheck.HIGH,
            is_required=True,
            na_value="no parent",
        )

    assert optional == []
    assert not result_passed(required[0])
    assert "real value is expected" in required[0].msgs[0]


def test_attribute_suite_dynamic_requirement_fails_safe(tmp_path, monkeypatch):
    import checks.utils

    monkeypatch.setattr(
        checks.utils,
        "has_parent_experiment_for_ds",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )
    with Dataset(tmp_path / "dynamic.nc", "w") as dataset:
        results = checker.check_attribute_suite(
            dataset,
            "parent_experiment_id",
            BaseCheck.HIGH,
            is_required="has_parent_experiment",
        )
    assert not result_passed(results[0])


def test_attribute_suite_vocabulary_lookup_failures_are_results(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "_ESGVOC_AVAILABLE", True)
    monkeypatch.setattr(
        checker.voc,
        "get_all_terms_in_collection",
        lambda **_kwargs: [],
    )
    with Dataset(tmp_path / "vocabulary.nc", "w") as dataset:
        dataset.institution = "Institution"
        empty = checker.check_attribute_suite(
            dataset,
            "institution",
            BaseCheck.HIGH,
            cv_source_collection="institution_id",
            cv_source_collection_key="description",
            project_name="cmip6",
        )
    assert "empty or could not be retrieved" in empty[-1].msgs[0]

    monkeypatch.setattr(checker, "_ESGVOC_AVAILABLE", False)
    with Dataset(tmp_path / "unavailable.nc", "w") as dataset:
        dataset.source_id = "MODEL"
        unavailable = checker.check_attribute_suite(
            dataset,
            "source_id",
            BaseCheck.HIGH,
            cv_source_collection="source_id",
            project_name="cmip6",
        )
    assert unavailable[-1].msgs == ["ESGVOC library not available."]


def test_gregorian_calendar_is_accepted_with_advisory(tmp_path):
    with Dataset(tmp_path / "calendar.nc", "w") as dataset:
        time = dataset.createVariable("time", "f8")
        time.calendar = "gregorian"
        results = checker.check_attribute_suite(
            dataset,
            "calendar",
            BaseCheck.MEDIUM,
            var_name="time",
            enum=("standard", "gregorian", "proleptic_gregorian"),
        )
    assert result_passed(results[-1])
    assert "proleptic_gregorian" in results[-1].msgs[0]
