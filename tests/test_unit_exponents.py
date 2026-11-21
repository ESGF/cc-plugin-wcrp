"""Exponent spelling must not relax unit symbols or numerical scales."""

import pytest
from netCDF4 import Dataset

from checks.units import normalize_unit_exponents
from checks.coordinate_checks.utils import compare_units
from checks.attribute_checks.check_attribute_suite import check_attribute_suite
from checks.coordinate_checks.validation import Findings, check_attributes
from plugins.wcrp_schema import ExponentSpellingRule
from pydantic import ValidationError


CASES = [
    ("1e-5 km2 1E10 N 1E+02 degC", "1e-5 km2 1e10 N 1e2 degC", True),
    ("1e-5 km2 1E11 N 1E+02 degC", "1e-5 km2 1e10 N 1e2 degC", False),
    ("1e-5 km2 1E10 N 1E+002 degC", "1e-5 km2 1e10 N 1e2 degC", False),
    ("1E-09", "1e-9", True),
    ("1e-4", "1E-04", True),
    ("1e3", "1E+03", True),
    ("1E06", "1e+6", True),
    ("1.5E-09 kg m-2", "1.5e-9 kg m-2", True),
    ("1.5E-009 kg m-2", "1.5e-9 kg m-2", False),
    ("1E00", "1e0", True),
    ("1E000", "1e0", False),
    ("1E12", "1e+12", True),
    ("1E012", "1e12", False),
    ("1E-012", "1e-12", False),
    (".5E03 m", ".5e3 m", True),
    ("1E-09 N m-1", "1e-9 n M-2", False),
    ("N m-1", "n M-2", False),
    ("1e-3", "1e-6", False),
    ("2e3", "1e3", False),
    ("1.0e3", "1e3", False),
    ("m", "km", False),
    ("1e0003", "1E0003", False),
    ("1e0003", "1e0003", True),
]


@pytest.mark.parametrize("actual,expected", [
    ("1e-3", "1E-03"),
    ("1E-09", "1e-9"),
    ("1e3", "1E+03"),
    ("1E06", "1e+6"),
    ("1.5E-09 kg m-2", "1.5e-9 kg m-2"),
    ("1e-5 km2 1E10 N 1E+02 degC", "1e-5 km2 1e10 N 1e2 degC"),
])
def test_exponent_spellings_are_equivalent_in_udunits(actual, expected):
    try:
        cf_units = pytest.importorskip("cf_units", exc_type=ImportError)
    except OSError as exc:
        pytest.skip(f"UDUNITS shared library is unavailable: {exc}")

    # Parse the original strings, not our normalized versions. Convertibility
    # alone is insufficient: equivalent spellings must preserve the scale too.
    actual_unit = cf_units.Unit(actual)
    expected_unit = cf_units.Unit(expected)
    assert actual_unit == expected_unit
    for value in (0.0, 1.0, -2.5):
        assert actual_unit.convert(value, expected_unit) == pytest.approx(value)
    assert cf_units.Unit("1e-3") != cf_units.Unit("1e-6")


@pytest.mark.parametrize("actual,expected", [
    ("1E003", "1e3"),
    ("1.0e9", "1e9"),
    ("1.000e+00008", "1E8"),
    ("1E005", "1e005"),
])
def test_stricter_spelling_is_rejected_despite_udunits_equivalence(tmp_path, actual, expected):
    try:
        cf_units = pytest.importorskip("cf_units", exc_type=ImportError)
    except OSError as exc:
        pytest.skip(f"UDUNITS shared library is unavailable: {exc}")

    assert cf_units.Unit(actual) == cf_units.Unit(expected)
    assert normalize_unit_exponents(actual) != normalize_unit_exponents(expected)
    assert not compare_units(actual, expected, qualifier="required")[0]

    # Excessive padding and changed mantissas remain High even with the spelling
    # exception enabled, for both data-variable and coordinate attributes.
    policy = ExponentSpellingRule(enabled=True, severity="L")
    with Dataset(tmp_path / "excessive-padding.nc", "w") as ds:
        var = ds.createVariable("tas", "f8")
        var.units = actual
        attributes = check_attribute_suite(
            ds, "units", 3, var_name="tas", cv_source_term_key="units",
            expected_term={"units": expected}, unit_exponent_policy=policy,
        )
        findings = Findings({"attributes": 3}, unit_exponent_policy=policy)
        check_attributes(findings, var, "tas", {"units": expected}, fallback_long_name=False)
        for results in (attributes, findings.results()):
            failures = [result for result in results if result.msgs]
            assert len(failures) == 1
            assert failures[0].weight == 3


@pytest.mark.parametrize("actual,expected,matches", CASES)
def test_exponents_in_coordinate_and_data_variable_units(tmp_path, actual, expected, matches):
    assert (normalize_unit_exponents(actual) == normalize_unit_exponents(expected)) == matches
    assert compare_units(actual, expected, qualifier="required")[0] == matches
    with Dataset(tmp_path / "units.nc", "w") as ds:
        variable = ds.createVariable("tas", "f4")
        variable.units = actual
        results = check_attribute_suite(
            ds, "units", 3, var_name="tas", cv_source_term_key="units",
            expected_term={"units": ["other", expected]},
            unit_exponent_policy=ExponentSpellingRule(enabled=False),
        )
        failures = [message for result in results for message in result.msgs]
        assert bool(failures) == (actual != expected)


@pytest.mark.parametrize("enabled,severity,weight", [(True, "L", 1), (True, "M", 2), (False, "L", 3)])
@pytest.mark.parametrize("actual,expected", [
    ("1E-09", "1e-9"), ("1e-9", "1e-9"), ("1e-6", "1e-9"),
    ("1E09", "1e9"), ("1E-09", "1E09"), ("1E-09", "1e9"),
    ("1E009", "1E09"), ("1E009", "1e9"), ("1E0000009", "1e9"),
    ("1E-009", "1e-9"),
    ("1E012", "1e12"), ("1E-012", "1e-12"), ("1E12", "1e12"),
])
def test_configured_exponent_results(tmp_path, enabled, severity, weight, actual, expected):
    policy = ExponentSpellingRule(enabled=enabled, severity=severity)
    with Dataset(tmp_path / "units.nc", "w") as ds:
        var = ds.createVariable("tas", "f8")
        var.units = actual
        attributes = check_attribute_suite(
            ds, "units", 3, var_name="tas", cv_source_term_key="units",
            expected_term={"units": ["other", expected]}, unit_exponent_policy=policy,
        )
        findings = Findings({"attributes": 3}, unit_exponent_policy=policy)
        check_attributes(findings, var, "tas", {"units": expected}, fallback_long_name=False)
        for results in (attributes, findings.results()):
            failures = [r for r in results if r.msgs]
            if actual == expected:
                assert not failures
            else:
                assert len(failures) == 1
                equivalent = normalize_unit_exponents(actual) == normalize_unit_exponents(expected)
                assert failures[0].weight == (weight if equivalent else 3)


@pytest.mark.parametrize("severity", ["L", "M"])
@pytest.mark.parametrize("actual,matches", [
    ("1e-5 km2 1E10 N 1E+02 degC", True),
    ("1e-5 km2 1E11 N 1E+02 degC", False),
    ("1e-5 km2 1E10 N 1E+002 degC", False),
])
def test_multiple_exponents_result_severity(tmp_path, severity, actual, matches):
    expected = "1e-5 km2 1e10 N 1e2 degC"
    policy = ExponentSpellingRule(severity=severity)
    with Dataset(tmp_path / "multiple_exponents.nc", "w") as ds:
        var = ds.createVariable("tas", "f8")
        var.units = actual
        attributes = check_attribute_suite(
            ds, "units", 3, var_name="tas", cv_source_term_key="units",
            expected_term={"units": expected}, unit_exponent_policy=policy,
        )
        findings = Findings({"attributes": 3}, unit_exponent_policy=policy)
        check_attributes(findings, var, "tas", {"units": expected}, fallback_long_name=False)
        for results in (attributes, findings.results()):
            failures = [result for result in results if result.msgs]
            assert len(failures) == 1
            assert failures[0].weight == ({"L": 1, "M": 2}[severity] if matches else 3)


@pytest.mark.parametrize("text", ["N m-1", "n M-2", "name1E03", "1E03name", "mE03", "1E-", "1E03.2"])
def test_non_numeric_tokens_are_untouched(text):
    assert normalize_unit_exponents(text) == text


def test_other_attributes_keep_literal_comparison(tmp_path):
    with Dataset(tmp_path / "comment.nc", "w") as ds:
        variable = ds.createVariable("tas", "f4")
        variable.comment = "1E-09"
        results = check_attribute_suite(
            ds, "comment", 1, var_name="tas", cv_source_term_key="comment",
            expected_term={"comment": ["1e-9"]},
        )
        assert any(result.msgs for result in results)


def test_exponent_exception_cannot_be_high():
    with pytest.raises(ValidationError):
        ExponentSpellingRule(severity="H")


def test_disabled_coordinate_attributes_do_not_report_spelling(tmp_path):
    with Dataset(tmp_path / "disabled.nc", "w") as ds:
        var = ds.createVariable("lat", "f8")
        var.units = "1E-09"
        findings = Findings({"identity": 3}, unit_exponent_policy=ExponentSpellingRule())
        check_attributes(findings, var, "lat", {"units": "1e-9"})
        assert not any(result.msgs for result in findings.results())
