"""Unit coverage for variable existence checks."""

from compliance_checker.base import BaseCheck
from compliance_checker.tests import BaseTestCase
from compliance_checker.tests.resources import STATIC_FILES

from checks.variable_checks.check_variable_existence import check_variable_existence


class TestVariableExistence(BaseTestCase):
    def test_check_variable_exists(self):
        dataset = self.load_dataset(STATIC_FILES["climatology"])
        results = check_variable_existence(
            dataset,
            "temperature",
            severity=BaseCheck.HIGH,
        )

        assert len(results) == 1
        self.assert_result_is_good(results[0])
        assert results[0].name.startswith("[VAR001]")
        assert results[0].weight == BaseCheck.HIGH

    def test_check_variable_exists_fails(self):
        dataset = self.load_dataset(STATIC_FILES["climatology"])
        results = check_variable_existence(
            dataset,
            "missing",
            severity=BaseCheck.MEDIUM,
        )

        assert len(results) == 1
        self.assert_result_is_bad(results[0])
        assert results[0].name.startswith("[VAR001]")
        assert results[0].msgs == ["Variable 'missing' is missing."]
