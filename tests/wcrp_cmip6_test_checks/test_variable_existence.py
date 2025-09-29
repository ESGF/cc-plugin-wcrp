from ...checks.variable_checks import check_variable_existence as checker
from compliance_checker.tests import BaseTestCase
from compliance_checker.tests.resources import STATIC_FILES


class TestVariableExistence(BaseTestCase):
    def test_check_variable_exists(self):
        dataset = self.load_dataset(STATIC_FILES["climatology"])
        check_id = "variable_exists"

        results = checker.check_variable_existence(dataset, "temperature", check_id=check_id)

        assert len(results) == 1
        self.assert_result_is_good(results[0])
        assert results[0].check_id == check_id

    def test_check_variable_exists_fails(self):
        dataset = self.load_dataset(STATIC_FILES["climatology"])
        check_id = "variable_does_not_exist"
        var_name = "missing"

        results = checker.check_variable_existence(dataset, var_name, check_id=check_id)

        assert len(results) == 1
        self.assert_result_is_bad(results[0])
        assert results[0].check_id == check_id
        assert results[0].msgs[0] == f"Variable '{var_name}' is missing."
