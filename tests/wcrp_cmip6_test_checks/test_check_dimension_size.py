#!/usr/bin/env python
"""
Test for check_dimension_size.py
Author : Iliana GHAZALI ''Meteo France''
"""

from compliance_checker.base import BaseCheck
from ...checks.dimension_checks import check_dimension_size as checker
from compliance_checker.tests import BaseTestCase
from compliance_checker.tests.resources import STATIC_FILES


class TestDimSize(BaseTestCase):

    # NOMINAL TEST CASES

    def test_check_dimension_size_is_equals_to(self):
        # Given
        dataset = self.load_dataset(STATIC_FILES["climatology"])
        # When
        results = checker.check_dimension_size_is_equals_to(dataset, "nv", 2, BaseCheck.MEDIUM)
        # Then
        assert len(results) == 1
        self.assert_result_is_good(results[0])

    def test_check_dimension_size_is_greater_than(self):
        # Given
        dataset = self.load_dataset(STATIC_FILES["climatology"])
        # When
        results = checker.check_dimension_size_is_strictly_greater_than(dataset, "time", 0, BaseCheck.MEDIUM)
        # Then
        assert len(results) == 1
        self.assert_result_is_good(results[0])

    # ERROR TEST CASES

    def test_check_dimension_size_is_equals_to_fails(self):
        # Given
        dataset = self.load_dataset(STATIC_FILES["climatology"])
        dim_name = "nv"
        # When
        expected_size = 3
        results = checker.check_dimension_size_is_equals_to(dataset, dim_name, expected_size, BaseCheck.MEDIUM)
        # Then
        assert len(results) == 1
        result = results[0]
        self.assert_result_is_bad(result)
        assert (
                result.msgs[0] ==
                f"Unexpected dimension size for '{dim_name}'. Got 2, expected {expected_size}."
        )

    def test_check_dimension_size_is_greater_than_fails(self):
        # Given
        dataset = self.load_dataset(STATIC_FILES["climatology"])
        dim_name = "time"
        lower_bound = 10
        # When
        results = checker.check_dimension_size_is_strictly_greater_than(dataset, dim_name, lower_bound, BaseCheck.MEDIUM)
        # Then
        assert len(results) == 1
        result = results[0]
        self.assert_result_is_bad(result)
        assert (
                result.msgs[0] ==
                f"Unexpected dimension size for '{dim_name}'. Got 1, expected strictly greater than {lower_bound}."
        )
