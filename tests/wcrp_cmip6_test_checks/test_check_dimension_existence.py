#!/usr/bin/env python
"""
Test for check_dimension_existence.py
Author : Iliana GHAZALI ''Meteo France''
"""

from compliance_checker.base import BaseCheck
from ...checks.checks.dimension_checks import check_dimension_existence as checker
from compliance_checker.tests import BaseTestCase
from compliance_checker.tests.resources import STATIC_FILES


class TestDimExistence(BaseTestCase):

    def test_check_dimension_exists(self):
        # Given
        dataset = self.load_dataset(STATIC_FILES["2d-regular-grid"])
        # When
        results = checker.check_dimension_existence(dataset, "lat", BaseCheck.MEDIUM)
        # Then
        assert len(results) == 1
        self.assert_result_is_good(results[0])

    def test_check_dimension_does_not_exists(self):
        # Given
        dataset = self.load_dataset(STATIC_FILES["2d-regular-grid"])
        dim_name = "not_existing"
        # When
        results = checker.check_dimension_existence(dataset, dim_name, BaseCheck.MEDIUM)
        # Then
        assert len(results) == 1
        result = results[0]
        self.assert_result_is_bad(result)
        assert (
                result.msgs[0]
                == f"Dimension '{dim_name}' is missing."
        )
