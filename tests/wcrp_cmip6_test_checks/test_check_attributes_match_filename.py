
#!/usr/bin/env python
"""
Test for check_attributes_match_filename.py
Author: Ayoub NACHITE ''IPSL''
"""

import os
from netCDF4 import Dataset
from compliance_checker.base import BaseCheck
from compliance_checker.checks.consistency_checks import check_attributes_match_filename as checker
from compliance_checker.tests import BaseTestCase

class TestCheckAttributesMatchFilename(BaseTestCase):

    def test_check_filename_vs_global_attrs(self):
        file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", "CMIP6", "CMIP", "IPSL", "IPSL-CM5A2-INCA", "historical", "r1i1p1f1", "Amon", "pr", "gr", "v20240619", "pr_Amon_IPSL-CM5A2-INCA_historical_r1i1p1f1_gr_185001-201412.nc"
        ))
        dataset = Dataset(file_path, mode="r")
        results = checker.check_filename_vs_global_attrs(dataset, severity=BaseCheck.HIGH)
        assert len(results) == 1
        for res in results:
            self.assert_result_is_good(res) 

