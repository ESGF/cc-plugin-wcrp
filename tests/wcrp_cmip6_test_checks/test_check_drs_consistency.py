
#!/usr/bin/env python
"""
Test for check_drs_consistency.py
Author: Ayoub NACHITE ''IPSL''
"""

import os
from netCDF4 import Dataset
from compliance_checker.base import BaseCheck
from ...checks.consistency_checks import check_drs_consistency as checker
from compliance_checker.tests import BaseTestCase

class TestCheckDrsConsistency(BaseTestCase):

    def test_check_attributes_match_directory_structure(self):
        file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", "CMIP6", "CMIP",  "IPSL", "IPSL-CM5A2-INCA", "historical", "r1i1p1f1", "Amon", "pr", "gr", "v20240619", "pr_Amon_IPSL-CM5A2-INCA_historical_r1i1p1f1_gr_185001-201412.nc"
        ))
        dataset = Dataset(file_path, mode="r")
        results = checker.check_attributes_match_directory_structure(dataset, severity=BaseCheck.MEDIUM, project_id="CMIP6")
        assert len(results) == 1
        for res in results:
            self.assert_result_is_good(res) 


    def test_check_filename_matches_directory_structure(self):
        file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", "CMIP6", "CMIP", "IPSL", "IPSL-CM5A2-INCA", "historical", "r1i1p1f1", "Amon", "pr", "gr", "v20240619", "pr_Amon_IPSL-CM5A2-INCA_historical_r1i1p1f1_gr_185001-201412.nc"
        ))
        dataset = Dataset(file_path, mode="r")
        results = checker.check_filename_matches_directory_structure(dataset, severity=BaseCheck.MEDIUM, project_id="CMIP6")
        assert len(results) == 1
        for res in results:
            self.assert_result_is_good(res) 

