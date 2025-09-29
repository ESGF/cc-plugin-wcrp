
#!/usr/bin/env python
"""
Test for check_frequency_table_consistency.py
Author: Ayoub NACHITE ''IPSL''
"""

import os
from netCDF4 import Dataset
from compliance_checker.base import BaseCheck
from ...checks.consistency_checks import check_frequency_table_consistency as checker
from compliance_checker.tests import BaseTestCase

class TestCheckFrequencyTableConsistency(BaseTestCase):

    def test_check_frequency_table_id_consistency(self):
        file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", "CMIP6", "CMIP", "IPSL", "IPSL-CM5A2-INCA",
            "historical", "r1i1p1f1", "Amon", "pr", "gr", "v20240619",
            "pr_Amon_IPSL-CM5A2-INCA_historical_r1i1p1f1_gr_185001-201412.nc"
        ))

        mapping_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "mapping.toml"
        ))

        dataset = Dataset(file_path, mode="r")
        results = checker.check_frequency_table_id_consistency(dataset, mapping_file, severity=BaseCheck.MEDIUM)
        assert len(results) == 1
        for res in results:
            self.assert_result_is_good(res) 

