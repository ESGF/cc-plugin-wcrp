"""
WCRP Data Plausibility Checks Package

This package contains all checks related to the physical and statistical plausibility
of NetCDF data for WCRP projects (CMIP, CORDEX, C3S-ATLAS, etc.).
Each module implements a specific DATAxxx check.

Modules included:
- check_nan_inf_v451.py
- check_fill_missing_v453.py
- check_constant_v453.py
- detect_physically_impossible_outlier_v452.py
- check_spatial_statistical_outliers_v454.py
- check_chunk_size_v141.py
"""

from checks.data_plausibility_checks.check_nan_inf import check_nan_inf
from checks.data_plausibility_checks.check_fill_missing import check_fillvalues_timeseries
from checks.data_plausibility_checks.check_constant import check_constants
from checks.data_plausibility_checks.detect_physically_impossible_outlier import check_outliers
from checks.data_plausibility_checks.check_spatial_statistical_outliers import check_spatial_statistical_outliers
from checks.data_plausibility_checks.check_chunk_size import check_chunk_size
__all__ = [
    "check_nan_inf",
    "check_fillvalues_timeseries",
    "check_constants",
    "check_physically_impossible_outlier",
    "check_spatial_statistical_outliers",
    "check_chunk_size",
]

