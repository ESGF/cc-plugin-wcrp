from types import SimpleNamespace

import cftime
import numpy as np

from checks.time_checks.check_time_cordex_cmip6 import check_time_chunking


def test_monthly_360_day_time_chunking_uses_numeric_offset():
    units = "days since 2000-01-01 00:00:00"
    calendar = "360_day"
    times = cftime.date2num(
        [
            cftime.Datetime360Day(2001, 1, 16),
            cftime.Datetime360Day(2010, 12, 16),
        ],
        units,
        calendar,
    )
    checker = SimpleNamespace(
        frequency="mon",
        time=np.asarray(times),
        calendar=calendar,
        timeunits=units,
        cell_methods="time: mean",
        _get_attr=lambda *_args: None,
    )

    result = check_time_chunking(checker)

    assert result[0].msgs == []
