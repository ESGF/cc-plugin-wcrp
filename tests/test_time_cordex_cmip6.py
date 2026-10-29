from types import SimpleNamespace

import cftime
import numpy as np
import pytest

from checks.time_checks.check_time_cordex_cmip6 import (
    check_calendar,
    check_time_chunking,
    check_time_range,
    check_time_units,
)


def _passed(result):
    value = result.value
    return value[0] == value[1] if isinstance(value, tuple) else bool(value)


def _checker(**overrides):
    values = {
        "frequency": "mon",
        "time": np.asarray([15.0, 45.0]),
        "calendar": "standard",
        "timeunits": "days since 2000-01-01",
        "cell_methods": "time: mean",
        "drs_fn": {"time_range": "200001-200002"},
        "_get_attr": lambda *_args: None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def test_cordex_time_checks_skip_fixed_and_reject_unknown_frequency():
    fixed = _checker(frequency="fx", time=None)
    assert _passed(check_time_chunking(fixed)[0])
    assert _passed(check_time_range(fixed)[0])

    unsupported = _checker(frequency="yr")
    assert "not supported" in check_time_chunking(unsupported)[0].msgs[0]
    assert "not supported" in check_time_range(unsupported)[0].msgs[0]


def test_cordex_time_checks_report_missing_prerequisites_without_conversion_errors():
    missing_time = _checker(time=None)
    for check in (check_time_chunking, check_time_range):
        result = check(missing_time)[0]
        assert result.msgs == ["Coordinate variable 'time' not found in file."]

    missing_metadata = _checker(calendar=None, timeunits=None)
    result = check_time_chunking(missing_metadata)[0]
    assert result.msgs == [
        "'time' variable has no 'calendar' attribute.",
        "'time' variable has no 'units' attribute.",
    ]


@pytest.mark.parametrize(
    ("frequency", "dates", "label"),
    [
        ("mon", [(2000, 1, 16, 0), (2000, 2, 15, 0)], "200001-200002"),
        ("day", [(2000, 1, 1, 12), (2000, 1, 2, 12)], "20000101-20000102"),
        ("6hr", [(2000, 1, 1, 3), (2000, 1, 1, 9)], "200001010300-200001010900"),
    ],
)
def test_cordex_time_range_formats_supported_frequencies(frequency, dates, label):
    units = "hours since 2000-01-01 00:00:00"
    times = cftime.date2num(
        [cftime.DatetimeGregorian(*date) for date in dates], units, "standard"
    )
    checker = _checker(
        frequency=frequency,
        time=np.asarray(times),
        timeunits=units,
        drs_fn={"time_range": label},
    )
    assert _passed(check_time_range(checker)[0])

    checker.drs_fn["time_range"] = "wrong"
    assert f"Expected time_range '{label}'" in check_time_range(checker)[0].msgs[0]


@pytest.mark.parametrize(
    ("calendar", "passes", "fragment"),
    [
        ("standard", True, None),
        ("proleptic_gregorian", True, None),
        ("gregorian", False, "deprecated"),
        ("360_day", False, "inherited from the driving model"),
    ],
)
def test_cordex_calendar_policy(calendar, passes, fragment):
    result = check_calendar(_checker(calendar=calendar))[0]
    assert _passed(result) is passes
    if fragment:
        assert fragment in result.msgs[0]


def test_cordex_time_unit_policy_and_no_time_short_circuit():
    allowed = _checker(timeunits="days since 1850-01-01T00:00:00Z")
    assert _passed(check_time_units(allowed)[0])
    assert check_time_units(_checker(time=None, timeunits="bad")) == []

    invalid = check_time_units(_checker(timeunits="hours since 2000-01-01"))[0]
    assert not _passed(invalid)
    assert "differs from the allowed values" in invalid.msgs[0]


def test_cordex_time_metadata_reports_missing_prerequisites_without_owner():
    missing_time = _checker(time=None, calendar=None, timeunits=None)

    calendar = check_calendar(
        missing_time, time_presence_delegated=False
    )[0]
    units = check_time_units(
        missing_time, time_presence_delegated=False
    )[0]

    assert "Coordinate variable 'time' is missing" in calendar.msgs[0]
    assert "Coordinate variable 'time' is missing" in units.msgs[0]
