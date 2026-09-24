"""Joint TIME001/TIME003 coverage for every configured time frequency."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import cftime
import pytest
import toml
from netCDF4 import Dataset

from checks.time_checks.check_time_range_vs_filename import (
    check_time_range_vs_filename,
)
from checks.time_checks.check_time_squareness import check_time_squareness


DRS_CONFIG = Path(__file__).parents[1] / "plugins/cmip7/config/wcrp/drs.toml"
PRECISION = toml.load(DRS_CONFIG)["drs"]["time_range"]["label_precision"]
INCREMENTS_CONFIG = (
    Path(__file__).parents[1]
    / "plugins/cmip7/config/wcrp/mappings/table_id_to_time_increment.toml"
)
INCREMENTS = toml.load(INCREMENTS_CONFIG)["time_increment_mapping"]

# frequency: (increment, unit, point sample, climatological bounds, table_id)
FREQUENCIES = {
    "dec": (10, "years", False, False, "None"),
    "yr": (1, "years", False, False, "None"),
    "yrPt": (1, "years", True, False, "None"),
    "sem": (3, "months", False, False, "None"),
    "mon": (1, "months", False, False, "None"),
    "monC": (1, "months", False, True, "Amon"),
    "monClim": (1, "months", False, True, "Aclim"),
    "monPt": (1, "months", True, False, "None"),
    "day": (1, "days", False, False, "None"),
    "6hr": (6, "hours", False, False, "None"),
    "6hrPt": (6, "hours", True, False, "None"),
    "3hr": (3, "hours", False, False, "None"),
    "3hrPt": (3, "hours", True, False, "None"),
    "1hr": (1, "hours", False, False, "None"),
    "1hrCM": (1, "hours", False, True, "E1hrClimMon"),
    "1hrPt": (1, "hours", True, False, "None"),
    "subhr": (30, "minutes", False, False, "None"),
    "subhrPt": (30, "minutes", True, False, "None"),
}


def _advance(date, amount, unit):
    if unit == "years":
        return date.replace(year=date.year + amount)
    if unit == "months":
        ordinal = date.year * 12 + date.month - 1 + amount
        return date.replace(year=ordinal // 12, month=ordinal % 12 + 1)
    return date + timedelta(**{unit: amount})


def _label(date, precision):
    fields = {
        "yyyy": (date.year,),
        "yyyyMM": (date.year, date.month),
        "yyyyMMdd": (date.year, date.month, date.day),
        "yyyyMMddhhmm": (
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
        ),
        "yyyyMMddhhmmss": (
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second,
        ),
    }[precision]
    widths = (4, 2, 2, 2, 2, 2)
    return "".join(f"{value:0{width}d}" for value, width in zip(fields, widths))


def _time_definition(frequency, calendar="360_day"):
    increment, unit, point, climatology, table_id = FREQUENCIES[frequency]
    units = "days since 1800-01-01 00:00:00"

    if climatology:
        bounds_dates = [
            (
                cftime.datetime(2000, 1, 1, calendar=calendar),
                cftime.datetime(2101, 1, 1, calendar=calendar),
            ),
            (
                cftime.datetime(2000, 2, 1, calendar=calendar),
                cftime.datetime(2101, 2, 1, calendar=calendar),
            ),
        ]
        precision = PRECISION["climatology"]
        start_label = _label(bounds_dates[0][0], precision)
        # An exact first-of-month endpoint represents the preceding month.
        end_label = "210101"
    else:
        start = (
            cftime.datetime(1990, 1, 1, calendar=calendar)
            if frequency == "dec"
            else cftime.datetime(2000, 1, 1, calendar=calendar)
        )
        starts = [start, _advance(start, increment, unit)]
        bounds_dates = [
            (interval_start, _advance(interval_start, increment, unit))
            for interval_start in starts
        ]

    bounds = [
        [
            cftime.date2num(begin, units=units, calendar=calendar),
            cftime.date2num(end, units=units, calendar=calendar),
        ]
        for begin, end in bounds_dates
    ]
    times = [pair[0] if point else 0.5 * (pair[0] + pair[1]) for pair in bounds]

    if not climatology:
        dates = cftime.num2date(times, units=units, calendar=calendar)
        precision = PRECISION[frequency]
        start_label = _label(dates[0], precision)
        end_label = _label(dates[-1], precision)

    return table_id, units, times, bounds, start_label, end_label, climatology, point


def _make_file(tmp_path, frequency, *, calendar="360_day", wrong_end=False):
    (
        table_id,
        units,
        times,
        bounds,
        start_label,
        end_label,
        climatology,
        point,
    ) = _time_definition(frequency, calendar)
    filename_end = start_label if wrong_end else end_label
    path = tmp_path / f"tas_{frequency}_test_{start_label}-{filename_end}.nc"
    dataset = Dataset(path, "w")
    dataset.frequency = frequency
    dataset.table_id = table_id
    dataset.variable_id = "tas"
    dataset.createDimension("time", 2)
    time = dataset.createVariable("time", "f8", ("time",))
    time.units = units
    time.calendar = calendar
    time[:] = times
    bounds_name = None
    if not point:
        dataset.createDimension("bnds", 2)
        bounds_name = "climatology_bnds" if climatology else "time_bnds"
        if climatology:
            time.climatology = bounds_name
        else:
            time.bounds = bounds_name
        dataset.createVariable(bounds_name, "f8", ("time", "bnds"))[:] = bounds
    data = dataset.createVariable("tas", "f4", ("time",))
    data.cell_methods = "time: point" if point else "time: mean"
    return dataset, bounds_name


def _messages(results):
    return [message for result in results for message in result.msgs]


def _time003(dataset):
    return check_time_range_vs_filename(
        dataset,
        precision_by_frequency=PRECISION,
        climatology_suffix="",
    )


def test_frequency_matrix_matches_configured_precisions():
    frequencies_with_increments = {key.split(".", 1)[1] for key in INCREMENTS}
    assert set(FREQUENCIES) == (
        (set(PRECISION) - {"climatology"}) & frequencies_with_increments
    )


@pytest.mark.parametrize(
    ("calendar", "expected"),
    [("360_day", "1995-2005"), ("gregorian", "1995-2004")],
)
def test_decadal_mean_uses_time_value_year_labels(calendar, expected):
    _, _, _, _, start, end, _, _ = _time_definition("dec", calendar)
    assert f"{start}-{end}" == expected


def test_fixed_frequency_has_no_time_range_checks(tmp_path):
    path = tmp_path / "orog_fx_test.nc"
    with Dataset(path, "w") as dataset:
        dataset.frequency = "fx"
        dataset.createVariable("orog", "f4")
        assert check_time_squareness(dataset) == []
        assert _messages(_time003(dataset)) == []


@pytest.mark.parametrize("frequency", FREQUENCIES, ids=FREQUENCIES)
@pytest.mark.parametrize("calendar", ["360_day", "gregorian"])
def test_time001_and_time003_pass_for_supported_frequency(
    tmp_path,
    frequency,
    calendar,
):
    with _make_file(tmp_path, frequency, calendar=calendar)[0] as dataset:
        assert _messages(check_time_squareness(dataset)) == []
        assert _messages(_time003(dataset)) == []


@pytest.mark.parametrize("frequency", FREQUENCIES, ids=FREQUENCIES)
@pytest.mark.parametrize("calendar", ["360_day", "gregorian"])
def test_bad_time_or_bounds_fails_time001_only(tmp_path, frequency, calendar):
    dataset, bounds_name = _make_file(tmp_path, frequency, calendar=calendar)
    with dataset:
        if bounds_name:
            dataset.variables[bounds_name][0, 1] += 1.0
            expected_message = (
                "midpoint"
                if FREQUENCIES[frequency][3]
                else "time-bounds interval"
            )
        else:
            # TIME003 only compares the endpoints at filename precision. A
            # sub-second displacement is nevertheless visible to TIME001.
            dataset.variables["time"][1] += 0.25 / 86400.0
            expected_message = "First incident at index 1"
        assert any(
            expected_message in message
            for message in _messages(check_time_squareness(dataset))
        )
        assert _messages(_time003(dataset)) == []


def test_irregular_gregorian_decadal_bounds_fail_with_unchanged_midpoint(tmp_path):
    dataset, bounds_name = _make_file(tmp_path, "dec", calendar="gregorian")
    with dataset:
        bounds = dataset.variables[bounds_name]
        bounds[1, 0] = bounds[1, 0] - 1.0
        bounds[1, 1] = bounds[1, 1] + 1.0

        found = _messages(check_time_squareness(dataset))

        assert any("First incident at index 1" in message for message in found)
        assert _messages(_time003(dataset)) == []


@pytest.mark.parametrize("frequency", FREQUENCIES, ids=FREQUENCIES)
@pytest.mark.parametrize("calendar", ["360_day", "gregorian"])
def test_wrong_filename_end_fails_time003_only(tmp_path, frequency, calendar):
    with _make_file(
        tmp_path,
        frequency,
        calendar=calendar,
        wrong_end=True,
    )[0] as dataset:
        assert _messages(check_time_squareness(dataset)) == []
        assert _messages(_time003(dataset))
