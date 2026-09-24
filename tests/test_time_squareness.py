from __future__ import annotations

import cftime
import numpy as np
import pytest
from netCDF4 import Dataset

from checks.time_checks.check_time_squareness import check_time_squareness
from checks.time_checks.check_time_bounds import check_time_bounds
from checks.time_checks.check_time_range_vs_filename import check_time_range_vs_filename


def messages(results):
    return [message for result in results for message in result.msgs]


def make_time_file(path, *, frequency, table_id, times, bounds, climatology=False):
    dataset = Dataset(path, "w")
    dataset.frequency = frequency
    dataset.table_id = table_id
    dataset.variable_id = "tas"
    dataset.createDimension("time", len(times))
    dataset.createDimension("bnds", 2)
    time = dataset.createVariable("time", "f8", ("time",))
    time.units = "days since 2000-01-01 00:00:00"
    time.calendar = "360_day"
    time[:] = times
    bounds_name = "climatology_bnds" if climatology else "time_bnds"
    if climatology:
        time.climatology = bounds_name
    else:
        time.bounds = bounds_name
    bounds_variable = dataset.createVariable(bounds_name, "f8", ("time", "bnds"))
    bounds_variable[:] = bounds
    data = dataset.createVariable("tas", "f4", ("time",))
    data.cell_methods = "time: mean"
    return dataset


def test_time_squareness_checks_regular_declared_bounds_midpoints(tmp_path):
    path = tmp_path / "tas_Amon_model_exp_r1i1p1f1_gn_200001-200002.nc"
    with make_time_file(
        path,
        frequency="mon",
        table_id="Amon",
        times=[15.0, 45.0],
        bounds=[[0.0, 30.0], [30.0, 60.0]],
    ) as dataset:
        assert messages(check_time_squareness(dataset)) == []


def test_time_squareness_reports_non_midpoint_regular_time(tmp_path):
    path = tmp_path / "tas_Amon_model_exp_r1i1p1f1_gn_200001-200002.nc"
    with make_time_file(
        path,
        frequency="mon",
        table_id="Amon",
        times=[14.0, 45.0],
        bounds=[[0.0, 30.0], [30.0, 60.0]],
    ) as dataset:
        found = messages(check_time_squareness(dataset))
    assert any("does not match the expected axis" in message for message in found)


def test_subdaily_non_point_cell_method_uses_interval_center(tmp_path):
    path = tmp_path / "tas_3hr_model_exp_r1i1p1f1_gn_200001010130-200001010130.nc"
    with make_time_file(
        path,
        frequency="3hr",
        table_id="3hr",
        times=[0.0625],
        bounds=[[0.0, 0.125]],
    ) as dataset:
        assert messages(check_time_squareness(dataset)) == []
        assert messages(check_time_range_vs_filename(dataset)) == []


@pytest.mark.parametrize(
    "frequency,hours,label",
    [
        ("1hr", 1, "200001010030-200001010130"),
        ("3hr", 3, "200001010130-200001010430"),
        ("6hr", 6, "200001010300-200001010900"),
    ],
)
def test_subdaily_mean_time_checks_agree_and_detect_wrong_spacing(
    tmp_path, frequency, hours, label
):
    path = tmp_path / f"tas_{frequency}_model_exp_r1i1p1f1_gn_{label}.nc"
    step = hours / 24.0
    with make_time_file(
        path,
        frequency=frequency,
        table_id=frequency,
        times=[step / 2, step * 1.5],
        bounds=[[0, step], [step, step * 2]],
    ) as dataset:
        assert messages(check_time_squareness(dataset)) == []
        assert messages(check_time_range_vs_filename(dataset)) == []
        # Preserve midpoint placement but shift the second interval and timestamp.
        dataset.variables["time"][1] += step
        dataset.variables["time_bnds"][1, :] += step
        assert any(
            "First incident at index 1" in msg
            for msg in messages(check_time_squareness(dataset))
        )


def test_climatology_uses_multi_year_bounds_midpoints(tmp_path):
    path = tmp_path / "tas_Amon_model_exp_r1i1p1f1_gn_200001-210012.nc"
    bounds = np.asarray([[0.0, 36030.0], [30.0, 36060.0]])
    with make_time_file(
        path,
        frequency="monC",
        table_id="Amon",
        times=bounds.mean(axis=1),
        bounds=bounds,
        climatology=True,
    ) as dataset:
        # An ordinary monthly reconstruction would expect [15, 45]. TIME001
        # must instead accept the centers of the multi-year climatology bounds.
        assert messages(check_time_squareness(dataset)) == []


def test_climatology_reports_non_midpoint_time(tmp_path):
    path = tmp_path / "tas_Amon_model_exp_r1i1p1f1_gn_200001-210012.nc"
    with make_time_file(
        path,
        frequency="monC",
        table_id="Amon",
        times=[18014.0, 18045.0],
        bounds=[[0.0, 36030.0], [30.0, 36060.0]],
        climatology=True,
    ) as dataset:
        found = messages(check_time_squareness(dataset))
    assert any("midpoint of its climatology interval" in message for message in found)


def test_time001_reports_count_numeric_and_decoded_first_incident(tmp_path):
    path = tmp_path / "tas_day_model_exp_r1i1p1f1_gn_20000101-20000103.nc"
    with Dataset(path, "w") as dataset:
        dataset.frequency = "day"
        dataset.table_id = "day"
        dataset.variable_id = "tas"
        dataset.createDimension("time", 3)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01 00:00:00"
        time.calendar = "standard"
        time[:] = [0.0, 2.0, 4.0]
        data = dataset.createVariable("tas", "f4", ("time",))
        data.cell_methods = "time: point"

        found = messages(check_time_squareness(dataset))

    assert len(found) == 1
    assert "2 time values" in found[0]
    assert "First incident at index 1" in found[0]
    assert "file contains 2.0 (decoded: 2000-01-03 00:00)" in found[0]
    assert "expected value is 1.0 (decoded: 2000-01-02 00:00)" in found[0]


def test_time002_reports_count_numeric_and_decoded_first_incident(tmp_path):
    path = tmp_path / "tas_day_model_exp_r1i1p1f1_gn_20000101-20000103.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 3)
        dataset.createDimension("bnds", 2)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01 00:00:00"
        time.calendar = "standard"
        time.bounds = "time_bnds"
        time[:] = [1.0, 4.0, 5.0]
        bounds = dataset.createVariable("time_bnds", "f8", ("time", "bnds"))
        bounds[:] = [[0.0, 2.0], [2.0, 3.0], [6.0, 7.0]]

        found = messages(check_time_bounds(dataset))

    assert len(found) == 1
    assert "2 'time' values lie outside declared bounds" in found[0]
    assert "First incident at index 1" in found[0]
    assert "file contains 4.0 (decoded: 2000-01-05 00:00)" in found[0]
    assert "bounds are [2.0, 3.0] (decoded: [2000-01-03 00:00" in found[0]


def test_time001_reports_multiple_irregular_bounds_with_first_incident(tmp_path):
    path = tmp_path / "tas_Amon_model_exp_r1i1p1f1_gn_200001-200003.nc"
    with make_time_file(
        path,
        frequency="mon",
        table_id="Amon",
        times=[15.0, 45.0, 75.0],
        bounds=[[0.0, 30.0], [29.0, 61.0], [59.0, 91.0]],
    ) as dataset:
        found = messages(check_time_squareness(dataset))

    assert len(found) == 1
    assert "2 time-bounds intervals do not match regular 1-month cells" in found[0]
    assert "First incident at index 1" in found[0]
    assert "file contains [29.0, 61.0] (decoded: [2000-01-30 00:00" in found[0]
    assert "expected interval is [30.0, 60.0] (decoded: [2000-02-01 00:00" in found[0]


@pytest.mark.parametrize("year", [1000, 3000])
def test_time_checks_support_gregorian_dates_outside_datetime64_ns_range(
    tmp_path, year
):
    path = tmp_path / f"tas_Amon_test_{year:04d}01-{year:04d}03.nc"
    units = "days since 0001-01-01 00:00:00"
    calendar = "proleptic_gregorian"
    edges = [
        cftime.datetime(year, month, 1, calendar=calendar)
        for month in range(1, 5)
    ]
    numeric_edges = cftime.date2num(edges, units=units, calendar=calendar)
    bounds_values = np.column_stack((numeric_edges[:-1], numeric_edges[1:]))

    with Dataset(path, "w") as dataset:
        dataset.frequency = "mon"
        dataset.table_id = "Amon"
        dataset.variable_id = "tas"
        dataset.createDimension("time", 3)
        dataset.createDimension("bnds", 2)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = units
        time.calendar = calendar
        time.bounds = "time_bnds"
        time[:] = bounds_values.mean(axis=1)
        dataset.createVariable("time_bnds", "f8", ("time", "bnds"))[:] = (
            bounds_values
        )
        data = dataset.createVariable("tas", "f4", ("time",))
        data.cell_methods = "time: mean"

        assert messages(check_time_squareness(dataset)) == []
        assert messages(check_time_range_vs_filename(dataset)) == []


def test_million_year_offsets_do_not_raise_datetime_overflow(tmp_path):
    path = tmp_path / "tas_day_test_00010101-00010102.nc"
    base = 360_000_000.0
    with Dataset(path, "w") as dataset:
        dataset.frequency = "day"
        dataset.table_id = "day"
        dataset.variable_id = "tas"
        dataset.createDimension("time", 2)
        dataset.createDimension("bnds", 2)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 0001-01-01 00:00:00"
        time.calendar = "360_day"
        time.bounds = "time_bnds"
        time[:] = [base + 0.5, base + 1.75]
        dataset.createVariable("time_bnds", "f8", ("time", "bnds"))[:] = [
            [base, base + 1.0],
            [base + 1.0, base + 2.0],
        ]
        data = dataset.createVariable("tas", "f4", ("time",))
        data.cell_methods = "time: mean"

        time001 = messages(check_time_squareness(dataset))
        time003 = messages(check_time_range_vs_filename(dataset))

    assert time001
    assert any("decoded: unavailable" in message for message in time001)
    assert time003
    assert any("Error converting time values" in message for message in time003)


def test_single_decadal_mean_reconstructs_bounds_from_representative_time(tmp_path):
    path = tmp_path / "masscello_dec_test_1855-1855.nc"
    units = "days since 1850-01-01 00:00:00"
    calendar = "proleptic_gregorian"
    expected_start = cftime.datetime(1850, 1, 1, calendar=calendar)
    expected_end = cftime.datetime(1860, 1, 1, calendar=calendar)
    expected_values = cftime.date2num(
        [expected_start, expected_end], units=units, calendar=calendar
    )
    midpoint = 0.5 * (expected_values[0] + expected_values[1])
    wrong_end = cftime.date2num(
        cftime.datetime(1856, 1, 1, calendar=calendar),
        units=units,
        calendar=calendar,
    )

    with Dataset(path, "w") as dataset:
        dataset.frequency = "dec"
        dataset.variable_id = "masscello"
        dataset.createDimension("time", 1)
        dataset.createDimension("bnds", 2)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = units
        time.calendar = calendar
        time.bounds = "time_bnds"
        time[:] = [midpoint]
        dataset.createVariable("time_bnds", "f8", ("time", "bnds"))[:] = [
            [midpoint, wrong_end]
        ]
        data = dataset.createVariable("masscello", "f4", ("time",))
        data.cell_methods = "time: mean"

        found = messages(check_time_squareness(dataset))
        assert messages(check_time_range_vs_filename(dataset)) == []

    assert len(found) == 1
    assert "time-bounds interval" in found[0]
    assert "expected interval is [0.0, 3652.0]" in found[0]
    assert "decoded: [1850-01-01 00:00, 1860-01-01 00:00]" in found[0]
    assert "time value" not in found[0]


def test_invalid_decadal_representative_time_has_clear_failure(tmp_path):
    dataset = make_time_file(
        tmp_path / "tas_dec_test_2005-2005.nc",
        frequency="dec",
        table_id="None",
        times=[1801.0],
        bounds=[[0.0, 3600.0]],
    )
    with dataset:
        found = messages(check_time_squareness(dataset))

    assert len(found) == 1
    assert "does not identify a unique calendar-aligned 10-year interval" in found[0]
    assert "Technical reason" not in found[0]
