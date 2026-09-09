from __future__ import annotations

import numpy as np
import pytest
from netCDF4 import Dataset

from checks.time_checks.check_time_squareness import check_time_squareness
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
    assert any("midpoint of its bounds interval" in message for message in found)


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
            "Mismatch at index 1" in msg
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
