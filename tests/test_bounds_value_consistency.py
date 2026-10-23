"""Unit coverage for legacy interval containment checks."""

import numpy as np
import pytest
from netCDF4 import Dataset

from checks.variable_checks.check_bounds_value_consistency import (
    check_bounds_value_consistency,
)
from tests.wcrp_cmip6_test_checks.conftest import result_passed

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning"
)


def _messages(results):
    return [message for result in results for message in result.msgs]


def _dataset(path, *, values, bounds):
    dataset = Dataset(path, "w")
    dataset.createDimension("lev", len(values))
    dataset.createDimension("bnds", np.shape(bounds)[-1])
    coordinate = dataset.createVariable("lev", "f8", ("lev",))
    coordinate.bounds = "lev_bnds"
    coordinate[:] = values
    bound = dataset.createVariable("lev_bnds", "f8", ("lev", "bnds"))
    for index, row in enumerate(bounds):
        bound[index, :] = row
    return dataset


def test_bounds_values_accept_increasing_and_decreasing_intervals(tmp_path):
    with _dataset(
        tmp_path / "increasing.nc",
        values=[1.0, 3.0],
        bounds=[[0.0, 2.0], [2.0, 4.0]],
    ) as dataset:
        assert result_passed(check_bounds_value_consistency(dataset, "lev")[0])

    with _dataset(
        tmp_path / "decreasing.nc",
        values=[1000.0, 900.0],
        bounds=[[1050.0, 950.0], [950.0, 850.0]],
    ) as dataset:
        assert result_passed(check_bounds_value_consistency(dataset, "lev")[0])


def test_bounds_values_report_outside_values_and_missing_inputs(tmp_path):
    with _dataset(
        tmp_path / "outside.nc",
        values=[1.0, 8.0],
        bounds=[[0.0, 2.0], [2.0, 4.0]],
    ) as dataset:
        results = check_bounds_value_consistency(dataset, "lev")
        assert not result_passed(results[0])
        assert "1/8.0∉[2.0, 4.0]" in _messages(results)[0]

        dataset.variables["lev"].bounds = "missing_bnds"
        results = check_bounds_value_consistency(dataset, "lev")
        assert not result_passed(results[0])
        assert "not found" in _messages(results)[0]

    results = check_bounds_value_consistency(SimpleDataset(), "missing")
    assert not result_passed(results[0])


def test_bounds_values_report_total_failure_count_but_limit_examples(tmp_path):
    with _dataset(
        tmp_path / "many-outside.nc",
        values=[10.0] * 6,
        bounds=[[0.0, 1.0]] * 6,
    ) as dataset:
        message = _messages(check_bounds_value_consistency(dataset, "lev"))[0]

    assert message.startswith("6 value(s) lie outside declared bounds.")
    assert "4/10.0" in message
    assert "5/10.0" not in message


def test_bounds_values_skip_absent_and_polygon_bounds(tmp_path):
    with _dataset(
        tmp_path / "polygon.nc",
        values=[1.0, 3.0],
        bounds=[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]],
    ) as dataset:
        assert check_bounds_value_consistency(dataset, "lev") == []
        del dataset.variables["lev"].bounds
        assert check_bounds_value_consistency(dataset, "lev") == []


class SimpleDataset:
    def __init__(self):
        self.variables = {}
