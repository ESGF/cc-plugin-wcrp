import numpy as np
import pytest
from netCDF4 import Dataset

from checks.data_plausibility_checks.utils.dimensions import (
    get_ds_dimensions,
    get_filtered_dimensions,
)
from plugins.data_plausibility import wcrp_data
from plugins.data_plausibility.wcrp_data import DatapluginProjectCheck


@pytest.fixture
def curvilinear_ocean_dataset(tmp_path):
    path = tmp_path / "ocean.nc"
    dataset = Dataset(path, "w")
    dataset.createDimension("time", 2)
    dataset.createDimension("y", 3)
    dataset.createDimension("x", 4)

    time = dataset.createVariable("time", "f8", ("time",))
    time.standard_name = "time"
    time.axis = "T"
    time[:] = [0, 1]

    nav_lon = dataset.createVariable("nav_lon", "f8", ("y", "x"))
    nav_lon.standard_name = "longitude"
    nav_lon[:] = np.tile(np.arange(4), (3, 1))

    nav_lat = dataset.createVariable("nav_lat", "f8", ("y", "x"))
    nav_lat.standard_name = "latitude"
    nav_lat[:] = np.tile(np.arange(3)[:, None], (1, 4))

    variable = dataset.createVariable(
        "agessc", "f4", ("time", "y", "x"), fill_value=False
    )
    variable.standard_name = "sea_water_age_since_surface_contact"
    variable[:] = 1.0
    dataset.variable_id = "agessc"

    yield dataset
    dataset.close()


def test_curvilinear_coordinates_map_to_dimensions(curvilinear_ocean_dataset):
    dimensions = get_ds_dimensions(curvilinear_ocean_dataset)

    assert dimensions["x"] == "x"
    assert dimensions["y"] == "y"
    assert dimensions["t"] == "time"
    assert get_filtered_dimensions(
        curvilinear_ocean_dataset, "agessc"
    ) == ["time"]


def test_ocean_file_without_fill_or_physical_threshold_does_not_abort(
    curvilinear_ocean_dataset,
):
    checker = DatapluginProjectCheck()
    checker.config = {
        "data_plausibility_checks": {
            "check_fill": {
                "enabled": True,
                "severity": "M",
                "parameter": "FillValue",
            },
            "check_physically_impossible_outlier": {
                "enabled": True,
                "severity": "H",
            },
            "check_chunk_size": {
                "enabled": True,
                "severity": "M",
            },
        }
    }

    results = checker.check_Data_Plausibility(curvilinear_ocean_dataset)

    assert len(results) == 3
    descriptions = [result.name for result in results]
    assert any("[DATA003]" in description for description in descriptions)
    assert any("[DATA006]" in description for description in descriptions)
    assert any("[DATA009]" in description for description in descriptions)
    data006 = next(
        result for result in results if "[DATA006]" in result.name
    )
    assert data006.value[0] == data006.value[1]


def test_one_data_check_exception_does_not_hide_other_results(
    curvilinear_ocean_dataset, monkeypatch
):
    def raise_internal_error(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        wcrp_data, "check_fillvalues_timeseries", raise_internal_error
    )
    checker = DatapluginProjectCheck()
    checker.config = {
        "data_plausibility_checks": {
            "check_fill": {
                "enabled": True,
                "severity": "M",
                "parameter": "FillValue",
            },
            "check_physically_impossible_outlier": {
                "enabled": True,
                "severity": "H",
            },
        }
    }

    results = checker.check_Data_Plausibility(curvilinear_ocean_dataset)

    assert len(results) == 2
    data003 = next(
        result for result in results if "[DATA003]" in result.name
    )
    data006 = next(
        result for result in results if "[DATA006]" in result.name
    )
    assert data003.value[0] < data003.value[1]
    assert data006.value[0] == data006.value[1]
