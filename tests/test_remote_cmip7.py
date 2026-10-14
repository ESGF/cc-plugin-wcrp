"""Focused CMIP7 regressions using the remote curvilinear-ocean fixture."""

import pytest
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from plugins.cmip7.cmip7 import Cmip7ProjectCheck
from tests.remote.cmip7 import CURVILINEAR_OCEAN
from tests.remote.fetch import resolve_remote_file

pytestmark = pytest.mark.remote_data


def test_old_grid_variable_ranges_report_coordinates_and_vertices():
    """Retain coverage for the superseded 0..360 grid-longitude metadata."""
    path = resolve_remote_file(CURVILINEAR_OCEAN.file)
    with Dataset(path) as dataset:
        checker = Cmip7ProjectCheck()
        checker.setup(dataset)

        assert checker._coordinate_setup_error is None
        assert checker._coordinate_grid_topology == "curvilinear"
        catalog = checker._coordinate_catalog
        assert catalog is not None

        old_ranges = {
            "latitude": (-90.0, 90.0),
            "vertices_latitude": (-90.0, 90.0),
            "longitude": (0.0, 360.0),
            "vertices_longitude": (0.0, 360.0),
        }
        for name, (valid_min, valid_max) in old_ranges.items():
            catalog.grid_variables[name].update(
                valid_min=valid_min,
                valid_max=valid_max,
            )

        results = checker.check_Coordinate_Standard(dataset)
        longitude_at_reported_index = float(dataset.variables["longitude"][48, 107])
        vertices_at_reported_index = dataset.variables["vertices_longitude"][
            80, 106, :
        ].tolist()

    latitude = next(
        result for result in results if result.name.endswith("Grid-latitude valid range")
    )
    assert latitude.weight == BaseCheck.MEDIUM
    assert latitude.value == (1, 1)
    assert not latitude.msgs

    longitude = next(
        result
        for result in results
        if result.name.endswith("Grid-longitude valid range")
    )
    assert longitude.weight == BaseCheck.MEDIUM
    assert longitude.value == (0, 2)
    assert len(longitude.msgs) == 2
    assert (
        "'longitude': 58534 of 118800 grid cells have values below the "
        "recommended valid_min=0.0" in longitude.msgs[0]
    )
    assert "Lowest value: -179.9965362548828 at index (48, 107)" in longitude.msgs[0]
    assert longitude_at_reported_index == pytest.approx(-179.9965362548828)
    assert (
        "'vertices_longitude': 58502 of 118800 grid cells have at least one "
        "vertex below the recommended valid_min=0.0" in longitude.msgs[1]
    )
    assert "circular longitude width=1.0" in longitude.msgs[1]
    assert "at index (80, 106)" in longitude.msgs[1]
    assert vertices_at_reported_index == pytest.approx(
        [179.0, 180.0, -180.0, 179.0]
    )
