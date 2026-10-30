"""Local regression tests for the legacy CMIP6 bounds-shape check."""

from netCDF4 import Dataset

from checks.variable_checks.check_bounds_shape import check_bounds_shape
from tests.wcrp_cmip6_test_checks.conftest import result_passed


def _messages(results):
    return [message for result in results for message in result.msgs]


def test_bounds_shape_accepts_regular_scalar_and_curvilinear_coordinates(tmp_path):
    path = tmp_path / "bounds-shapes.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("lat", 3)
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 3)
        dataset.createDimension("bnds", 2)
        dataset.createDimension("vertices", 4)

        scalar = dataset.createVariable("height", "f8")
        scalar.bounds = "height_bnds"
        dataset.createVariable("height_bnds", "f8", ("bnds",))

        regular = dataset.createVariable("lat", "f8", ("lat",))
        regular.bounds = "lat_bnds"
        dataset.createVariable("lat_bnds", "f8", ("lat", "bnds"))

        curvilinear = dataset.createVariable("latitude", "f8", ("y", "x"))
        curvilinear.bounds = "vertices_latitude"
        dataset.createVariable("vertices_latitude", "f8", ("y", "x", "vertices"))

        for name in ("height", "lat", "latitude"):
            results = check_bounds_shape(dataset, name)
            assert len(results) == 1
            assert result_passed(results[0])


def test_bounds_shape_accepts_shared_cell_polygons_and_rejects_bad_shapes(tmp_path):
    path = tmp_path / "unstructured-bounds.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("cell", 2)
        dataset.createDimension("bad_cell", 1)
        dataset.createDimension("vertices", 3)
        dataset.createDimension("bad_vertices", 4)

        latitude = dataset.createVariable("latitude", "f8", ("cell",))
        latitude.bounds = "vertices_latitude"
        longitude = dataset.createVariable("longitude", "f8", ("cell",))
        longitude.bounds = "vertices_longitude"
        dataset.createVariable("vertices_latitude", "f8", ("cell", "vertices"))
        dataset.createVariable("vertices_longitude", "f8", ("bad_cell", "bad_vertices"))

        assert result_passed(check_bounds_shape(dataset, "latitude")[0])
        results = check_bounds_shape(dataset, "longitude")
        assert not result_passed(results[0])
        assert "len(longitude)=2" in _messages(results)[0]

        latitude.bounds = "missing_vertices"
        results = check_bounds_shape(dataset, "latitude")
        assert not result_passed(results[0])
        assert "not found" in _messages(results)[0]


def test_bounds_shape_skips_irrelevant_variables_and_reports_wrong_interval(tmp_path):
    path = tmp_path / "wrong-bounds.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("lat", 3)
        dataset.createDimension("vertices", 3)
        latitude = dataset.createVariable("lat", "f8", ("lat",))

        assert check_bounds_shape(dataset, "missing") == []
        assert check_bounds_shape(dataset, "lat") == []

        latitude.bounds = "lat_bnds"
        dataset.createVariable("lat_bnds", "f8", ("lat", "vertices"))
        results = check_bounds_shape(dataset, "lat")
        assert not result_passed(results[0])
        assert "(n, 2)" in _messages(results)[0]
