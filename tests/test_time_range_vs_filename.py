from cftime import DatetimeGregorian, date2num
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.time_checks.check_time_range_vs_filename import (
    _extract_time_range_token,
    _infer_is_climatology,
    _nearest_month_beginning_label,
    _nearest_month_ending_label,
    check_time_range_vs_filename,
)
from checks.time_checks.check_time_squareness import check_time_squareness


def _messages(result):
    return [str(message) for message in result.msgs]


def _make_climatology_file(
    path,
    *,
    climatology_attribute="climatology_bnds",
    first_bound=DatetimeGregorian(2000, 1, 1),
    last_bound=DatetimeGregorian(2010, 12, 31),
):
    with Dataset(path, "w") as ds:
        ds.frequency = "mon"
        ds.createDimension("time", 2)
        ds.createDimension("bnds", 2)

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time.calendar = "standard"
        if climatology_attribute is not None:
            time.climatology = climatology_attribute

        bounds = ds.createVariable("climatology_bnds", "f8", ("time", "bnds"))
        dates = (
            first_bound,
            DatetimeGregorian(2000, 12, 31),
            DatetimeGregorian(2010, 1, 1),
            last_bound,
        )
        numeric = date2num(dates, units=time.units, calendar=time.calendar)
        bounds[:] = ((numeric[0], numeric[1]), (numeric[2], numeric[3]))
        time[:] = (
            (numeric[0] + numeric[1]) / 2,
            (numeric[2] + numeric[3]) / 2,
        )


def test_extract_time_range_token_accepts_climatology_suffix():
    assert _extract_time_range_token("tas_Amon_200001-201012-clim.nc") == (
        "200001",
        "201012",
    )


def test_climatology_endpoint_month_rounding():
    assert _nearest_month_beginning_label(DatetimeGregorian(2000, 1, 31)) == (
        2000,
        2,
    )
    assert _nearest_month_ending_label(DatetimeGregorian(2101, 1, 1)) == (
        2100,
        12,
    )


def test_climatology_attribute_selects_climatology_bounds(tmp_path):
    path = tmp_path / "tas_Amon_200001-201012.nc"
    _make_climatology_file(path)

    with Dataset(path) as ds:
        assert _infer_is_climatology(ds) is True
        result = check_time_range_vs_filename(ds, severity=BaseCheck.HIGH)[0]

    assert result.value == (1, 1)


def test_missing_climatology_attribute_uses_regular_time_coverage(tmp_path):
    path = tmp_path / "tas_Amon_200007-201007.nc"
    _make_climatology_file(path, climatology_attribute=None)

    with Dataset(path) as ds:
        assert _infer_is_climatology(ds) is False
        result = check_time_range_vs_filename(ds, severity=BaseCheck.HIGH)[0]

    assert result.value == (1, 1)


def test_declared_climatology_requires_named_bounds_variable(tmp_path):
    path = tmp_path / "tas_Amon_200001-201012.nc"
    _make_climatology_file(path, climatology_attribute="missing_bnds")

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(ds, severity=BaseCheck.HIGH)[0]

    assert result.value[0] < result.value[1]
    assert any(
        "Missing climatology bounds variable" in msg for msg in _messages(result)
    )


def test_exact_next_month_boundary_uses_preceding_end_label(tmp_path):
    path = tmp_path / "tas_Amon_200001-201012.nc"
    _make_climatology_file(path, last_bound=DatetimeGregorian(2011, 1, 1))

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(ds, severity=BaseCheck.HIGH)[0]

    assert result.value == (1, 1)


def test_configured_climatology_suffix_is_accepted(tmp_path):
    path = tmp_path / "tas_Amon_200001-201012-clim.nc"
    _make_climatology_file(path)

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(
            ds,
            severity=BaseCheck.HIGH,
            climatology_suffix="-clim",
            expected_is_climatology=True,
        )[0]

    assert result.value == (1, 1)


def test_coordinate_metadata_requires_climatology_attribute(tmp_path):
    path = tmp_path / "tas_Amon_200001-201012-clim.nc"
    _make_climatology_file(path, climatology_attribute=None)

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(
            ds,
            severity=BaseCheck.HIGH,
            climatology_suffix="-clim",
            expected_is_climatology=True,
        )[0]

    assert any(
        "coordinate definition identifies climatological time" in message
        for message in _messages(result)
    )


def test_coordinate_metadata_rejects_unexpected_climatology_attribute(tmp_path):
    path = tmp_path / "tas_Amon_200007-201007.nc"
    _make_climatology_file(path)

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(
            ds,
            severity=BaseCheck.HIGH,
            expected_is_climatology=False,
        )[0]

    assert any(
        "coordinate definition identifies regular time" in message
        for message in _messages(result)
    )


def test_configured_climatology_suffix_is_required(tmp_path):
    path = tmp_path / "tas_Amon_200001-201012.nc"
    _make_climatology_file(path)

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(
            ds, severity=BaseCheck.HIGH, climatology_suffix="-clim"
        )[0]

    assert result.value[0] < result.value[1]
    assert any("expected '-clim'" in msg for msg in _messages(result))


def test_default_climatology_suffix_rejects_suffix(tmp_path):
    path = tmp_path / "tas_Amon_200001-201012-clim.nc"
    _make_climatology_file(path)

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(ds, severity=BaseCheck.HIGH)[0]

    assert result.value[0] < result.value[1]
    assert any("expected no suffix" in msg for msg in _messages(result))


def test_climatology_suffix_requires_file_attribute(tmp_path):
    path = tmp_path / "tas_Amon_200007-201007-clim.nc"
    _make_climatology_file(path, climatology_attribute=None)

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(
            ds, severity=BaseCheck.HIGH, climatology_suffix="-clim"
        )[0]

    assert result.value[0] < result.value[1]
    assert any(
        "does not define a climatology attribute" in msg for msg in _messages(result)
    )


def test_time003_reports_both_numeric_and_decoded_endpoints(tmp_path):
    path = tmp_path / "tas_Amon_199901-201107.nc"
    _make_climatology_file(path, climatology_attribute=None)

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(ds, severity=BaseCheck.HIGH)[0]

    found = _messages(result)
    assert len(found) == 1
    assert "filename time range '199901-201107'" in found[0]
    assert "resolve to '200007-201007'" in found[0]
    assert "first endpoint is" in found[0]
    assert "last endpoint is" in found[0]
    assert found[0].count("(decoded:") == 2


def test_two_dimensional_time_has_one_structural_owner(tmp_path):
    path = tmp_path / "tas_Amon_200001-200002.nc"
    with Dataset(path, "w") as ds:
        ds.frequency = "mon"
        ds.createDimension("time", 2)
        ds.createDimension("extra", 1)
        time = ds.createVariable("time", "f8", ("time", "extra"))
        time.units = "days since 2000-01-01"
        time[:] = [[15.0], [45.0]]

    with Dataset(path) as ds:
        primary = check_time_squareness(ds, report_structural_prerequisites=True)
        dependent = check_time_range_vs_filename(
            ds, report_time_structure_prerequisite=False
        )

    assert len(primary) == 1
    assert "not one-dimensional" in _messages(primary[0])[0]
    assert dependent == []


def test_invalid_time_units_do_not_cascade_into_time003(tmp_path):
    path = tmp_path / "tas_Amon_200001-200002.nc"
    with Dataset(path, "w") as ds:
        ds.frequency = "mon"
        ds.createDimension("time", 2)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "not a CF time unit"
        time[:] = [0.0, 1.0]

    with Dataset(path) as ds:
        primary = check_time_squareness(ds)
        dependent = check_time_range_vs_filename(
            ds, report_time_units_prerequisite=False
        )

    assert len(primary) == 1
    assert "Invalid time units or calendar" in _messages(primary[0])[0]
    assert dependent == []


def test_masked_coverage_endpoint_is_not_replaced_by_an_interior_value(tmp_path):
    path = tmp_path / "tas_Amon_200001-200003.nc"
    with Dataset(path, "w") as ds:
        ds.frequency = "mon"
        ds.createDimension("time", 3)
        time = ds.createVariable("time", "f8", ("time",), fill_value=-999.0)
        time.units = "days since 2000-01-01"
        time[:] = [-999.0, 45.0, 75.0]

    with Dataset(path) as ds:
        result = check_time_range_vs_filename(ds)[0]

    assert result.value[0] < result.value[1]
    assert "first or last time value is missing" in _messages(result)[0]


def test_malformed_climatology_bounds_can_be_owned_by_structural_check(tmp_path):
    path = tmp_path / "tas_Amon_200001-201012.nc"
    with Dataset(path, "w") as ds:
        ds.frequency = "mon"
        ds.createDimension("time", 2)
        ds.createDimension("three", 3)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time.climatology = "climatology_bnds"
        time[:] = [180.0, 3830.0]
        ds.createVariable("climatology_bnds", "f8", ("time", "three"))[:] = 0.0

    with Dataset(path) as ds:
        assert (
            check_time_range_vs_filename(
                ds, report_climatology_bounds_prerequisite=False
            )
            == []
        )
        result = check_time_range_vs_filename(
            ds, report_climatology_bounds_prerequisite=True
        )[0]

    assert "must have dimensions" in _messages(result)[0]


def test_filename_precision_error_is_left_to_file001_when_configured(tmp_path):
    path = tmp_path / "tas_Amon_2000-2000.nc"
    _make_climatology_file(path, climatology_attribute=None)

    with Dataset(path) as ds:
        assert check_time_range_vs_filename(ds, filename_structure_delegated=True) == []


def test_missing_filename_time_range_is_not_repeated_by_time001(tmp_path):
    path = tmp_path / "tas_Amon.nc"
    _make_climatology_file(path, climatology_attribute=None)

    with Dataset(path) as ds:
        assert check_time_squareness(ds, filename_structure_delegated=True) == []


def test_unsupported_frequency_has_one_time_check_owner(tmp_path):
    path = tmp_path / "tas_unknown_2000-2000.nc"
    with Dataset(path, "w") as ds:
        ds.frequency = "unknown"
        ds.createDimension("time", 1)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time[:] = [0.0]

    with Dataset(path) as ds:
        primary = check_time_squareness(ds)
        dependent = check_time_range_vs_filename(
            ds, report_frequency_prerequisite=False
        )

    assert len(primary) == 1
    assert "Cannot resolve increment" in _messages(primary[0])[0]
    assert dependent == []
