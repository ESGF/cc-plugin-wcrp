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
            ds, severity=BaseCheck.HIGH, climatology_suffix="-clim"
        )[0]

    assert result.value == (1, 1)


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
