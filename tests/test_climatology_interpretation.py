"""Climatology related tests - categories come from project metadata, not file cell_methods."""

from pathlib import Path
from datetime import timedelta

import cftime
import numpy as np
import pytest
import toml

from checks.time_checks.check_time_squareness import check_time_squareness
from checks.time_checks.check_time_bounds import check_time_bounds
from plugins.wcrp_schema import TimeSquarenessRule
from tests.test_time_squareness import make_time_file, messages


CALENDARS = ["noleap", "365_day", "360_day", "gregorian"]


@pytest.mark.parametrize("kind", ["monthly_climatology", "daily_cycle"])
@pytest.mark.parametrize("problem", ["invalid", "masked", "nonfinite"])
def test_bad_interval_does_not_hide_other_midpoints(tmp_path, kind, problem):
    daily = kind == "daily_cycle"
    bounds = np.array([[0, 3600 + 1 / 24], [1 / 24, 3600 + 2 / 24],
                       [2 / 24, 3600 + 3 / 24]]) if daily else np.array(
        [[0., 3630], [30, 3660], [60, 3690]])
    times = [0.5 / 24, 1.6 / 24, 2.5 / 24] if daily else [15, 46, 75]
    with make_time_file(
        tmp_path / "tas_200001-200912-clim.nc", frequency="1hr" if daily else "mon",
        table_id="Amon", times=times, bounds=bounds, climatology=True,
    ) as ds:
        if problem == "invalid":
            ds["climatology_bnds"][0, 1] = -1
        elif problem == "masked":
            ds["climatology_bnds"][0, :] = np.ma.masked
        else:
            ds["climatology_bnds"][0, 0] = np.nan
        found = messages(check_time_squareness(
            ds, expected_is_climatology=True, climatology_type=kind,
        ))
        assert any("missing or non-finite" in m or "could not be interpreted" in m for m in found)
        midpoint = next(m for m in found if "does not match the midpoint" in m)
        assert "1 time value" in midpoint and "index 1:" in midpoint
        assert "decoded:" in midpoint


def test_daily_cycle_preserves_indices_after_missing_interval(tmp_path):
    with make_time_file(
        tmp_path / "tas_200001-200912-clim.nc", frequency="1hr", table_id="Amon",
        times=[0.5 / 24, 2 / 24], bounds=[[0, 3600 + 1 / 24], [1 / 24, 3600 + 3 / 24]],
        climatology=True,
    ) as ds:
        ds["climatology_bnds"][0, :] = np.ma.masked
        found = messages(check_time_squareness(
            ds, expected_is_climatology=True, climatology_type="daily_cycle",
        ))
        failure = next(m for m in found if "incorrect start or end times" in m)
        assert "index 1:" in failure and "lasting 2 hours" in failure


@pytest.mark.parametrize("kind", ["monthly_climatology", "daily_cycle"])
def test_all_invalid_intervals_skip_midpoints_without_derivative_errors(tmp_path, kind):
    with make_time_file(
        tmp_path / "tas_200001-200912-clim.nc", frequency="1hr" if kind == "daily_cycle" else "mon",
        table_id="Amon", times=[15, 45], bounds=[[0, -1], [30, -1]], climatology=True,
    ) as ds:
        found = messages(check_time_squareness(
            ds, expected_is_climatology=True, climatology_type=kind,
        ))
        assert len(found) == 1
        assert "2 climatology intervals" in found[0]
        assert "Midpoint verification was skipped" in found[0]


@pytest.mark.parametrize("climatology", [False, True])
def test_regular_overlap_and_missing_bounds_do_not_hide_time_errors(tmp_path, climatology):
    with make_time_file(
        tmp_path / "tas_200001-200003.nc", frequency="mon", table_id="Amon",
        times=[15, 40, 76], bounds=[[0, 30], [20, 60], [60, 90]], climatology=climatology,
    ) as ds:
        options = dict(expected_is_climatology=climatology,
                       climatology_type="monthly_daily_statistics" if climatology else None)
        for masked in (False, True):
            if masked:
                ds["climatology_bnds" if climatology else "time_bnds"][0, :] = np.ma.masked
            found = messages(check_time_squareness(ds, **options))
            assert any("do not match regular" in m or "does not match regular" in m for m in found)
            assert any("2 time values do not match" in m for m in found)


@pytest.mark.parametrize("calendar", CALENDARS)
def test_monthly_invalid_boundaries_include_count_and_example(tmp_path, calendar):
    units = "days since 1850-01-01"
    dates = [[cftime.datetime(1850, m, 1, calendar=calendar),
              cftime.datetime(1860, 1, 1, calendar=calendar)] for m in (10, 11)]
    bounds = cftime.date2num(dates, units, calendar)
    with make_time_file(
        tmp_path / "tas_185010-185911-clim.nc", frequency="mon", table_id="Oclim",
        times=np.mean(bounds, axis=1), bounds=bounds, climatology=True,
    ) as ds:
        ds["time"].units = units
        ds["time"].calendar = calendar
        found = messages(check_time_squareness(
            ds, expected_is_climatology=True, climatology_type="monthly_climatology",
        ))
        message = next(m for m in found if "could not be interpreted" in m)
        assert "2 climatology intervals" in message
        assert "First incident at index 0:" in message
        assert "1850-10-01" in message and "1860-01-01" in message
        assert "decoded:" in message
        assert "next month boundary" in message


@pytest.mark.parametrize("problem", [None, "missing_attribute", "missing_variable", "outside", "shape"])
def test_time002_climatology_envelope(tmp_path, problem):
    with make_time_file(
        tmp_path / "tas_200001-201002-clim.nc", frequency="monClim", table_id="Oclim",
        times=[15, 45], bounds=[[0, 3630], [30, 3660]], climatology=True,
    ) as ds:
        if problem == "missing_attribute":
            ds["time"].delncattr("climatology")
        elif problem == "missing_variable":
            ds["time"].climatology = "absent"
        elif problem == "outside":
            ds["time"][0] = -1
        elif problem == "shape":
            ds["time"].climatology = "time"
        found = messages(check_time_bounds(
            ds, expected_is_climatology=True, report_bounds_prerequisites=True,
        ))
        assert bool(found) == (problem is not None)
        assert not any("has no bounds attribute" in m for m in found)
        if problem in {"missing_attribute", "missing_variable"}:
            assert not messages(check_time_bounds(
                ds, expected_is_climatology=True, report_bounds_prerequisites=False,
            ))


@pytest.mark.parametrize("calendar", CALENDARS)
@pytest.mark.parametrize("unit", ["days", "hours", "seconds"])
@pytest.mark.parametrize(
    "kind", [None, "monthly_climatology", "daily_cycle", "monthly_daily_statistics"]
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_climatology_uses_regular_time_numeric_precision(
    tmp_path, calendar, unit, kind, sign
):
    units = f"{unit} since 1999-01-01"
    date = lambda y, m, d=1, h=0: cftime.datetime(y, m, d, h, calendar=calendar)
    if kind == "daily_cycle":
        pairs = [
            [date(2000, 1, 1, 23), date(2000, 2)],
            [date(2000, 2, 1, 23), date(2000, 3)],
        ]
        stamps = [
            date(2000, 1, 16, 23) + timedelta(minutes=30),
            date(2000, 2, 16, 23) + timedelta(minutes=30),
        ]
        times = cftime.date2num(stamps, units, calendar)
    else:
        pairs = [[date(2000, 1), date(2000, 2)], [date(2000, 2), date(2000, 3)]]
        times = np.mean(cftime.date2num(pairs, units, calendar), axis=1)
        if kind == "monthly_climatology":
            pairs = [[date(1999, 1), date(2001, 2)], [date(1999, 2), date(2001, 3)]]
    bounds = np.asarray(cftime.date2num(pairs, units, calendar), dtype=float)
    with make_time_file(
        tmp_path / "tas_200001-200002.nc",
        frequency="1hr" if kind == "daily_cycle" else "mon",
        table_id="None",
        times=times,
        bounds=bounds,
        climatology=kind is not None,
    ) as ds:
        time = ds.variables["time"]
        time.units, time.calendar = units, calendar
        bound_var = ds.variables["time_bnds" if kind is None else "climatology_bnds"]
        options = dict(expected_is_climatology=kind is not None, climatology_type=kind)
        # The same offsets in stored units must pass/fail for regular and
        # climatological axes. Boundary perturbations include midnight and Feb.
        for target in ("bounds", "time"):
            for offset, passes in [(0.1e-6, True), (1.2e-6, False), (20e-6, False)]:
                time[:] = times
                bound_var[:] = bounds
                if target == "bounds":
                    changed = bounds.copy()
                    changed[1] += sign * offset
                    bound_var[:] = changed
                else:
                    time[1] = times[1] + sign * offset
                found = messages(check_time_squareness(ds, **options))
                assert bool(found) is not passes, (target, offset, found)


@pytest.mark.parametrize("calendar", CALENDARS)
@pytest.mark.parametrize("year", [1855, 1856])
def test_monthly_climatology_representative_year(tmp_path, calendar, year):
    units = "days since 1850-01-01"
    date = lambda y, m: cftime.datetime(y, m, 1, calendar=calendar)
    next_month = lambda y, m: date(y + (m == 12), m % 12 + 1)
    bounds = np.asarray(
        [
            cftime.date2num(
                [date(1850, month), next_month(1859, month)], units, calendar
            )
            for month in range(1, 13)
        ]
    )
    times = [
        np.mean(
            cftime.date2num(
                [date(year, month), next_month(year, month)], units, calendar
            )
        )
        for month in range(1, 13)
    ]
    with make_time_file(
        tmp_path / "tas_185001-185912.nc",
        frequency="mon",
        table_id="Amon",
        times=times,
        bounds=bounds,
        climatology=True,
    ) as ds:
        ds.variables["time"].units = units
        ds.variables["time"].calendar = calendar
        options = dict(
            expected_is_climatology=True, climatology_type="monthly_climatology"
        )
        assert not messages(check_time_squareness(ds, **options))
        # The old whole-envelope midpoint must fail, including the reported
        # January example whose envelope midpoint lies in July.
        ds.variables["time"][:] = bounds.mean(axis=1)
        assert messages(check_time_squareness(ds, **options))


@pytest.mark.parametrize("calendar", CALENDARS)
@pytest.mark.parametrize(
    "start_hour,end_hour,hour,minute",
    [(0, 1, 0, 30), (23, 0, 23, 30)],
    ids=["ordinary", "crossing_midnight"],
)
def test_daily_cycle(tmp_path, calendar, start_hour, end_hour, hour, minute):
    units = "days since 2000-01-01"
    # Multiple representative days include February and a year boundary.
    representative_days = [(2000, 2, 28), (2000, 3, 1), (2001, 1, 1)]
    dates = [
        cftime.datetime(2000, 1, 1, start_hour, calendar=calendar),
        cftime.datetime(2001, 1, 30, end_hour, calendar=calendar),
    ]
    values = cftime.date2num(
        [
            cftime.datetime(*day, hour, minute, calendar=calendar)
            for day in representative_days
        ],
        units,
        calendar,
    )
    with make_time_file(
        tmp_path / "tas_200001-200001.nc",
        frequency="1hr",
        table_id="None",
        times=values,
        bounds=[cftime.date2num(dates, units, calendar)] * len(values),
        climatology=True,
    ) as ds:
        ds.variables["time"].calendar = calendar
        options = dict(expected_is_climatology=True, climatology_type="daily_cycle")
        assert not messages(check_time_squareness(ds, **options))
        ds.variables["time"][1] += 0.1
        assert any(
            "First incident at index 1" in m
            for m in messages(check_time_squareness(ds, **options))
        )


@pytest.mark.parametrize("climatology", [False, True])
@pytest.mark.parametrize("calendar", CALENDARS)
def test_daily_statistics_preserve_regular_spacing_checks(
    tmp_path, climatology, calendar
):
    units = "days since 1999-12-01"
    edges = cftime.date2num(
        [
            cftime.datetime(year, month, 1, calendar=calendar)
            for year, month in [(1999, 12), (2000, 1), (2000, 2), (2000, 3), (2000, 4)]
        ],
        units,
        calendar,
    )
    bounds = np.column_stack((edges[:-1], edges[1:]))
    times = bounds.mean(axis=1)
    with make_time_file(
        tmp_path / "tas_199912-200003.nc",
        frequency="mon",
        table_id="None",
        times=times,
        bounds=bounds,
        climatology=climatology,
    ) as ds:
        ds.variables["time"].units = units
        ds.variables["time"].calendar = calendar
        options = dict(
            expected_is_climatology=climatology,
            climatology_type="monthly_daily_statistics" if climatology else None,
            expected_cell_methods=["time: maximum within days time: mean over days"],
        )
        assert not messages(check_time_squareness(ds, **options))
        name = "climatology_bnds" if climatology else "time_bnds"
        # Widen February without changing its midpoint: spacing must fail.
        ds.variables[name][2, :] = bounds[2] + [-1, 1]
        assert any(
            "time-bounds interval" in m
            for m in messages(check_time_squareness(ds, **options))
        )
        ds.variables[name][:] = bounds
        ds.variables["time"][2] += 0.1
        assert any(
            "First incident at index 2" in m
            for m in messages(check_time_squareness(ds, **options))
        )


@pytest.mark.parametrize("calendar", CALENDARS)
@pytest.mark.parametrize(
    "frequency,start,end,passes",
    [
        ("1hr", 23, 24, True),
        ("1hr", 23, 25, False),  # Correct midpoint, wrong width.
        ("1hr", 0.25, 1.25, False),  # Correct width, shifted endpoints.
        ("1hr", 0.25, 2.25, False),  # Both duration and alignment are wrong.
        ("1hr", 0, 24, False),  # Equal clocks must not bypass width validation.
        ("3hr", 21, 24, True),
        ("3hr", 22, 25, False),
        ("subhr", 23.5, 24, True),
        ("subhr", 0.25, 0.75, False),
        ("day", 0, 24, True),
        ("day", 1, 25, False),
    ],
)
def test_daily_cycle_frequency_alignment(
    tmp_path, calendar, frequency, start, end, passes
):
    # Two contributing envelopes; their representative timestamps are exact
    # clock midpoints so only boundary alignment/width can cause a failure.
    bounds = [[start / 24, 30 + (end % 24) / 24]] * 2
    times = [10 + (start + end) / 48, 11 + (start + end) / 48]
    with make_time_file(
        tmp_path / "tas_200001-200002.nc",
        frequency=frequency,
        table_id="None",
        times=times,
        bounds=bounds,
        climatology=True,
    ) as ds:
        ds.variables["time"].calendar = calendar
        found = messages(
            check_time_squareness(
                ds, expected_is_climatology=True, climatology_type="daily_cycle"
            )
        )
        if passes:
            assert not found
        else:
            assert len(found) == 1
            assert "2 daily-cycle intervals" in found[0]
            assert "First incident at index 0" in found[0]
            assert f"For frequency '{frequency}'" in found[0]
            assert "decoded:" in found[0]
            if frequency == "1hr" and start == 23:
                assert (
                    "from 23:00 to 01:00 the following day, lasting 2 hours instead of 1 hour"
                    in found[0]
                )
                assert "endpoints are also" not in found[0]
            if frequency == "1hr" and start == 0.25:
                if end == 1.25:
                    assert (
                        "Its duration is correct, but its endpoints are not on full hours"
                        in found[0]
                    )
                else:
                    assert "lasting 2 hours instead of 1 hour" in found[0]
                    assert "Its endpoints are also not on full hours" in found[0]
            if frequency == "3hr":
                assert "00:00, 03:00, 06:00" in found[0]
            if frequency == "subhr":
                assert "00:00, 00:30, 01:00" in found[0]
            if frequency == "day":
                assert "last 1 day and start and end at midnight" in found[0]


@pytest.mark.parametrize(
    "frequency,delegated,expected",
    [
        (None, False, "Missing global attribute 'frequency'"),
        (None, True, None),
        ("unknown", False, "Cannot resolve increment"),
        ("mon", False, "must divide one day"),
    ],
)
def test_daily_cycle_frequency_prerequisite(tmp_path, frequency, delegated, expected):
    with make_time_file(
        tmp_path / "tas_200001-200001.nc",
        frequency=frequency or "1hr",
        table_id="None",
        times=[10 + 0.5 / 24],
        bounds=[[0, 30 + 1 / 24]],
        climatology=True,
    ) as ds:
        if frequency is None:
            ds.delncattr("frequency")
        found = messages(
            check_time_squareness(
                ds,
                expected_is_climatology=True,
                climatology_type="daily_cycle",
                report_frequency_prerequisite=not delegated,
            )
        )
        if expected:
            assert len(found) == 1
            assert expected in found[0]
        else:
            assert not found


def test_metadata_does_not_adopt_unrequested_file_climatology(tmp_path):
    with make_time_file(
        tmp_path / "tas_200001-200002.nc",
        frequency="mon",
        table_id="None",
        times=[15, 45],
        bounds=[[0, 36030], [30, 36060]],
        climatology=True,
    ) as ds:
        # The vocabulary requires ordinary time, so the erroneous climatology
        # attribute must not select a climatology algorithm (COORD owns it).
        assert not messages(check_time_squareness(ds, expected_is_climatology=False))


@pytest.mark.parametrize("delegated", [False, True])
def test_required_climatology_attribute_ownership(tmp_path, delegated):
    with make_time_file(
        tmp_path / "tas_200001-200002.nc",
        frequency="mon",
        table_id="None",
        times=[15, 45],
        bounds=[[0, 30], [30, 60]],
    ) as ds:
        found = messages(
            check_time_squareness(
                ds,
                expected_is_climatology=True,
                climatology_type="monthly_climatology",
                report_bounds_prerequisites=not delegated,
            )
        )
        assert bool(found) is not delegated


def test_unknown_climatology_type_is_not_guessed(tmp_path):
    with make_time_file(
        tmp_path / "tas_200001-200002.nc",
        frequency="mon",
        table_id="None",
        times=[15, 45],
        bounds=[[0, 30], [30, 60]],
        climatology=True,
    ) as ds:
        assert any(
            "no climatology type is configured" in m
            for m in messages(check_time_squareness(ds, expected_is_climatology=True))
        )


@pytest.mark.parametrize(
    "project", ["cmip7", "cmip6", "cmip6plus", "cordex_cmip6", "c3scmip6/c3scmip6"]
)
def test_project_climatology_mapping(project):
    config = (
        Path(__file__).parents[1]
        / "plugins"
        / project
        / "config/wcrp/coordinate_variables.toml"
    )
    rule = TimeSquarenessRule.model_validate(
        toml.load(config)["coordinates"]["time"]["squareness"]
    )
    expected = {"time2": "monthly_climatology", "time3": "daily_cycle"}
    if project == "cmip7":
        expected["time4"] = "monthly_daily_statistics"
    assert rule.climatology_types == expected


@pytest.mark.parametrize(
    "project", ["cmip7", "cmip6", "cmip6plus", "cordex_cmip6", "c3scmip6"]
)
@pytest.mark.parametrize(
    "coordinate_id,is_climatology",
    [
        ("time", False),
        ("time1", False),
        ("time2", True),
        ("time3", True),
        ("custom_time", True),
    ],
)
def test_plugins_pass_vocabulary_interpretation(
    tmp_path, monkeypatch, project, coordinate_id, is_climatology
):
    import importlib
    from checks.coordinate_checks.model import Catalog
    from checks.variable_checks.known_branded_variable import ExpectedVariableMetadata

    modules = {
        "cmip7": ("plugins.cmip7.cmip7", "Cmip7ProjectCheck"),
        "cmip6": ("plugins.cmip6.cmip6", "Cmip6ProjectCheck"),
        "cmip6plus": ("plugins.cmip6plus.cmip6plus", "Cmip6PlusProjectCheck"),
        "cordex_cmip6": (
            "plugins.cordex_cmip6.cordex_cmip6",
            "CordexCmip6ProjectCheck",
        ),
        "c3scmip6": ("plugins.c3scmip6.c3scmip6.c3scmip6", "C3SCmip6ProjectCheck"),
    }
    module_name, class_name = modules[project]
    module = importlib.import_module(module_name)
    checker = getattr(module, class_name)()
    checker._load_split_config()
    if coordinate_id == "custom_time":
        checker.config.coordinates.variables["time"].squareness.climatology_types[
            coordinate_id
        ] = "daily_cycle"
    entry = dict(
        id=coordinate_id, axis="T", out_name="time", is_climatology=is_climatology
    )
    if project in {"cmip7", "cordex_cmip6"}:
        checker._coordinate_catalog = Catalog(
            project_id=checker.project_name,
            branded_variable_id="tas",
            branded_variable={"cell_methods": ["time: mean"]},
            coordinate_ids=(coordinate_id,),
            data_coordinates={coordinate_id: entry},
            file_variable_name="tas",
        )
    else:
        expected = ExpectedVariableMetadata(
            dimensions=["longitude", coordinate_id, "latitude"],
            cell_methods=["time: mean"],
        )
        monkeypatch.setattr(
            checker, "_get_expected_from_registry", lambda *a, **k: (expected, [])
        )
    captured = []
    monkeypatch.setattr(
        module, "check_time_squareness", lambda *a, **k: captured.append(k) or []
    )
    coverage = []
    monkeypatch.setattr(
        module, "check_time_range_vs_filename", lambda *a, **k: coverage.append(k) or []
    )
    with make_time_file(
        tmp_path / "tas_200001-200002.nc",
        frequency="mon",
        table_id="None",
        times=[15, 45],
        bounds=[[0, 30], [30, 60]],
        climatology=is_climatology,
    ) as ds:
        results = checker.check_Coordinates(ds)
    assert not [m for r in results if "TIME002" in r.name for m in r.msgs]
    assert len(captured) == 1
    assert captured[0]["expected_is_climatology"] is is_climatology
    assert len(coverage) == 1
    assert coverage[0]["expected_is_climatology"] is is_climatology
    assert captured[0]["climatology_type"] == {
        "time2": "monthly_climatology",
        "time3": "daily_cycle",
        "custom_time": "daily_cycle",
    }.get(coordinate_id)
