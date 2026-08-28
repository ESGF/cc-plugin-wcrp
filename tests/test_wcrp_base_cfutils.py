import json
from pathlib import Path

import pytest
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.variable_checks.check_coords_cordex_cmip6 import (
    check_horizontal_axes_bounds,
    check_lat_lon_bounds,
    check_lon_value_range,
)
from checks.variable_checks.check_data_types import (
    check_coord_data_types,
    check_var_data_type,
)
from checks.time_checks.check_time_cordex_cmip6 import check_time_range
from checks.time_checks.time_constants import FREQ_INC
from checks.utils import infer_frequency
from plugins import wcrp_base
from plugins.cordex_cmip6 import cordex_cmip6 as cordex_cmip6_module
from plugins.cordex_cmip6.cordex_cmip6 import CordexCmip6ProjectCheck
from plugins.wcrp_base import WCRPBaseCheck


class ExampleProjectCheck(WCRPBaseCheck):
    _cc_spec = "wcrp_example"

    def __init__(self, options=None):
        super().__init__(options)
        self.project_name = "example"


def _create_dataset(path, *, formula_terms=False, frequency="mon"):
    """Create a small CF dataset whose data variable has no standard_name."""
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 2)
        dataset.createDimension("latitude", 2)
        dataset.createDimension("longitude", 3)
        dataset.createDimension("bnds", 2)
        if formula_terms:
            dataset.createDimension("lev", 2)

        if frequency is not None:
            dataset.frequency = frequency
        dataset.variable_id = "tas"

        time = dataset.createVariable("time", "f8", ("time",))
        time.axis = "T"
        time.units = "days since 2000-01-01"
        time.calendar = "standard"
        time.bounds = "time_bounds"
        time[:] = [15.5, 45.0]

        time_bounds = dataset.createVariable(
            "time_bounds",
            "f8",
            ("time", "bnds"),
        )
        time_bounds[:] = [[0.0, 31.0], [31.0, 60.0]]

        latitude = dataset.createVariable("latitude", "f8", ("latitude",))
        latitude.units = "degrees_north"
        latitude.axis = "Y"
        latitude.bounds = "latitude_vertices"
        latitude[:] = [-10.0, 10.0]

        longitude = dataset.createVariable("longitude", "f8", ("longitude",))
        longitude.units = "degrees_east"
        longitude.axis = "X"
        longitude.bounds = "longitude_vertices"
        longitude[:] = [-20.0, 0.0, 20.0]

        latitude_vertices = dataset.createVariable(
            "latitude_vertices",
            "f8",
            ("latitude", "bnds"),
        )
        latitude_vertices[:] = [[-20.0, 0.0], [0.0, 20.0]]

        longitude_vertices = dataset.createVariable(
            "longitude_vertices",
            "f8",
            ("longitude", "bnds"),
        )
        longitude_vertices[:] = [[-30.0, -10.0], [-10.0, 10.0], [10.0, 30.0]]

        dimensions = ("time", "latitude", "longitude")
        if formula_terms:
            lev = dataset.createVariable("lev", "f8", ("lev",))
            lev.axis = "Z"
            lev.standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
            lev.formula_terms = "a: a b: b ps: ps"
            lev[:] = [1.0, 2.0]

            dataset.createVariable("a", "f8", ("lev",))[:] = [0.1, 0.2]
            dataset.createVariable("b", "f8", ("lev",))[:] = [0.9, 0.8]
            surface_pressure = dataset.createVariable(
                "ps",
                "f4",
                ("time", "latitude", "longitude"),
            )
            surface_pressure.units = "Pa"
            surface_pressure[:] = 100000.0
            dimensions = ("time", "lev", "latitude", "longitude")

        tas = dataset.createVariable("tas", "f4", dimensions)
        tas.long_name = "Near-Surface Air Temperature"
        tas.units = "K"
        tas.cell_methods = "area: time: mean"
        tas.coordinates = "latitude longitude"
        if formula_terms:
            tas.coordinates += " lev"
        tas[:] = 280.0


def _create_time_dataset(
    path,
    values,
    *,
    units,
    bounds=None,
    calendar="standard",
):
    """Create a minimal dataset for frequency-inference tests."""
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", len(values))
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = units
        time.calendar = calendar
        time[:] = values
        if bounds is not None:
            dataset.createDimension("bnds", 2)
            time.bounds = "time_bounds"
            time_bounds = dataset.createVariable(
                "time_bounds",
                "f8",
                ("time", "bnds"),
            )
            time_bounds[:] = bounds


def _create_dataset_without_time_coordinate(path):
    """Create time-dependent model data without a time coordinate variable."""
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", None)
        dataset.createDimension("cell", 2)
        dataset.variable_id = "hfls"

        dataset.createVariable("lat", "f8", ("cell",))[:] = [-10.0, 10.0]
        dataset.createVariable("lon", "f8", ("cell",))[:] = [0.0, 20.0]
        model_data = dataset.createVariable("hfls", "f4", ("time", "cell"))
        model_data.units = "W m-2"


@pytest.fixture
def cf_dataset(tmp_path):
    path = tmp_path / "tas.nc"
    _create_dataset(path)
    with Dataset(path) as dataset:
        yield dataset


def test_setup_uses_cfutils_without_requiring_standard_name(cf_dataset):
    checker = WCRPBaseCheck()

    checker.setup(cf_dataset)

    assert checker.varname == ["tas"]
    assert checker.time.name == "time"
    assert set(checker.coords) >= {"time", "latitude", "longitude"}
    assert checker.bounds == {
        "time_bounds",
        "latitude_vertices",
        "longitude_vertices",
    }
    assert checker.frequency == "mon"
    assert checker.frequency_inferred is False
    assert not hasattr(checker, "xrds")


def test_setup_infers_missing_frequency(tmp_path):
    path = tmp_path / "tas-without-frequency.nc"
    _create_dataset(path, frequency=None)

    with Dataset(path) as dataset:
        checker = WCRPBaseCheck()
        checker.setup(dataset)

        assert checker.frequency == "mon"
        assert checker.frequency_inferred is True


def test_setup_leaves_missing_time_coordinate_to_cf_checker(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "missing-time-coordinate.nc"
    output_path = tmp_path / "consistency.json"
    _create_dataset_without_time_coordinate(dataset_path)
    monkeypatch.setattr(wcrp_base, "ESG_VOCAB_AVAILABLE", False)

    with Dataset(dataset_path) as dataset:
        checker = WCRPBaseCheck({"consistency_output": str(output_path)})
        checker.setup(dataset)

        setup_result = checker.check_setup_warnings(dataset)[0]
        consistency_result = checker.check_consistency_output(dataset)[0]

    assert checker.time is None
    assert setup_result.value == (1, 1)
    assert consistency_result.value == (1, 1)
    assert json.loads(output_path.read_text(encoding="utf-8"))["time_info"] == {}


def test_time_discovery_exception_is_only_a_setup_warning(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "broken-time-discovery.nc"
    _create_dataset_without_time_coordinate(dataset_path)

    with Dataset(dataset_path) as dataset:
        checker = WCRPBaseCheck({"consistency_output": "unused.json"})
        checker.dataset = dataset
        checker.varname = ["hfls"]
        checker.consistency_output = "unused.json"
        checker.consistency_output_error = None
        checker.setup_warnings = []

        def fail_time_discovery(_dataset):
            raise RuntimeError("broken discovery")

        monkeypatch.setattr(
            wcrp_base.cfutil,
            "get_time_variable",
            fail_time_discovery,
        )

        checker._initialize_time_info()

    assert len(checker.setup_warnings) == 1
    assert "Could not identify the time coordinate" in checker.setup_warnings[0]
    assert checker.setup_warnings[0].endswith(": broken discovery")
    assert checker.consistency_output_error is None


@pytest.mark.parametrize("frequency", ["", "unknown", "unset", 3])
def test_setup_does_not_replace_invalid_present_frequency(tmp_path, frequency):
    path = tmp_path / "tas-with-invalid-frequency.nc"
    _create_dataset(path, frequency=frequency)

    with Dataset(path) as dataset:
        checker = WCRPBaseCheck()
        checker.setup(dataset)

        assert checker.frequency == frequency
        assert checker.frequency_inferred is False


def test_missing_requested_tables_are_recorded_without_aborting_setup(
    cf_dataset,
    tmp_path,
):
    missing_tables = tmp_path / "missing-tables"
    checker = WCRPBaseCheck({"tables": str(missing_tables)})

    checker.setup(cf_dataset)

    assert checker.time.name == "time"
    assert any(
        "Could not load the requested CV and CMOR tables" in warning
        for warning in checker.setup_warnings
    )


def test_cordex_table_retrieval_failure_is_recorded(
    cf_dataset,
    monkeypatch,
):
    checker = CordexCmip6ProjectCheck()

    def fail_retrieval(*_args, **_kwargs):
        raise OSError("table service unavailable")

    monkeypatch.setattr(cordex_cmip6_module, "retrieve", fail_retrieval)

    checker.setup(cf_dataset)

    assert any(
        "Could not retrieve the CORDEX-CMIP6 CMOR tables" in warning
        for warning in checker.setup_warnings
    )


@pytest.mark.parametrize(
    ("values", "units", "expected"),
    [
        ([0.0, 0.5, 1.0], "hours since 2000-01-01", "subhr"),
        ([0.0, 12.0, 24.0], "minutes since 2000-01-01", "subhr"),
        ([0.0, 20.0, 40.0], "minutes since 2000-01-01", "subhr"),
        ([0.0, 17.0, 34.0], "minutes since 2000-01-01", "unknown"),
        ([0.0, 31.0, 62.0], "minutes since 2000-01-01", "unknown"),
        ([0.0, 10.0, 25.0], "minutes since 2000-01-01", "unknown"),
        ([0.0, 1.0, 2.0], "hours since 2000-01-01", "1hr"),
        ([0.0, 3.0, 6.0], "hours since 2000-01-01", "3hr"),
        ([0.0, 6.0, 12.0], "hours since 2000-01-01", "6hr"),
        ([0.0, 1.0, 2.0], "days since 2000-01-01", "day"),
        ([0.0, 31.0, 60.0, 91.0], "days since 2000-01-01", "mon"),
        ([0.0, 366.0, 731.0], "days since 2000-01-01", "yr"),
        ([0.0, 1.0, 3.0], "days since 2000-01-01", "unknown"),
        ([2.0, 1.0, 0.0], "days since 2000-01-01", "unknown"),
    ],
)
def test_infer_frequency_from_coordinate_intervals(
    tmp_path,
    values,
    units,
    expected,
):
    path = tmp_path / "time.nc"
    _create_time_dataset(path, values, units=units)

    with Dataset(path) as dataset:
        assert infer_frequency(dataset.variables["time"]) == expected


def test_infer_frequency_from_single_time_bounds(tmp_path):
    path = tmp_path / "single-time.nc"
    _create_time_dataset(
        path,
        [0.5],
        units="days since 2000-01-01",
        bounds=[[0.0, 1.0]],
    )

    with Dataset(path) as dataset:
        assert (
            infer_frequency(
                dataset.variables["time"],
                dataset.variables["time_bounds"],
            )
            == "day"
        )


def test_infer_frequency_uses_calendar_months_for_360_day_calendar(tmp_path):
    path = tmp_path / "360-day-monthly-time.nc"
    _create_time_dataset(
        path,
        [0.0, 30.0, 60.0],
        units="days since 2000-01-01",
        calendar="360_day",
    )

    with Dataset(path) as dataset:
        assert infer_frequency(dataset.variables["time"]) == "mon"


def test_infer_frequency_uses_calendar_months_for_seasonal_data(tmp_path):
    path = tmp_path / "seasonal-time.nc"
    _create_time_dataset(
        path,
        [0.0, 91.0, 182.0],
        units="days since 2000-01-01",
    )

    with Dataset(path) as dataset:
        assert infer_frequency(dataset.variables["time"]) == "sem"


def test_infer_frequency_reads_generic_freq_inc_entries(tmp_path, monkeypatch):
    path = tmp_path / "custom-subhourly-time.nc"
    _create_time_dataset(
        path,
        [0.0, 15.0, 30.0],
        units="minutes since 2000-01-01",
    )
    monkeypatch.setitem(FREQ_INC, ("None", "subhr"), (15, "minutes"))

    with Dataset(path) as dataset:
        assert infer_frequency(dataset.variables["time"]) == "subhr"

    above_configured_limit = tmp_path / "above-custom-subhourly-limit.nc"
    _create_time_dataset(
        above_configured_limit,
        [0.0, 20.0, 40.0],
        units="minutes since 2000-01-01",
    )
    with Dataset(above_configured_limit) as dataset:
        assert infer_frequency(dataset.variables["time"]) == "unknown"


def test_infer_frequency_rejects_conflicting_bounds(tmp_path):
    path = tmp_path / "conflicting-time.nc"
    _create_time_dataset(
        path,
        [0.5, 2.5, 4.5],
        units="days since 2000-01-01",
        bounds=[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
    )

    with Dataset(path) as dataset:
        assert (
            infer_frequency(
                dataset.variables["time"],
                dataset.variables["time_bounds"],
            )
            == "unknown"
        )


def test_infer_frequency_handles_invalid_metadata(tmp_path):
    path = tmp_path / "invalid-time.nc"
    _create_time_dataset(
        path,
        [0.0, 1.0],
        units="not valid CF time units",
    )

    with Dataset(path) as dataset:
        assert infer_frequency(dataset.variables["time"]) == "unknown"
    assert infer_frequency(None) == "unknown"


def test_setup_preserves_formula_term_coordinates(tmp_path):
    path = tmp_path / "formula_terms.nc"
    _create_dataset(path, formula_terms=True)

    with Dataset(path) as dataset:
        checker = WCRPBaseCheck()
        checker.setup(dataset)

        assert checker.formula_terms["lev"] == {
            "a": "a",
            "b": "b",
            "ps": "ps",
        }
        assert set(checker.coords) >= {"lev", "a", "b", "ps"}
        assert checker.varname == ["tas"]
        assert not check_coord_data_types(
            checker,
            ctype="double",
            auxtype="real",
        )[0].msgs


def test_derived_plugin_writes_consistency_output_with_existing_schema(
    cf_dataset,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(wcrp_base, "ESG_VOCAB_AVAILABLE", False)
    output = tmp_path / "consistency.json"
    checker = ExampleProjectCheck({"consistency_output": str(output)})

    checker.setup(cf_dataset)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == {
        "global_attributes",
        "global_attributes_non_required",
        "global_attributes_dtypes",
        "variable_attributes",
        "variable_attributes_dtypes",
        "variable_dtypes",
        "dimensions",
        "coordinates",
        "time_info",
    }
    assert payload["dimensions"]["time"] == "n"
    assert payload["variable_attributes"]["tas"]["long_name"] == (
        "Near-Surface Air Temperature"
    )
    assert checker.consistency_output_error is None


def test_cordex_defers_consistency_output_until_after_table_setup(
    cf_dataset,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(wcrp_base, "ESG_VOCAB_AVAILABLE", False)
    output = tmp_path / "consistency.json"
    checker = CordexCmip6ProjectCheck({"consistency_output": str(output)})

    WCRPBaseCheck.setup(checker, cf_dataset)

    assert not output.exists()
    assert checker._write_consistency_output()
    assert output.exists()


def test_cordex_does_not_repeat_cf_discovery_after_loading_tables(
    cf_dataset,
    tmp_path,
    monkeypatch,
):
    checker = CordexCmip6ProjectCheck({"tables_dir": str(tmp_path)})
    calls = {"time": 0, "coordinates": 0}

    def initialize_time():
        calls["time"] += 1
        checker.time = cf_dataset.variables["time"]
        checker.calendar = checker.time.calendar
        checker.timeunits = checker.time.units
        checker.timebnds = checker.time.bounds

    def initialize_coordinates():
        calls["coordinates"] += 1
        checker.coords = ["time", "latitude", "longitude"]
        checker.bounds = {
            "time_bounds",
            "latitude_vertices",
            "longitude_vertices",
        }
        checker.formula_terms = {}
        checker.external_variables = []

    def initialize_tables(_tables_path):
        checker.varname = ["tas"]

    monkeypatch.setattr(checker, "_initialize_time_info", initialize_time)
    monkeypatch.setattr(checker, "_initialize_coords_info", initialize_coordinates)
    monkeypatch.setattr(checker, "_initialize_CV_info", initialize_tables)
    monkeypatch.setattr(
        cordex_cmip6_module,
        "retrieve",
        lambda _url, filename, path, force=False: str(Path(path) / filename),
    )

    checker.setup(cf_dataset)

    assert calls == {"time": 1, "coordinates": 1}


def test_time_invariant_variables_are_selected_lazily(cf_dataset):
    checker = WCRPBaseCheck()
    checker.dataset = cf_dataset
    checker.time = cf_dataset.variables["time"]
    checker.varname = ["tas"]

    variables = checker._get_time_invariant_vars()

    assert "tas" not in variables
    assert "time" not in variables
    assert set(variables) >= {
        "latitude",
        "longitude",
        "latitude_vertices",
        "longitude_vertices",
    }


def test_cmip7_writes_consistency_output(cf_dataset, tmp_path, monkeypatch):
    try:
        from plugins.cmip7.cmip7 import Cmip7ProjectCheck
    except ImportError as exc:
        pytest.skip(f"CMIP7 dependencies are unavailable: {exc}")

    monkeypatch.setattr(wcrp_base, "ESG_VOCAB_AVAILABLE", False)
    output = tmp_path / "cmip7-consistency.json"
    checker = Cmip7ProjectCheck({"consistency_output": str(output)})

    checker.setup(cf_dataset)

    assert output.exists()
    assert checker.consistency_output_error is None


def test_data_plugin_writes_consistency_output_once(cf_dataset, monkeypatch):
    from plugins.data_plausibility.wcrp_data import DatapluginProjectCheck

    checker = DatapluginProjectCheck({"consistency_output": "unused.json"})
    writes = []
    monkeypatch.setattr(
        checker, "_write_consistency_output", lambda: writes.append(True)
    )

    checker.setup(cf_dataset)

    assert writes == [True]


def test_cf_discovery_failure_does_not_abort_setup(cf_dataset, monkeypatch):
    def fail_discovery(_dataset):
        raise RuntimeError("broken discovery")

    monkeypatch.setattr(
        wcrp_base.cfutil,
        "get_geophysical_variables",
        fail_discovery,
    )
    checker = WCRPBaseCheck()

    checker.setup(cf_dataset)

    assert checker.varname == []
    assert any("geophysical variables" in warning for warning in checker.setup_warnings)


def test_consistency_output_failure_does_not_abort_setup(
    cf_dataset,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "consistency.json"
    checker = WCRPBaseCheck({"consistency_output": str(output)})

    def fail_output():
        raise RuntimeError("broken output")

    monkeypatch.setattr(checker, "_build_consistency_output", fail_output)

    checker.setup(cf_dataset)

    assert checker.consistency_output_error == (
        "Could not build the requested consistency summary: broken output"
    )
    assert not output.exists()
    result = checker.check_consistency_output(cf_dataset)[0]
    assert result.weight == BaseCheck.LOW
    assert result.value == (0, 1)
    assert result.name == "[TOOL002] Cross-file consistency summary"
    assert result.msgs == [
        "The plugin could not fully prepare or write the JSON summary used to "
        "compare metadata across files. The consistency summary may therefore "
        "be missing or incomplete. Technical details: Could not build the "
        "requested consistency summary: broken output"
    ]


def test_consistency_output_check_passes_without_recorded_error():
    checker = WCRPBaseCheck()
    checker.consistency_output_error = None

    result = checker.check_consistency_output(None)[0]

    assert result.weight == BaseCheck.LOW
    assert result.value == (1, 1)
    assert result.msgs == []


def test_setup_warning_check_reports_all_recorded_warnings():
    checker = WCRPBaseCheck()
    checker.setup_warnings = [
        "Could not identify the time coordinate: broken time metadata",
        "Could not identify coordinate bounds: broken bounds metadata",
    ]

    result = checker.check_setup_warnings(None)[0]

    assert result.weight == BaseCheck.LOW
    assert result.value == (0, 2)
    assert result.name == (
        "[TOOL001] Plugin initialization and file metadata discovery"
    )
    assert result.msgs == [
        "The plugin encountered a problem while examining the file structure "
        "and metadata. Some compliance checks may have been skipped or may be "
        f"incomplete. Technical details: {warning}"
        for warning in checker.setup_warnings
    ]


def test_setup_warning_recording_deduplicates_identical_problems():
    checker = WCRPBaseCheck()
    checker.setup_warnings = []

    checker._record_setup_warning("Could not identify time", ValueError("broken"))
    checker._record_setup_warning("Could not identify time", ValueError("broken"))

    assert checker.setup_warnings == ["Could not identify time: broken"]


def test_setup_warning_check_passes_without_recorded_warnings():
    checker = WCRPBaseCheck()
    checker.setup_warnings = []

    result = checker.check_setup_warnings(None)[0]

    assert result.weight == BaseCheck.LOW
    assert result.value == (1, 1)
    assert result.msgs == []


def test_consistency_output_serialization_failure_is_recorded(
    tmp_path,
    monkeypatch,
):
    checker = WCRPBaseCheck({"consistency_output": str(tmp_path / "output.json")})
    checker.consistency_output = checker.options["consistency_output"]
    checker.consistency_output_error = None
    monkeypatch.setattr(checker, "_build_consistency_output", lambda: object())

    assert not checker._write_consistency_output()
    assert checker.consistency_output_error.startswith(
        "Could not serialize the requested consistency summary as JSON"
    )


def test_consistency_output_write_failure_is_recorded(tmp_path, monkeypatch):
    output = tmp_path / "missing-directory" / "output.json"
    checker = WCRPBaseCheck({"consistency_output": str(output)})
    checker.consistency_output = checker.options["consistency_output"]
    checker.consistency_output_error = None
    monkeypatch.setattr(checker, "_build_consistency_output", lambda: {})

    assert not checker._write_consistency_output()
    assert checker.consistency_output_error.startswith(
        f"Could not write the requested consistency summary to '{output}'"
    )


def test_cordex_consumers_use_netCDF_and_cfutils(cf_dataset):
    checker = WCRPBaseCheck()
    checker.setup(cf_dataset)
    checker.drs_fn = {"time_range": "200001-200002"}

    results = [
        check_var_data_type(checker, vartype="real")[0],
        check_lon_value_range(checker)[0],
        check_horizontal_axes_bounds(checker)[0],
        check_lat_lon_bounds(checker)[0],
        check_time_range(checker)[0],
    ]

    assert all(not result.msgs for result in results)
