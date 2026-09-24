from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.attribute_checks.check_attribute_suite import check_attribute_suite
from checks.coordinate_checks.esgvoc import (
    CoordinateMetadataError,
    load_catalog,
    require_supported_version,
)
from checks.coordinate_checks.model import Catalog
from checks.coordinate_checks.suite import check_coordinate_catalog
from checks.coordinate_checks.topology import (
    GridTopologyConfig,
    load_grid_topology_config,
    resolve_grid_topology,
)
from checks.coordinate_checks.utils import expected_formula_terms
from checks.coordinate_checks.validation import Findings, check_direct_vertical_values
from checks.time_checks.check_time_calendar import check_calendar_recommendation
from plugins.cmip7.cmip7 import Cmip7ProjectCheck


FAMILIES = {
    "identity": BaseCheck.HIGH,
    "dimension_order": BaseCheck.HIGH,
    "attributes": BaseCheck.HIGH,
    "recommendations": BaseCheck.MEDIUM,
    "direction": BaseCheck.HIGH,
    "valid_range": BaseCheck.HIGH,
    "requested_values": BaseCheck.HIGH,
    "bounds": BaseCheck.HIGH,
    "bounds_name": BaseCheck.MEDIUM,
    "associations": BaseCheck.HIGH,
    "grid": BaseCheck.HIGH,
    "formula": BaseCheck.HIGH,
}
FAMILIES_WITH_INFORMATION = {
    **FAMILIES,
    "allowed_when_unset": BaseCheck.LOW,
}


@pytest.fixture
def nc(tmp_path):
    dataset = Dataset(tmp_path / "coordinate.nc", "w")
    yield dataset
    dataset.close()


def messages(results):
    return [message for result in results for message in result.msgs]


def catalog(coordinates, *, dimensions=None, **collections):
    dimensions = dimensions or list(coordinates)
    return Catalog(
        project_id="cmip7",
        branded_variable_id="ta_ti-u-hxy-air",
        branded_variable={"id": "ta_ti-u-hxy-air", "out_name": "ta"},
        coordinate_ids=tuple(dimensions),
        data_coordinates=coordinates,
        model_levels=collections.get("model_levels", {}),
        formula_terms=collections.get("formula_terms", {}),
        grid_variables=collections.get("grid_variables", {}),
        grid_axes=collections.get("grid_axes", {}),
    )


def coordinate(identifier, kind, out_name, **values):
    return {
        "id": identifier,
        "coordinate_type": kind,
        "data_type": values.pop("data_type", "double"),
        "out_name": out_name,
        "bounds_required": values.pop("bounds_required", False),
        **values,
    }


def test_esgvoc_version_is_at_least_5_1_0():
    with pytest.raises(CoordinateMetadataError, match=r"esgvoc>=5\.1\.0"):
        require_supported_version("5.1.0.dev1")
    assert require_supported_version("5.1.0") == "5.1.0"


def test_catalog_reads_each_coordinate_collection_once():
    class API:
        def __init__(self):
            self.calls = []

        def get_term_in_data_descriptor(self, descriptor, identifier, fields):
            self.calls.append(("one", descriptor))
            return {
                "id": identifier,
                "out_name": "ta",
                "dimensions": ["time1"],
            }

        def get_all_terms_in_collection(self, project, descriptor, fields):
            self.calls.append(("all", descriptor))
            if descriptor == "data_coordinate":
                return [coordinate("time1", "standard_1d", "time")]
            return [{"id": f"test_{descriptor}"}]

    api = API()
    result = load_catalog("ta_ti-u-hxy-air", api=api, installed_version="5.1.0")
    assert result.coordinate_ids == ("time1",)
    assert api.calls.count(("all", "data_coordinate")) == 1
    assert [call for call in api.calls if call[0] == "all"] == [
        ("all", "data_coordinate"),
        ("all", "model_level_coordinate"),
        ("all", "formula_term"),
        ("all", "grid_variable"),
        ("all", "grid_axis"),
    ]


def test_catalog_normalizes_mixed_case_branded_variable_id():
    class API:
        requested_id = None

        def get_term_in_data_descriptor(self, descriptor, identifier, fields):
            self.requested_id = identifier
            return {
                "id": "baresoilfrac_tavg-u-hxy-u",
                "out_name": "baresoilFrac",
                "dimensions": ["time1"],
            }

        def get_all_terms_in_collection(self, project, descriptor, fields):
            if descriptor == "data_coordinate":
                return [coordinate("time1", "standard_1d", "time")]
            return [{"id": f"test_{descriptor}"}]

    api = API()
    result = load_catalog(
        "baresoilFrac_tavg-u-hxy-u",
        api=api,
        installed_version="5.1.0",
    )

    assert api.requested_id == "baresoilfrac_tavg-u-hxy-u"
    assert result.branded_variable_id == "baresoilfrac_tavg-u-hxy-u"
    assert result.data_variable_name == "baresoilFrac"


def test_setup_failure_is_one_verbose_high_result(nc, monkeypatch):
    nc.branded_variable = "ta_ti-u-hxy-air"
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise CoordinateMetadataError(
            "ESGVoc could not read CMIP7 'grid_axis' coordinate metadata: "
            "DatabaseError: database is locked"
        )

    monkeypatch.setattr("plugins.cmip7.cmip7.load_catalog", fail)
    checker = Cmip7ProjectCheck()
    checker.setup(nc)
    setup_results = checker.check_Coordinate_Metadata_Setup(nc)
    assert len(calls) == 1
    assert len(setup_results) == 1
    assert setup_results[0].weight == BaseCheck.HIGH
    assert "database is locked" in setup_results[0].msgs[0]
    assert checker.check_Coordinate_Standard(nc) == []


def test_time001_still_runs_when_coordinate_catalogue_is_unavailable(nc):
    nc.frequency = "monC"
    nc.table_id = "Amon"
    nc.variable_id = "tas"
    nc.createDimension("time", 3)
    nc.createDimension("bnds", 2)
    time = nc.createVariable("time", "f8", ("time",))
    time.units = "days since 2000-01-01"
    time.calendar = "standard"
    time.climatology = "climatology_bnds"
    time[:] = [15.0, 44.0, 74.0]
    nc.createVariable("climatology_bnds", "f8", ("time", "bnds"))[:] = [
        [0.0, 30.0],
        [30.0, 60.0],
        [60.0, 90.0],
    ]
    nc.createVariable("tas", "f4", ("time",))

    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checker._coordinate_catalog = None
    checker._coordinate_setup_error = "coordinate catalogue unavailable"

    results = checker.check_Coordinates(nc)

    time001 = next(result for result in results if "TIME001" in result.name)
    assert "2 time values" in time001.msgs[0]
    assert "First incident at index 1" in time001.msgs[0]


def test_calendar_message_wording_follows_configured_severity(nc):
    nc.createDimension("time", 1)
    time = nc.createVariable("time", "f8", ("time",))
    time.units = "days since 2000-01-01"
    time.calendar = "standard"

    result = check_calendar_recommendation(nc, severity=BaseCheck.HIGH)

    assert any(
        "It is required to use 'proleptic_gregorian'" in msg for msg in messages(result)
    )


def test_coordinate_calendar_enum_advisory_uses_configured_severity(nc):
    nc.createDimension("time", 1)
    time = nc.createVariable("time", "f8", ("time",))
    time.calendar = "gregorian"

    result = check_attribute_suite(
        nc,
        attribute_name="calendar",
        severity=BaseCheck.LOW,
        is_required=False,
        enum=["gregorian"],
        var_name="time",
        context="Coordinate",
    )

    assert any(
        "it is suggested to use 'proleptic_gregorian'" in msg
        for msg in messages(result)
    )


def test_grid_topology_mapping_supports_label_and_future_emd_precedence():
    config = GridTopologyConfig(
        grid_labels={"g123": "rectilinear"},
        grid_types={"tripolar": "curvilinear"},
        grid_type_and_mapping={("tripolar", "special_projection"): "unstructured"},
    )
    assert resolve_grid_topology(config, grid_label="g123") == (
        "rectilinear",
        None,
    )
    assert resolve_grid_topology(
        config,
        grid_label="g123",
        grid_type={"id": "tripolar"},
        grid_mapping={"id": "special_projection"},
    ) == ("unstructured", None)
    assert resolve_grid_topology(
        config, grid_label="g123", grid_type="not_yet_mapped"
    ) == ("rectilinear", None)


def test_default_grid_topology_mapping_is_loaded_from_toml():
    config_path = Path(__file__).parents[1] / (
        "plugins/cmip7/config/wcrp/mappings/grid_topology.toml"
    )
    config = load_grid_topology_config(config_path)
    assert resolve_grid_topology(config, grid_label="g100") == (
        "rectilinear",
        None,
    )
    assert resolve_grid_topology(config, grid_label="g117") == (
        "unstructured",
        None,
    )
    assert resolve_grid_topology(config, grid_label="g102") == (
        "curvilinear",
        None,
    )
    assert config.allow_standard_name_fallback is True


def test_grid_topology_mapping_can_disable_standard_name_fallback(tmp_path):
    config_path = tmp_path / "grid_topology.toml"
    config_path.write_text(
        "[coordinate_identification]\n"
        "allow_standard_name_fallback = false\n",
        encoding="utf-8",
    )

    config = load_grid_topology_config(config_path)

    assert config.allow_standard_name_fallback is False


def test_cmip7_config_sets_recommended_bounds_dimension_names():
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    naming = checker.config.coordinates.registry.bounds_name
    assert naming.severity == "M"
    assert naming.bounds_dimension_name == "bnds"
    assert naming.vertices_dimension_name == "vertices"
    assert naming.climatology_bounds_name == "climatology_bnds"


def test_cmip7_config_enables_optional_physical_direction_checks():
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    direction = checker.config.coordinates.registry.direction
    assert direction.check_direct_physical_values is True
    assert direction.check_formula_derived_profile is True


def test_cmip7_config_allows_unset_computed_standard_name():
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    assert checker.config.coordinates.registry.attributes.allowed_when_unset == [
        "computed_standard_name"
    ]
    assert (
        checker.config.coordinates.registry.attributes.allowed_when_unset_severity
        == "L"
    )


def test_grid_without_resolvable_metadata_reports_one_decisive_failure(nc):
    result = check_coordinate_catalog(
        nc,
        grid_catalog(),
        severities=FAMILIES,
        grid_resolution_error=(
            "The file has no non-empty global 'grid_label' attribute."
        ),
    )
    found = messages(result)
    assert sum("horizontal grid could not be verified" in msg for msg in found) == 1
    assert not any("coordinate order" in msg for msg in found)


def test_cmip7_setup_resolves_grid_label_once(nc, monkeypatch):
    nc.branded_variable = "ta_ti-u-hxy-air"
    nc.grid_label = "g100"
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.load_catalog", lambda *args, **kw: grid_catalog()
    )
    checker = Cmip7ProjectCheck()
    checker.setup(nc)
    assert checker._coordinate_grid_topology == "rectilinear"
    assert checker._coordinate_grid_error is None


def test_cmip7_missing_grid_label_is_one_high_grid_failure(nc, monkeypatch):
    nc.branded_variable = "ta_ti-u-hxy-air"
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.load_catalog", lambda *args, **kw: grid_catalog()
    )
    checker = Cmip7ProjectCheck()
    checker.setup(nc)
    results = checker.check_Coordinate_Standard(nc)
    grid_result = next(result for result in results if "COORD011" in result.name)
    assert grid_result.weight == BaseCheck.HIGH
    assert (
        sum("horizontal grid could not be verified" in msg for msg in grid_result.msgs)
        == 1
    )
    assert "grid_label" in grid_result.msgs[0]


def test_numeric_coordinate_follows_declared_bounds_and_uses_cmor_tolerance(nc):
    nc.createDimension("plev", 2)
    nc.createDimension("bnds", 2)
    plev = nc.createVariable("plev", "f8", ("plev",))
    plev[:] = [1000.5, 500.0]
    plev.axis = "Z"
    plev.standard_name = "air_pressure"
    plev.units = "Pa"
    plev.positive = "down"
    plev.bounds = "pressure_edges"
    bounds = nc.createVariable("pressure_edges", "f8", ("plev", "bnds"))
    bounds[:] = [[1250.0, 750.0], [750.0, 250.0]]
    nc.createVariable("ta", "f4", ("plev",))

    entry = coordinate(
        "plev2",
        "standard_1d",
        "plev",
        axis="Z",
        cf_standard_name="air_pressure",
        units="Pa",
        positive="down",
        stored_direction="decreasing",
        coordinate_values=[1000.0, 500.0],
        coordinate_bounds=[1250.0, 750.0, 250.0],
        tolerance=1.0,
        valid_min=0.0,
        valid_max=1100.0,
        bounds_required=True,
    )
    result = check_coordinate_catalog(
        nc, catalog({"plev2": entry}), severities=FAMILIES
    )
    found = messages(result)
    assert any("recommended name is 'plev_bnds'" in msg for msg in found)
    assert not any("does not exist" in msg for msg in found)
    assert not any("does not contain requested" in msg for msg in found)
    assert not any("overlapping" in msg for msg in found)


def test_numeric_coordinate_without_stored_direction_is_strictly_monotonic(nc):
    nc.createDimension("rho", 3)
    rho = nc.createVariable("rho", "f8", ("rho",))
    rho[:] = [1027.0, 1026.0, 1025.0]
    nc.createVariable("ta", "f4", ("rho",))
    entry = coordinate("rho", "standard_1d", "rho")

    result = check_coordinate_catalog(nc, catalog({"rho": entry}), severities=FAMILIES)
    direction = next(item for item in result if "COORD005" in item.name)
    assert direction.value == (1, 1)

    rho[:] = [1027.0, 1025.0, 1026.0]
    result = check_coordinate_catalog(nc, catalog({"rho": entry}), severities=FAMILIES)
    assert any(
        "'rho' is not strictly monotonic (increasing or decreasing)" in msg
        for msg in messages(result)
    )


@pytest.mark.parametrize("attribute", ["axis", "standard_name", "units", "positive"])
@pytest.mark.parametrize("file_value", ["unexpected", ""])
def test_ordinary_empty_cv_attribute_requires_file_attribute_absence(
    nc, attribute, file_value
):
    nc.createDimension("z", 2)
    variable = nc.createVariable("z", "f8", ("z",))
    variable[:] = [0.0, 1.0]
    variable.long_name = "z coordinate"
    variable.setncattr(attribute, file_value)
    nc.createVariable("ta", "f4", ("z",))

    result = check_coordinate_catalog(
        nc,
        catalog({"z": coordinate("z", "standard_1d", "z")}),
        severities=FAMILIES,
    )

    metadata = next(item for item in result if "COORD003" in item.name)
    assert metadata.weight == BaseCheck.HIGH
    assert any(
        f"defines {attribute}=" in message and "attribute to be absent" in message
        for message in metadata.msgs
    )

    result = check_coordinate_catalog(
        nc,
        catalog({"z": coordinate("z", "standard_1d", "z")}),
        severities=FAMILIES_WITH_INFORMATION,
        attributes_allowed_when_unset=(attribute,),
    )
    assert not any(
        f"defines {attribute}=" in message and "attribute to be absent" in message
        for message in messages(result)
    )
    information = next(item for item in result if "COORD004a" in item.name)
    assert information.weight == BaseCheck.LOW
    assert any(
        f"defines {attribute}=" in message
        and "permitted by the configured allowed_when_unset" in message
        for message in information.msgs
    )


@pytest.mark.parametrize(
    ("standard_name", "positive", "coordinate_values"),
    [
        ("air_pressure", "down", [1000.0, -1.0]),
        ("height", "up", [-1.0, 0.0]),
        ("depth", "down", [-1.0, 0.0]),
    ],
)
def test_direct_vertical_coordinates_have_valid_physical_values(
    nc, standard_name, positive, coordinate_values
):
    nc.createDimension("z", 2)
    variable = nc.createVariable("z", "f8", ("z",))
    variable[:] = coordinate_values
    variable.axis = "Z"
    variable.standard_name = standard_name
    variable.positive = positive
    nc.createVariable("ta", "f4", ("z",))
    entry = coordinate(
        "z",
        "standard_1d",
        "z",
        axis="Z",
        cf_standard_name=standard_name,
        positive=positive,
        stored_direction=(
            "decreasing" if standard_name == "air_pressure" else "increasing"
        ),
    )

    result = check_coordinate_catalog(nc, catalog({"z": entry}), severities=FAMILIES)

    assert any(
        f"standard_name='{standard_name}'" in message and "physical values" in message
        for message in messages(result)
    )

    result = check_coordinate_catalog(
        nc,
        catalog({"z": entry}),
        severities=FAMILIES,
        check_direct_physical_values=False,
    )
    assert not any("physical values" in message for message in messages(result))


def test_direct_physical_value_exception_is_contained(nc, monkeypatch):
    variable = nc.createVariable("z", "f8")
    findings = Findings({"direction": BaseCheck.HIGH})
    monkeypatch.setattr(
        "checks.coordinate_checks.validation.numeric_values",
        lambda unused: (_ for _ in ()).throw(RuntimeError("unreadable values")),
    )

    check_direct_vertical_values(
        findings,
        variable,
        "z",
        {"cf_standard_name": "height"},
    )

    result = findings.results()[0]
    assert any("RuntimeError: unreadable values" in message for message in result.msgs)


def test_latitude_stored_direction_is_checked_by_coordinate_catalog(nc):
    nc.createDimension("lat", 3)
    lat = nc.createVariable("lat", "f8", ("lat",))
    lat[:] = [45.0, 0.0, -45.0]
    nc.createVariable("ta", "f4", ("lat",))
    entry = coordinate(
        "latitude",
        "standard_1d",
        "lat",
        cf_standard_name="latitude",
        stored_direction="increasing",
    )

    result = check_coordinate_catalog(
        nc, catalog({"latitude": entry}), severities=FAMILIES
    )
    assert any(
        "'lat' is strictly decreasing; the required stored_direction is "
        "strictly increasing" in msg
        for msg in messages(result)
    )


def test_time_monotonicity_remains_outside_coordinate_catalog(nc):
    nc.createDimension("time", 3)
    time = nc.createVariable("time", "f8", ("time",))
    time[:] = [0.0, 2.0, 1.0]
    time.axis = "T"
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time.calendar = "standard"
    nc.createVariable("ta", "f4", ("time",))
    entry = coordinate(
        "time",
        "standard_1d",
        "time",
        axis="T",
        cf_standard_name="time",
        units="days since ?",
    )

    result = check_coordinate_catalog(nc, catalog({"time": entry}), severities=FAMILIES)
    direction = next(item for item in result if "COORD005" in item.name)
    assert direction.value == (1, 1)


def test_time002_uses_esgvoc_time_out_name_without_duplicating_coord008(nc):
    nc.createDimension("forecast_time", 2)
    nc.createDimension("bnds", 2)
    time = nc.createVariable("forecast_time", "f8", ("forecast_time",))
    time[:] = [1.0, 4.0]
    time.axis = "T"
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time.calendar = "standard"
    time.bounds = "forecast_time_bnds"
    nc.createVariable("forecast_time_bnds", "f8", ("forecast_time", "bnds"))[:] = [
        [0.0, 2.0],
        [2.0, 3.0],
    ]
    nc.createVariable("ta", "f4", ("forecast_time",))
    entry = coordinate(
        "time_custom",
        "standard_1d",
        "forecast_time",
        axis="T",
        cf_standard_name="time",
        units="days since ?",
        bounds_required=True,
    )

    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checker._coordinate_catalog = catalog({"time_custom": entry})
    result = checker.check_Coordinate_Standard(nc)

    time002 = next(item for item in result if "TIME002" in item.name)
    assert time002.weight == BaseCheck.HIGH
    assert any("'forecast_time' value lies outside" in msg for msg in time002.msgs)
    coord008 = next(item for item in result if "COORD008" in item.name)
    assert not any("outside its bounds" in msg for msg in coord008.msgs)


def test_time002_owns_regular_time_bounds_shape(nc):
    nc.createDimension("forecast_time", 2)
    nc.createDimension("three", 3)
    time = nc.createVariable("forecast_time", "f8", ("forecast_time",))
    time[:] = [1.0, 4.0]
    time.axis = "T"
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time.calendar = "standard"
    time.bounds = "forecast_time_bnds"
    nc.createVariable("forecast_time_bnds", "f8", ("forecast_time", "three"))
    nc.createVariable("ta", "f4", ("forecast_time",))
    entry = coordinate(
        "time_custom",
        "standard_1d",
        "forecast_time",
        axis="T",
        cf_standard_name="time",
        units="days since ?",
        bounds_required=True,
    )

    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checker._coordinate_catalog = catalog({"time_custom": entry})
    result = checker.check_Coordinate_Standard(nc)

    time002 = next(item for item in result if "TIME002" in item.name)
    assert any(
        "required for forecast_time_bnds to have shape (n, 2)" in msg
        for msg in time002.msgs
    )
    coord008 = next(item for item in result if "COORD008" in item.name)
    assert not any("expected dimensions" in msg for msg in coord008.msgs)


def test_coord008_retains_time_bounds_parent_dimension_identity(nc):
    nc.createDimension("time", 2)
    nc.createDimension("other", 2)
    nc.createDimension("bnds", 2)
    time = nc.createVariable("time", "f8", ("time",))
    time[:] = [1.0, 3.0]
    time.axis = "T"
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time.calendar = "standard"
    time.bounds = "time_bnds"
    nc.createVariable("time_bnds", "f8", ("other", "bnds"))[:] = [
        [0.0, 2.0],
        [2.0, 4.0],
    ]
    nc.createVariable("ta", "f4", ("time",))
    entry = coordinate(
        "time",
        "standard_1d",
        "time",
        axis="T",
        cf_standard_name="time",
        units="days since ?",
        bounds_required=True,
    )

    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checker._coordinate_catalog = catalog({"time": entry})
    result = checker.check_Coordinate_Standard(nc)

    time002 = next(item for item in result if "TIME002" in item.name)
    assert time002.value == (1, 1)
    coord008 = next(item for item in result if "COORD008" in item.name)
    assert any(
        "expected the parent time coordinate dimension 'time'" in msg
        for msg in coord008.msgs
    )


def test_original_calendar_enum_is_retained_without_duplicate_presence_checks(nc):
    nc.createDimension("time", 2)
    time = nc.createVariable("time", "f8", ("time",))
    time[:] = [0.0, 1.0]
    time.axis = "T"
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time.calendar = "julian"
    nc.createVariable("ta", "f4", ("time",))
    entry = coordinate(
        "time",
        "standard_1d",
        "time",
        axis="T",
        cf_standard_name="time",
        units="days since ?",
    )

    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checker._coordinate_catalog = catalog({"time": entry})
    result = checker.check_Coordinates(nc)

    calendar = [item for item in result if "calendar" in item.name.lower()]
    enum = [item for item in calendar if "[ATTR004]" in item.name]
    assert len(enum) == 1
    assert any("not in allowed values" in msg for msg in enum[0].msgs)
    assert not any("[ATTR001]" in item.name for item in calendar)

    time.delncattr("calendar")
    standard_result = checker.check_Coordinate_Standard(nc)
    original_result = checker.check_Coordinates(nc)
    missing = [
        msg
        for msg in messages(standard_result + original_result)
        if "missing calendar" in msg
    ]
    assert len(missing) == 1
    assert not any("[ATTR001]" in item.name for item in original_result)


def test_bounds_follow_inferred_parent_direction_inside_and_between_pairs(nc):
    nc.createDimension("lat", 2)
    nc.createDimension("bnds", 2)
    lat = nc.createVariable("lat", "f8", ("lat",))
    lat[:] = [-45.0, 45.0]
    lat.bounds = "lat_bnds"
    bounds = nc.createVariable("lat_bnds", "f8", ("lat", "bnds"))
    nc.createVariable("ta", "f4", ("lat",))
    entry = coordinate(
        "latitude",
        "standard_1d",
        "lat",
        cf_standard_name="latitude",
        bounds_required=True,
    )

    bounds[:] = [[-10.0, -90.0], [10.0, 90.0]]
    result = check_coordinate_catalog(
        nc, catalog({"latitude": entry}), severities=FAMILIES
    )
    assert any(
        "bounds pairs that are not strictly increasing" in msg
        for msg in messages(result)
    )

    bounds[:] = [[-90.0, -10.0], [-100.0, 90.0]]
    result = check_coordinate_catalog(
        nc, catalog({"latitude": entry}), severities=FAMILIES
    )
    assert any(
        "not strictly increasing between successive cells" in msg
        for msg in messages(result)
    )


def test_bounds_dimension_name_is_a_configurable_recommendation(nc):
    nc.createDimension("lat", 2)
    nc.createDimension("nv", 2)
    lat = nc.createVariable("lat", "f8", ("lat",))
    lat[:] = [-45.0, 45.0]
    lat.bounds = "lat_bnds"
    nc.createVariable("lat_bnds", "f8", ("lat", "nv"))[:] = [
        [-90.0, 0.0],
        [0.0, 90.0],
    ]
    nc.createVariable("ta", "f4", ("lat",))
    entry = coordinate(
        "latitude",
        "standard_1d",
        "lat",
        cf_standard_name="latitude",
        bounds_required=True,
    )

    result = check_coordinate_catalog(
        nc, catalog({"latitude": entry}), severities=FAMILIES
    )
    naming = next(item for item in result if "COORD009" in item.name)
    assert naming.weight == BaseCheck.MEDIUM
    assert any(
        "trailing dimension 'nv'" in msg and "recommended name is 'bnds'" in msg
        for msg in naming.msgs
    )

    high_naming_severities = {**FAMILIES, "bounds_name": BaseCheck.HIGH}
    result = check_coordinate_catalog(
        nc,
        catalog({"latitude": entry}),
        severities=high_naming_severities,
    )
    naming = next(item for item in result if "COORD009" in item.name)
    assert naming.weight == BaseCheck.HIGH
    assert any("configured required name is 'bnds'" in msg for msg in naming.msgs)
    assert not any("recommended name" in msg for msg in naming.msgs)

    result = check_coordinate_catalog(
        nc,
        catalog({"latitude": entry}),
        severities=FAMILIES,
        bounds_dimension_name="nv",
    )
    naming = next(item for item in result if "COORD009" in item.name)
    assert naming.value == (1, 1)


def test_identity_message_wording_follows_configured_severity(nc):
    entry = coordinate("depth", "standard_1d", "depth")
    severities = {**FAMILIES, "identity": BaseCheck.MEDIUM}

    result = check_coordinate_catalog(
        nc,
        catalog({"depth": entry}),
        severities=severities,
    )

    identity = next(item for item in result if "COORD001" in item.name)
    assert identity.weight == BaseCheck.MEDIUM
    assert any(
        "recommended one-dimensional coordinate 'depth'" in msg for msg in identity.msgs
    )
    assert not any(
        "required one-dimensional coordinate" in msg for msg in identity.msgs
    )


@pytest.mark.parametrize(
    ("severity", "noun"),
    [
        (BaseCheck.HIGH, "requirements"),
        (BaseCheck.MEDIUM, "recommendations"),
        (BaseCheck.LOW, "suggestions"),
    ],
)
def test_coord004_label_wording_follows_configured_severity(nc, severity, noun):
    entry = coordinate("depth", "standard_1d", "depth")
    severities = {**FAMILIES, "recommendations": severity}

    result = check_coordinate_catalog(
        nc,
        catalog({"depth": entry}),
        severities=severities,
    )

    recommendations = next(item for item in result if "COORD004" in item.name)
    assert recommendations.weight == severity
    assert noun in recommendations.name


@pytest.mark.parametrize(
    "dimensions",
    [("bnds", "lat"), ("other", "bnds")],
)
def test_bounds_require_parent_dimension_first_and_size_two_last(nc, dimensions):
    nc.createDimension("lat", 3)
    nc.createDimension("other", 3)
    nc.createDimension("bnds", 2)
    lat = nc.createVariable("lat", "f8", ("lat",))
    lat[:] = [-45.0, 0.0, 45.0]
    lat.bounds = "lat_bnds"
    nc.createVariable("lat_bnds", "f8", dimensions)
    nc.createVariable("ta", "f4", ("lat",))
    entry = coordinate(
        "latitude",
        "standard_1d",
        "lat",
        cf_standard_name="latitude",
        bounds_required=True,
    )

    result = check_coordinate_catalog(
        nc, catalog({"latitude": entry}), severities=FAMILIES
    )
    structural = next(item for item in result if "COORD008" in item.name)
    assert structural.weight == BaseCheck.HIGH
    assert any(
        "expected dimensions ('lat', <size-2>)" in msg for msg in structural.msgs
    )


def test_required_bounds_attribute_is_not_inferred_from_expected_variable(nc):
    nc.createDimension("lat", 2)
    nc.createDimension("bnds", 2)
    lat = nc.createVariable("lat", "f8", ("lat",))
    lat[:] = [-30, 30]
    lat.axis = "Y"
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    nc.createVariable("lat_bnds", "f8", ("lat", "bnds"))
    nc.createVariable("ta", "f4", ("lat",))
    entry = coordinate(
        "latitude_band",
        "standard_1d",
        "lat",
        axis="Y",
        cf_standard_name="latitude",
        units="degrees_north",
        stored_direction="increasing",
        bounds_required=True,
    )
    result = check_coordinate_catalog(
        nc, catalog({"latitude_band": entry}), severities=FAMILIES
    )
    found = messages(result)
    assert any(
        "required for 'lat' to have a 'bounds' attribute" in msg for msg in found
    )
    assert not any("recommended name" in msg for msg in found)


def test_scalar_uses_units_and_valid_range_but_not_tolerance(nc):
    height = nc.createVariable("height", "f8")
    height.assignValue(2.5)
    height.axis = "Z"
    height.standard_name = "height"
    height.units = "cm"
    height.positive = "up"
    ta = nc.createVariable("ta", "f4")
    ta.coordinates = "height"
    entry = coordinate(
        "height2m",
        "scalar",
        "height",
        axis="Z",
        cf_standard_name="height",
        units="m",
        positive="up",
        coordinate_values=[2.0],
        tolerance=1000.0,
        valid_min=0.0,
        valid_max=3.0,
    )
    result = check_coordinate_catalog(
        nc, catalog({"height2m": entry}), severities=FAMILIES
    )
    found = messages(result)
    assert any("convertible to, but not identical" in msg for msg in found)
    assert any("recommended value is 2.0" in msg for msg in found)
    assert not any("permitted absolute" in msg for msg in found)


def test_data_dimensions_follow_reverse_esgvoc_cmor_order(nc):
    nc.createDimension("plev", 2)
    nc.createDimension("time", 3)
    for name, axis, standard_name, units in (
        ("plev", "Z", "air_pressure", "Pa"),
        ("time", "T", "time", "days since 2000-01-01"),
    ):
        variable = nc.createVariable(name, "f8", (name,))
        variable.axis = axis
        variable.standard_name = standard_name
        variable.units = units
    nc.variables["plev"].positive = "down"
    nc.variables["time"].calendar = "standard"
    nc.createVariable("ta", "f4", ("plev", "time"))
    coordinates = {
        "plev": coordinate(
            "plev",
            "standard_1d",
            "plev",
            axis="Z",
            cf_standard_name="air_pressure",
            units="Pa",
            positive="down",
        ),
        "time": coordinate(
            "time",
            "standard_1d",
            "time",
            axis="T",
            cf_standard_name="time",
            units="days since ?",
        ),
    }
    result = check_coordinate_catalog(
        nc,
        catalog(coordinates, dimensions=["plev", "time"]),
        severities=FAMILIES,
    )
    assert any(
        "['time', 'plev'] as their required order" in msg for msg in messages(result)
    )


def test_text_auxiliary_requires_sector_but_accepts_extra_unordered_labels(nc):
    nc.createDimension("basin", 3)
    nc.createDimension("strlen", 8)
    sector = nc.createVariable("sector", "S1", ("basin", "strlen"))
    sector[:] = np.asarray(
        [list("atlantic"), list("pacific "), list("indian  ")], dtype="S1"
    )
    sector.standard_name = "region"
    ta = nc.createVariable("ta", "f4", ("basin",))
    ta.coordinates = "sector"
    entry = coordinate(
        "basin",
        "auxiliary",
        "basin",
        data_type="character",
        cf_standard_name="region",
        coordinate_values=["indian", "atlantic"],
    )
    result = check_coordinate_catalog(
        nc, catalog({"basin": entry}), severities=FAMILIES
    )
    assert not any("missing requested label" in msg for msg in messages(result))
    ta.coordinates = ""
    result = check_coordinate_catalog(
        nc, catalog({"basin": entry}), severities=FAMILIES
    )
    assert any(
        "required for 'ta' coordinates attribute to include text auxiliary" in msg
        for msg in messages(result)
    )


def test_site_requires_linked_latitude_and_longitude(nc):
    nc.createDimension("site", 2)
    ta = nc.createVariable("ta", "f4", ("site",))
    lat = nc.createVariable("station_lat", "f8", ("site",))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lon = nc.createVariable("station_lon", "f8", ("site",))
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    ta.coordinates = "station_lat"
    entry = coordinate("site", "site", "site", data_type="integer")
    result = check_coordinate_catalog(nc, catalog({"site": entry}), severities=FAMILIES)
    assert any("standard_name='longitude'" in msg for msg in messages(result))


def test_climatology_rejects_regular_bounds_unless_climatology_names_them(nc):
    nc.createDimension("time", 2)
    nc.createDimension("bnds", 2)
    time = nc.createVariable("time", "f8", ("time",))
    time[:] = [15, 45]
    time.axis = "T"
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time.calendar = "standard"
    time.climatology = "clim"
    time.bounds = "time_bnds"
    nc.createVariable("clim", "f8", ("time", "bnds"))[:] = [[0, 30], [30, 60]]
    nc.createVariable("time_bnds", "f8", ("time", "bnds"))[:] = [[0, 30], [30, 60]]
    nc.createVariable("ta", "f4", ("time",))
    entry = coordinate(
        "time2",
        "standard_1d",
        "time",
        axis="T",
        cf_standard_name="time",
        units="days since ?",
        stored_direction="increasing",
        bounds_required=True,
        is_climatology=True,
    )
    result = check_coordinate_catalog(
        nc, catalog({"time2": entry}), severities=FAMILIES
    )
    found = messages(result)
    assert any(
        "required that climatological time coordinate 'time' does not define a bounds attribute"
        in msg
        for msg in found
    )
    assert any(
        "required that climatological time coordinate 'time' does not define regular bounds variable"
        in msg
        for msg in found
    )


def test_climatology_bounds_name_is_a_configurable_recommendation(nc):
    nc.createDimension("time", 2)
    nc.createDimension("bnds", 2)
    time = nc.createVariable("time", "f8", ("time",))
    time[:] = [15.0, 45.0]
    time.axis = "T"
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time.calendar = "standard"
    time.climatology = "model_climatology"
    nc.createVariable("model_climatology", "f8", ("time", "bnds"))[:] = [
        [0.0, 30.0],
        [30.0, 60.0],
    ]
    nc.createVariable("ta", "f4", ("time",))
    entry = coordinate(
        "time_climatology",
        "standard_1d",
        "time",
        axis="T",
        cf_standard_name="time",
        units="days since ?",
        bounds_required=True,
        is_climatology=True,
    )

    result = check_coordinate_catalog(
        nc, catalog({"time_climatology": entry}), severities=FAMILIES
    )
    naming = next(item for item in result if "COORD009" in item.name)
    assert naming.weight == BaseCheck.MEDIUM
    assert any(
        "recommended climatology bounds name is 'climatology_bnds'" in msg
        for msg in naming.msgs
    )

    result = check_coordinate_catalog(
        nc,
        catalog({"time_climatology": entry}),
        severities=FAMILIES,
        climatology_bounds_name="model_climatology",
    )
    naming = next(item for item in result if "COORD009" in item.name)
    assert naming.value == (1, 1)


def grid_catalog():
    coordinates = {
        role: coordinate(
            role,
            "generic_horizontal",
            "lat" if role == "latitude" else "lon",
            axis="Y" if role == "latitude" else "X",
            cf_standard_name=role,
            units="degrees_north" if role == "latitude" else "degrees_east",
            bounds_required=True,
        )
        for role in ("longitude", "latitude")
    }
    grid_variables = {}
    for role in ("latitude", "longitude"):
        grid_variables[role] = {
            "id": role,
            "out_name": role,
            "data_type": "double",
            "cf_standard_name": role,
            "units": "degrees_north" if role == "latitude" else "degrees_east",
            "dimensions": ["longitude", "latitude"],
        }
        grid_variables[f"vertices_{role}"] = {
            "id": f"vertices_{role}",
            "out_name": f"vertices_{role}",
            "data_type": "double",
            "units": "degrees_north" if role == "latitude" else "degrees_east",
            "dimensions": ["vertices", "longitude", "latitude"],
        }
    axes = {
        "x": {
            "id": "x",
            "out_name": "x",
            "axis": "X",
            "data_type": "double",
            "cf_standard_name": "projection_x_coordinate",
            "units": "m",
        },
        "y": {
            "id": "y",
            "out_name": "y",
            "axis": "Y",
            "data_type": "double",
            "cf_standard_name": "projection_y_coordinate",
            "units": "m",
        },
        "x_deg": {
            "id": "x_deg",
            "out_name": "x",
            "axis": "X",
            "data_type": "double",
            "cf_standard_name": "projection_x_angular_coordinate",
            "units": "degrees",
        },
        "y_deg": {
            "id": "y_deg",
            "out_name": "y",
            "axis": "Y",
            "data_type": "double",
            "cf_standard_name": "projection_y_angular_coordinate",
            "units": "degrees",
        },
        "grid_longitude": {
            "id": "grid_longitude",
            "out_name": "rlon",
            "axis": "X",
            "data_type": "double",
            "cf_standard_name": "grid_longitude",
            "units": "degrees",
        },
        "grid_latitude": {
            "id": "grid_latitude",
            "out_name": "rlat",
            "axis": "Y",
            "data_type": "double",
            "cf_standard_name": "grid_latitude",
            "units": "degrees",
        },
        "i_index": {
            "id": "i_index",
            "out_name": "i",
            "data_type": "integer",
            "units": "1",
        },
        "j_index": {
            "id": "j_index",
            "out_name": "j",
            "data_type": "integer",
            "units": "1",
        },
    }
    return catalog(
        coordinates,
        dimensions=["longitude", "latitude"],
        grid_variables=grid_variables,
        grid_axes=axes,
    )


def test_curvilinear_grid_uses_esgvoc_grid_variables_axes_and_vertices(nc):
    nc.createDimension("y", 2)
    nc.createDimension("x", 3)
    nc.createDimension("vertices", 4)
    for name, axis, standard in (
        ("x", "X", "projection_x_coordinate"),
        ("y", "Y", "projection_y_coordinate"),
    ):
        var = nc.createVariable(name, "f8", (name,))
        var.axis = axis
        var.standard_name = standard
        var.units = "m"
        var[:] = np.arange(len(nc.dimensions[name]), dtype="float64")
    for role, units in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
        var = nc.createVariable(role, "f8", ("y", "x"))
        var.standard_name = role
        var.units = units
        var.bounds = f"vertices_{role}"
        nc.createVariable(f"vertices_{role}", "f8", ("y", "x", "vertices"))
    ta = nc.createVariable("ta", "f4", ("y", "x"))
    ta.coordinates = "latitude longitude"
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    assert messages(result) == []

    nc.variables["x"][:] = [0.0, 2.0, 1.0]
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    assert any(
        "'x' is not strictly monotonic (increasing or decreasing)" in msg
        for msg in messages(result)
    )


def test_horizontal_standard_name_fallback_can_be_disabled(nc):
    nc.createDimension("y", 2)
    nc.createDimension("x", 3)
    nc.createDimension("vertices", 4)
    for role, name, units in (
        ("latitude", "lat", "degrees_north"),
        ("longitude", "lon", "degrees_east"),
    ):
        variable = nc.createVariable(name, "f8", ("y", "x"))
        variable.standard_name = role
        variable.units = units
        variable.bounds = f"vertices_{role}"
        nc.createVariable(f"vertices_{role}", "f8", ("y", "x", "vertices"))
    ta = nc.createVariable("ta", "f4", ("y", "x"))
    ta.coordinates = "lat lon"

    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    assert sum("identified by standard_name" in msg for msg in messages(result)) == 2

    result = check_coordinate_catalog(
        nc,
        grid_catalog(),
        severities=FAMILIES,
        grid_topology="curvilinear",
        allow_standard_name_fallback=False,
    )
    found = messages(result)
    assert not any("identified by standard_name" in msg for msg in found)
    assert any(
        "required horizontal coordinate(s) are absent: ['longitude', 'latitude']"
        in msg
        for msg in found
    )


def test_rectilinear_grid_always_prefers_esgvoc_name(nc):
    nc.createDimension("lat", 2)
    nc.createDimension("lon", 3)
    nc.createDimension("bnds", 2)
    for role, name, alias, axis, units, data, bounds_data in (
        (
            "latitude",
            "lat",
            "latitude",
            "Y",
            "degrees_north",
            [-45, 45],
            [[-90, 0], [0, 90]],
        ),
        (
            "longitude",
            "lon",
            "longitude",
            "X",
            "degrees_east",
            [0, 120, 240],
            [[-60, 60], [60, 180], [180, 300]],
        ),
    ):
        variable = nc.createVariable(name, "f8", (name,))
        variable[:] = data
        variable.standard_name = role
        variable.axis = axis
        variable.units = units
        variable.bounds = f"{name}_bnds"
        nc.createVariable(f"{name}_bnds", "f8", (name, "bnds"))[:] = bounds_data
        fallback = nc.createVariable(alias, "f8", (name,))
        fallback.standard_name = role
    ta = nc.createVariable("ta", "f4", ("lat", "lon"))
    ta.coordinates = "latitude longitude"

    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="rectilinear"
    )

    assert not any("standard_name=" in msg for msg in messages(result))


@pytest.mark.parametrize(
    ("x_name", "y_name", "x_standard", "y_standard", "units", "dtype"),
    [
        (
            "x",
            "y",
            "projection_x_angular_coordinate",
            "projection_y_angular_coordinate",
            "degrees",
            "f8",
        ),
        ("i", "j", "", "", "1", "i4"),
    ],
)
def test_curvilinear_grid_accepts_angular_and_index_axes(
    nc, x_name, y_name, x_standard, y_standard, units, dtype
):
    nc.createDimension(y_name, 2)
    nc.createDimension(x_name, 3)
    nc.createDimension("vertices", 4)
    for name, axis, standard in (
        (x_name, "X", x_standard),
        (y_name, "Y", y_standard),
    ):
        variable = nc.createVariable(name, dtype, (name,))
        if standard:
            variable.axis = axis
            variable.standard_name = standard
        variable.units = units
        variable[:] = np.arange(len(nc.dimensions[name]), dtype=variable.dtype)
    for role, role_units in (
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
    ):
        variable = nc.createVariable(role, "f8", (y_name, x_name))
        variable.standard_name = role
        variable.units = role_units
        variable.bounds = f"vertices_{role}"
        nc.createVariable(f"vertices_{role}", "f8", (y_name, x_name, "vertices"))
    ta = nc.createVariable("ta", "f4", (y_name, x_name))
    ta.coordinates = "latitude longitude"
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    assert messages(result) == []


def test_unstructured_grid_requires_shared_cell_dimension_and_vertices(nc):
    nc.createDimension("cell", 3)
    nc.createDimension("vertices", 4)
    for role, units in (
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
    ):
        variable = nc.createVariable(role, "f8", ("cell",))
        variable.standard_name = role
        variable.units = units
        variable.bounds = f"vertices_{role}"
        nc.createVariable(f"vertices_{role}", "f8", ("cell", "vertices"))
    ta = nc.createVariable("ta", "f4", ("cell",))
    ta.coordinates = "latitude longitude"
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="unstructured"
    )
    assert messages(result) == []


def test_vertex_dimension_name_is_a_configurable_recommendation(nc):
    nc.createDimension("cell", 3)
    nc.createDimension("nv", 4)
    for role, units in (
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
    ):
        variable = nc.createVariable(role, "f8", ("cell",))
        variable.standard_name = role
        variable.units = units
        variable.bounds = f"vertices_{role}"
        nc.createVariable(f"vertices_{role}", "f8", ("cell", "nv"))
    ta = nc.createVariable("ta", "f4", ("cell",))
    ta.coordinates = "latitude longitude"

    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="unstructured"
    )
    naming = next(item for item in result if "COORD009" in item.name)
    assert naming.weight == BaseCheck.MEDIUM
    assert (
        len([msg for msg in naming.msgs if "recommended name is 'vertices'" in msg])
        == 2
    )

    result = check_coordinate_catalog(
        nc,
        grid_catalog(),
        severities=FAMILIES,
        grid_topology="unstructured",
        vertices_dimension_name="nv",
    )
    naming = next(item for item in result if "COORD009" in item.name)
    assert naming.value == (1, 1)


def test_vertices_require_the_parent_coordinate_dimensions(nc):
    nc.createDimension("cell", 3)
    nc.createDimension("other", 3)
    nc.createDimension("vertices", 4)
    for role, units in (
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
    ):
        variable = nc.createVariable(role, "f8", ("cell",))
        variable.standard_name = role
        variable.units = units
        variable.bounds = f"vertices_{role}"
        dimension = "other" if role == "latitude" else "cell"
        nc.createVariable(f"vertices_{role}", "f8", (dimension, "vertices"))
    ta = nc.createVariable("ta", "f4", ("cell",))
    ta.coordinates = "latitude longitude"

    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="unstructured"
    )
    grid = next(item for item in result if "COORD011" in item.name)
    assert grid.weight == BaseCheck.HIGH
    assert any(
        "required for vertex variable 'vertices_latitude' to have the coordinate dimensions ['cell']"
        in msg
        for msg in grid.msgs
    )


def test_curvilinear_grid_accepts_implicit_index_dimensions(nc):
    nc.createDimension("row", 2)
    nc.createDimension("column", 3)
    nc.createDimension("vertices", 4)
    for role, units in (
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
    ):
        variable = nc.createVariable(role, "f8", ("row", "column"))
        variable.standard_name = role
        variable.units = units
        variable.bounds = f"vertices_{role}"
        nc.createVariable(f"vertices_{role}", "f8", ("row", "column", "vertices"))
    ta = nc.createVariable("ta", "f4", ("row", "column"))
    ta.coordinates = "latitude longitude"
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    assert messages(result) == []


def test_rotated_grid_accepts_rlon_rlat_axis_pair(nc):
    nc.createDimension("rlat", 2)
    nc.createDimension("rlon", 3)
    nc.createDimension("vertices", 4)
    for name, axis, standard in (
        ("rlon", "X", "grid_longitude"),
        ("rlat", "Y", "grid_latitude"),
    ):
        variable = nc.createVariable(name, "f8", (name,))
        variable.axis = axis
        variable.standard_name = standard
        variable.units = "degrees"
        variable[:] = np.arange(len(nc.dimensions[name]), dtype="float64")
    for role, units in (
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
    ):
        variable = nc.createVariable(role, "f8", ("rlat", "rlon"))
        variable.standard_name = role
        variable.units = units
        variable.bounds = f"vertices_{role}"
        nc.createVariable(f"vertices_{role}", "f8", ("rlat", "rlon", "vertices"))
    mapping = nc.createVariable("rotated_pole", "i4")
    mapping.grid_mapping_name = "rotated_latitude_longitude"
    ta = nc.createVariable("ta", "f4", ("rlat", "rlon"))
    ta.coordinates = "latitude longitude"
    ta.grid_mapping = "rotated_pole"
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    assert messages(result) == []


def test_rectilinear_grid_uses_generic_coordinate_records(nc):
    nc.createDimension("lat", 2)
    nc.createDimension("lon", 3)
    nc.createDimension("bnds", 2)
    for role, name, axis, units, data in (
        ("latitude", "lat", "Y", "degrees_north", [-45, 45]),
        ("longitude", "lon", "X", "degrees_east", [0, 120, 240]),
    ):
        variable = nc.createVariable(name, "f8", (name,))
        variable[:] = data
        variable.standard_name = role
        variable.axis = axis
        variable.units = units
        variable.bounds = f"{name}_bnds"
        bounds = nc.createVariable(f"{name}_bnds", "f8", (name, "bnds"))
        if role == "latitude":
            bounds[:] = [[-90, 0], [0, 90]]
        else:
            bounds[:] = [[-60, 60], [60, 180], [180, 300]]
    nc.createVariable("ta", "f4", ("lat", "lon"))
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="rectilinear"
    )
    assert messages(result) == []


def test_rotated_mapping_requires_rlon_rlat_axis_pair(nc):
    nc.createDimension("y", 2)
    nc.createDimension("x", 3)
    nc.createDimension("vertices", 4)
    for name, axis, standard in (
        ("x", "X", "projection_x_coordinate"),
        ("y", "Y", "projection_y_coordinate"),
    ):
        variable = nc.createVariable(name, "f8", (name,))
        variable.axis = axis
        variable.standard_name = standard
        variable.units = "m"
    for role, units in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
        variable = nc.createVariable(role, "f8", ("y", "x"))
        variable.standard_name = role
        variable.units = units
        variable.bounds = f"vertices_{role}"
        nc.createVariable(f"vertices_{role}", "f8", ("y", "x", "vertices"))
    mapping = nc.createVariable("rotated_pole", "i4")
    mapping.grid_mapping_name = "rotated_latitude_longitude"
    ta = nc.createVariable("ta", "f4", ("y", "x"))
    ta.coordinates = "latitude longitude"
    ta.grid_mapping = "rotated_pole"
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    assert any(
        "required axis pair for grid_mapping_name='rotated_latitude_longitude' "
        "is the grid_longitude/grid_latitude pair" in msg
        for msg in messages(result)
    )


def test_generic_vertical_is_selected_by_standard_name_and_validates_formula(nc):
    nc.createDimension("lev", 2)
    nc.createDimension("bnds", 2)
    lev = nc.createVariable("lev", "f8", ("lev",))
    lev[:] = [1.0, 0.5]
    lev.axis = "Z"
    lev.standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
    lev.units = "1"
    lev.positive = "down"
    lev.formula = "p = a*p0 + b*ps"
    lev.formula_terms = "p0: p0 a: a b: b ps: ps"
    lev.bounds = "lev_bnds"
    lev_bnds = nc.createVariable("lev_bnds", "f8", ("lev", "bnds"))
    lev_bnds[:] = [[1.2, 0.8], [0.8, 0.2]]
    lev_bnds.formula_terms = "p0: p0 a: a_bnds b: b_bnds ps: ps"
    nc.createVariable("p0", "f8")
    nc.createVariable("ps", "f8")
    nc.createVariable("a", "f8", ("lev",))
    nc.createVariable("b", "f8", ("lev",))
    nc.createVariable("a_bnds", "f8", ("lev", "bnds"))
    nc.createVariable("b_bnds", "f8", ("lev", "bnds"))
    nc.variables["p0"].assignValue(1000.0)
    nc.variables["ps"].assignValue(100000.0)
    nc.variables["a"][:] = [0.1, 0.05]
    nc.variables["b"][:] = [0.9, 0.45]
    for name in ("p0", "a", "b", "ps", "a_bnds", "b_bnds"):
        nc.variables[name].long_name = f"Formula term {name}"
    nc.createVariable("ta", "f4", ("lev",))

    generic = coordinate("alevel", "generic_vertical", "lev", axis="Z")
    level = {
        "id": "standard_hybrid_sigma",
        "generic_level_name": "alevel",
        "out_name": "lev",
        "axis": "Z",
        "data_type": "double",
        "cf_standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
        "units": "1",
        "positive": "down",
        "stored_direction": "decreasing",
        "bounds_required": True,
        "formula": "p = a*p0 + b*ps",
        "z_factors": ["p0", "a", "b", "ps"],
        "z_bounds_factors": ["p0", "a_bnds", "b_bnds", "ps"],
    }
    alternate_level = {
        **level,
        "id": "alternate_hybrid_sigma",
        "formula": "p = ap + b*ps",
        "z_factors": ["ap", "b", "ps"],
        "z_bounds_factors": ["ap_bnds", "b_bnds", "ps"],
    }
    terms = {
        name: {
            "id": name,
            "out_name": name,
            "data_type": "double",
            "long_name": f"Formula term {name}",
            "dimensions": ["alevel"] if name in {"a", "b", "a_bnds", "b_bnds"} else [],
        }
        for name in ("p0", "a", "b", "ps", "a_bnds", "b_bnds")
    }
    result = check_coordinate_catalog(
        nc,
        catalog(
            {"alevel": generic},
            model_levels={
                "standard_hybrid_sigma": level,
                "alternate_hybrid_sigma": alternate_level,
            },
            formula_terms=terms,
        ),
        severities=FAMILIES,
    )
    assert messages(result) == []
    assert "generic_level_name" not in lev.ncattrs()


@pytest.mark.parametrize("reverse_physical_profile", [False, True])
def test_generic_vertical_checks_formula_profile_at_one_horizontal_cell(
    nc, reverse_physical_profile
):
    nc.createDimension("lev", 3)
    nc.createDimension("lat", 2)
    nc.createDimension("lon", 2)
    nc.createDimension("bnds", 2)
    lev = nc.createVariable("lev", "f8", ("lev",))
    lev[:] = [1.0, 0.5, 0.0]
    lev.axis = "Z"
    lev.standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
    lev.units = "1"
    lev.positive = "down"
    lev.formula = "p = a + b*ps"
    lev.formula_terms = "a: a b: b ps: ps"
    a = nc.createVariable("a", "f8", ("lev",))
    a[:] = [0.0, 0.5, 1.0] if reverse_physical_profile else [1.0, 0.5, 0.0]
    nc.createVariable("b", "f8", ("lev",))[:] = 0.0
    nc.createVariable("ps", "f8", ("lat", "lon"))[:] = 100000.0

    for role, name, axis, units, data, bounds in (
        (
            "latitude",
            "lat",
            "Y",
            "degrees_north",
            [-45.0, 45.0],
            [[-90.0, 0.0], [0.0, 90.0]],
        ),
        (
            "longitude",
            "lon",
            "X",
            "degrees_east",
            [0.0, 180.0],
            [[-90.0, 90.0], [90.0, 270.0]],
        ),
    ):
        variable = nc.createVariable(name, "f8", (name,))
        variable[:] = data
        variable.axis = axis
        variable.standard_name = role
        variable.units = units
        variable.bounds = f"{name}_bnds"
        nc.createVariable(f"{name}_bnds", "f8", (name, "bnds"))[:] = bounds
    nc.createVariable("ta", "f4", ("lat", "lon", "lev"))

    horizontal = grid_catalog()
    generic = coordinate("alevel", "generic_vertical", "lev", axis="Z")
    level = {
        "id": "hybrid",
        "generic_level_name": "alevel",
        "out_name": "lev",
        "axis": "Z",
        "data_type": "double",
        "cf_standard_name": lev.standard_name,
        "units": "1",
        "positive": "down",
        "stored_direction": "decreasing",
        "formula": lev.formula,
        "z_factors": ["a", "b", "ps"],
        "bounds_required": False,
    }
    terms = {
        "a": {
            "id": "a",
            "out_name": "a",
            "data_type": "double",
            "dimensions": ["alevel"],
        },
        "b": {
            "id": "b",
            "out_name": "b",
            "data_type": "double",
            "dimensions": ["alevel"],
        },
        "ps": {
            "id": "ps",
            "out_name": "ps",
            "data_type": "double",
            "dimensions": ["longitude", "latitude"],
        },
    }
    cv = Catalog(
        project_id=horizontal.project_id,
        branded_variable_id=horizontal.branded_variable_id,
        branded_variable=horizontal.branded_variable,
        coordinate_ids=("alevel", "longitude", "latitude"),
        data_coordinates={**horizontal.data_coordinates, "alevel": generic},
        model_levels={"hybrid": level},
        formula_terms=terms,
        grid_variables=horizontal.grid_variables,
        grid_axes=horizontal.grid_axes,
    )

    result = check_coordinate_catalog(
        nc, cv, severities=FAMILIES, grid_topology="rectilinear"
    )
    direction = next(item for item in result if "COORD005" in item.name)
    if reverse_physical_profile:
        assert any(
            "Formula-derived physical profile" in message
            and "not strictly decreasing" in message
            for message in direction.msgs
        )
        result = check_coordinate_catalog(
            nc,
            cv,
            severities=FAMILIES,
            grid_topology="rectilinear",
            check_formula_derived_profile=False,
        )
        direction = next(item for item in result if "COORD005" in item.name)
        assert direction.value == (1, 1)
    else:
        assert direction.value == (1, 1)


def test_formula_profile_exception_is_contained(nc, monkeypatch):
    cv = _sigma_file_and_catalog(nc)

    def fail_profile(*args, **kwargs):
        raise RuntimeError("profile unavailable")

    monkeypatch.setattr(
        "checks.coordinate_checks.vertical.formula_vertical_profile",
        fail_profile,
    )

    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)

    assert len(result) == len(FAMILIES)
    direction = next(item for item in result if "COORD005" in item.name)
    assert any(
        "RuntimeError: profile unavailable" in message for message in direction.msgs
    )
    assert next(item for item in result if "COORD012" in item.name).value == (1, 1)


def test_formula_term_long_name_uses_complete_descriptor(nc):
    nc.createDimension("lev", 2)
    lev = nc.createVariable("lev", "f8", ("lev",))
    lev[:] = [1.0, 0.5]
    lev.axis = "Z"
    lev.standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
    lev.units = "1"
    lev.positive = "down"
    lev.formula = "p = ap + b*ps"
    lev.formula_terms = "ap: ap b: b ps: ps"
    for name, dimensions in (("ap", ("lev",)), ("b", ("lev",)), ("ps", ())):
        variable = nc.createVariable(name, "f8", dimensions)
        if name in {"ap", "b"}:
            variable.long_name = f"vertical coordinate formula term: {name}(k)"
    nc.createVariable("ta", "f4", ("lev",))

    generic = coordinate("alevel", "generic_vertical", "lev", axis="Z")
    level = {
        "id": "alternate_hybrid_sigma",
        "generic_level_name": "alevel",
        "out_name": "lev",
        "axis": "Z",
        "data_type": "double",
        "cf_standard_name": lev.standard_name,
        "units": "1",
        "positive": "down",
        "stored_direction": "decreasing",
        "formula": lev.formula,
        # Resolved relationship records do not necessarily carry long_name.
        "z_factors": [{"id": name, "out_name": name} for name in ("ap", "b", "ps")],
        "bounds_required": False,
    }
    terms = {
        name: {
            "id": name,
            "out_name": name,
            "data_type": "double",
            "long_name": (
                f"vertical coordinate formula term: {name}"
                if name in {"ap", "b"}
                else ""
            ),
            "dimensions": ["alevel"] if name in {"ap", "b"} else [],
        }
        for name in ("ap", "b", "ps")
    }
    result = check_coordinate_catalog(
        nc,
        catalog(
            {"alevel": generic},
            model_levels={"alternate_hybrid_sigma": level},
            formula_terms=terms,
        ),
        severities=FAMILIES,
    )
    for name in ("ap", "b"):
        assert any(
            f"'{name}' long_name='vertical coordinate formula term: {name}(k)'; "
            f"the recommended value is 'vertical coordinate formula term: {name}'"
            in msg
            for msg in messages(result)
        )


def test_formula_factor_resolution_distinguishes_main_and_bounds_aliases():
    dimensions = [{"id": "alevel"}]
    terms = {
        "ap": {
            "id": "ap",
            "out_name": "ap",
            "long_name": "vertical coordinate formula term: ap",
            "dimensions": dimensions,
        },
        "ap_bnds": {
            "id": "ap_bnds",
            "out_name": "ap_bnds",
            "long_name": "vertical coordinate formula term: ap(k+1/2)",
            "dimensions": dimensions,
        },
    }
    # This mirrors the resolved ESGVoc relationship where both fields expose
    # the bounds reference. The field context determines the intended record.
    level = {
        "out_name": "lev",
        "formula": "p = ap",
        "z_factors": [{"id": "ap_bnds", "out_name": "ap_bnds"}],
        "z_bounds_factors": [{"id": "ap_bnds", "out_name": "ap_bnds"}],
    }
    resolved = {}

    assert expected_formula_terms(
        level,
        terms,
        generic_id="alevel",
        resolved=resolved,
    ) == {"ap": "ap"}
    assert resolved["ap"]["long_name"] == "vertical coordinate formula term: ap"
    assert expected_formula_terms(
        level,
        terms,
        generic_id="alevel",
        bounds=True,
    ) == {"ap": "ap_bnds"}


def test_generic_vertical_reports_formula_terms_from_selected_esgvoc_entry(nc):
    nc.createDimension("lev", 2)
    lev = nc.createVariable("lev", "f8", ("lev",))
    lev[:] = [1, 0]
    lev.axis = "Z"
    lev.standard_name = "model_level_number"
    lev.units = "1"
    lev.positive = "down"
    lev.formula = "z = lev"
    lev.formula_terms = "sigma: wrong"
    nc.createVariable("ta", "f4", ("lev",))
    generic = coordinate("alevel", "generic_vertical", "lev", axis="Z")
    level = {
        "id": "model_levels",
        "generic_level_name": "alevel",
        "out_name": "lev",
        "axis": "Z",
        "data_type": "double",
        "cf_standard_name": "model_level_number",
        "units": "1",
        "positive": "down",
        "stored_direction": "decreasing",
        "formula": "z = lev",
        "z_factors": [],
        "bounds_required": False,
    }
    result = check_coordinate_catalog(
        nc,
        catalog({"alevel": generic}, model_levels={"model_levels": level}),
        severities=FAMILIES,
    )
    assert any("formula_terms={'sigma': 'wrong'}" in msg for msg in messages(result))


def ambiguous_model_levels(generic_id="alevel", out_name="lev"):
    common = {
        "generic_level_name": generic_id,
        "out_name": out_name,
        "axis": "Z",
        "data_type": "double",
        "cf_standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
        "units": "1",
        "positive": "down",
        "bounds_required": False,
    }
    return {
        "standard_hybrid_sigma": {
            **common,
            "id": "standard_hybrid_sigma",
            "formula": "p = a*p0 + b*ps",
        },
        "alternate_hybrid_sigma": {
            **common,
            "id": "alternate_hybrid_sigma",
            "formula": "p = ap + b*ps",
        },
    }


def test_generic_vertical_missing_variable_names_esgvoc_out_name(nc):
    generic = coordinate("alevel", "generic_vertical", "model_lev", axis="Z")
    result = check_coordinate_catalog(
        nc,
        catalog(
            {"alevel": generic},
            model_levels=ambiguous_model_levels(out_name="model_lev"),
        ),
        severities=FAMILIES,
    )
    assert any(
        "Coordinate 'alevel' could not be verified. Expected is a generic "
        "vertical coordinate with out_name 'model_lev'. Variable 'model_lev' "
        "does not exist" in msg
        for msg in messages(result)
    )


def test_generic_vertical_requires_standard_name_for_model_selection(nc):
    nc.createDimension("lev", 2)
    nc.createVariable("lev", "f8", ("lev",))
    nc.createVariable("ta", "f4", ("lev",))
    generic = coordinate("alevel", "generic_vertical", "lev", axis="Z")
    result = check_coordinate_catalog(
        nc,
        catalog({"alevel": generic}, model_levels=ambiguous_model_levels()),
        severities=FAMILIES,
    )
    assert any(
        "has no standard_name, which is required to select" in msg
        for msg in messages(result)
    )


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        (None, "has no formula with which to select one"),
        ("p = wrong", "does not match any candidate formula"),
    ],
)
def test_generic_vertical_formula_explains_ambiguous_selection(nc, formula, expected):
    nc.createDimension("lev", 2)
    lev = nc.createVariable("lev", "f8", ("lev",))
    lev.standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
    if formula is not None:
        lev.formula = formula
    nc.createVariable("ta", "f4", ("lev",))
    generic = coordinate("alevel", "generic_vertical", "lev", axis="Z")
    result = check_coordinate_catalog(
        nc,
        catalog({"alevel": generic}, model_levels=ambiguous_model_levels()),
        severities=FAMILIES,
    )
    found = messages(result)
    assert any("Coordinate 'alevel' could not be verified" in msg for msg in found)
    assert any(expected in msg for msg in found)


@pytest.mark.parametrize("invalid", [-999.0, np.nan, np.inf])
def test_nonfinite_coordinates_fail_without_losing_bounds_alignment(nc, invalid):
    nc.createDimension("z", 3)
    nc.createDimension("bnds", 2)
    var = nc.createVariable("z", "f8", ("z",), fill_value=-999.0)
    var[:] = [1.0, invalid, 3.0]
    var.long_name = "Model coordinate"
    var.bounds = "z_bnds"
    nc.createVariable("z_bnds", "f8", ("z", "bnds"))[:] = [
        [0.5, 1.5],
        [1.5, 2.5],
        [2.5, 3.5],
    ]
    nc.createVariable("ta", "f4", ("z",))
    result = check_coordinate_catalog(
        nc, catalog({"z": coordinate("z", "standard_1d", "z")}), severities=FAMILIES
    )
    for code in ("COORD005", "COORD008"):
        finding = next(r for r in result if code in r.name)
        assert finding.msgs, code
        assert finding.weight == BaseCheck.HIGH


def test_missing_bounds_values_fail_explicitly(nc):
    nc.createDimension("z", 2)
    nc.createDimension("bnds", 2)
    var = nc.createVariable("z", "f8", ("z",))
    var[:] = [1.0, 2.0]
    var.long_name = "Model coordinate"
    var.bounds = "z_bnds"
    bounds = nc.createVariable("z_bnds", "f8", ("z", "bnds"))
    bounds[:] = np.ma.array(
        [[0.5, 1.5], [1.5, 2.5]], mask=[[False, True], [False, False]]
    )
    result = check_coordinate_catalog(
        nc, catalog({"z": coordinate("z", "standard_1d", "z")}), severities=FAMILIES
    )
    assert any(
        "'z_bnds' contains missing or non-finite values" in msg
        for msg in messages(result)
    )


@pytest.mark.parametrize("dimensions", [(), ("lat", "extra")])
def test_malformed_rectilinear_coordinate_returns_findings(nc, dimensions):
    for name in ("lat", "lon", "extra"):
        nc.createDimension(name, 2)
    for role, name, dims in [
        ("latitude", "lat", dimensions),
        ("longitude", "lon", ("lon",)),
    ]:
        var = nc.createVariable(name, "f8", dims)
        var.standard_name = role
        var[:] = 1.0
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="rectilinear"
    )
    assert any("1-D coordinate variable" in msg for msg in messages(result))


@pytest.mark.parametrize("climatology", [False, True])
def test_scalar_time_does_not_crash_dependent_bounds_checks(nc, climatology):
    nc.createDimension("time", 1)
    nc.createDimension("bnds", 2)
    time = nc.createVariable("time", "f8")
    time.assignValue(0.5)
    time.axis = "T"
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time.calendar = "standard"
    time.setncattr("climatology" if climatology else "bounds", "time_bnds")
    nc.createVariable("time_bnds", "f8", ("time", "bnds"))[:] = [[0, 1]]
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checker._coordinate_catalog = catalog(
        {
            "time1": coordinate(
                "time1",
                "standard_1d",
                "time",
                axis="T",
                is_climatology=climatology,
                cf_standard_name="time",
                units="days since ?",
                bounds_required=True,
            )
        }
    )
    result = checker.check_Coordinate_Standard(nc)
    assert next(r for r in result if "COORD001" in r.name).msgs


def _projected_grid_file(nc, *, transpose=False, unrelated_axes=False):
    for name, length in (("x", 3), ("y", 2), ("unrelated", 4), ("vertices", 4)):
        nc.createDimension(name, length)
    for name, axis in (("x", "X"), ("y", "Y")):
        dimension = "unrelated" if unrelated_axes else name
        var = nc.createVariable(name, "f8", (dimension,))
        var.axis = axis
        var.standard_name = f"projection_{name}_coordinate"
        var.units = "m"
        var[:] = np.arange(len(nc.dimensions[dimension]))
    dimensions = ("x", "y") if transpose else ("y", "x")
    for role, units in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
        var = nc.createVariable(role, "f8", dimensions)
        var.standard_name = role
        var.units = units
        var.bounds = f"vertices_{role}"
        var[:] = 0.0
        nc.createVariable(f"vertices_{role}", "f8", (*dimensions, "vertices"))[:] = 0.0
    nc.createVariable("ta", "f4", dimensions).coordinates = "latitude longitude"


def test_explicit_grid_axes_require_their_own_dimensions(nc):
    _projected_grid_file(nc, unrelated_axes=True)
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    found = messages(result)
    assert any("'x(x)'" in msg and "unrelated" in msg for msg in found)
    assert any("'y(y)'" in msg and "unrelated" in msg for msg in found)


def test_transposed_curvilinear_grid_is_checked_against_cv_dimensions(nc):
    _projected_grid_file(nc, transpose=True)
    result = check_coordinate_catalog(
        nc, grid_catalog(), severities=FAMILIES, grid_topology="curvilinear"
    )
    for name in ("latitude", "longitude", "vertices_latitude", "vertices_longitude"):
        assert any(
            f"'{name}'" in msg and "dimension order" in msg for msg in messages(result)
        )
    assert next(r for r in result if "COORD002" in r.name).msgs


def test_vertex_dimensions_use_their_own_cv_record(nc):
    _projected_grid_file(nc)
    cv = grid_catalog()
    cv.grid_variables["vertices_latitude"]["dimensions"] = [
        "longitude",
        "latitude",
        "vertices",
    ]
    result = check_coordinate_catalog(
        nc, cv, severities=FAMILIES, grid_topology="curvilinear"
    )
    assert any(
        "'vertices_latitude'" in msg and "dimension order" in msg
        for msg in messages(result)
    )


def _sigma_file_and_catalog(nc, *, bounds_name="lev_bnds", ptop_dimensions=()):
    nc.createDimension("lev", 2)
    nc.createDimension("bnds", 2)
    lev = nc.createVariable("lev", "f8", ("lev",))
    lev[:] = [0.75, 0.25]
    lev.axis = "Z"
    lev.standard_name = "atmosphere_sigma_coordinate"
    lev.positive = "down"
    lev.formula = "p = ptop + sigma*(ps - ptop)"
    lev.formula_terms = "ptop: ptop ps: ps sigma: lev"
    lev.bounds = bounds_name
    bounds = nc.createVariable(bounds_name, "f8", ("lev", "bnds"))
    bounds[:] = [[1.0, 0.5], [0.5, 0.0]]
    bounds.formula_terms = f"ptop: ptop ps: ps sigma: {bounds_name}"
    nc.createVariable("ptop", "f8", ptop_dimensions)[:] = 100.0
    nc.createVariable("ps", "f8")[:] = 100000.0
    nc.createVariable("ta", "f4", ("lev",))
    level = {
        "id": "sigma",
        "generic_level_name": "alevel",
        "out_name": "lev",
        "axis": "Z",
        "data_type": "double",
        "cf_standard_name": lev.standard_name,
        "positive": "down",
        "stored_direction": "decreasing",
        "formula": lev.formula,
        "z_factors": ["ptop", "ps"],
        "z_bounds_factors": ["ptop", "ps"],
        "bounds_required": True,
    }
    return catalog(
        {"alevel": coordinate("alevel", "generic_vertical", "lev", axis="Z")},
        model_levels={"sigma": level},
        formula_terms={
            name: {"id": name, "out_name": name, "data_type": "double"}
            for name in ("ptop", "ps")
        },
    )


def test_generic_vertical_computed_standard_name_is_required_only_when_defined(nc):
    cv = _sigma_file_and_catalog(nc)
    lev = nc.variables["lev"]
    lev.computed_standard_name = "model_specific_height"

    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)
    assert any(
        "defines computed_standard_name='model_specific_height'" in msg
        and "attribute to be absent" in msg
        for msg in messages(result)
    )

    result = check_coordinate_catalog(
        nc,
        cv,
        severities=FAMILIES_WITH_INFORMATION,
        attributes_allowed_when_unset=("computed_standard_name",),
    )
    metadata = next(item for item in result if "COORD003" in item.name)
    assert metadata.value == (1, 1)
    information = next(item for item in result if "COORD004a" in item.name)
    assert information.weight == BaseCheck.LOW
    assert any(
        "computed_standard_name='model_specific_height'" in msg
        and "permitted by the configured allowed_when_unset" in msg
        for msg in information.msgs
    )

    cv.model_levels["sigma"]["computed_standard_name"] = "air_pressure"
    result = check_coordinate_catalog(
        nc,
        cv,
        severities=FAMILIES_WITH_INFORMATION,
        attributes_allowed_when_unset=("computed_standard_name",),
    )
    metadata = next(item for item in result if "COORD003" in item.name)
    assert metadata.weight == BaseCheck.HIGH
    assert any(
        "'lev' computed_standard_name='model_specific_height'; the required value "
        "is 'air_pressure'" in msg
        for msg in metadata.msgs
    )

    lev.computed_standard_name = "air_pressure"
    result = check_coordinate_catalog(
        nc,
        cv,
        severities=FAMILIES_WITH_INFORMATION,
        attributes_allowed_when_unset=("computed_standard_name",),
    )
    assert not any("computed_standard_name" in msg for msg in messages(result))


def test_cmip7_reports_allowed_computed_standard_name_at_low_severity(nc):
    cv = _sigma_file_and_catalog(nc)
    nc.variables["lev"].computed_standard_name = "model_specific_height"
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checker._coordinate_catalog = cv

    result = checker.check_Coordinate_Standard(nc)

    information = next(item for item in result if "COORD004a" in item.name)
    assert information.weight == BaseCheck.LOW
    assert any(
        "computed_standard_name='model_specific_height'" in message
        for message in information.msgs
    )


def test_generic_vertical_empty_cv_attribute_requires_file_attribute_absence(nc):
    cv = _sigma_file_and_catalog(nc)
    cv.model_levels["sigma"]["positive"] = ""

    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)

    metadata = next(item for item in result if "COORD003" in item.name)
    assert metadata.weight == BaseCheck.HIGH
    assert any(
        "defines positive='down'" in message and "attribute to be absent" in message
        for message in metadata.msgs
    )


@pytest.mark.parametrize("file_value", ["z = unexpected", ""])
def test_generic_vertical_empty_cv_formula_requires_attribute_absence(nc, file_value):
    cv = _sigma_file_and_catalog(nc)
    cv.model_levels["sigma"]["formula"] = ""
    nc.variables["lev"].formula = file_value

    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)

    formula = next(item for item in result if "COORD012" in item.name)
    assert formula.weight == BaseCheck.HIGH
    assert any(
        "formula" in message and "attribute to be absent" in message
        for message in formula.msgs
    )

    result = check_coordinate_catalog(
        nc,
        cv,
        severities=FAMILIES_WITH_INFORMATION,
        attributes_allowed_when_unset=("formula",),
    )
    assert not any("attribute to be absent" in message for message in messages(result))
    information = next(item for item in result if "COORD004a" in item.name)
    assert information.weight == BaseCheck.LOW
    assert any("defines formula=" in message for message in information.msgs)


@pytest.mark.parametrize("file_value", ["sigma: lev", ""])
def test_generic_vertical_without_formula_terms_requires_attribute_absence(
    nc, file_value
):
    cv = _sigma_file_and_catalog(nc)
    cv.model_levels["sigma"]["formula"] = ""
    cv.model_levels["sigma"]["z_factors"] = []
    nc.variables["lev"].delncattr("formula")
    nc.variables["lev"].formula_terms = file_value

    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)

    formula = next(item for item in result if "COORD012" in item.name)
    assert formula.weight == BaseCheck.HIGH
    assert any(
        "defines formula_terms=" in message and "attribute to be absent" in message
        for message in formula.msgs
    )

    result = check_coordinate_catalog(
        nc,
        cv,
        severities=FAMILIES_WITH_INFORMATION,
        attributes_allowed_when_unset=("formula_terms",),
    )
    assert not any("attribute to be absent" in message for message in messages(result))
    information = next(item for item in result if "COORD004a" in item.name)
    assert information.weight == BaseCheck.LOW
    assert any("defines formula_terms=" in message for message in information.msgs)


@pytest.mark.parametrize("dimensions", [(), ("lev",)])
def test_scalar_formula_terms_are_dimensionless(nc, dimensions):
    cv = _sigma_file_and_catalog(nc, ptop_dimensions=dimensions)
    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)
    if dimensions:
        assert any(
            "Formula term 'ptop' has dimensions ['lev']" in msg
            for msg in messages(result)
        )
    else:
        assert messages(result) == []


@pytest.mark.parametrize("alternate_first", [False, True])
def test_formula_factor_descriptor_selection_is_independent_of_collection_order(
    nc, alternate_first
):
    cv = _sigma_file_and_catalog(nc)
    # The same file name can refer to a descriptor with a different type/dimensions.
    alternate = {
        "ps1": {
            "id": "ps1",
            "out_name": "ps",
            "data_type": "real",
            "dimensions": ["alevel"],
        }
    }
    records = (
        {**alternate, **cv.formula_terms}
        if alternate_first
        else {**cv.formula_terms, **alternate}
    )
    cv.formula_terms.clear()
    cv.formula_terms.update(records)
    assert messages(check_coordinate_catalog(nc, cv, severities=FAMILIES)) == []


@pytest.mark.parametrize("embedded_reference", [False, True])
def test_formula_factor_uses_referenced_id_with_distinct_out_name(
    nc, embedded_reference
):
    cv = _sigma_file_and_catalog(nc)
    cv.formula_terms["ps1"] = {
        "id": "ps1",
        "out_name": "ps",
        "data_type": "double",
    }
    reference = {"id": "ps1"} if embedded_reference else "ps1"
    cv.model_levels["sigma"]["z_factors"] = ["ptop", reference]
    cv.model_levels["sigma"]["z_bounds_factors"] = ["ptop", reference]
    # Only the referenced descriptor is authoritative, even with the same out_name.
    cv.formula_terms["ps"]["data_type"] = "integer"
    assert messages(check_coordinate_catalog(nc, cv, severities=FAMILIES)) == []


def test_missing_formula_descriptor_is_reported(nc):
    cv = _sigma_file_and_catalog(nc)
    del cv.formula_terms["ptop"]
    assert any(
        "'ptop' has no usable coordinate-catalogue descriptor" in msg
        for msg in messages(check_coordinate_catalog(nc, cv, severities=FAMILIES))
    )


def test_renamed_vertical_bounds_do_not_rewrite_cv_formula_terms(nc):
    cv = _sigma_file_and_catalog(nc, bounds_name="custom_bounds")
    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)
    formula = next(r for r in result if "COORD012" in r.name)
    assert formula.weight == BaseCheck.HIGH
    assert any("'sigma': 'lev_bnds'" in msg for msg in formula.msgs)
    assert any("'lev_bnds'" in msg and "absent" in msg for msg in formula.msgs)
    assert next(r for r in result if "COORD009" in r.name).weight == BaseCheck.MEDIUM


@pytest.mark.parametrize("formula", [None, "p = ptop + sigma*(ps - ptop)", "p = wrong"])
def test_model_bounds_formula_is_optional_but_must_match_parent(nc, formula):
    cv = _sigma_file_and_catalog(nc)
    if formula is not None:
        nc.variables["lev_bnds"].formula = formula
    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)
    if formula == "p = wrong":
        assert any(
            "same formula as its parent 'lev'" in msg for msg in messages(result)
        )
    else:
        assert messages(result) == []


@pytest.mark.parametrize("kind", ["ordinary", "time", "model"])
def test_bounds_reject_unexpected_attributes(nc, kind):
    if kind == "model":
        cv = _sigma_file_and_catalog(nc)
        bounds = nc.variables["lev_bnds"]
    else:
        name = "time" if kind == "time" else "z"
        nc.createDimension(name, 2)
        nc.createDimension("bnds", 2)
        var = nc.createVariable(name, "f8", (name,))
        var[:] = [0.5, 1.5]
        var.long_name = name
        var.bounds = f"{name}_bnds"
        entry = coordinate(name, "standard_1d", name)
        if kind == "time":
            var.axis = entry["axis"] = "T"
            var.calendar = "standard"
        bounds = nc.createVariable(var.bounds, "f8", (name, "bnds"))
        bounds[:] = [[0, 1], [1, 2]]
        cv = catalog({name: entry})
    bounds.units = "nonsense"
    result = check_coordinate_catalog(nc, cv, severities=FAMILIES)
    bounds_result = next(r for r in result if "COORD008" in r.name)
    assert any("unexpected attributes ['units']" in msg for msg in bounds_result.msgs)
    assert bounds_result.weight == BaseCheck.HIGH
