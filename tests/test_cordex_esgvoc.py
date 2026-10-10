from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.attribute_checks.check_attribute_suite import check_attribute_suite
from checks.attribute_checks.check_attrs_cordex_cmip6 import check_grid_mapping
from checks.consistency_checks import (
    check_institution_source_consistency as descriptive_consistency,
)
from checks.coordinate_checks.cmor import (
    catalog_from_cmor,
    expected_variable_from_cmor,
)
from checks.coordinate_checks.esgvoc import CoordinateMetadataError, load_catalog
from checks.coordinate_checks.model import Catalog
from checks.coordinate_checks.topology import (
    load_grid_topology_config,
    resolve_grid_topology,
)
from checks.variable_checks.check_coords_cordex_cmip6 import (
    infer_horizontal_topology,
)
from checks.variable_checks.known_branded_variable import (
    lookup_expected_variable_metadata_in_collection,
)
from plugins.cordex_cmip6.cordex_cmip6 import CordexCmip6ProjectCheck
from plugins.wcrp_schema import AttributeRule, CoordinateGlobalRule


def _messages(results):
    return [message for result in results for message in result.msgs]


def test_descriptive_attributes_are_checked_against_their_own_ids(
    tmp_path, monkeypatch
):
    expected_by_collection = {
        "institution_id": ("GERICS", "institution", "Canonical Institution"),
        "source_id": ("REMO", "source", ["Canonical Source", "Canonical Source (2026)"]),
        "driving_source_id": (
            "MPI-ESM",
            "driving_source",
            "Canonical Driving Source",
        ),
    }

    def get_term_in_collection(project_id, collection_id, term_id):
        assert project_id == "cordex-cmip6"
        identifier, term_field, expected = expected_by_collection[collection_id]
        if term_id != identifier.lower():
            return None
        return SimpleNamespace(
            id=identifier.lower(),
            drs_name=identifier,
            **{term_field: expected},
        )

    monkeypatch.setattr(
        descriptive_consistency.voc,
        "get_term_in_collection",
        get_term_in_collection,
    )
    monkeypatch.setattr(descriptive_consistency, "ESG_VOCAB_AVAILABLE", True)

    for collection_id, (identifier, value_attribute, expected) in (
        expected_by_collection.items()
    ):
        accepted = expected[-1] if isinstance(expected, list) else expected
        with Dataset(tmp_path / f"{collection_id}.nc", "w") as dataset:
            dataset.setncattr(collection_id, identifier)
            dataset.setncattr(value_attribute, accepted)
            passing = descriptive_consistency.check_id_attribute_consistency(
                dataset,
                BaseCheck.HIGH,
                project_id="cordex-cmip6",
                id_attribute=collection_id,
                value_attribute=value_attribute,
            )
            assert _messages(passing) == []

            dataset.setncattr(value_attribute, "Incorrect value")
            failing = descriptive_consistency.check_id_attribute_consistency(
                dataset,
                BaseCheck.HIGH,
                project_id="cordex-cmip6",
                id_attribute=collection_id,
                value_attribute=value_attribute,
            )
            assert len(_messages(failing)) == 1
            assert f"{collection_id}={identifier!r}" in _messages(failing)[0]


def test_descriptive_consistency_passes_when_either_attribute_is_missing(tmp_path):
    pairs = (
        ("institution_id", "institution"),
        ("source_id", "source"),
        ("driving_source_id", "driving_source"),
    )
    for id_attribute, value_attribute in pairs:
        for present_attribute, present_value in (
            (id_attribute, "test-id"),
            (value_attribute, "test description"),
        ):
            path = tmp_path / f"{id_attribute}-{present_attribute}.nc"
            with Dataset(path, "w") as dataset:
                dataset.setncattr(present_attribute, present_value)
                results = descriptive_consistency.check_id_attribute_consistency(
                    dataset,
                    BaseCheck.HIGH,
                    project_id="cordex-cmip6",
                    id_attribute=id_attribute,
                    value_attribute=value_attribute,
                )
            assert _messages(results) == []


def test_cordex_config_enables_descriptive_consistency_checks():
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()
    consistency = checker.config.global_.consistency

    assert consistency.institution_id_vs_institution.severity == "H"
    assert consistency.source_id_vs_source.severity == "H"
    assert consistency.driving_source_id_vs_driving_source.severity == "H"


def test_cordex_catalog_uses_universe_when_project_coordinate_collections_are_empty():
    class API:
        def __init__(self):
            self.project_calls = []
            self.universe_calls = []

        def get_term_in_collection(self, project, collection, identifier, fields):
            assert (project, collection) == (
                "cordex-cmip6",
                "known_branded_variable",
            )
            return {
                "id": identifier,
                "variable_root_name": "tas",
                "dimensions": ["time"],
            }

        def get_all_terms_in_collection(self, project, collection, fields):
            self.project_calls.append(collection)
            return []

        def get_all_terms_in_data_descriptor(self, descriptor, fields):
            self.universe_calls.append(descriptor)
            if descriptor == "data_coordinate":
                return [
                    {
                        "id": "time",
                        "coordinate_type": "standard_1d",
                        "axis": "T",
                        "out_name": "time",
                    }
                ]
            return [{"id": f"test_{descriptor}"}]

    api = API()
    result = load_catalog(
        "tas_tavg-h2m-hxy-u",
        project_id="cordex-cmip6",
        api=api,
        installed_version="7.0.0",
        branded_collection="known_branded_variable",
        file_variable_name="tas",
        allow_universe_coordinate_fallback=True,
    )

    expected_collections = [
        "data_coordinate",
        "model_level_coordinate",
        "formula_term",
        "grid_variable",
        "grid_axis",
    ]
    assert api.project_calls == expected_collections
    assert api.universe_calls == expected_collections
    assert result.coordinate_ids == ("time",)
    assert result.data_variable_name == "tas"


def test_cordex_variable_lookup_uses_project_collection_names_and_cf_units():
    calls = []

    def get_term(*, project_id, collection_id, term_id, selected_term_fields):
        calls.append((project_id, collection_id, term_id))
        if collection_id == "known_branded_variable":
            return {
                "id": term_id,
                "cf_standard_name": "air_temperature",
                "cf_units": "K",
                "variable_root_name": "tas",
                "dimensions": ["time", "height2m"],
            }
        assert collection_id == "variable_id"
        return {"id": "tas", "long_name": "Near-Surface Air Temperature"}

    lookup = lookup_expected_variable_metadata_in_collection(
        get_term,
        "cordex-cmip6",
        "tas_tavg-h2m-hxy-u",
        fallback_variable_id="tas",
        branded_collection="known_branded_variable",
        variable_collection="variable_id",
    )

    assert lookup.expected.units == "K"
    assert lookup.expected.long_name == "Near-Surface Air Temperature"
    assert [call[1] for call in calls] == [
        "known_branded_variable",
        "variable_id",
    ]


def test_esgvoc_and_cmor_variable_metadata_produce_the_same_failure_message(tmp_path):
    path = tmp_path / "tas.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        variable = dataset.createVariable("tas", "f4", ("time",))
        variable.units = "s"

        source = {
            "standard_name": "air_temperature",
            "units": "K",
            "dimensions": "time",
            "cell_methods": "time: mean",
            "cell_measures": "",
            "long_name": "Near-Surface Air Temperature",
            "out_name": "tas",
        }
        cmor = expected_variable_from_cmor("tas", source)

        def get_term(*, project_id, collection_id, term_id, selected_term_fields):
            return {
                "id": term_id,
                "cf_standard_name": "air_temperature",
                "cf_units": "K",
                "dimensions": ["time"],
                "cell_methods": "time: mean",
                "cell_measures": "",
                "long_name": "Near-Surface Air Temperature",
                "variable_root_name": "tas",
            }

        esgvoc = lookup_expected_variable_metadata_in_collection(
            get_term,
            "cordex-cmip6",
            "tas_tavg-h2m-hxy-u",
            branded_collection="known_branded_variable",
            variable_collection="variable_id",
        ).expected

        def check(expected):
            return check_attribute_suite(
                dataset,
                var_name="tas",
                attribute_name="units",
                severity=BaseCheck.HIGH,
                value_type="str",
                expected_term=expected,
                cv_source_term_key="units",
            )

        assert _messages(check(esgvoc)) == _messages(check(cmor))


def test_cmor_catalog_uses_the_shared_coordinate_model():
    variable = {
        "dimensions": "longitude latitude time",
        "out_name": "tas",
        "cell_methods": "area: time: mean",
    }
    coordinates = {
        "axis_entry": {
            "longitude": {
                "standard_name": "longitude",
                "out_name": "lon",
                "axis": "X",
                "type": "double",
            },
            "latitude": {
                "standard_name": "latitude",
                "out_name": "lat",
                "axis": "Y",
                "type": "double",
            },
            "time": {
                "standard_name": "time",
                "out_name": "time",
                "axis": "T",
                "type": "double",
            },
        }
    }
    catalog = catalog_from_cmor(
        "tas_tavg-h2m-hxy-u",
        "tas",
        variable,
        coordinates,
        {"variable_entry": {}, "axis_entry": {}},
        {"formula_entry": {}},
    )

    assert catalog.data_variable_name == "tas"
    assert catalog.coordinate_ids == ("longitude", "latitude", "time")
    assert catalog.branded_variable["cell_methods"] == "area: time: mean"
    assert catalog.data_coordinates["longitude"]["coordinate_type"] == (
        "generic_horizontal"
    )


def test_cordex_non_latitude_longitude_topology_uses_grid_mapping():
    path = (
        Path(__file__).parents[1]
        / "plugins/cordex_cmip6/config/wcrp/mappings/grid_topology.toml"
    )
    config = load_grid_topology_config(path)

    assert resolve_grid_topology(config, grid_mapping="rotated_latitude_longitude") == (
        "curvilinear",
        None,
    )
    assert resolve_grid_topology(config, grid_mapping="lambert_azimuthal_equal_area") == (
        "curvilinear",
        None,
    )
    topology, error = resolve_grid_topology(config, grid_mapping="healpix")
    assert topology is None
    assert "No horizontal topology is configured" in error


def test_cordex_coordinate_loading_failure_is_reported_once(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise CoordinateMetadataError(
            "ESGVoc returned no data_coordinate records for CORDEX-CMIP6"
        )

    monkeypatch.setattr("plugins.cordex_cmip6.cordex_cmip6.load_catalog", fail)
    checker = CordexCmip6ProjectCheck()
    with Dataset(tmp_path / "tas.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        dataset.createDimension("time", 1)
        dataset.createVariable("tas", "f4", ("time",))
        checker.setup(dataset)
        result = checker.check_Coordinate_Metadata_Setup(dataset)

        assert len(result) == 1
        assert result[0].weight == BaseCheck.HIGH
        assert "no data_coordinate records" in result[0].msgs[0]
        assert checker.check_Coordinate_Standard(dataset) == []


def test_cordex_default_does_not_enable_cmor_tables(tmp_path, monkeypatch):
    checker = CordexCmip6ProjectCheck()

    def unexpected(*args, **kwargs):
        raise AssertionError("CMOR retrieval must not run by default")

    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.retrieve",
        unexpected,
    )
    monkeypatch.setattr(checker, "_initialize_coordinate_catalog", lambda ds: None)
    with Dataset(tmp_path / "tas.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        dataset.createDimension("time", 1)
        dataset.createVariable("tas", "f4", ("time",))
        checker.setup(dataset)

    assert checker.verification_against_tables is False
    assert checker.config.coordinates.registry is not None
    assert len(checker.variable_mapping) > 1000


def _horizontal_catalog():
    coordinates = {
        role: {
            "id": role,
            "coordinate_type": "generic_horizontal",
            "axis": axis,
            "out_name": role,
        }
        for role, axis in (("longitude", "X"), ("latitude", "Y"))
    }
    return Catalog(
        project_id="cordex-cmip6",
        branded_variable_id="tas_tavg-h2m-hxy-u",
        branded_variable={"id": "tas_tavg-h2m-hxy-u", "dimensions": list(coordinates)},
        coordinate_ids=tuple(coordinates),
        data_coordinates=coordinates,
        file_variable_name="tas",
    )


def _complete_horizontal_catalog():
    coordinates = {
        role: {
            "id": role,
            "coordinate_type": "generic_horizontal",
            "axis": axis,
            "out_name": name,
            "data_type": "double",
            "cf_standard_name": role,
            "units": units,
        }
        for role, axis, name, units in (
            ("longitude", "X", "lon", "degrees_east"),
            ("latitude", "Y", "lat", "degrees_north"),
        )
    }
    grid_variables = {}
    for role, name, units in (
        ("longitude", "lon", "degrees_east"),
        ("latitude", "lat", "degrees_north"),
    ):
        grid_variables[role] = {
            "id": role,
            "out_name": name,
            "data_type": "double",
            "cf_standard_name": role,
            "units": units,
            "dimensions": ["longitude", "latitude"],
        }
        grid_variables[f"vertices_{role}"] = {
            "id": f"vertices_{role}",
            "out_name": f"vertices_{name}",
            "data_type": "double",
            "units": units,
            "dimensions": ["vertices", "longitude", "latitude"],
        }
    grid_axes = {
        identifier: {
            "id": identifier,
            "out_name": name,
            "axis": axis,
            "data_type": "double",
            "cf_standard_name": standard_name,
            "units": units,
        }
        for identifier, name, axis, standard_name, units in (
            ("grid_longitude", "rlon", "X", "grid_longitude", "degrees"),
            ("grid_latitude", "rlat", "Y", "grid_latitude", "degrees"),
            ("x", "x", "X", "projection_x_coordinate", "m"),
            ("y", "y", "Y", "projection_y_coordinate", "m"),
        )
    }
    return Catalog(
        project_id="cordex-cmip6",
        branded_variable_id="tas_tavg-h2m-hxy-u",
        branded_variable={
            "id": "tas_tavg-h2m-hxy-u",
            "dimensions": ["longitude", "latitude"],
        },
        coordinate_ids=("longitude", "latitude"),
        data_coordinates=coordinates,
        grid_variables=grid_variables,
        grid_axes=grid_axes,
        file_variable_name="tas",
    )


def _cordex_coordinate_checker(monkeypatch, catalog=None):
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()
    checker._load_mappings()
    checker.variable_mapping = {"mon.tas": "tas_tavg-h2m-hxy-u"}
    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.load_catalog",
        lambda *args, **kwargs: catalog or _complete_horizontal_catalog(),
    )
    return checker


def _add_supported_horizontal_grid(dataset, layout):
    if layout == "rectilinear":
        dimensions = ("lat", "lon")
        coordinate_dimensions = {"lat": ("lat",), "lon": ("lon",)}
        mapping_name = ""
    elif layout in {"curvilinear", "latitude_longitude"}:
        dimensions = ("y", "x")
        coordinate_dimensions = {"lat": dimensions, "lon": dimensions}
        mapping_name = "latitude_longitude" if layout == "latitude_longitude" else ""
    elif layout == "rotated":
        dimensions = ("rlat", "rlon")
        coordinate_dimensions = {"lat": dimensions, "lon": dimensions}
        mapping_name = "rotated_latitude_longitude"
    elif layout == "projected":
        dimensions = ("y", "x")
        coordinate_dimensions = {"lat": dimensions, "lon": dimensions}
        mapping_name = "lambert_conformal_conic"
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"Unsupported test layout {layout!r}")

    for dimension, size in zip(dimensions, (2, 3)):
        dataset.createDimension(dimension, size)
    if layout != "rectilinear":
        dataset.createDimension("vertices", 4)

    if layout == "rotated":
        axes = (
            ("rlat", "Y", "grid_latitude", "degrees"),
            ("rlon", "X", "grid_longitude", "degrees"),
        )
    elif layout == "projected":
        axes = (
            ("y", "Y", "projection_y_coordinate", "m"),
            ("x", "X", "projection_x_coordinate", "m"),
        )
    else:
        axes = ()
    for name, axis, standard_name, units in axes:
        variable = dataset.createVariable(name, "f8", (name,))
        variable.axis = axis
        variable.standard_name = standard_name
        variable.units = units
        variable[:] = np.arange(len(dataset.dimensions[name]), dtype="float64")

    for name, standard_name, units in (
        ("lat", "latitude", "degrees_north"),
        ("lon", "longitude", "degrees_east"),
    ):
        variable = dataset.createVariable(
            name, "f8", coordinate_dimensions[name]
        )
        variable.standard_name = standard_name
        variable.units = units
        if layout == "rectilinear":
            variable.axis = "Y" if name == "lat" else "X"
            variable[:] = np.arange(variable.size, dtype="float64")
        else:
            variable.bounds = f"vertices_{name}"
            dataset.createVariable(
                f"vertices_{name}", "f8", dimensions + ("vertices",)
            )

    data = dataset.createVariable("tas", "f4", dimensions)
    data.coordinates = "lat lon"
    if mapping_name:
        mapping = dataset.createVariable("crs", "i4")
        mapping.grid_mapping_name = mapping_name
        mapping.earth_radius = 6371229.0
        data.grid_mapping = "crs"
    return data


def test_cordex_time003_uses_coordinate_catalogue_climatology(tmp_path, monkeypatch):
    time_entry = {
        "id": "time_climatology",
        "coordinate_type": "standard_1d",
        "axis": "T",
        "out_name": "time",
        "is_climatology": True,
    }
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()
    checker._coordinate_catalog = Catalog(
        project_id="cordex-cmip6",
        branded_variable_id="tas_tavg-h2m-hxy-u",
        branded_variable={"id": "tas_tavg-h2m-hxy-u", "cell_methods": ["time: mean"]},
        coordinate_ids=(time_entry["id"],),
        data_coordinates={time_entry["id"]: time_entry},
        file_variable_name="tas",
    )
    observed = {}

    def capture(*args, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.check_time_squareness",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.check_time_range_vs_filename",
        capture,
    )

    with Dataset(tmp_path / "tas.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.createDimension("time", 1)
        time = dataset.createVariable("time", "f8", ("time",))
        time[:] = [0.0]
        checker.check_Coordinates(dataset)

    assert observed["expected_is_climatology"] is True


def test_cordex_catalogue_topology_check_is_enabled_in_toml():
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()

    assert checker.config.coordinates.registry.grid.severity == "H"
    assert checker.config.coordinates.registry.grid.require_explicit_grid_axes is True


def test_cordex_enabled_topology_ignores_free_text_grid_attributes(
    tmp_path, monkeypatch
):
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()
    checker.config.coordinates.registry.grid = CoordinateGlobalRule(severity="H")
    checker.variable_mapping = {"mon.tas": "tas_tavg-h2m-hxy-u"}
    checker._grid_topology_config = object()
    observed = {}

    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.load_catalog",
        lambda *args, **kwargs: _horizontal_catalog(),
    )

    def resolve(_config, **kwargs):
        observed.update(kwargs)
        return "curvilinear", None

    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.resolve_grid_topology",
        resolve,
    )

    with Dataset(tmp_path / "tas.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        dataset.grid = "model-defined free text"
        dataset.grid_label = "must-not-be-used"
        dataset.createDimension("y", 1)
        dataset.createDimension("x", 1)
        data = dataset.createVariable("tas", "f4", ("y", "x"))
        data.grid_mapping = "rotated_pole"
        mapping = dataset.createVariable("rotated_pole", "i4")
        mapping.grid_mapping_name = "rotated_latitude_longitude"
        checker.dataset = dataset

        checker._initialize_coordinate_catalog(dataset)

    assert observed == {"grid_mapping": "rotated_latitude_longitude"}


def test_cordex_cmor_base_url_is_read_from_toml(tmp_path, monkeypatch):
    checker = CordexCmip6ProjectCheck({"tables_dir": str(tmp_path)})
    checker._load_split_config()
    checker.cordex_config["cmor_tables"]["base_url"] = (
        "https://metadata.example.test/custom-tables"
    )
    retrieved = []

    def capture(url, filename, destination, force=False):
        retrieved.append((url, filename, destination, force))

    monkeypatch.setattr("plugins.cordex_cmip6.cordex_cmip6.retrieve", capture)
    monkeypatch.setattr(checker, "_initialize_CV_info", lambda path: None)

    checker._load_cmor_tables()

    assert len(retrieved) == 10
    assert all(
        url == f"https://metadata.example.test/custom-tables/{filename}"
        for url, filename, _, _ in retrieved
    )
    assert all(destination == str(tmp_path) for _, _, destination, _ in retrieved)


def test_registry_comment_contains_one_allowed_comment(tmp_path):
    path = tmp_path / "tas.nc"
    expected = SimpleNamespace(
        comment=["Alternative description", "Near-Surface Air Temperature"]
    )

    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        variable = dataset.createVariable("tas", "f4", ("time",))
        variable.comment = "Near-Surface Air Temperature, sampled every three hours."

        passing = check_attribute_suite(
            dataset,
            var_name="tas",
            attribute_name="comment",
            severity=BaseCheck.LOW,
            expected_term=expected,
            cv_source_term_key="comment",
            expected_term_comparison="contains",
        )
        assert _messages(passing) == []

        variable.comment = "Unrelated metadata"
        failing = check_attribute_suite(
            dataset,
            var_name="tas",
            attribute_name="comment",
            severity=BaseCheck.LOW,
            expected_term=expected,
            cv_source_term_key="comment",
            expected_term_comparison="contains",
        )

    assert _messages(failing) == [
        "Expected 'Unrelated metadata' to contain one of "
        "['Alternative description', 'Near-Surface Air Temperature'] "
        "from registry key 'comment'."
    ]


def test_comment_is_catalogue_backed_for_all_requested_projects():
    root = Path(__file__).parents[1]
    configurations = (
        "plugins/cmip7/config/wcrp/geophysical_variable.toml",
        "plugins/cmip6/config/wcrp/geophysical_variable.toml",
        "plugins/cmip6plus/config/wcrp/geophysical_variable.toml",
        "plugins/cordex_cmip6/config/wcrp/geophysical_variable.toml",
    )

    for relative_path in configurations:
        data = toml.load(root / relative_path)
        rule = AttributeRule.model_validate(data["variable"]["attributes"]["comment"])
        assert rule.severity == "L"
        assert rule.is_required is False
        assert rule.cv_source_term_key == "comment"
        assert rule.expected_term_comparison == "contains"
        assert rule.report_missing_expected_term is True


def test_main_variable_data_type_policy_is_project_configured():
    root = Path(__file__).parents[1]
    configurations = {
        "plugins/cmip7/config/wcrp/geophysical_variable.toml": ["real", "double"],
        "plugins/cmip6/config/wcrp/geophysical_variable.toml": ["real", "double"],
        "plugins/cmip6plus/config/wcrp/geophysical_variable.toml": [
            "real",
            "double",
        ],
        "plugins/c3scmip6/c3scmip6/config/wcrp/geophysical_variable.toml": [
            "real",
            "double",
        ],
        "plugins/cordex_cmip6/config/wcrp/geophysical_variable.toml": ["real"],
    }

    for relative_path, expected in configurations.items():
        data = toml.load(root / relative_path)
        assert data["variable"]["type"]["data_type"] == expected


def test_cordex_drs_uses_project_templates(tmp_path, monkeypatch):
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()
    drs = checker.config.drs
    captured = []

    assert drs.directory_template_keys == [
        "project_id",
        "activity_id",
        "domain_id",
        "institution_id",
        "driving_source_id",
        "driving_experiment_id",
        "driving_variant_label",
        "source_id",
        "version_realization",
        "frequency",
        "variable_id",
        "version",
    ]
    assert drs.filename_template_keys == [
        "variable_id",
        "domain_id",
        "driving_source_id",
        "driving_experiment_id",
        "driving_variant_label",
        "institution_id",
        "source_id",
        "version_realization",
        "frequency",
        "time_range",
    ]

    def capture(*args, **kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.check_attributes_match_directory_structure",
        capture,
    )
    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.check_filename_matches_directory_structure",
        capture,
    )
    drs.filename = None
    drs.directory = None

    with Dataset(tmp_path / "tas.nc", "w") as dataset:
        checker.check_DRS(dataset)

    assert captured == [
        {
            "project_id": "cordex-cmip6",
            "dir_template_keys": drs.directory_template_keys,
            "filename_template_keys": drs.filename_template_keys,
            "report_directory_structure_error": True,
        },
        {
            "project_id": "cordex-cmip6",
            "dir_template_keys": drs.directory_template_keys,
            "filename_template_keys": drs.filename_template_keys,
            "report_directory_structure_error": False,
        },
    ]


def test_cordex_allows_rectilinear_grid_without_mapping_but_recommends_one(tmp_path):
    with Dataset(tmp_path / "tas.nc", "w") as dataset:
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        dataset.createVariable("lat", "f8", ("lat",))
        dataset.createVariable("lon", "f8", ("lon",))
        dataset.createVariable("tas", "f4", ("lat", "lon"))
        checker = SimpleNamespace(ds=dataset, varname=["tas"])

        results = check_grid_mapping(
            checker,
            severity=BaseCheck.HIGH,
            missing_severity=BaseCheck.MEDIUM,
            horizontal_topology="rectilinear",
        )

    assert results[0].msgs == []
    assert results[1].weight == BaseCheck.MEDIUM
    assert results[1].msgs == [
        "No grid_mapping variable was found. It is recommended to define one "
        "with information about the shape and size of the Earth used for the "
        "model grid, even for latitude-longitude grids (e.g., regular grids "
        "and curvilinear ocean grids)."
    ]


def test_cordex_allows_marked_ocean_grid_without_mapping_but_recommends_one(tmp_path):
    with Dataset(tmp_path / "curvilinear-no-mapping.nc", "w") as dataset:
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 3)
        dataset.createVariable("lat", "f8", ("y", "x"))
        dataset.grid = "Ocean native grid (no grid_mapping)"
        dataset.createVariable("lon", "f8", ("y", "x"))
        dataset.createVariable("tas", "f4", ("y", "x"))
        checker = SimpleNamespace(ds=dataset, varname=["tas"])

        results = check_grid_mapping(
            checker,
            horizontal_topology="curvilinear",
            severity=BaseCheck.HIGH,
            missing_severity=BaseCheck.MEDIUM,
        )

    assert results[0].msgs == []
    assert results[1].weight == BaseCheck.MEDIUM
    assert "shape and size of the Earth" in results[1].msgs[0]


def test_cordex_unmarked_curvilinear_grid_without_mapping_is_high_and_medium(
    tmp_path,
):
    with Dataset(tmp_path / "unmarked-ocean.nc", "w") as dataset:
        dataset.grid = "Ocean native grid"
        dataset.createVariable("tas", "f4")
        checker = SimpleNamespace(ds=dataset, varname=["tas"])

        results = check_grid_mapping(
            checker,
            horizontal_topology="curvilinear",
            severity=BaseCheck.HIGH,
            missing_severity=BaseCheck.MEDIUM,
        )

    assert any(
        "may be omitted for a rectilinear latitude-longitude grid" in message
        for message in results[0].msgs
    )
    assert results[1].weight == BaseCheck.MEDIUM


def test_cordex_infers_topology_from_cf_coordinates_without_name_assumptions(tmp_path):
    cases = {
        "rectilinear": (("j",), ("i",)),
        "curvilinear": (("row", "column"), ("row", "column")),
    }
    for topology, (lat_dimensions, lon_dimensions) in cases.items():
        with Dataset(tmp_path / f"{topology}.nc", "w") as dataset:
            for dimension in dict.fromkeys(lat_dimensions + lon_dimensions):
                dataset.createDimension(dimension, 2)
            latitude = dataset.createVariable("geographic_y", "f8", lat_dimensions)
            latitude.standard_name = "latitude"
            latitude.units = "degrees_north"
            longitude = dataset.createVariable("geographic_x", "f8", lon_dimensions)
            longitude.standard_name = "longitude"
            longitude.units = "degrees_east"

            assert infer_horizontal_topology(dataset) == (topology, None)


def test_cordex_rejects_unstructured_cf_coordinates(tmp_path):
    with Dataset(tmp_path / "unstructured.nc", "w") as dataset:
        dataset.createDimension("cell", 3)
        latitude = dataset.createVariable("geographic_y", "f8", ("cell",))
        latitude.standard_name = "latitude"
        latitude.units = "degrees_north"
        longitude = dataset.createVariable("geographic_x", "f8", ("cell",))
        longitude.standard_name = "longitude"
        longitude.units = "degrees_east"

        topology, error = infer_horizontal_topology(dataset)

    assert topology is None
    assert "does not currently support unstructured horizontal grids" in error
    assert "['cell']" in error


def test_cordex_grid_mapping_check_does_not_validate_coordinates(tmp_path):
    with Dataset(tmp_path / "mapping.nc", "w") as dataset:
        mapping = dataset.createVariable("crs", "i4")
        mapping.grid_mapping_name = "latitude_longitude"
        mapping.earth_radius = 6371229.0
        data = dataset.createVariable("tas", "f4")
        data.grid_mapping = "crs"
        checker = SimpleNamespace(ds=dataset, varname=["tas"])

        results = check_grid_mapping(
            checker,
            severity=BaseCheck.HIGH,
        )

    assert _messages(results) == []


def test_cordex_grid_mapping_name_validity_is_delegated_to_cf(tmp_path):
    with Dataset(tmp_path / "mapping.nc", "w") as dataset:
        mapping = dataset.createVariable("crs", "i4")
        mapping.grid_mapping_name = "lambert_azimuthal_equal_area"
        mapping.earth_radius = 6371229.0
        data = dataset.createVariable("tas", "f4")
        data.grid_mapping = "crs"
        checker = SimpleNamespace(ds=dataset, varname=["tas"])

        results = check_grid_mapping(checker, severity=BaseCheck.HIGH)

    assert _messages(results) == []


def test_cordex_version_realization_info_is_conditionally_recommended(tmp_path):
    checker = CordexCmip6ProjectCheck()
    checker.cordex_config = {
        "attribute_checks": {
            "check_version_realization_info": {"severity": "M"},
        }
    }
    checker.verification_against_tables = False
    with Dataset(tmp_path / "tas.nc", "w") as dataset:
        dataset.version_realization = "v2-r1"
        checker.dataset = dataset
        checker.ds = dataset

        missing = checker.check_attributes_cordex(dataset)
        assert len(_messages(missing)) == 1
        assert "version_realization_info" in _messages(missing)[0]

        dataset.version_realization_info = "Corrected postprocessing"
        present = checker.check_attributes_cordex(dataset)
        assert _messages(present) == []

        dataset.version_realization = "v1-r1"
        del dataset.version_realization_info
        first_release = checker.check_attributes_cordex(dataset)
        assert _messages(first_release) == []


def test_cordex_enabled_topology_infers_rectilinear_without_grid_mapping(
    tmp_path, monkeypatch
):
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()
    checker.config.coordinates.registry.grid = CoordinateGlobalRule(severity="H")
    checker.variable_mapping = {"mon.tas": "tas_tavg-h2m-hxy-u"}
    checker._grid_topology_config = object()

    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.load_catalog",
        lambda *args, **kwargs: _horizontal_catalog(),
    )

    with Dataset(tmp_path / "tas.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        latitude = dataset.createVariable("lat", "f8", ("lat",))
        latitude.standard_name = "latitude"
        longitude = dataset.createVariable("lon", "f8", ("lon",))
        longitude.standard_name = "longitude"
        dataset.createVariable("tas", "f4", ("lat", "lon"))
        checker.dataset = dataset

        checker._initialize_coordinate_catalog(dataset)

    assert checker._coordinate_grid_topology == "rectilinear"
    assert checker._coordinate_grid_error is None


@pytest.mark.parametrize(
    ("layout", "expected_topology"),
    [
        ("rectilinear", "rectilinear"),
        ("curvilinear", "curvilinear"),
        ("latitude_longitude", "curvilinear"),
        ("rotated", "curvilinear"),
        ("projected", "curvilinear"),
    ],
)
def test_cordex_supported_topologies_run_through_project_coordinate_check(
    tmp_path, monkeypatch, layout, expected_topology
):
    checker = _cordex_coordinate_checker(monkeypatch)
    with Dataset(tmp_path / f"{layout}.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        _add_supported_horizontal_grid(dataset, layout)
        checker.dataset = checker.ds = dataset
        checker.varname = ["tas"]

        checker._initialize_coordinate_catalog(dataset)
        results = checker.check_Coordinate_Standard(dataset)

    assert checker._coordinate_grid_topology == expected_topology
    assert checker._coordinate_grid_error is None
    assert _messages(results) == []


@pytest.mark.parametrize("malformation", ["wrong_rank", "different_dimensions"])
def test_cordex_malformed_projected_grid_stops_derivative_coordinate_checks(
    tmp_path, monkeypatch, malformation
):
    checker = _cordex_coordinate_checker(monkeypatch)
    with Dataset(tmp_path / f"{malformation}.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 3)
        dataset.createDimension("other_y", 2)
        latitude_dimensions = (
            ("y",)
            if malformation == "wrong_rank"
            else ("other_y", "x")
        )
        latitude = dataset.createVariable("lat", "f8", latitude_dimensions)
        latitude.standard_name = "latitude"
        latitude.units = "degrees_north"
        longitude = dataset.createVariable("lon", "f8", ("y", "x"))
        longitude.standard_name = "longitude"
        longitude.units = "degrees_east"
        mapping = dataset.createVariable("crs", "i4")
        mapping.grid_mapping_name = "lambert_conformal_conic"
        data = dataset.createVariable("tas", "f4", ("y", "x"))
        data.coordinates = "lat lon"
        data.grid_mapping = "crs"
        checker.dataset = checker.ds = dataset
        checker.varname = ["tas"]

        checker._initialize_coordinate_catalog(dataset)
        results = checker.check_Coordinate_Standard(dataset)

    grid = next(result for result in results if "COORD011" in result.name)
    assert grid.weight == BaseCheck.HIGH
    assert len(grid.msgs) == 1
    assert all(
        result.msgs == []
        for result in results
        if not result.name.startswith("[COORD011]")
    )


def test_cordex_project_rejects_unstructured_grid_once(tmp_path, monkeypatch):
    checker = _cordex_coordinate_checker(monkeypatch)
    with Dataset(tmp_path / "unstructured-project.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        dataset.createDimension("cell", 3)
        for name, standard_name, units in (
            ("lat", "latitude", "degrees_north"),
            ("lon", "longitude", "degrees_east"),
        ):
            variable = dataset.createVariable(name, "f8", ("cell",))
            variable.standard_name = standard_name
            variable.units = units
        data = dataset.createVariable("tas", "f4", ("cell",))
        data.coordinates = "lat lon"
        checker.dataset = checker.ds = dataset
        checker.varname = ["tas"]

        checker._initialize_coordinate_catalog(dataset)
        coordinate_results = checker.check_Coordinate_Standard(dataset)
        attribute_results = checker.check_attributes_cordex(dataset)

    assert checker._coordinate_grid_topology is None
    assert "does not currently support unstructured horizontal grids" in (
        checker._coordinate_grid_error or ""
    )
    grid = next(result for result in coordinate_results if "COORD011" in result.name)
    assert len(grid.msgs) == 1
    assert "does not currently support unstructured horizontal grids" in grid.msgs[0]
    assert all(
        result.msgs == []
        for result in coordinate_results
        if not result.name.startswith("[COORD011]")
    )
    mapping_results = [
        result for result in attribute_results if "grid_mapping" in result.name
    ]
    assert _messages(mapping_results) == [
        (
            "No grid_mapping variable was found. It is recommended to define one "
            "with information about the shape and size of the Earth used for the "
            "model grid, even for latitude-longitude grids (e.g., regular grids "
            "and curvilinear ocean grids)."
        )
    ]


def test_cordex_rejects_unstructured_topology_from_future_mapping(
    tmp_path, monkeypatch
):
    checker = _cordex_coordinate_checker(monkeypatch)
    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.resolve_grid_topology",
        lambda *args, **kwargs: ("unstructured", None),
    )
    with Dataset(tmp_path / "future-unstructured-mapping.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        mapping = dataset.createVariable("crs", "i4")
        mapping.grid_mapping_name = "future_unstructured_mapping"
        data = dataset.createVariable("tas", "f4")
        data.grid_mapping = "crs"
        checker.dataset = checker.ds = dataset
        checker.varname = ["tas"]

        checker._initialize_coordinate_catalog(dataset)
        results = checker.check_Coordinate_Standard(dataset)

    assert checker._coordinate_grid_topology is None
    assert checker._coordinate_grid_error == (
        "The plugin does not currently support unstructured horizontal grids "
        "for CORDEX-CMIP6. Please open a GitHub issue and provide test data so "
        "that support can be discussed and, if appropriate, implemented."
    )
    grid = next(result for result in results if "COORD011" in result.name)
    assert grid.msgs == [
        (
            "The horizontal grid could not be verified. The plugin does not "
            "currently support unstructured horizontal grids for CORDEX-CMIP6. "
            "Please open a GitHub issue and provide test data so that support "
            "can be discussed and, if appropriate, implemented."
        )
    ]


def test_cordex_coordinate_failures_match_for_esgvoc_and_cmor_routes(
    tmp_path, monkeypatch
):
    catalog = _complete_horizontal_catalog()
    checker_esgvoc = _cordex_coordinate_checker(monkeypatch, catalog)
    checker_cmor = _cordex_coordinate_checker(monkeypatch, catalog)
    checker_cmor.verification_against_tables = True
    checker_cmor.CTcoords = {}
    checker_cmor.CTgrids = {}
    checker_cmor.CTformulas = {}
    checker_cmor._cmor_variable_entry = lambda ds: ("tas", {})
    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.catalog_from_cmor",
        lambda *args, **kwargs: catalog,
    )

    with Dataset(tmp_path / "route-parity.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 3)
        latitude = dataset.createVariable("lat", "f8", ("y",))
        latitude.standard_name = "latitude"
        latitude.units = "degrees_north"
        longitude = dataset.createVariable("lon", "f8", ("y", "x"))
        longitude.standard_name = "longitude"
        longitude.units = "degrees_east"
        mapping = dataset.createVariable("crs", "i4")
        mapping.grid_mapping_name = "rotated_latitude_longitude"
        data = dataset.createVariable("tas", "f4", ("y", "x"))
        data.coordinates = "lat lon"
        data.grid_mapping = "crs"

        route_messages = []
        for checker in (checker_esgvoc, checker_cmor):
            checker.dataset = checker.ds = dataset
            checker.varname = ["tas"]
            checker._initialize_coordinate_catalog(dataset)
            route_messages.append(_messages(checker.check_Coordinate_Standard(dataset)))

    assert route_messages[0] == route_messages[1]
    assert len(route_messages[0]) == 1
    assert "2-D for this grid topology" in route_messages[0][0]


def test_cordex_ambiguous_cf_coordinates_report_once_and_stop_topology(
    tmp_path, monkeypatch
):
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()
    checker.variable_mapping = {"mon.tas": "tas_tavg-h2m-hxy-u"}
    checker._grid_topology_config = SimpleNamespace(
        allow_standard_name_fallback=True
    )

    monkeypatch.setattr(
        "plugins.cordex_cmip6.cordex_cmip6.load_catalog",
        lambda *args, **kwargs: _horizontal_catalog(),
    )

    with Dataset(tmp_path / "ambiguous.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.variable_id = "tas"
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 3)
        first = dataset.createVariable("first_latitude", "f8", ("y", "x"))
        first.standard_name = "latitude"
        second = dataset.createVariable("second_latitude", "f8", ("y", "x"))
        second.standard_name = "latitude"
        longitude = dataset.createVariable("geographic_x", "f8", ("y", "x"))
        longitude.standard_name = "longitude"
        dataset.createVariable("tas", "f4", ("y", "x"))
        checker.dataset = checker.ds = dataset
        checker.varname = ["tas"]

        checker._initialize_coordinate_catalog(dataset)
        results = checker.check_Coordinate_Standard(dataset)
        attribute_results = checker.check_attributes_cordex(dataset)
        project_specific_results = (
            checker.check_lat_lon_bounds(dataset)
            + checker.check_horizontal_axes_bounds(dataset)
            + checker.check_lon_value_range(dataset)
        )

    assert checker._coordinate_grid_topology is None
    assert "exactly one of each is required" in checker._coordinate_grid_error
    grid = next(result for result in results if "COORD011" in result.name)
    assert grid.weight == BaseCheck.HIGH
    assert len(grid.msgs) == 1
    assert any(
        "horizontal grid could not be verified" in message for message in grid.msgs
    )
    mapping_results = [
        result for result in attribute_results if "grid_mapping" in result.name
    ]
    assert _messages(mapping_results) == [
        (
            "No grid_mapping variable was found. It is recommended to define one "
            "with information about the shape and size of the Earth used for the "
            "model grid, even for latitude-longitude grids (e.g., regular grids "
            "and curvilinear ocean grids)."
        )
    ]
    assert len(project_specific_results) == 3
