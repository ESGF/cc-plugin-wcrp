from types import SimpleNamespace

from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

import checks.consistency_checks.check_experiment_consistency as experiment_checks
from checks.coordinate_checks.grid import check_grid_cell_count
from checks.coordinate_checks.model import Catalog
from checks.coordinate_checks.suite import (
    check_coordinate_catalog,
    coordinate_metadata_setup_result,
    missing_configured_coordinate_result,
)
from checks.coordinate_checks.validation import Findings
from checks.time_checks.check_time_squareness import check_time_squareness
from checks.variable_checks.check_bounds_value_consistency import (
    check_bounds_value_consistency,
)
from plugins.cmip7.cmip7 import Cmip7ProjectCheck
from plugins.wcrp_schema import CoordinateVariableConfig, WCRPConfig


def _messages(results):
    return [message for result in results for message in result.msgs]


def _catalog(coordinates, *, dimensions=None):
    return Catalog(
        project_id="cmip7",
        branded_variable_id="ta_ti-u-hxy-air",
        branded_variable={"id": "ta_ti-u-hxy-air", "out_name": "ta"},
        coordinate_ids=tuple(dimensions or coordinates),
        data_coordinates=coordinates,
    )


def _coordinate(identifier, kind="standard_1d", out_name=None, **values):
    return {
        "id": identifier,
        "coordinate_type": kind,
        "data_type": "double",
        "out_name": out_name or identifier,
        "bounds_required": False,
        **values,
    }


def test_missing_coordinate_falls_back_to_enabled_metadata_family(tmp_path):
    with Dataset(tmp_path / "missing-coordinate.nc", "w") as ds:
        results = check_coordinate_catalog(
            ds,
            _catalog({"depth": _coordinate("depth")}),
            severities={"attributes": BaseCheck.MEDIUM},
        )

    assert len(results) == 1
    assert "[COORD003]" in results[0].name
    assert results[0].weight == BaseCheck.MEDIUM
    assert "'depth'" in results[0].msgs[0]
    assert "could not be evaluated" in results[0].msgs[0]


def test_missing_coordinate_stays_disabled_without_consumers(tmp_path):
    with Dataset(tmp_path / "disabled-coordinate.nc", "w") as ds:
        results = check_coordinate_catalog(
            ds,
            _catalog({"depth": _coordinate("depth")}),
            severities={},
        )

    assert results == []


def test_unresolved_grid_falls_back_to_enabled_grid_range_family(tmp_path):
    coordinates = {
        "latitude": _coordinate("latitude", "generic_horizontal", "latitude"),
        "longitude": _coordinate("longitude", "generic_horizontal", "longitude"),
    }
    with Dataset(tmp_path / "unresolved-grid.nc", "w") as ds:
        results = check_coordinate_catalog(
            ds,
            _catalog(coordinates),
            severities={"grid_longitude_valid_range": BaseCheck.MEDIUM},
            grid_resolution_error="No registered topology is available.",
        )

    assert len(results) == 1
    assert "Grid-longitude valid range" in results[0].name
    assert "horizontal grid could not be verified" in results[0].msgs[0]


def test_grid_metadata_failure_falls_back_to_enabled_emd_family(tmp_path):
    with Dataset(tmp_path / "missing-grid-metadata.nc", "w") as ds:
        results = check_coordinate_catalog(
            ds,
            _catalog({}),
            severities={"grid_mapping_consistency": BaseCheck.MEDIUM},
            registered_grid_metadata_error="The grid collection could not be read.",
        )

    assert len(results) == 1
    assert "Registered grid mapping" in results[0].name
    assert "grid collection could not be read" in results[0].msgs[0]


def test_missing_registered_cell_count_falls_back_to_consistency_family(tmp_path):
    findings = Findings({"grid_cell_count_consistency": BaseCheck.MEDIUM})
    with Dataset(tmp_path / "missing-cell-count.nc", "w") as ds:
        check_grid_cell_count(
            findings,
            ds,
            _catalog({"latitude": _coordinate("latitude")}),
            None,
            topology="rectilinear",
            registered_grid_metadata={"id": "g999"},
            grid_topology_config=None,
            allow_standard_name_fallback=True,
        )

    results = findings.results()
    assert len(results) == 1
    assert "Horizontal grid cell count" in results[0].name
    assert "does not define n_cells" in results[0].msgs[0]


def test_missing_data_variable_falls_back_when_existence_owner_is_disabled(tmp_path):
    with Dataset(tmp_path / "missing-data-variable.nc", "w") as ds:
        results = check_coordinate_catalog(
            ds,
            _catalog({}),
            severities={"dimension_order": BaseCheck.HIGH},
            data_variable_presence_delegated=False,
        )

    assert len(results) == 1
    assert "[COORD002]" in results[0].name
    assert "data variable 'ta' is absent" in results[0].msgs[0]


def test_disabled_setup_still_reports_failure_needed_by_enabled_check():
    config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "coordinates": {
                "registry": {"attributes": {"severity": "M"}},
            }
        }
    )
    checker = Cmip7ProjectCheck()
    results = coordinate_metadata_setup_result(
        config=config,
        catalog=None,
        error="database is unavailable",
        project_label="CMIP7",
        get_severity=checker.get_severity,
    )

    assert len(results) == 1
    assert "[COORD000]" in results[0].name
    assert results[0].weight == BaseCheck.MEDIUM
    assert "database is unavailable" in results[0].msgs[0]


def test_disabled_setup_omits_success_result():
    config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "coordinates": {
                "registry": {"attributes": {"severity": "H"}},
            }
        }
    )
    checker = Cmip7ProjectCheck()

    assert coordinate_metadata_setup_result(
        config=config,
        catalog=object(),
        error=None,
        project_label="CMIP7",
        get_severity=checker.get_severity,
    ) == []


def test_missing_configured_time_uses_first_enabled_consumer():
    rule = CoordinateVariableConfig.model_validate(
        {
            "monotonicity": {"severity": "M", "direction": "increasing"},
            "squareness": {"severity": "H"},
        }
    )
    checker = Cmip7ProjectCheck()
    results = missing_configured_coordinate_result(
        "time", rule, checker.get_severity
    )

    assert len(results) == 1
    assert "[VAR005]" in results[0].name
    assert results[0].weight == BaseCheck.MEDIUM
    assert "could not be evaluated" in results[0].msgs[0]


def test_time001_can_report_invalid_shape_when_identity_owner_is_disabled(tmp_path):
    with Dataset(tmp_path / "tas_Amon_200001-200002.nc", "w") as ds:
        ds.createDimension("x", 2)
        ds.createDimension("y", 2)
        ds.createVariable("time", "f8", ("x", "y"))
        results = check_time_squareness(
            ds,
            report_structural_prerequisites=True,
        )

    assert len(results) == 1
    assert "not one-dimensional" in results[0].msgs[0]


def test_missing_time_does_not_emit_a_time003_pass(tmp_path, monkeypatch):
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checker.config.coordinates.registry.identity = None
    checker._coordinate_catalog = _catalog(
        {
            "time": _coordinate(
                "time",
                axis="T",
                units="days since ?",
                cf_standard_name="time",
            )
        }
    )
    called = False

    def capture(*_args, **_kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "plugins.cmip7.cmip7.check_time_range_vs_filename",
        capture,
    )
    with Dataset(tmp_path / "ta_mon_200001-200012.nc", "w") as ds:
        results = checker.check_Coordinates(ds)

    assert called is False
    assert any("Coordinate variable 'time' is missing" in msg for msg in _messages(results))


def test_parent_consistency_reports_missing_prerequisite_without_attr_owner(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(experiment_checks, "ESG_VOCAB_AVAILABLE", True)
    monkeypatch.setattr(
        experiment_checks,
        "resolve_experiment_term",
        lambda *_args, **_kwargs: SimpleNamespace(
            parent_experiment_id=["piControl"]
        ),
    )
    with Dataset(tmp_path / "missing-parent.nc", "w") as ds:
        ds.experiment_id = "abrupt-4xCO2"
        results = experiment_checks.check_experiment_id_vs_parent_experiment_id(
            ds,
            BaseCheck.HIGH,
            report_missing=True,
        )

    assert len(results) == 1
    assert "parent_experiment_id" in results[0].msgs[0]
    assert "piControl" in results[0].msgs[0]


def test_missing_referenced_bounds_result_is_not_discarded(tmp_path):
    with Dataset(tmp_path / "missing-bounds.nc", "w") as ds:
        ds.createDimension("time", 1)
        time = ds.createVariable("time", "f8", ("time",))
        time.bounds = "time_bnds"
        results = check_bounds_value_consistency(ds, "time", BaseCheck.HIGH)

    assert len(results) == 1
    assert "not found" in results[0].msgs[0]
