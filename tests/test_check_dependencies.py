from types import SimpleNamespace
import importlib

import pytest
from compliance_checker.base import BaseCheck, TestCtx as ComplianceTestCtx
from netCDF4 import Dataset

import checks.consistency_checks.check_experiment_consistency as experiment_checks
import checks.consistency_checks.check_variant_label_consistency as variant_checks
from checks.attribute_checks.check_attribute_suite import check_attribute_suite
from checks.consistency_checks.check_attributes_match_filename import (
    check_filename_vs_global_attrs,
)
from checks.consistency_checks.check_drs_consistency import (
    check_filename_matches_directory_structure,
)
from checks.consistency_checks.check_drs_filename_cv import check_drs_directory
from checks.consistency_checks.check_frequency_table_consistency import (
    check_frequency_table_id_consistency,
)
from checks.consistency_checks.check_frequency_branded_variable_consistency import (
    check_frequency_branded_variable_consistency,
)
from checks.consistency_checks.check_region_branded_variable_consistency import (
    check_region_branded_variable_consistency,
)
from checks.consistency_checks.check_institution_source_consistency import (
    check_id_attribute_consistency,
)
from checks.coordinate_checks.grid import check_grid_cell_count
from checks.coordinate_checks.grid import _validate_vertices
from checks.coordinate_checks.model import Catalog
from checks.coordinate_checks.ordinary import _identity_fallbacks
from checks.coordinate_checks.suite import (
    check_coordinate_catalog,
    coordinate_catalog_required,
    coordinate_metadata_setup_result,
    missing_configured_coordinate_result,
)
from checks.coordinate_checks.validation import Findings, PREREQUISITE_FALLBACKS
from checks.time_checks.check_time_bounds import check_time_bounds
from checks.time_checks.check_time_range_vs_filename import (
    check_time_range_vs_filename,
)
from checks.time_checks.check_time_squareness import check_time_squareness
from checks.variable_checks.check_bounds_value_consistency import (
    check_bounds_value_consistency,
)
from checks.variable_checks.known_branded_variable import ExpectedVariableMetadata
from plugins.c3scmip6.c3scmip6.c3scmip6 import C3SCmip6ProjectCheck
from plugins.cmip6.cmip6 import Cmip6ProjectCheck
from plugins.cmip6plus.cmip6plus import Cmip6PlusProjectCheck
from plugins.cmip7.cmip7 import Cmip7ProjectCheck
from plugins.cordex_cmip6.cordex_cmip6 import CordexCmip6ProjectCheck
from plugins.wcrp_schema import CoordinateVariableConfig, WCRPConfig


def _messages(results):
    return [message for result in results for message in result.msgs]


@pytest.mark.parametrize(
    ("kind", "entry", "present", "absent"),
    (
        (
            "standard_1d",
            {"data_type": "double"},
            {"attributes", "direction", "dimension_order", "recommendations"},
            {"formula", "requested_values", "bounds", "associations"},
        ),
        (
            "standard_1d",
            {
                "data_type": "double",
                "valid_min": 0,
                "coordinate_values": [1],
                "bounds_required": True,
            },
            {"valid_range", "requested_values", "bounds", "bounds_name"},
            {"formula", "associations"},
        ),
        (
            "scalar",
            {"data_type": "double", "valid_max": 1},
            {"attributes", "missing_value_prohibited", "valid_range", "associations"},
            {"formula", "direction", "dimension_order"},
        ),
        (
            "auxiliary",
            {"data_type": "character", "coordinate_values": ["atlantic"]},
            {"attributes", "missing_value_prohibited", "associations", "requested_values"},
            {"formula", "direction", "valid_range", "bounds", "dimension_order"},
        ),
    ),
)
def test_coordinate_identity_fallbacks_are_context_specific(
    kind, entry, present, absent
):
    fallbacks = _identity_fallbacks(entry, kind)

    assert present <= set(fallbacks)
    assert set(fallbacks).isdisjoint(absent)
    advice = ("recommendations", "bounds_name_recommendations")
    assert fallbacks[-len(set(fallbacks) & set(advice)) :] == tuple(
        family for family in advice if family in fallbacks
    )


def test_grid_fallbacks_exclude_unrelated_coordinate_checks_and_end_in_advice():
    fallbacks = PREREQUISITE_FALLBACKS["grid"]

    assert "formula" not in fallbacks
    assert "requested_values" not in fallbacks
    assert fallbacks.index("grid_geometry_missing_values") < fallbacks.index(
        "grid_latitude_valid_range"
    )
    assert fallbacks[-2:] == ("recommendations", "bounds_name_recommendations")


@pytest.mark.parametrize("weights", [(1, 3, 2), (3, 2, 1), (2, 1, 3)])
@pytest.mark.parametrize("disabled", range(7))
def test_complete_missing_time_chain(weights, disabled):
    chain = [
        "monotonicity",
        "squareness",
        "coverage",
        "calendar",
        "units",
        "calendar_recommendation",
    ]
    enabled = set(chain[disabled:])
    config = {
        key: {"severity": "LMH"[weights[chain.index(key) % 3] - 1]}
        for key in ("monotonicity", "squareness", "coverage", "calendar_recommendation")
        if key in enabled
    }
    if "monotonicity" in config:
        config["monotonicity"]["direction"] = "increasing"
    config["attributes"] = {
        key: {"severity": "LMH"[weights[chain.index(key) % 3] - 1]}
        for key in ("calendar", "units")
        if key in enabled
    }
    results = missing_configured_coordinate_result(
        "time", CoordinateVariableConfig.model_validate(config), Cmip7ProjectCheck().get_severity
    )
    if disabled == len(chain):
        assert results == []
    else:
        assert len(results) == 1
        labels = ["VAR005", "TIME001", "TIME002", "ATTR004", "ATTR004", "TIME003a"]
        assert results[0].name.startswith(f"[{labels[disabled]}]")
        assert results[0].weight == weights[disabled % 3]
        assert len(results[0].msgs) == 1
        if 3 <= disabled <= 4:
            assert f"'{chain[disabled]}'" in results[0].name


@pytest.mark.parametrize("role", ["latitude", "longitude"])
@pytest.mark.parametrize("problem", ["attribute", "variable", "shape"])
@pytest.mark.parametrize("weights", [(1, 3, 2), (3, 2, 1), (2, 1, 3)])
@pytest.mark.parametrize("disabled", range(len(PREREQUISITE_FALLBACKS["grid"]) + 2))
def test_vertex_explicit_fallback_chain(tmp_path, role, problem, weights, disabled):
    chain = list(dict.fromkeys(("grid", "grid_geometry_missing_values",
                               f"grid_{role}_valid_range",
                               *PREREQUISITE_FALLBACKS["grid"])))
    severities = {key: weights[i % 3] for i, key in enumerate(chain) if i >= disabled}
    findings = Findings(severities)
    with Dataset(tmp_path / "vertices.nc", "w") as ds:
        ds.createDimension("cells", 2)
        coord = ds.createVariable(role, "f8", ("cells",))
        if problem != "attribute":
            coord.bounds = f"vertices_{role}"
        if problem == "shape":
            ds.createVariable(coord.bounds, "f8", ("cells",))
        # Repeated consumers must not report the same root cause twice.
        for _ in range(2):
            _validate_vertices(findings, ds, coord, role, {}, _catalog({}))
    populated = {key: messages for key, messages in findings.messages.items() if messages}
    if disabled >= len(chain):
        assert populated == {}
    else:
        assert list(populated) == [chain[disabled]]
        assert len(populated[chain[disabled]]) == 1
        failures = [r for r in findings.results() if r.msgs]
        assert len(failures) == 1
        assert failures[0].weight == weights[disabled % 3]


_EMD_CHAIN = list(dict.fromkeys((
    "grid", "grid_mapping_consistency", "grid_region_consistency",
    "grid_label_recommendation", "grid_cell_count_availability",
    "grid_cell_count_consistency", *PREREQUISITE_FALLBACKS["grid"],
)))


@pytest.mark.parametrize("weights", [(1, 3, 2), (3, 2, 1), (2, 1, 3)])
@pytest.mark.parametrize("disabled", range(len(_EMD_CHAIN) + 1))
def test_unavailable_emd_explicit_fallback_chain(tmp_path, weights, disabled):
    severities = {key: weights[i % 3] for i, key in enumerate(_EMD_CHAIN) if i >= disabled}
    with Dataset(tmp_path / "emd.nc", "w") as ds:
        results = check_coordinate_catalog(
            ds, _catalog({}), severities=severities,
            registered_grid_metadata_error="Grid database unavailable",
        )
    failures = [r for r in results if r.msgs]
    if disabled == len(_EMD_CHAIN):
        assert failures == []
    else:
        expected = Findings(severities)
        expected.add(_EMD_CHAIN[disabled], "Grid database unavailable")
        expected_result = next(r for r in expected.results() if r.msgs)
        assert len(failures) == 1
        assert failures[0].name == expected_result.name
        assert failures[0].weight == weights[disabled % 3]
        assert len(failures[0].msgs) == 1


@pytest.mark.parametrize("checker_class", [Cmip6ProjectCheck, Cmip6PlusProjectCheck, C3SCmip6ProjectCheck])
@pytest.mark.parametrize("owner_severity,consumer_severity", [("L", "H"), ("H", "M"), ("M", "L")])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("owner,flags", [
    ("name", ("report_time_structure_prerequisite",)),
    ("units", ("report_time_units_prerequisite",)),
    ("squareness", ("report_time_values_prerequisite", "report_climatology_bounds_prerequisite", "report_frequency_prerequisite")),
])
def test_legacy_plugin_local_time_delegation(tmp_path, monkeypatch, checker_class,
                                           owner_severity, consumer_severity, enabled, owner, flags):
    checker = checker_class()
    rule = {}
    if enabled:
        if owner == "units":
            rule["attributes"] = {"units": {"severity": owner_severity}}
        elif owner == "name":
            rule["name"] = {"severity": owner_severity, "variable_name": "time"}
        else:
            rule[owner] = {"severity": owner_severity}
    checker.config = WCRPConfig.model_validate({
        "project_name": "test", "project_version": "1",
        "coordinates": {"time": rule},
        "drs": {"time_range": {"severity": consumer_severity}},
    })
    monkeypatch.setattr(checker, "_get_expected_from_registry", lambda *a, **k: (
        ExpectedVariableMetadata(dimensions=["time"], cell_methods="time: point"), []))
    calls = []
    def capture(ds, severity, **kwargs):
        calls.append((severity, kwargs))
        return []
    monkeypatch.setattr(importlib.import_module(checker_class.__module__),
                        "check_time_range_vs_filename", capture)
    with Dataset(tmp_path / "time.nc", "w") as ds:
        ds.createDimension("time", 2)
        time = ds.createVariable("time", "f8", ("time",))
        time[:] = [0, 1]
        time.units = "days since 2000-01-01"
        time.calendar = "standard"
        ds.frequency = "day"
        checker.check_Coordinates(ds)
    assert len(calls) == 1
    assert calls[0][0] == checker.get_severity(consumer_severity)
    for flag in flags:
        assert calls[0][1][flag] is not enabled
    for flag in ("report_time_structure_prerequisite", "report_time_units_prerequisite",
                 "report_time_values_prerequisite", "report_climatology_bounds_prerequisite",
                 "report_frequency_prerequisite"):
        if flag not in flags:
            assert calls[0][1][flag] is True


@pytest.mark.parametrize("checker_class", [Cmip7ProjectCheck, Cmip6ProjectCheck,
    Cmip6PlusProjectCheck, C3SCmip6ProjectCheck, CordexCmip6ProjectCheck])
@pytest.mark.parametrize("owner_severity", ["L", "M", "H"])
def test_project_ownership_uses_enabled_rules_not_severity(checker_class, owner_severity):
    checker = checker_class()
    checker.config = WCRPConfig.model_validate({
        "project_name": "test", "project_version": "1",
        "global": {"attributes": {"frequency": {"severity": owner_severity}}},
        "variable": {"existence": {"severity": owner_severity},
                     "attributes": {"units": {"severity": owner_severity, "cv_source_term_key": "units"}}},
        "coordinates": {"registry": {"attributes": {"severity": owner_severity}}},
        "drs": {"filename": {"severity": owner_severity}},
    })
    assertions = [
        lambda: checker._global_attribute_owns("frequency"),
        lambda: checker._variable_existence_owns(),
        lambda: checker._variable_registry_owns(),
        lambda: checker._coordinate_registry_owns("attributes"),
        lambda: checker._drs_rule_owns("filename"),
    ]
    assert all(check() for check in assertions)
    assert not checker._global_attribute_owns("region")
    assert not checker._coordinate_registry_owns("bounds")
    checker.config.global_.attributes.clear()
    checker.config.variable.existence = None
    checker.config.variable.attributes.clear()
    checker.config.coordinates.registry.attributes = None
    checker.config.drs.filename = None
    assert not any(check() for check in assertions)


@pytest.mark.parametrize("checker_class", [Cmip7ProjectCheck, CordexCmip6ProjectCheck])
def test_unrelated_coordinate_check_cannot_own_time_prerequisites(checker_class):
    checker = checker_class()
    checker.config = WCRPConfig.model_validate({
        "project_name": "test", "project_version": "1",
        "coordinates": {"latitude": {"squareness": {"severity": "H"}},
                        "time": {"squareness": {"severity": "L"}}},
    })
    checker._coordinate_catalog = _catalog({"time": _coordinate("time", axis="T")})
    assert checker._coordinate_rule_enabled_for_axis("T", "squareness")
    checker.config.coordinates.variables["time"].squareness = None
    assert not checker._coordinate_rule_enabled_for_axis("T", "squareness")


@pytest.mark.parametrize("checker_class", [Cmip7ProjectCheck, CordexCmip6ProjectCheck])
@pytest.mark.parametrize("enabled", [True, False])
def test_time_bounds_delegation_keeps_low_owner_and_ignores_other_axes(tmp_path, monkeypatch, checker_class, enabled):
    import importlib

    checker = checker_class()
    checker.config = WCRPConfig.model_validate({
        "project_name": "test", "project_version": "1",
        "coordinates": {"registry": {"bounds": {"severity": "H"}},
                        "latitude": {"coverage": {"severity": "H"}},
                        "time": {"coverage": {"severity": "L"}} if enabled else {}},
    })
    checker._coordinate_catalog = _catalog({"time": _coordinate("time", axis="T")})
    for name in ("_coordinate_grid_topology", "_coordinate_grid_error", "_coordinate_grid_metadata",
                 "_coordinate_grid_metadata_error", "_grid_topology_config"):
        setattr(checker, name, None)
    captured = {}
    def capture(*args, **kwargs):
        captured.update(kwargs)
        return []
    monkeypatch.setattr(importlib.import_module(checker_class.__module__), "check_coordinate_catalog", capture)
    with Dataset(tmp_path / "bounds.nc", "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("bnds", 2)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time.bounds = "time_bnds"
        time[:] = [3]
        ds.createVariable("time_bnds", "f8", ("time", "bnds"))[:] = [[0, 1]]
        results = checker.check_Coordinate_Standard(ds)
    assert captured["time_bounds_delegated"] == enabled
    assert [r.weight for r in results if r.msgs] == ([BaseCheck.LOW] if enabled else [])


@pytest.mark.parametrize("owner_enabled,fallback_enabled,expected", [
    (True, True, 1), (False, True, 2), (False, False, None),
])
def test_ncells_ownership_does_not_inherit_severity(owner_enabled, fallback_enabled, expected):
    severities = {}
    if owner_enabled:
        severities["grid_cell_count_availability"] = 1
    if fallback_enabled:
        severities["grid_cell_count_consistency"] = 2
    findings = Findings(severities)
    for _ in range(2):
        findings.add_prerequisite("grid_cell_count_availability", "missing n_cells",
            fallbacks=("grid_cell_count_consistency",), issue="n_cells")
    failures = [r for r in findings.results() if r.msgs]
    assert [r.weight for r in failures] == ([] if expected is None else [expected])
    assert all(r.msgs == ["missing n_cells"] for r in failures)


_COORDINATE_PREREQUISITE_WALK = [
    pytest.param(owner, chain[index:], chain[index], id=f"{owner}-{chain[index]}")
    for owner, fallbacks in PREREQUISITE_FALLBACKS.items()
    for chain in [(owner, *fallbacks)]
    for index in range(len(chain))
] + [
    pytest.param(owner, (), None, id=f"{owner}-all-disabled")
    for owner in PREREQUISITE_FALLBACKS
]

_BLOCKED_DATA_VARIABLE_WALK = [
    pytest.param(chain[index:], chain[index], id=chain[index])
    for chain in [
        (
            "dimension_order",
            "associations",
            "grid_mapping_consistency",
            "grid_cell_count_consistency",
        )
    ]
    for index in range(len(chain))
] + [pytest.param((), None, id="all-disabled")]


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


def test_missing_data_variable_can_fall_back_to_cell_count_check(tmp_path):
    with Dataset(tmp_path / "missing-cell-count-variable.nc", "w") as ds:
        results = check_coordinate_catalog(
            ds,
            _catalog({}),
            severities={"grid_cell_count_consistency": BaseCheck.MEDIUM},
            data_variable_presence_delegated=False,
            registered_grid_metadata={"id": "g001", "n_cells": 1},
        )

    assert len(results) == 1
    assert "grid cell count" in results[0].name
    assert "data variable 'ta' is absent" in results[0].msgs[0]


def test_time_direction_falls_back_to_coord005_when_var005_is_disabled(tmp_path):
    catalog = _catalog(
        {
            "time": _coordinate(
                "time", axis="T", stored_direction="increasing"
            )
        }
    )
    with Dataset(tmp_path / "time-direction.nc", "w") as ds:
        ds.createDimension("time", 2)
        ds.createVariable("time", "f8", ("time",))[:] = [1.0, 0.0]
        fallback = check_coordinate_catalog(
            ds,
            catalog,
            severities={"direction": BaseCheck.HIGH},
            time_monotonicity_delegated=False,
        )
        delegated = check_coordinate_catalog(
            ds,
            catalog,
            severities={"direction": BaseCheck.HIGH},
            time_monotonicity_delegated=True,
        )

    assert any("increasing" in message for message in _messages(fallback))
    assert _messages(delegated) == []


def test_non_time_monotonicity_rule_does_not_own_time_direction():
    checker = Cmip7ProjectCheck()
    checker.config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "coordinates": {
                "latitude": {
                    "name": {"severity": "H", "variable_name": "lat"},
                    "monotonicity": {
                        "severity": "H",
                        "direction": "increasing",
                    },
                },
                "time": {
                    "name": {"severity": "H", "variable_name": "time"}
                },
            },
        }
    )
    checker._coordinate_catalog = _catalog(
        {
            "latitude": _coordinate("lat", axis="Y"),
            "time": _coordinate("time", axis="T"),
        }
    )

    assert not checker._coordinate_rule_enabled_for_axis("T", "monotonicity")
    assert checker._coordinate_rule_enabled_for_axis("Y", "monotonicity")


def test_time002_reports_missing_bounds_only_without_coord008_owner(tmp_path):
    with Dataset(tmp_path / "time-bounds.nc", "w") as ds:
        ds.createDimension("time", 1)
        ds.createVariable("time", "f8", ("time",))[:] = [0.5]
        fallback = check_time_bounds(ds, report_bounds_prerequisites=True)
        delegated = check_time_bounds(ds, report_bounds_prerequisites=False)

    assert "has no bounds attribute" in _messages(fallback)[0]
    assert delegated == []


def test_time002_reports_missing_referenced_bounds_without_coord008_owner(
    tmp_path,
):
    with Dataset(tmp_path / "missing-time-bounds-variable.nc", "w") as ds:
        ds.createDimension("time", 1)
        time = ds.createVariable("time", "f8", ("time",))
        time[:] = [0.5]
        time.bounds = "time_bnds"
        fallback = check_time_bounds(
            ds,
            report_bounds_prerequisites=True,
        )
        delegated = check_time_bounds(
            ds,
            report_bounds_prerequisites=False,
        )

    assert "bounds variable 'time_bnds'" in _messages(fallback)[0]
    assert delegated == []


def test_time001_leaves_missing_frequency_to_attr_owner(tmp_path):
    with Dataset(tmp_path / "tas_mon_200001-200001.nc", "w") as ds:
        ds.createDimension("time", 1)
        time = ds.createVariable("time", "f8", ("time",))
        time[:] = [15.5]
        time.units = "days since 2000-01-01"
        delegated = check_time_squareness(
            ds, report_frequency_prerequisite=False
        )
        fallback = check_time_squareness(ds, report_frequency_prerequisite=True)

    assert delegated == []
    assert "Missing global attribute 'frequency'" in _messages(fallback)[0]


@pytest.mark.parametrize(
    ("delegated", "expected_count"),
    [(True, 0), (False, 1)],
)
def test_attr008_missing_frequency_respects_attribute_owner(
    tmp_path, delegated, expected_count
):
    with Dataset(tmp_path / "missing-frequency-attr008.nc", "w") as ds:
        ds.branded_variable = "tas_tavg-h2m-hxy-air"
        results = check_frequency_branded_variable_consistency(
            ds,
            ["mon"],
            BaseCheck.LOW,
            project_label="CMIP7",
            missing_attributes_delegated=("frequency",) if delegated else (),
        )

    assert len(_messages(results)) == expected_count


@pytest.mark.parametrize(
    ("delegated", "expected_count"),
    [(True, 0), (False, 1)],
)
def test_attr008_invalid_frequency_respects_vocabulary_owner(
    tmp_path, delegated, expected_count
):
    with Dataset(tmp_path / "invalid-frequency-attr008.nc", "w") as ds:
        ds.frequency = "invalid"
        ds.branded_variable = "tas_tavg-h2m-hxy-air"
        results = check_frequency_branded_variable_consistency(
            ds,
            ["mon"],
            BaseCheck.LOW,
            project_label="CMIP7",
            frequency_is_registered=False,
            invalid_frequency_delegated=delegated,
        )

    assert len(_messages(results)) == expected_count


@pytest.mark.parametrize(
    ("delegated", "expected_count"),
    [(True, 0), (False, 1)],
)
def test_attr008_registry_failure_respects_metadata_owner(
    tmp_path, delegated, expected_count
):
    with Dataset(tmp_path / "registry-failure-attr008.nc", "w") as ds:
        ds.frequency = "mon"
        ds.branded_variable = "tas_tavg-h2m-hxy-air"
        results = check_frequency_branded_variable_consistency(
            ds,
            None,
            BaseCheck.LOW,
            project_label="CMIP7",
            metadata_error="database unavailable",
            metadata_error_delegated=delegated,
        )

    assert len(_messages(results)) == expected_count


@pytest.mark.parametrize(
    ("delegated", "expected_count"),
    [(True, 0), (False, 1)],
)
def test_attr012_missing_region_respects_attribute_owner(
    tmp_path, delegated, expected_count
):
    with Dataset(tmp_path / "missing-region-attr012.nc", "w") as ds:
        ds.branded_variable = "tas_tavg-h2m-hxy-u"
        results = check_region_branded_variable_consistency(
            ds,
            ["glb"],
            BaseCheck.LOW,
            project_label="CMIP7",
            missing_attributes_delegated=("region",) if delegated else (),
        )

    assert len(_messages(results)) == expected_count


@pytest.mark.parametrize(
    ("delegated", "expected_count"),
    [(True, 0), (False, 1)],
)
def test_attr012_invalid_region_respects_vocabulary_owner(
    tmp_path, delegated, expected_count
):
    with Dataset(tmp_path / "invalid-region-attr012.nc", "w") as ds:
        ds.region = "invalid"
        ds.branded_variable = "tas_tavg-h2m-hxy-u"
        results = check_region_branded_variable_consistency(
            ds,
            ["glb"],
            BaseCheck.LOW,
            project_label="CMIP7",
            region_is_registered=False,
            invalid_region_delegated=delegated,
        )

    assert len(_messages(results)) == expected_count


@pytest.mark.parametrize(
    ("delegated", "expected_count"),
    [(True, 0), (False, 1)],
)
def test_attr012_registry_failure_respects_metadata_owner(
    tmp_path, delegated, expected_count
):
    with Dataset(tmp_path / "registry-failure-attr012.nc", "w") as ds:
        ds.region = "glb"
        ds.branded_variable = "tas_tavg-h2m-hxy-u"
        results = check_region_branded_variable_consistency(
            ds,
            None,
            BaseCheck.LOW,
            project_label="CMIP7",
            metadata_error="database unavailable",
            metadata_error_delegated=delegated,
        )

    assert len(_messages(results)) == expected_count


@pytest.mark.parametrize(
    ("disabled_prefix", "expected_check"),
    (
        ((), "ATTR001"),
        (("attribute",), "TIME001"),
        (("attribute", "squareness"), "TIME003"),
        (("attribute", "squareness", "time_range"), None),
    ),
)
@pytest.mark.parametrize("weights", [(3, 3, 3), (1, 3, 2), (3, 2, 1)])
def test_missing_frequency_hierarchy_walks_enabled_toml_rules(
    tmp_path,
    disabled_prefix,
    expected_check,
    weights,
):
    enabled = {"attribute", "squareness", "time_range"} - set(disabled_prefix)
    config = {
        "project_name": "cmip7",
        "project_version": "1.0",
        "global": {
            "attributes": (
                {"frequency": {"severity": {1: "L", 2: "M", 3: "H"}[weights[0]]}}
                if "attribute" in enabled
                else {}
            )
        },
        "coordinates": {
            "time": {
                "name": {"severity": "H", "variable_name": "time"},
                **(
                    {"squareness": {"severity": "H"}}
                    if "squareness" in enabled
                    else {}
                ),
            }
        },
        "drs": (
            {"time_range": {"severity": "H"}}
            if "time_range" in enabled
            else {}
        ),
    }
    checker = Cmip7ProjectCheck()
    checker.config = WCRPConfig.model_validate(config)

    path = tmp_path / "ta_mon_200001-200001.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time[:] = [15.5]
        results = []
        if "attribute" in enabled:
            results.extend(
                check_attribute_suite(
                    ds,
                    "frequency",
                    weights[0],
                    is_required=True,
                )
            )
        if "squareness" in enabled:
            results.extend(
                check_time_squareness(
                    ds,
                    severity=weights[1],
                    expected_cell_methods="time: mean",
                    report_frequency_prerequisite=not checker._global_attribute_owns("frequency"),
                )
            )
        if "time_range" in enabled:
            squareness_owns = "squareness" in enabled
            results.extend(
                check_time_range_vs_filename(
                    ds,
                    severity=weights[2],
                    report_frequency_prerequisite=not (
                        checker._global_attribute_owns("frequency")
                        or squareness_owns
                    ),
                )
            )

    failures = [result for result in results if result.msgs]
    assert len(failures) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert failures[0].name.startswith(f"[{expected_check}]")
        assert failures[0].weight == weights[("ATTR001", "TIME001", "TIME003").index(expected_check)]


@pytest.mark.parametrize(
    ("disabled_prefix", "expected_check"),
    (
        ((), "COORD003"),
        (("coordinate_attributes",), "TIME001"),
        (("coordinate_attributes", "squareness"), "TIME003"),
        (("coordinate_attributes", "squareness", "time_range"), None),
    ),
)
@pytest.mark.parametrize("weights", [(3, 3, 3), (1, 3, 2), (3, 2, 1)])
def test_missing_time_units_hierarchy_walks_enabled_toml_rules(
    tmp_path,
    disabled_prefix,
    expected_check,
    weights,
):
    enabled = {
        "coordinate_attributes",
        "squareness",
        "time_range",
    } - set(disabled_prefix)
    checker = Cmip7ProjectCheck()
    checker.config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "coordinates": {
                "registry": (
                    {"attributes": {"severity": "H"}}
                    if "coordinate_attributes" in enabled
                    else {}
                ),
                "time": {
                    "name": {"severity": "H", "variable_name": "time"},
                    **(
                        {"squareness": {"severity": "H"}}
                        if "squareness" in enabled
                        else {}
                    ),
                },
            },
            "drs": (
                {"time_range": {"severity": "H"}}
                if "time_range" in enabled
                else {}
            ),
        }
    )
    catalog = _catalog(
        {
            "time": _coordinate(
                "time",
                axis="T",
                units="days since ?",
                cf_standard_name="time",
            )
        }
    )

    path = tmp_path / "ta_mon_200001-200001.nc"
    with Dataset(path, "w") as ds:
        ds.frequency = "mon"
        ds.createDimension("time", 1)
        ds.createVariable("time", "f8", ("time",))[:] = [15.5]
        results = []
        if "coordinate_attributes" in enabled:
            results.extend(
                check_coordinate_catalog(
                    ds,
                    catalog,
                    severities={"attributes": weights[0]},
                )
            )
        if "squareness" in enabled:
            results.extend(
                check_time_squareness(
                    ds,
                    severity=weights[1],
                    expected_cell_methods="time: mean",
                    report_units_prerequisite=(
                        "coordinate_attributes" not in enabled
                    ),
                )
            )
        if "time_range" in enabled:
            results.extend(
                check_time_range_vs_filename(
                    ds,
                    severity=weights[2],
                    report_time_units_prerequisite=not (
                        "coordinate_attributes" in enabled
                        or "squareness" in enabled
                    ),
                )
            )

    failures = [result for result in results if result.msgs]
    assert len(failures) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert failures[0].name.startswith(f"[{expected_check}]")
        assert failures[0].weight == weights[("COORD003", "TIME001", "TIME003").index(expected_check)]


@pytest.mark.parametrize(
    ("coordinate_bounds_enabled", "time002_enabled", "expected_check"),
    (
        (True, True, "COORD008"),
        (False, True, "TIME002"),
        (False, False, None),
    ),
)
@pytest.mark.parametrize("weights", [(3, 3), (1, 3), (3, 2)])
def test_missing_regular_time_bounds_walks_enabled_toml_rules(
    tmp_path,
    coordinate_bounds_enabled,
    time002_enabled,
    expected_check,
    weights,
):
    catalog = _catalog(
        {
            "time": _coordinate(
                "time",
                axis="T",
                bounds_required=True,
            )
        }
    )
    with Dataset(tmp_path / "time-bounds-hierarchy.nc", "w") as ds:
        ds.createDimension("time", 1)
        ds.createVariable("time", "f8", ("time",))[:] = [0.5]
        results = []
        if coordinate_bounds_enabled:
            results.extend(
                check_coordinate_catalog(
                    ds,
                    catalog,
                    severities={"bounds": weights[0]},
                )
            )
        if time002_enabled:
            results.extend(
                check_time_bounds(
                    ds,
                    weights[1],
                    report_bounds_prerequisites=not coordinate_bounds_enabled,
                )
            )

    failures = [result for result in results if result.msgs]
    assert len(failures) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert failures[0].name.startswith(f"[{expected_check}]")
        assert failures[0].weight == weights[("COORD008", "TIME002").index(expected_check)]


def test_grid_region_reports_missing_attribute_without_attr_owner(tmp_path):
    from checks.coordinate_checks.grid import check_grid_region_consistency

    findings = Findings({"grid_region_consistency": BaseCheck.MEDIUM})
    with Dataset(tmp_path / "region.nc", "w") as ds:
        check_grid_region_consistency(
            findings,
            ds,
            {"id": "g001", "region": {"drs_name": "glb"}},
            attribute_presence_delegated=False,
        )

    assert "Global attribute 'region' is missing" in _messages(findings.results())[0]


def test_filename_consistency_reports_unowned_missing_attribute(tmp_path):
    filename = "tas_Amon_Model_historical_r1i1p1f1_gn_200001-200012.nc"
    with Dataset(tmp_path / filename, "w") as ds:
        results = check_filename_vs_global_attrs(
            ds,
            BaseCheck.HIGH,
            project_id="cmip6",
            missing_attributes_delegated={"table_id", "source_id"},
        )

    messages = _messages(results)
    assert any("variable_id" in message and "missing" in message for message in messages)
    assert not any("table_id" in message for message in messages)


def test_path002_leaves_missing_drs_specs_to_attr001(tmp_path):
    with Dataset(tmp_path / "example.nc", "w") as ds:
        delegated = check_filename_matches_directory_structure(
            ds,
            BaseCheck.HIGH,
            project_id="cmip7",
            missing_attributes_delegated={"drs_specs"},
        )
        fallback = check_filename_matches_directory_structure(
            ds,
            BaseCheck.HIGH,
            project_id="cmip7",
        )

    assert delegated == []
    assert "missing global attribute 'drs_specs'" in _messages(fallback)[0]


def test_frequency_consistency_reports_only_unowned_missing_attribute(tmp_path):
    with Dataset(tmp_path / "frequency.nc", "w") as ds:
        results = check_frequency_table_id_consistency(
            ds,
            {},
            BaseCheck.HIGH,
            missing_attributes_delegated={"frequency"},
        )

    assert _messages(results) == [
        (
            "Global attribute 'table_id' is missing, so frequency/table_id "
            "consistency could not be checked."
        )
    ]


def test_configured_attributes_own_their_prerequisites():
    checker = Cmip6ProjectCheck()
    checker._load_split_config()

    assert checker._global_attribute_owns("source")
    assert checker._global_attribute_owns("institution")
    assert checker._global_attribute_owns("source_id")


def test_variant_prerequisite_defers_to_enabled_attribute_owner(tmp_path):
    with Dataset(tmp_path / "variant-severity.nc", "w") as dataset:
        dataset.variant_label = "r1i1p1f1"
        results = variant_checks.check_variant_label_consistency_group(
            dataset,
            [
                (
                    variant_checks.check_variant_vs_realization_index,
                    BaseCheck.HIGH,
                )
            ],
            attribute_severities={
                "variant_label": BaseCheck.HIGH,
                "realization_index": BaseCheck.MEDIUM,
            },
        )

    assert _messages(results) == []


def test_attr011_reports_unowned_missing_attribute(tmp_path):
    with Dataset(tmp_path / "ids.nc", "w") as ds:
        results = check_id_attribute_consistency(
            ds,
            BaseCheck.HIGH,
            project_id="cordex-cmip6",
            id_attribute="source_id",
            value_attribute="source",
            missing_attributes_delegated={"source_id"},
        )

    assert _messages(results) == [
        (
            "Global attribute 'source' is missing, so 'source_id' could not be "
            "compared with 'source'."
        )
    ]


def test_expected_data_variable_cannot_be_masked_by_unrelated_cf_candidate(tmp_path):
    checker = Cmip7ProjectCheck()
    checker.config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "variable": {"existence": {"severity": "H"}},
        }
    )
    checker._coordinate_catalog = _catalog({})
    checker._geo_var_cache = None
    with Dataset(tmp_path / "wrong-variable.nc", "w") as ds:
        ds.createDimension("x", 1)
        ds.createVariable("unrelated", "f4", ("x",))
        name, results = checker._get_geo_var(ds, BaseCheck.HIGH)

    assert name is None
    assert len(results) == 1
    assert "Variable 'ta' is missing" in _messages(results)[0]


def test_expected_data_variable_is_authoritative_for_every_project_plugin(tmp_path):
    checker_classes = (
        Cmip7ProjectCheck,
        Cmip6ProjectCheck,
        Cmip6PlusProjectCheck,
        C3SCmip6ProjectCheck,
        CordexCmip6ProjectCheck,
    )
    for checker_class in checker_classes:
        checker = checker_class()
        checker.config = WCRPConfig.model_validate(
            {
                "project_name": checker.project_name,
                "project_version": "1.0",
                "variable": {"existence": {"severity": "H"}},
            }
        )
        checker._geo_var_cache = None
        with Dataset(tmp_path / f"{checker.project_name}-wrong-variable.nc", "w") as ds:
            ds.variable_id = "ta"
            ds.createDimension("x", 1)
            ds.createVariable("unrelated", "f4", ("x",))
            name, results = checker._get_geo_var(ds, BaseCheck.HIGH)

        assert name is None
        assert len(results) == 1
        assert "Variable 'ta' is missing" in _messages(results)[0]


def test_known_missing_data_variable_uses_existence_owner(tmp_path):
    checker = Cmip7ProjectCheck()
    checker.config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "variable": {
                "existence": {"severity": "M"},
                "type": {"severity": "M", "data_type": "real"},
                "attributes": {
                    "units": {"severity": "H", "is_required": True}
                },
            },
        }
    )
    checker._coordinate_catalog = _catalog({})
    checker._geo_var_cache = None
    with Dataset(tmp_path / "missing-expected-variable.nc", "w") as ds:
        name, results = checker._get_geo_var(
            ds, checker._configured_variable_severity()
        )

    assert name is None
    assert results[0].weight == BaseCheck.MEDIUM
    assert "Variable 'ta' is missing" in _messages(results)[0]


def test_cordex_compression_delegates_to_enabled_variable_owner(
    monkeypatch,
):
    checker = CordexCmip6ProjectCheck()
    checker.config = WCRPConfig.model_validate(
        {
            "project_name": "cordex-cmip6",
            "project_version": "1.0",
            "file": {"compression": {"severity": "H"}},
            "variable": {"type": {"severity": "M", "data_type": "real"}},
        }
    )
    ctx = ComplianceTestCtx(BaseCheck.HIGH, "Geophysical Variable Detection")
    ctx.add_failure("data variable missing")
    detection = [ctx.to_result()]
    monkeypatch.setattr(
        checker, "_get_geo_var", lambda *_args, **_kwargs: (None, detection)
    )

    assert checker.check_File_Compression(SimpleNamespace()) == []

    checker.config.variable.type.severity = "H"
    assert checker.check_File_Compression(SimpleNamespace()) == []
    checker.config.variable.type = None
    assert checker.check_File_Compression(SimpleNamespace()) == detection


def test_cmip6_family_reports_required_coordinate_missing_from_file(
    tmp_path, monkeypatch
):
    for checker_class in (
        Cmip6ProjectCheck,
        Cmip6PlusProjectCheck,
        C3SCmip6ProjectCheck,
    ):
        checker = checker_class()
        checker._load_split_config()
        monkeypatch.setattr(
            checker,
            "_get_expected_from_registry",
            lambda *_args, **_kwargs: (
                ExpectedVariableMetadata(
                    dimensions=[
                        "longitude",
                        "latitude",
                        "air_pressure",
                        "height2m",
                        "depth",
                        "alevel",
                        "time",
                    ],
                    coordinate_out_names={
                        "longitude": "lon",
                        "latitude": "lat",
                        "air_pressure": "plev",
                        "height2m": "height",
                        "depth": "depth",
                        "alevel": "lev",
                        "time": "time",
                    },
                    time_is_climatology=None,
                    cell_methods=None,
                ),
                [],
            ),
        )
        with Dataset(tmp_path / f"{checker.project_name}-missing-time.nc", "w") as ds:
            results = checker.check_Coordinates(ds)

        messages = _messages(results)
        for name in ("lon", "lat", "plev", "height", "depth", "lev", "time"):
            assert any(f"Variable '{name}' is missing" in message for message in messages)


def test_cmip6_bounds_shape_owns_required_bounds_when_attribute_rule_is_disabled(
    tmp_path, monkeypatch
):
    checker = Cmip6ProjectCheck()
    checker._load_split_config()
    checker.config.coordinates.variables["latitude"].attributes.pop("bounds")
    monkeypatch.setattr(
        checker,
        "_get_expected_from_registry",
        lambda *_args, **_kwargs: (
            ExpectedVariableMetadata(
                dimensions=["latitude"],
                coordinate_out_names={"latitude": "lat"},
                coordinate_bounds_required={"latitude": True},
            ),
            [],
        ),
    )

    with Dataset(tmp_path / "missing-required-bounds.nc", "w") as ds:
        ds.createDimension("lat", 2)
        ds.createVariable("lat", "f8", ("lat",))[:] = [-45.0, 45.0]
        results = checker.check_Coordinates(ds)

    assert any(
        result.name.startswith("[VAR004]")
        and "requires bounds but has no bounds attribute" in message
        for result in results
        for message in result.msgs
    )


def test_cmip6_time002_owns_missing_bounds_when_attribute_is_optional(
    tmp_path, monkeypatch
):
    checker = Cmip6ProjectCheck()
    checker._load_split_config()
    monkeypatch.setattr(
        checker,
        "_get_expected_from_registry",
        lambda *_args, **_kwargs: (
            ExpectedVariableMetadata(
                dimensions=["time"],
                coordinate_out_names={"time": "time"},
                coordinate_bounds_required={"time": True},
                cell_methods="time: mean",
            ),
            [],
        ),
    )

    with Dataset(tmp_path / "tas_Amon_200001-200001.nc", "w") as ds:
        ds.frequency = "mon"
        ds.createDimension("time", 1)
        time = ds.createVariable("time", "f8", ("time",))
        time[:] = [15.5]
        time.units = "days since 2000-01-01"
        results = checker.check_Coordinates(ds)

    assert any(
        result.name.startswith("[TIME002]")
        and "has no bounds attribute" in message
        for result in results
        for message in result.msgs
    )


def test_cmip6_time003_owns_missing_values_when_time001_is_disabled(
    tmp_path, monkeypatch
):
    checker = Cmip6ProjectCheck()
    checker._load_split_config()
    checker.config.coordinates.variables["time"].squareness = None
    monkeypatch.setattr(
        checker,
        "_get_expected_from_registry",
        lambda *_args, **_kwargs: (
            ExpectedVariableMetadata(
                dimensions=["time"],
                coordinate_out_names={"time": "time"},
                time_is_climatology=False,
            ),
            [],
        ),
    )

    filename = "tas_Amon_Model_historical_r1i1p1f1_gn_200001-200001.nc"
    with Dataset(tmp_path / filename, "w") as ds:
        ds.frequency = "mon"
        ds.createDimension("time", 1)
        time = ds.createVariable("time", "f8", ("time",), fill_value=1.0e20)
        time.units = "days since 2000-01-01"
        results = checker.check_Coordinates(ds)

    assert any(
        result.name.startswith("[TIME003]")
        and "first or last time value" in message
        for result in results
        for message in result.msgs
    )

def test_cmip6_coordinate_dispatcher_does_not_discard_registry_failure(
    tmp_path, monkeypatch
):
    checker = Cmip6ProjectCheck()
    checker._load_split_config()
    checker.config.drs.time_range = None
    ctx = ComplianceTestCtx(BaseCheck.HIGH, "Variable Registry")
    ctx.add_failure("registry unavailable")
    monkeypatch.setattr(
        checker,
        "_get_expected_from_registry",
        lambda *_args, **_kwargs: (None, [ctx.to_result()]),
    )

    with Dataset(tmp_path / "registry-failure.nc", "w") as ds:
        results = checker.check_Coordinates(ds)

    assert _messages(results).count("registry unavailable") == 1


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


def test_setup_failure_uses_highest_enabled_dependent_severity():
    config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "coordinates": {
                "registry": {
                    "setup": {"severity": "L"},
                    "attributes": {"severity": "H"},
                },
            },
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
    assert results[0].weight == BaseCheck.HIGH


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


def test_omitting_drs_time_range_really_disables_catalogue_dependency():
    config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "drs": {"filename": {"severity": "H"}},
        }
    )

    assert config.drs.time_range is None
    assert coordinate_catalog_required(config) is False


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


def test_missing_configured_time_uses_stable_hierarchy_for_equal_severities():
    rule = CoordinateVariableConfig.model_validate(
        {
            "monotonicity": {"severity": "H", "direction": "increasing"},
            "squareness": {"severity": "H"},
        }
    )
    checker = Cmip7ProjectCheck()
    results = missing_configured_coordinate_result(
        "time", rule, checker.get_severity
    )

    assert len(results) == 1
    assert "[VAR005]" in results[0].name
    assert results[0].weight == BaseCheck.HIGH


def test_coordinate_prerequisite_keeps_enabled_owner():
    findings = Findings(
        {
            "identity": BaseCheck.MEDIUM,
            "attributes": BaseCheck.HIGH,
        }
    )
    owner = findings.add_prerequisite(
        "identity",
        lambda family: f"reported by {family}",
        issue="missing-coordinate",
    )

    assert owner == "identity"
    results = findings.results()
    failure = next(result for result in results if result.msgs)
    assert failure.weight == BaseCheck.MEDIUM
    assert failure.msgs == ["reported by identity"]


def test_coordinate_prerequisite_tie_uses_hard_coded_hierarchy():
    # Reverse dictionary insertion order to ensure it cannot decide ownership.
    findings = Findings(
        {
            "attributes": BaseCheck.HIGH,
            "identity": BaseCheck.HIGH,
        }
    )
    owner = findings.add_prerequisite(
        "identity",
        "missing coordinate",
        issue="missing-coordinate",
    )

    assert owner == "identity"


@pytest.mark.parametrize(
    ("owner", "enabled_families", "expected_owner"),
    _COORDINATE_PREREQUISITE_WALK,
)
@pytest.mark.parametrize("weights", [(3, 3, 3), (1, 3, 2), (2, 1, 3), (3, 2, 1)])
def test_coordinate_prerequisite_hierarchies_walk_enabled_fallbacks(
    owner,
    enabled_families,
    expected_owner,
    weights,
):
    """Each disabled owner hands one root cause to the next configured check."""
    findings = Findings(
        {family: weights[i % len(weights)] for i, family in enumerate(enabled_families)}
    )

    selected = findings.add_prerequisite(
        owner,
        lambda family: f"reported by {family}",
        issue=(owner, "unusable prerequisite"),
    )

    assert selected == expected_owner
    failures = [result for result in findings.results() if result.msgs]
    if expected_owner is not None:
        assert len(failures) == 1
        assert failures[0].weight == findings.severities[expected_owner]
    populated = {
        family: messages
        for family, messages in findings.messages.items()
        if messages
    }
    assert populated == (
        {expected_owner: [f"reported by {expected_owner}"]}
        if expected_owner is not None
        else {}
    )


@pytest.mark.parametrize(
    ("enabled_families", "expected_owner"),
    _BLOCKED_DATA_VARIABLE_WALK,
)
@pytest.mark.parametrize("weights", [(3, 3, 3), (1, 3, 2), (2, 1, 3), (3, 2, 1)])
def test_missing_data_variable_hierarchy_walks_enabled_consumers(
    enabled_families,
    expected_owner,
    weights,
):
    findings = Findings(
        {family: weights[i % len(weights)] for i, family in enumerate(enabled_families)}
    )

    selected = findings.add_blocked(
        (
            "dimension_order",
            "associations",
            "grid_mapping_consistency",
            "grid_cell_count_consistency",
        ),
        "missing data variable",
        issue="missing-data-variable",
    )

    assert selected == expected_owner
    populated = {
        family: messages
        for family, messages in findings.messages.items()
        if messages
    }
    assert populated == (
        {expected_owner: ["missing data variable"]}
        if expected_owner is not None
        else {}
    )


def test_blocked_prerequisite_uses_argument_order():
    findings = Findings(
        {
            "dimension_order": BaseCheck.MEDIUM,
            "associations": BaseCheck.HIGH,
            "grid_mapping_consistency": BaseCheck.HIGH,
        }
    )
    owner = findings.add_blocked(
        ("dimension_order", "associations", "grid_mapping_consistency"),
        "missing data variable",
        issue="missing-data-variable",
    )

    assert owner == "dimension_order"


@pytest.mark.parametrize(
    ("enabled_checks", "expected_check"),
    tuple(
        pytest.param(checks[index:], check_id, id=check_id)
        for checks in [
            (
                variant_checks.check_variant_vs_realization_index,
                variant_checks.check_variant_vs_initialization_index,
                variant_checks.check_variant_vs_physics_index,
                variant_checks.check_variant_vs_forcing_index,
            )
        ]
        for index, check_id in enumerate(
            ("ATTR006a", "ATTR006b", "ATTR006c", "ATTR006d")
        )
    )
    + (pytest.param((), None, id="all-disabled"),),
)
@pytest.mark.parametrize("weights", [(3, 3, 3), (1, 3, 2), (2, 1, 3), (3, 2, 1)])
def test_variant_selector_hierarchy_walks_enabled_toml_checks(
    tmp_path,
    enabled_checks,
    expected_check,
    weights,
):
    with Dataset(tmp_path / f"{expected_check}.nc", "w") as dataset:
        dataset.variant_label = "invalid"
        dataset.realization_index = 1
        dataset.initialization_index = 1
        dataset.physics_index = 1
        dataset.forcing_index = 1
        results = variant_checks.check_variant_label_consistency_group(
            dataset,
            [(check, weights[i % len(weights)]) for i, check in enumerate(enabled_checks)],
        )

    failures = [result for result in results if result.msgs]
    assert len(failures) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert failures[0].name.startswith(f"[{expected_check}]")


@pytest.mark.parametrize(
    ("enabled_checks", "expected_check"),
    tuple(
        pytest.param(checks[index:], check_id, id=check_id)
        for checks in [
            (
                experiment_checks.check_experiment_id_vs_activity_id,
                experiment_checks.check_experiment_id_vs_experiment,
                experiment_checks.check_experiment_id_vs_parent_experiment_id,
                experiment_checks.check_experiment_id_vs_sub_experiment_id,
            )
        ]
        for index, check_id in enumerate(
            ("ATTR007a", "ATTR007b", "ATTR007c", "ATTR007d")
        )
    )
    + (pytest.param((), None, id="all-disabled"),),
)
@pytest.mark.parametrize("weights", [(3, 3, 3), (1, 3, 2), (2, 1, 3), (3, 2, 1)])
def test_experiment_selector_hierarchy_walks_enabled_toml_checks(
    tmp_path,
    monkeypatch,
    enabled_checks,
    expected_check,
    weights,
):
    monkeypatch.setattr(experiment_checks, "ESG_VOCAB_AVAILABLE", True)
    monkeypatch.setattr(
        experiment_checks,
        "resolve_experiment_term",
        lambda *_args, **_kwargs: None,
    )
    with Dataset(tmp_path / f"{expected_check}.nc", "w") as dataset:
        dataset.experiment_id = "invalid"
        dataset.activity_id = "CMIP"
        dataset.experiment = "invalid"
        dataset.parent_experiment_id = "piControl"
        dataset.sub_experiment_id = "none"
        results = experiment_checks.check_experiment_consistency_group(
            dataset,
            "cmip6",
            [
                (check, weights[i % len(weights)], {"report_missing": True})
                for i, check in enumerate(enabled_checks)
            ],
        )

    failures = [result for result in results if result.msgs]
    assert len(failures) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert failures[0].name.startswith(f"[{expected_check}]")


def test_variant_selector_owner_uses_fixed_priority(tmp_path):
    with Dataset(tmp_path / "variant.nc", "w") as dataset:
        dataset.variant_label = "invalid"
        dataset.realization_index = 1
        dataset.forcing_index = 1
        dataset.physics_index = 1
        results = variant_checks.check_variant_label_consistency_group(
            dataset,
            [
                (
                    variant_checks.check_variant_vs_realization_index,
                    BaseCheck.MEDIUM,
                ),
                (
                    variant_checks.check_variant_vs_forcing_index,
                    BaseCheck.HIGH,
                ),
            ],
        )
        tied_results = variant_checks.check_variant_label_consistency_group(
            dataset,
            [
                (
                    variant_checks.check_variant_vs_forcing_index,
                    BaseCheck.HIGH,
                ),
                (
                    variant_checks.check_variant_vs_physics_index,
                    BaseCheck.HIGH,
                ),
            ],
        )

    assert len(results) == 1
    assert results[0].name.startswith("[ATTR006a]")
    assert results[0].weight == BaseCheck.MEDIUM
    assert tied_results[0].name.startswith("[ATTR006c]")


def test_variant_selector_rejects_an_unknown_check_function(tmp_path):
    def unsupported_variant_check(
        _dataset,
        _severity,
        *,
        report_selector_error=True,
        report_missing=True,
    ):
        return []

    with Dataset(tmp_path / "variant-unsupported.nc", "w") as dataset:
        dataset.variant_label = "r1i1p1f1"
        with pytest.raises(
            ValueError,
            match=(
                "Unsupported variant-label consistency check: "
                "unsupported_variant_check"
            ),
        ):
            variant_checks.check_variant_label_consistency_group(
                dataset,
                [(unsupported_variant_check, BaseCheck.HIGH)],
            )


def test_experiment_selector_owner_uses_fixed_priority(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(experiment_checks, "ESG_VOCAB_AVAILABLE", True)
    monkeypatch.setattr(
        experiment_checks, "resolve_experiment_term", lambda *_args, **_kwargs: None
    )
    with Dataset(tmp_path / "experiment.nc", "w") as dataset:
        dataset.experiment_id = "invalid"
        dataset.activity_id = "CMIP"
        dataset.experiment = "invalid"
        results = experiment_checks.check_experiment_consistency_group(
            dataset,
            "cmip6",
            [
                (
                    experiment_checks.check_experiment_id_vs_activity_id,
                    BaseCheck.MEDIUM,
                    {"report_missing": True},
                ),
                (
                    experiment_checks.check_experiment_id_vs_experiment,
                    BaseCheck.HIGH,
                    {"report_missing": True},
                ),
            ],
        )
        tied_results = experiment_checks.check_experiment_consistency_group(
            dataset,
            "cmip6",
            [
                (
                    experiment_checks.check_experiment_id_vs_experiment,
                    BaseCheck.HIGH,
                    {"report_missing": True},
                ),
                (
                    experiment_checks.check_experiment_id_vs_activity_id,
                    BaseCheck.HIGH,
                    {"report_missing": True},
                ),
            ],
        )

    assert len(results) == 1
    assert results[0].name.startswith("[ATTR007a]")
    assert results[0].weight == BaseCheck.MEDIUM
    assert tied_results[0].name.startswith("[ATTR007a]")


def test_consistency_groups_leave_missing_selectors_to_attr001(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(experiment_checks, "ESG_VOCAB_AVAILABLE", True)
    with Dataset(tmp_path / "delegated-selectors.nc", "w") as dataset:
        dataset.activity_id = "CMIP"
        experiment = experiment_checks.check_experiment_consistency_group(
            dataset,
            "cmip6",
            [
                (
                    experiment_checks.check_experiment_id_vs_activity_id,
                    BaseCheck.HIGH,
                    {"report_missing": True},
                )
            ],
            selector_error_delegated=True,
        )
        variant = variant_checks.check_variant_label_consistency_group(
            dataset,
            [
                (
                    variant_checks.check_variant_vs_realization_index,
                    BaseCheck.HIGH,
                )
            ],
            missing_attributes_delegated={"variant_label", "realization_index"},
        )

    assert experiment == []
    assert variant == []


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


def test_cmip7_directory_check_uses_mip_drs7_without_legacy_fallback(tmp_path):
    with Dataset(tmp_path / "ta.nc", "w") as dataset:
        dataset.drs_specs = "MIP-DRS7"
        results = check_drs_directory(
            dataset,
            BaseCheck.HIGH,
            project_id="cmip7",
        )

    assert len(results) == 1
    assert results[0].name.startswith("[PATH003]")
    assert "'MIP-DRS7' not found" in results[0].msgs[0]
    assert "root 'cmip7'" not in results[0].msgs[0]


@pytest.mark.parametrize(
    ("disabled_rules", "expected_check"),
    (
        ((), "PATH003"),
        (("directory",), "PATH001"),
        (("directory", "attributes_vs_directory"), "PATH002"),
        (
            ("directory", "attributes_vs_directory", "filename_vs_directory"),
            None,
        ),
    ),
)
@pytest.mark.parametrize("weights", [("H", "H", "H"), ("L", "H", "M"), ("H", "M", "L")])
def test_drs_directory_hierarchy_walks_enabled_toml_rules(
    tmp_path,
    monkeypatch,
    disabled_rules,
    expected_check,
    weights,
):
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    for name, weight in zip(("directory", "attributes_vs_directory", "filename_vs_directory"), weights):
        getattr(checker.config.drs, name).severity = weight
    for rule_name in disabled_rules:
        setattr(checker.config.drs, rule_name, None)
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.check_drs_filename",
        lambda *_args, **_kwargs: [],
    )

    with Dataset(tmp_path / "ta.nc", "w") as dataset:
        dataset.drs_specs = "MIP-DRS7"
        results = checker.check_DRS(dataset)

    assert len(results) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert results[0].name.startswith(f"[{expected_check}]")
        assert results[0].weight == checker.get_severity(weights[("PATH003", "PATH001", "PATH002").index(expected_check)])
        assert "MIP-DRS7" in results[0].msgs[0]


def test_first_path_consumer_owns_structure_when_path003_is_disabled(
    tmp_path, monkeypatch
):
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    drs = checker.config.drs
    drs.directory = None
    drs.attributes_vs_directory.severity = "M"
    drs.filename_vs_directory.severity = "H"
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.check_drs_filename",
        lambda *_args, **_kwargs: [],
    )

    with Dataset(tmp_path / "ta.nc", "w") as dataset:
        dataset.drs_specs = "MIP-DRS7"
        results = checker.check_DRS(dataset)

    assert len(results) == 1
    assert results[0].name.startswith("[PATH001]")
    assert results[0].weight == BaseCheck.MEDIUM
    assert "MIP-DRS7" in results[0].msgs[0]


def test_equal_severity_path_consumers_use_fixed_priority():
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    drs = checker.config.drs
    drs.directory = None
    drs.attributes_vs_directory.severity = "H"
    drs.filename_vs_directory.severity = "H"

    assert checker._drs_directory_structure_owner(drs) == "attributes_vs_directory"
