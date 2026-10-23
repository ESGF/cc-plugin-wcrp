from types import SimpleNamespace

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
from checks.consistency_checks.check_institution_source_consistency import (
    check_id_attribute_consistency,
)
from checks.coordinate_checks.grid import check_grid_cell_count
from checks.coordinate_checks.model import Catalog
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
    ("disabled_prefix", "expected_check"),
    (
        ((), "ATTR001"),
        (("attribute",), "TIME001"),
        (("attribute", "squareness"), "TIME003"),
        (("attribute", "squareness", "time_range"), None),
    ),
)
def test_missing_frequency_hierarchy_walks_enabled_toml_rules(
    tmp_path,
    disabled_prefix,
    expected_check,
):
    enabled = {"attribute", "squareness", "time_range"} - set(disabled_prefix)
    config = {
        "project_name": "cmip7",
        "project_version": "1.0",
        "global": {
            "attributes": (
                {"frequency": {"severity": "H"}}
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
                    BaseCheck.HIGH,
                    is_required=True,
                )
            )
        if "squareness" in enabled:
            results.extend(
                check_time_squareness(
                    ds,
                    expected_cell_methods="time: mean",
                    report_frequency_prerequisite=not checker._global_attribute_owns(
                        "frequency", BaseCheck.HIGH
                    ),
                )
            )
        if "time_range" in enabled:
            squareness_owns = "squareness" in enabled
            results.extend(
                check_time_range_vs_filename(
                    ds,
                    report_frequency_prerequisite=not (
                        checker._global_attribute_owns(
                            "frequency", BaseCheck.HIGH
                        )
                        or squareness_owns
                    ),
                )
            )

    failures = [result for result in results if result.msgs]
    assert len(failures) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert failures[0].name.startswith(f"[{expected_check}]")


@pytest.mark.parametrize(
    ("disabled_prefix", "expected_check"),
    (
        ((), "COORD003"),
        (("coordinate_attributes",), "TIME001"),
        (("coordinate_attributes", "squareness"), "TIME003"),
        (("coordinate_attributes", "squareness", "time_range"), None),
    ),
)
def test_missing_time_units_hierarchy_walks_enabled_toml_rules(
    tmp_path,
    disabled_prefix,
    expected_check,
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
                    severities={"attributes": BaseCheck.HIGH},
                )
            )
        if "squareness" in enabled:
            results.extend(
                check_time_squareness(
                    ds,
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


@pytest.mark.parametrize(
    ("coordinate_bounds_enabled", "time002_enabled", "expected_check"),
    (
        (True, True, "COORD008"),
        (False, True, "TIME002"),
        (False, False, None),
    ),
)
def test_missing_regular_time_bounds_walks_enabled_toml_rules(
    tmp_path,
    coordinate_bounds_enabled,
    time002_enabled,
    expected_check,
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
                    severities={"bounds": BaseCheck.HIGH},
                )
            )
        if time002_enabled:
            results.extend(
                check_time_bounds(
                    ds,
                    BaseCheck.HIGH,
                    report_bounds_prerequisites=not coordinate_bounds_enabled,
                )
            )

    failures = [result for result in results if result.msgs]
    assert len(failures) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert failures[0].name.startswith(f"[{expected_check}]")


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


def test_attribute_prerequisite_owner_must_not_lower_severity():
    checker = Cmip6ProjectCheck()
    checker._load_split_config()

    assert not checker._global_attribute_owns("source", BaseCheck.HIGH)
    assert not checker._global_attribute_owns("institution", BaseCheck.HIGH)
    assert checker._global_attribute_owns("source_id", BaseCheck.HIGH)
    assert checker._global_attribute_owns("source", BaseCheck.LOW)


def test_variant_prerequisite_owner_is_selected_per_attribute_severity(tmp_path):
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

    assert _messages(results) == [
        "Missing required global attribute: 'realization_index'."
    ]


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


def test_missing_data_variable_fallback_uses_highest_dependent_severity(tmp_path):
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
    assert results[0].weight == BaseCheck.HIGH
    assert "dependent variable checks" in _messages(results)[0]


def test_cordex_compression_delegates_only_to_equal_severity_variable_owner(
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

    assert checker.check_File_Compression(SimpleNamespace()) == detection

    checker.config.variable.type.severity = "H"
    assert checker.check_File_Compression(SimpleNamespace()) == []


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


def test_missing_configured_time_uses_highest_severity_consumer():
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
    assert "[TIME001]" in results[0].name
    assert results[0].weight == BaseCheck.HIGH
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


def test_coordinate_prerequisite_uses_highest_severity_fallback():
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

    assert owner == "attributes"
    results = findings.results()
    failure = next(result for result in results if result.msgs)
    assert failure.weight == BaseCheck.HIGH
    assert failure.msgs == ["reported by attributes"]


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
def test_coordinate_prerequisite_hierarchies_walk_enabled_fallbacks(
    owner,
    enabled_families,
    expected_owner,
):
    """Each disabled owner hands one root cause to the next configured check."""
    findings = Findings(
        {family: BaseCheck.HIGH for family in enabled_families}
    )

    selected = findings.add_prerequisite(
        owner,
        lambda family: f"reported by {family}",
        issue=(owner, "unusable prerequisite"),
    )

    assert selected == expected_owner
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
def test_missing_data_variable_hierarchy_walks_enabled_consumers(
    enabled_families,
    expected_owner,
):
    findings = Findings(
        {family: BaseCheck.HIGH for family in enabled_families}
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


def test_blocked_prerequisite_uses_severity_then_argument_order():
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

    assert owner == "associations"


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
def test_variant_selector_hierarchy_walks_enabled_toml_checks(
    tmp_path,
    enabled_checks,
    expected_check,
):
    with Dataset(tmp_path / f"{expected_check}.nc", "w") as dataset:
        dataset.variant_label = "invalid"
        dataset.realization_index = 1
        dataset.initialization_index = 1
        dataset.physics_index = 1
        dataset.forcing_index = 1
        results = variant_checks.check_variant_label_consistency_group(
            dataset,
            [(check, BaseCheck.HIGH) for check in enabled_checks],
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
def test_experiment_selector_hierarchy_walks_enabled_toml_checks(
    tmp_path,
    monkeypatch,
    enabled_checks,
    expected_check,
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
                (check, BaseCheck.HIGH, {"report_missing": True})
                for check in enabled_checks
            ],
        )

    failures = [result for result in results if result.msgs]
    assert len(failures) == (1 if expected_check is not None else 0)
    if expected_check is not None:
        assert failures[0].name.startswith(f"[{expected_check}]")


def test_variant_selector_owner_uses_severity_then_fixed_priority(tmp_path):
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
    assert results[0].name.startswith("[ATTR006d]")
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


def test_experiment_selector_owner_uses_severity_then_fixed_priority(
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
    assert results[0].name.startswith("[ATTR007b]")
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
def test_drs_directory_hierarchy_walks_enabled_toml_rules(
    tmp_path,
    monkeypatch,
    disabled_rules,
    expected_check,
):
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
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
        assert "MIP-DRS7" in results[0].msgs[0]


def test_highest_severity_path_consumer_owns_structure_when_path003_is_disabled(
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
    assert results[0].name.startswith("[PATH002]")
    assert results[0].weight == BaseCheck.HIGH
    assert "MIP-DRS7" in results[0].msgs[0]


def test_equal_severity_path_consumers_use_fixed_priority():
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    drs = checker.config.drs
    drs.directory = None
    drs.attributes_vs_directory.severity = "H"
    drs.filename_vs_directory.severity = "H"

    assert checker._drs_directory_structure_owner(drs) == "attributes_vs_directory"
