"""Ensure configured global consistency checks can be executed and are executed properly."""

import pytest
from compliance_checker.base import BaseCheck
from compliance_checker.base import TestCtx as CheckContext
from netCDF4 import Dataset

from checks.consistency_checks.check_attributes_match_filename import (
    check_filename_vs_global_attrs,
)
from checks.consistency_checks.check_frequency_branded_variable_consistency import (
    check_frequency_branded_variable_consistency,
)
from checks.consistency_checks.check_frequency_table_consistency import (
    check_frequency_table_id_consistency,
)
from checks.coordinate_checks.model import Catalog
from plugins.c3scmip6.c3scmip6 import c3scmip6 as c3scmip6_module
from plugins.c3scmip6.c3scmip6.c3scmip6 import C3SCmip6ProjectCheck
from plugins.cmip6 import cmip6 as cmip6_module
from plugins.cmip6.cmip6 import Cmip6ProjectCheck
from plugins.cmip6plus import cmip6plus as cmip6plus_module
from plugins.cmip6plus.cmip6plus import Cmip6PlusProjectCheck
from plugins.cmip7 import cmip7 as cmip7_module
from plugins.cmip7.cmip7 import Cmip7ProjectCheck
from plugins.wcrp_schema import WCRPConfig

_COMMON_FUNCTIONS = (
    "check_filename_vs_global_attrs",
    "check_experiment_id_vs_activity_id",
    "check_experiment_id_vs_experiment",
    "check_experiment_id_vs_parent_experiment_id",
    "check_institution_consistency",
    "check_variant_vs_realization_index",
    "check_variant_vs_initialization_index",
    "check_variant_vs_physics_index",
    "check_variant_vs_forcing_index",
)

_CMIP6_FUNCTIONS = (
    "check_experiment_id_vs_sub_experiment_id",
    "check_source_consistency",
    "check_frequency_table_id_consistency",
)

_CMIP7_FUNCTIONS = ("check_frequency_branded_variable_consistency",)

_VARIANT_FUNCTIONS = (
    "check_variant_vs_realization_index",
    "check_variant_vs_initialization_index",
    "check_variant_vs_physics_index",
    "check_variant_vs_forcing_index",
)


@pytest.mark.parametrize(
    ("module", "checker_class", "extra_functions"),
    [
        (cmip7_module, Cmip7ProjectCheck, _CMIP7_FUNCTIONS),
        (cmip6_module, Cmip6ProjectCheck, _CMIP6_FUNCTIONS),
        (cmip6plus_module, Cmip6PlusProjectCheck, _CMIP6_FUNCTIONS),
        (c3scmip6_module, C3SCmip6ProjectCheck, _CMIP6_FUNCTIONS),
    ],
)
def test_project_consistency_configuration_reaches_every_check(
    tmp_path,
    monkeypatch,
    module,
    checker_class,
    extra_functions,
):
    checker = checker_class()
    checker._load_split_config()
    checker.table_id_to_frequency = {}
    called = []
    function_names = _COMMON_FUNCTIONS + extra_functions

    def replacement(name):
        def check(*_args, **_kwargs):
            called.append(name)
            context = CheckContext(BaseCheck.HIGH, f"[TEST] {name}")
            context.add_pass()
            return [context.to_result()]

        return check

    for function_name in function_names:
        if function_name in _VARIANT_FUNCTIONS:
            continue
        monkeypatch.setattr(module, function_name, replacement(function_name))

    def variant_group(_dataset, configured_checks, **_kwargs):
        results = []
        for check, _severity in configured_checks:
            name = check.__name__
            called.append(name)
            context = CheckContext(BaseCheck.HIGH, f"[TEST] {name}")
            context.add_pass()
            results.append(context.to_result())
        return results

    # The dispatcher deliberately identifies these four checks by function
    # object. Record its configured inputs without replacing those objects.
    monkeypatch.setattr(
        module,
        "check_variant_label_consistency_group",
        variant_group,
    )

    with Dataset(tmp_path / f"{checker.project_name}.nc", "w") as dataset:
        # The group dispatcher deliberately skips dependent comparisons when
        # their shared selector is unusable. Supply a valid selector here so
        # this test remains about configuration reachability.
        dataset.variant_label = "r1i1p1f1"
        dataset.realization_index = 1
        dataset.initialization_index = 1
        dataset.physics_index = 1
        dataset.forcing_index = 1
        results = checker.check_Global_Consistency(dataset)

    assert len(results) == len(function_names)
    assert set(called) == set(function_names)


def test_cmip6_consistency_entry_point_reports_inconsistent_metadata(
    tmp_path,
    monkeypatch,
):
    """Exercise configured checks on a small, deliberately inconsistent file."""
    checker = Cmip6ProjectCheck()
    checker._load_split_config()
    checker._load_mappings()

    # Keep the test independent of an installed ESGVoc database. The checks
    # exercised below use only the filename, file metadata, and TOML mappings.
    for function_name in (
        "check_experiment_id_vs_activity_id",
        "check_experiment_id_vs_experiment",
        "check_experiment_id_vs_parent_experiment_id",
        "check_experiment_id_vs_sub_experiment_id",
        "check_institution_consistency",
        "check_source_consistency",
    ):
        monkeypatch.setattr(cmip6_module, function_name, lambda *_args, **_kwargs: [])

    filename = "tas_Amon_Model_historical_r1i1p1f1_gn_200001-200012.nc"
    with Dataset(tmp_path / filename, "w") as dataset:
        dataset.variable_id = "tas"
        dataset.table_id = "Omon"
        dataset.source_id = "Model"
        dataset.experiment_id = "historical"
        dataset.variant_label = "r2i2p2f2"
        dataset.grid_label = "gn"
        dataset.frequency = "day"
        dataset.realization_index = 1
        dataset.initialization_index = 1
        dataset.physics_index = 1
        dataset.forcing_index = 1

        results = checker.check_Global_Consistency(dataset)

    result_by_id = {result.name.split("]", 1)[0] + "]": result for result in results}
    expected_failures = {
        "[ATTR005]",
        "[ATTR006a]",
        "[ATTR006b]",
        "[ATTR006c]",
        "[ATTR006d]",
        "[ATTR008]",
    }

    assert set(result_by_id) == expected_failures
    assert all(result.value[0] < result.value[1] for result in result_by_id.values())


def test_cmip6_filename_consistency_reports_unexpected_token_count(tmp_path):
    filename = "ta_tavg-al-hxy-u_mon_glb_g122_Model_piControl_r1i1p1f1_185101-185113.nc"
    with Dataset(tmp_path / filename, "w") as dataset:
        results = check_filename_vs_global_attrs(
            dataset,
            severity=BaseCheck.HIGH,
            project_id="cmip6",
        )

    assert len(results) == 1
    assert results[0].name.startswith("[ATTR005]")
    assert results[0].value[0] < results[0].value[1]
    assert "does not have the expected 7 components" in results[0].msgs[0]


def test_cmip7_additional_branded_variable_frequency_is_low(tmp_path):
    with Dataset(tmp_path / "additional-frequency.nc", "w") as dataset:
        dataset.frequency = "day"
        dataset.branded_variable = "tas_tavg-h2m-hxy-air"
        results = check_frequency_branded_variable_consistency(
            dataset,
            ["mon"],
            BaseCheck.LOW,
            project_label="CMIP7",
        )

    assert len(results) == 1
    assert results[0].name.startswith("[ATTR008]")
    assert results[0].weight == BaseCheck.LOW
    assert results[0].value[0] < results[0].value[1]
    assert "Registered frequencies are ['mon']" in results[0].msgs[0]
    assert "Submission at additional frequencies is allowed" in results[0].msgs[0]


def test_frequency_association_check_uses_the_calling_project_label(tmp_path):
    with Dataset(tmp_path / "cordex-cmip7-frequency.nc", "w") as dataset:
        dataset.frequency = "day"
        dataset.branded_variable = "tas-example"
        results = check_frequency_branded_variable_consistency(
            dataset,
            ["mon"],
            BaseCheck.LOW,
            project_label="CORDEX-CMIP7",
        )

    assert "in the CORDEX-CMIP7 Data Request" in results[0].msgs[0]


def test_additional_frequency_permission_is_only_stated_at_low_severity(tmp_path):
    with Dataset(tmp_path / "medium-frequency-policy.nc", "w") as dataset:
        dataset.frequency = "day"
        dataset.branded_variable = "tas-example"
        results = check_frequency_branded_variable_consistency(
            dataset,
            ["mon"],
            BaseCheck.MEDIUM,
            project_label="Example",
        )

    assert "Submission at additional frequencies" not in results[0].msgs[0]


def test_cmip7_registered_branded_variable_frequency_passes(tmp_path):
    with Dataset(tmp_path / "registered-frequency.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.branded_variable = "tas_tavg-h2m-hxy-air"
        results = check_frequency_branded_variable_consistency(
            dataset,
            ["mon", "day"],
            BaseCheck.LOW,
            project_label="CMIP7",
        )

    assert len(results) == 1
    assert results[0].value[0] == results[0].value[1]
    assert results[0].msgs == []


def test_frequency_table_mismatch_uses_neutral_wording(tmp_path):
    with Dataset(tmp_path / "frequency-table.nc", "w") as dataset:
        dataset.table_id = "Amon"
        dataset.frequency = "day"
        result = check_frequency_table_id_consistency(
            dataset,
            {"Amon": ["mon"]},
            BaseCheck.HIGH,
        )[0]

    assert result.msgs == [
        "For table_id 'Amon', the allowed frequencies are ['mon'], but the "
        "file defines 'day'."
    ]


def test_cmip7_absent_branded_variable_frequency_list_is_silent(tmp_path):
    with Dataset(tmp_path / "unavailable-frequency-list.nc", "w") as dataset:
        dataset.frequency = "mon"
        dataset.branded_variable = "tas_tavg-h2m-hxy-air"
        results = check_frequency_branded_variable_consistency(
            dataset,
            None,
            BaseCheck.LOW,
            project_label="CMIP7",
        )

    assert results == []


def test_cmip7_project_reports_valid_additional_frequency(tmp_path, monkeypatch):
    checker = Cmip7ProjectCheck()
    checker.config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "1.0",
            "global": {
                "attributes": {
                    "frequency": {"severity": "H"},
                    "branded_variable": {"severity": "H"},
                },
                "consistency": {
                    "frequency_vs_branded_variable": {"severity": "L"}
                },
            },
        }
    )
    checker._coordinate_catalog = Catalog(
        project_id="cmip7",
        branded_variable_id="tas_tavg-h2m-hxy-air",
        branded_variable={
            "id": "tas_tavg-h2m-hxy-air",
            "frequency": [{"id": "mon"}],
        },
        coordinate_ids=(),
    )

    class Api:
        @staticmethod
        def get_term_in_collection(*_args, **_kwargs):
            return {"id": "day"}

    monkeypatch.setattr(cmip7_module, "esgvoc_api", Api())
    with Dataset(tmp_path / "cmip7-additional-frequency.nc", "w") as dataset:
        dataset.frequency = "day"
        dataset.branded_variable = "tas_tavg-h2m-hxy-air"
        results = checker.check_Global_Consistency(dataset)

    failures = [result for result in results if result.msgs]
    assert len(failures) == 1
    assert failures[0].name.startswith("[ATTR008]")
    assert failures[0].weight == BaseCheck.LOW


@pytest.mark.parametrize(
    "filename",
    [
        "tas_Amon_Model_historical_r1i1p1f1.nc",
        "tas_Amon_Model_historical_r1i1p1f1_gn_200001-200012_extra.nc",
    ],
)
def test_filename_structure_is_left_to_file001_when_configured(tmp_path, filename):
    with Dataset(tmp_path / filename, "w") as dataset:
        results = check_filename_vs_global_attrs(
            dataset,
            severity=BaseCheck.HIGH,
            project_id="cmip6",
            filename_structure_delegated=True,
        )

    assert results == []


def test_filename_value_mismatch_remains_owned_by_attr005(tmp_path):
    filename = "tas_Amon_Model_historical_r1i1p1f1_gn_200001-200012.nc"
    with Dataset(tmp_path / filename, "w") as dataset:
        dataset.variable_id = "pr"
        results = check_filename_vs_global_attrs(
            dataset,
            severity=BaseCheck.HIGH,
            project_id="cmip6",
            filename_structure_delegated=True,
        )

    assert len(results) == 1
    assert results[0].name.startswith("[ATTR005]")
    assert any("variable_id" in message for message in results[0].msgs)
