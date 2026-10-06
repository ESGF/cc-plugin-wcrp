"""Ensure configured global consistency checks can be executed and are executed properly."""

import pytest
from compliance_checker.base import BaseCheck
from compliance_checker.base import TestCtx as CheckContext
from netCDF4 import Dataset

from checks.consistency_checks.check_attributes_match_filename import (
    check_filename_vs_global_attrs,
)
from plugins.c3scmip6.c3scmip6 import c3scmip6 as c3scmip6_module
from plugins.c3scmip6.c3scmip6.c3scmip6 import C3SCmip6ProjectCheck
from plugins.cmip6 import cmip6 as cmip6_module
from plugins.cmip6.cmip6 import Cmip6ProjectCheck
from plugins.cmip6plus import cmip6plus as cmip6plus_module
from plugins.cmip6plus.cmip6plus import Cmip6PlusProjectCheck
from plugins.cmip7 import cmip7 as cmip7_module
from plugins.cmip7.cmip7 import Cmip7ProjectCheck

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


@pytest.mark.parametrize(
    ("module", "checker_class", "extra_functions"),
    [
        (cmip7_module, Cmip7ProjectCheck, ()),
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
        monkeypatch.setattr(module, function_name, replacement(function_name))

    with Dataset(tmp_path / f"{checker.project_name}.nc", "w") as dataset:
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
    filename = (
        "ta_tavg-al-hxy-u_mon_glb_g122_Model_piControl_"
        "r1i1p1f1_185101-185113.nc"
    )
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
