"""Regression tests for state shared across project checker instances."""

from types import SimpleNamespace

import pytest
from compliance_checker.base import BaseCheck
from netCDF4 import Dataset

from checks.time_checks.check_time_squareness import FREQ_INC, _resolve_increment
from plugins.c3scmip6.c3scmip6.c3scmip6 import C3SCmip6ProjectCheck
from plugins.cmip6.cmip6 import Cmip6ProjectCheck
from plugins.cmip6plus.cmip6plus import Cmip6PlusProjectCheck
from plugins.cmip7.cmip7 import Cmip7ProjectCheck
from plugins.cordex_cmip6.cordex_cmip6 import CordexCmip6ProjectCheck


class _Attributes:
    def __init__(self, **attributes):
        self.attributes = attributes

    def getncattr(self, name):
        return self.attributes[name]


@pytest.mark.parametrize(
    "order",
    [
        ("cmip6", "cmip7", "cordex"),
        ("cordex", "cmip7", "cmip6"),
    ],
)
def test_project_time_increment_mappings_do_not_leak_between_checkers(order):
    original_defaults = dict(FREQ_INC)
    checkers = {
        "cmip6": Cmip6ProjectCheck(),
        "cmip7": Cmip7ProjectCheck(),
        "cordex": CordexCmip6ProjectCheck(),
    }
    checkers["cmip6"].table_id_to_time_increment = {"Amon.mon": [1, "months"]}
    checkers["cmip7"].table_id_to_time_increment = {"tavg.mon": [2, "months"]}
    checkers["cordex"].table_id_to_time_increment = {"None.mon": [3, "months"]}

    for project in order:
        checker = checkers[project]
        if project == "cmip7":
            checker._install_time_increment_mapping(
                _Attributes(table_id="tavg", frequency="mon")
            )
        else:
            checker._install_time_increment_mapping()

    assert _resolve_increment(
        "Amon", "mon", None, checkers["cmip6"]._time_increment_mapping
    ) == (1, "months")
    assert _resolve_increment(
        "tavg", "mon", None, checkers["cmip7"]._time_increment_mapping
    ) == (2, "months")
    assert _resolve_increment(
        "Other", "mon", None, checkers["cordex"]._time_increment_mapping
    ) == (3, "months")
    assert FREQ_INC == original_defaults


def test_geophysical_variable_cache_is_not_reused_for_another_dataset(monkeypatch):
    checker = Cmip7ProjectCheck()
    checker._geo_var_cache = "tas"
    first = SimpleNamespace(variables={"tas": object()})
    second = SimpleNamespace(variables={"pr": object()}, variable_id="pr")

    assert checker._get_geo_var(first, BaseCheck.HIGH) == ("tas", [])
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.get_geophysical_variables", lambda _ds: ["pr"]
    )
    selected, results = checker._get_geo_var(second, BaseCheck.HIGH)

    assert selected == "pr"
    assert results == []
    assert checker._geo_var_cache == "pr"


@pytest.mark.parametrize(
    ("candidates", "variable_id", "expected", "message"),
    [
        ([], None, None, "No geophysical variable detected"),
        (["tas", "areacella"], "tas", "tas", None),
        (
            ["tas", "areacella"],
            "pr",
            None,
            "variable_id='pr' is not among the detected candidates",
        ),
        (
            ["tas", "areacella"],
            None,
            None,
            "No variable_id global attribute available",
        ),
    ],
)
def test_geophysical_variable_selection_branches(
    monkeypatch, candidates, variable_id, expected, message
):
    checker = Cmip7ProjectCheck()
    variables = {name: object() for name in set(candidates) | {"pr", "tas"}}
    dataset = SimpleNamespace(variables=variables)
    if variable_id is not None:
        dataset.variable_id = variable_id
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.get_geophysical_variables",
        lambda _ds: candidates,
    )

    selected, results = checker._get_geo_var(dataset, BaseCheck.HIGH)

    assert selected == expected
    if message is None:
        assert results == []
    else:
        assert len(results) == 1
        assert message in results[0].msgs[0]


def test_flag_variable_falls_back_to_existing_variable_id(monkeypatch):
    checker = Cmip7ProjectCheck()
    dataset = SimpleNamespace(variables={"basin": object()}, variable_id="basin")
    monkeypatch.setattr("plugins.cmip7.cmip7.get_geophysical_variables", lambda _ds: [])

    selected, results = checker._get_geo_var(dataset, BaseCheck.HIGH)

    assert selected == "basin"
    assert results == []


def test_registry_lookup_failure_is_reported_once_and_dependents_skip(
    tmp_path, monkeypatch
):
    checker = Cmip6ProjectCheck()
    checker._load_split_config()
    checker.variable_id_to_branded_variable = {}
    checker._expected_term_cache = None
    checker._expected_term_lookup_attempted = False
    monkeypatch.setattr(
        "plugins.cmip6.cmip6.get_geophysical_variables", lambda _ds: ["tas"]
    )

    path = tmp_path / "tas_Amon_Model_historical_r1i1p1f1_gn_200001.nc"
    with Dataset(path, "w") as dataset:
        dataset.variable_id = "tas"
        dataset.table_id = "Amon"
        dataset.frequency = "mon"
        dataset.createDimension("time", 1)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time.calendar = "standard"
        time[:] = [15.0]
        dataset.createVariable("tas", "f4", ("time",))[:] = [280.0]

        results = [
            *checker.check_Geophysical_Variable(dataset),
            *checker.check_Coordinates(dataset),
        ]

    registry = [result for result in results if result.name == "Variable Registry"]
    assert len(registry) == 1
    assert "No branded variable mapping" in registry[0].msgs[0]
    assert not any(result.name.startswith("[TIME001]") for result in results)
    assert not any(result.name.startswith("[TIME003]") for result in results)
    assert any(result.name.startswith("[TIME003a]") for result in results)


def test_flag_main_variable_skips_only_float_specific_rules(
    tmp_path, monkeypatch
):
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    checked_attributes = []
    type_checks = []
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.get_geophysical_variables", lambda _ds: []
    )
    monkeypatch.setattr(
        checker,
        "_get_expected_from_registry",
        lambda *_args: (SimpleNamespace(), []),
    )
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.check_variable_type",
        lambda *_args, **_kwargs: type_checks.append(True) or [],
    )
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.check_attribute_suite",
        lambda *_args, **kwargs: checked_attributes.append(
            kwargs["attribute_name"]
        )
        or [],
    )

    with Dataset(tmp_path / "basin.nc", "w") as dataset:
        dataset.variable_id = "basin"
        dataset.createDimension("basin", 2)
        variable = dataset.createVariable("basin", "i4", ("basin",))
        variable.flag_values = [0, 1]
        variable.flag_meanings = "atlantic_ocean pacific_ocean"
        checker.check_Geophysical_Variable(dataset)

    assert type_checks == []
    assert "_FillValue" not in checked_attributes
    assert "missing_value" not in checked_attributes
    assert "long_name" in checked_attributes
    assert "standard_name" in checked_attributes


def test_missing_main_variable_stops_registry_backed_variable_checks(
    tmp_path, monkeypatch
):
    checker = Cmip7ProjectCheck()
    checker._load_split_config()
    monkeypatch.setattr(
        "plugins.cmip7.cmip7.get_geophysical_variables", lambda _ds: []
    )
    registry_calls = []
    monkeypatch.setattr(
        checker,
        "_get_expected_from_registry",
        lambda *_args: registry_calls.append(True) or (None, []),
    )

    with Dataset(tmp_path / "empty.nc", "w") as dataset:
        results = checker.check_Geophysical_Variable(dataset)

    assert len(results) == 1
    assert "No geophysical variable" in results[0].msgs[0]
    assert registry_calls == []


@pytest.mark.parametrize(
    "checker_class",
    [
        Cmip6ProjectCheck,
        Cmip6PlusProjectCheck,
        C3SCmip6ProjectCheck,
        Cmip7ProjectCheck,
        CordexCmip6ProjectCheck,
    ],
)
def test_wrong_project_metadata_produces_findings_but_never_exceptions(
    tmp_path, monkeypatch, checker_class
):
    # Deliberately use a CMIP7-like file name and too little metadata for all
    # project plugins. The contract is robustness, not acceptance.
    path = tmp_path / "ta_tavg-u_mon_glb_g999_Model_exp_r1i1p1f1_200001.nc"
    with Dataset(path, "w") as dataset:
        dataset.variable_id = "ta"
        dataset.frequency = "mon"
        dataset.createDimension("time", 1)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time[:] = [15.0]
        dataset.createVariable("ta", "f4", ("time",))[:] = [280.0]

    checker = checker_class()
    monkeypatch.setattr(checker, "_initialize_esgvoc_project_specs", lambda: None)
    with Dataset(path) as dataset:
        checker.setup(dataset)
        results = []
        for name in dir(checker):
            if name.startswith("check_") and callable(getattr(checker, name)):
                results.extend(getattr(checker, name)(dataset) or [])

    assert results
    assert any(result.msgs for result in results)
