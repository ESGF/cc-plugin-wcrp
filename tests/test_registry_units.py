"""Unit-schema compatibility through the shared registry adapter."""

from types import SimpleNamespace

import pytest
from esgvoc.api.pydantic_handler import instantiate_pydantic_term
from netCDF4 import Dataset

from checks.variable_checks.known_branded_variable import (
    normalize_known_branded_variable,
)
from plugins.cmip7 import cmip7
from plugins.wcrp_schema import WCRPConfig


UNIT_SCHEMAS = [
    pytest.param({"units": "K"}, "K", id="new-universe"),
    pytest.param({"cf_units": "K"}, "K", id="old-universe"),
    pytest.param(
        {"units": "K", "cf_units": "degC"},
        "K",
        id="prefer-units",
    ),
    pytest.param(
        {"units": None, "cf_units": "K"},
        "K",
        id="null-fallback",
    ),
    pytest.param(
        {"units": "", "cf_units": "K"},
        "K",
        id="empty-fallback",
    ),
    pytest.param({}, None, id="missing-both"),
    pytest.param(
        {"units": None, "cf_units": None},
        None,
        id="null-both",
    ),
]


@pytest.mark.parametrize(("unit_fields", "expected_units"), UNIT_SCHEMAS)
def test_registry_unit_schema_normalization(unit_fields, expected_units):
    """Normalize new, legacy, and partially populated ESGVoc subsets."""
    term = SimpleNamespace(
        id="tas_tavg-h2m-hxy-u",
        specs={"type": "known_branded_variable", **unit_fields},
    )
    subset = instantiate_pydantic_term(term, ["units", "cf_units"])

    expected = normalize_known_branded_variable(subset)

    assert expected.units == expected_units
    assert expected.cf_units == expected_units


def test_legacy_cf_units_config_reaches_attr004_through_shared_adapter(
    tmp_path,
    monkeypatch,
):
    """Keep external configurations using the old registry key functional."""
    calls = []

    def find_terms(
        *,
        expression,
        data_descriptor_id,
        selected_term_fields=None,
        only_id=False,
    ):
        calls.append(data_descriptor_id)
        assert expression == "tas_tavg-h2m-hxy-u"
        assert data_descriptor_id == "known_branded_variable"
        assert selected_term_fields is None
        assert only_id is True
        return [
            SimpleNamespace(
                id=expression,
                units="K",
                long_name="Near-Surface Air Temperature",
            )
        ]

    monkeypatch.setattr(cmip7, "find_terms_in_data_descriptor", find_terms)
    checker = cmip7.Cmip7ProjectCheck.__new__(cmip7.Cmip7ProjectCheck)
    checker._expected_term_cache = None
    checker._geo_var_cache = "tas"
    checker.project_name = "cmip7"
    checker.config = WCRPConfig.model_validate(
        {
            "project_name": "cmip7",
            "project_version": "test",
            "variable": {
                "attributes": {
                    "units": {
                        "value_type": "str",
                        "cv_source_term_key": "cf_units",
                    }
                }
            },
        }
    )

    with Dataset(tmp_path / "tas.nc", "w") as ds:
        ds.variable_id = "tas"
        ds.branded_variable = "tas_tavg-h2m-hxy-u"
        variable = ds.createVariable("tas", "f4")
        variable.units = "K"

        matching = checker.check_Geophysical_Variable(ds)
        matching_attr004 = [r for r in matching if "[ATTR004]" in r.name]
        assert len(matching_attr004) == 1
        assert matching_attr004[0].value[0] == matching_attr004[0].value[1]
        assert checker._expected_term_cache.cf_units == "K"

        variable.units = "wrong-unit"
        mismatching = checker.check_Geophysical_Variable(ds)
        mismatching_attr004 = [r for r in mismatching if "[ATTR004]" in r.name]
        assert len(mismatching_attr004) == 1
        assert mismatching_attr004[0].value[0] < mismatching_attr004[0].value[1]

    assert calls == ["known_branded_variable"]
