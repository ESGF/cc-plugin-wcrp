from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml
from compliance_checker.base import BaseCheck

from checks.variable_checks.known_branded_variable import (
    KnownBrandedVariableLookupError,
    lookup_expected_variable_metadata,
    normalize_known_branded_variable,
)
from plugins.c3scmip6.c3scmip6.c3scmip6 import C3SCmip6ProjectCheck
from plugins.cmip6.cmip6 import Cmip6ProjectCheck
from plugins.cmip6plus.cmip6plus import Cmip6PlusProjectCheck
from plugins.cmip7.cmip7 import Cmip7ProjectCheck


NEW_RECORD = {
    "id": "tas_ti-u-hxy-air",
    "variable_root_name": "tas",
    "cf_standard_name": "air_temperature",
    "units": "K",
    "long_name": "Near-Surface Air Temperature",
    "out_name": "tas",
    "dimensions": ["time", "latitude", "longitude"],
    "cell_methods": "area: time: mean",
    "cell_measures": "area: areacella",
    "flag_values": None,
    "flag_meanings": None,
}


def test_new_known_branded_variable_uses_canonical_fields_directly():
    calls = []

    def find_terms(**kwargs):
        calls.append(kwargs)
        assert kwargs["data_descriptor_id"] == "known_branded_variable"
        return [NEW_RECORD]

    lookup = lookup_expected_variable_metadata(
        find_terms,
        NEW_RECORD["id"],
        fallback_variable_id="wrong-fallback",
    )

    assert lookup.warning is None
    assert lookup.expected.units == "K"
    assert lookup.expected.cf_units == "K"
    assert lookup.expected.long_name == "Near-Surface Air Temperature"
    assert lookup.expected.out_name == "tas"
    assert len(calls) == 1


def test_legacy_known_branded_variable_maps_units_and_looks_up_long_name():
    legacy = {
        "id": NEW_RECORD["id"],
        "variable_root_name": "tas",
        "cf_standard_name": "air_temperature",
        "cf_units": "K",
        "dimensions": ["time", "latitude", "longitude"],
        "cell_methods": "area: time: mean",
        "cell_measures": "area: areacella",
    }
    calls = []

    def find_terms(**kwargs):
        calls.append(kwargs)
        if kwargs["data_descriptor_id"] == "known_branded_variable":
            return [legacy]
        return [{"id": "tas", "long_name": "Near-Surface Air Temperature"}]

    lookup = lookup_expected_variable_metadata(find_terms, legacy["id"])

    assert lookup.warning is None
    assert lookup.expected.units == "K"
    assert lookup.expected.long_name == "Near-Surface Air Temperature"
    assert [call["data_descriptor_id"] for call in calls] == [
        "known_branded_variable",
        "variable",
    ]


def test_legacy_long_name_lookup_failure_preserves_other_metadata():
    legacy = {
        "id": NEW_RECORD["id"],
        "variable_root_name": "tas",
        "cf_standard_name": "air_temperature",
        "cf_units": "K",
    }

    def find_terms(**kwargs):
        if kwargs["data_descriptor_id"] == "known_branded_variable":
            return [legacy]
        raise RuntimeError("database is locked")

    lookup = lookup_expected_variable_metadata(find_terms, legacy["id"])

    assert lookup.expected.units == "K"
    assert lookup.expected.long_name is None
    assert "Only 'long_name' is unavailable" in lookup.warning
    assert "database is locked" in lookup.warning


def test_lookup_requires_an_exact_known_branded_variable_id():
    def find_terms(**kwargs):
        return [{**NEW_RECORD, "id": f"{NEW_RECORD['id']}-other"}]

    with pytest.raises(KnownBrandedVariableLookupError, match="was not found"):
        lookup_expected_variable_metadata(find_terms, NEW_RECORD["id"])


def test_lookup_normalizes_drs_case_but_preserves_variable_metadata():
    calls = []
    branded = {
        **NEW_RECORD,
        "id": "baresoilfrac_tavg-u-hxy-u",
        "variable_root_name": "baresoilFrac",
        "long_name": None,
        "out_name": "baresoilFrac",
    }

    def find_terms(**kwargs):
        calls.append((kwargs["data_descriptor_id"], kwargs["expression"]))
        if kwargs["data_descriptor_id"] == "known_branded_variable":
            return [branded]
        return [{"id": "baresoilfrac", "long_name": "Bare Soil Percentage"}]

    lookup = lookup_expected_variable_metadata(
        find_terms,
        "baresoilFrac_tavg-u-hxy-u",
    )

    assert calls == [
        ("known_branded_variable", "baresoilfrac_tavg-u-hxy-u"),
        ("variable", "baresoilfrac"),
    ]
    assert lookup.warning is None
    assert lookup.expected.out_name == "baresoilFrac"
    assert lookup.expected.long_name == "Bare Soil Percentage"


@pytest.mark.parametrize(
    ("module_name", "checker_class", "uses_file_branded_variable"),
    [
        ("plugins.cmip6.cmip6", Cmip6ProjectCheck, False),
        ("plugins.cmip6plus.cmip6plus", Cmip6PlusProjectCheck, False),
        (
            "plugins.c3scmip6.c3scmip6.c3scmip6",
            C3SCmip6ProjectCheck,
            False,
        ),
        ("plugins.cmip7.cmip7", Cmip7ProjectCheck, True),
    ],
)
def test_all_registry_backed_plugins_use_new_known_branded_variable_model(
    monkeypatch,
    module_name,
    checker_class,
    uses_file_branded_variable,
):
    calls = []

    def find_terms(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(**NEW_RECORD)]

    monkeypatch.setattr(
        importlib.import_module(module_name),
        "find_terms_in_data_descriptor",
        find_terms,
    )
    checker = checker_class()
    attributes = {"variable_id": "tas", "table_id": "Amon"}
    if uses_file_branded_variable:
        attributes["branded_variable"] = NEW_RECORD["id"]
    else:
        checker.variable_id_to_branded_variable = {"Amon.tas": NEW_RECORD["id"]}
    dataset = SimpleNamespace(getncattr=lambda name: attributes[name])

    expected, results = checker._get_expected_from_registry(dataset, BaseCheck.HIGH)

    assert results == []
    assert expected.units == "K"
    assert expected.long_name == "Near-Surface Air Temperature"
    assert len(calls) == 1


def test_builtin_configs_use_canonical_known_branded_variable_units_key():
    root = Path(__file__).parents[1]
    paths = [
        root / "plugins/cmip6/config/wcrp/geophysical_variable.toml",
        root / "plugins/cmip6plus/config/wcrp/geophysical_variable.toml",
        root / "plugins/c3scmip6/c3scmip6/config/wcrp/geophysical_variable.toml",
        root / "plugins/cmip7/config/wcrp/geophysical_variable.toml",
    ]

    for path in paths:
        config = toml.load(path)
        units = config["variable"]["attributes"]["units"]
        assert units["cv_source_term_key"] == "units"


def test_normalizer_prefers_new_units_over_legacy_alias():
    expected = normalize_known_branded_variable(
        {**NEW_RECORD, "cf_units": "legacy-units"}
    )
    assert expected.units == "K"


def test_normalizer_reduces_resolved_references_to_stable_ids():
    expected = normalize_known_branded_variable(
        {
            **NEW_RECORD,
            "variable_root_name": SimpleNamespace(id="tas"),
            "dimensions": [
                SimpleNamespace(id="longitude"),
                SimpleNamespace(id="latitude"),
            ],
        }
    )
    assert expected.variable_root_name == "tas"
    assert expected.dimensions == ["longitude", "latitude"]
