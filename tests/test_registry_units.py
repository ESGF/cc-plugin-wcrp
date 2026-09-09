"""Unit schema compatibility through ESGVOC subsets and plugin ATTR004 checks."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from esgvoc.api.pydantic_handler import instantiate_pydantic_term
from netCDF4 import Dataset

from plugins.wcrp_schema import WCRPConfig


PLUGINS = [
    ("plugins.cmip6.cmip6", "Cmip6ProjectCheck"),
    ("plugins.cmip6plus.cmip6plus", "Cmip6PlusProjectCheck"),
    ("plugins.c3scmip6.c3scmip6.c3scmip6", "C3SCmip6ProjectCheck"),
    ("plugins.cmip7.cmip7", "Cmip7ProjectCheck"),
]
SCHEMAS = [
    pytest.param({"units": "K"}, "K", id="new-universe"),
    pytest.param({"cf_units": "K"}, "K", id="old-universe"),
    pytest.param({"units": "K", "cf_units": "degC"}, "K", id="prefer-units"),
    pytest.param({"units": None, "cf_units": "K"}, "K", id="null-fallback"),
    pytest.param({"units": "", "cf_units": "K"}, "", id="empty-is-not-null"),
    pytest.param({}, None, id="missing-both"),
    pytest.param({"units": None, "cf_units": None}, None, id="null-both"),
]


@pytest.mark.parametrize("module_name,class_name", PLUGINS)
@pytest.mark.parametrize("unit_fields,expected_units", SCHEMAS)
def test_registry_units(tmp_path, monkeypatch, module_name, class_name,
                        unit_fields, expected_units):
    module = import_module(module_name)
    calls = []

    def find_terms(*, expression, data_descriptor_id, selected_term_fields,
                   only_id=False):
        calls.append(data_descriptor_id)
        if data_descriptor_id == "known_branded_variable":
            assert expression == "tas_tavg-h2m-hxy-u"
            assert only_id is True
            specs = {"type": data_descriptor_id, **unit_fields}
        else:
            assert data_descriptor_id == "variable"
            assert expression == "tas"
            specs = {"type": "variable", "long_name": "Near-Surface Air Temperature"}
        # Use ESGVOC's real subset constructor, including absent-field handling.
        term = SimpleNamespace(id=expression, specs=specs)
        return [instantiate_pydantic_term(term, selected_term_fields)]

    monkeypatch.setattr(module, "find_terms_in_data_descriptor", find_terms)
    # Bypass setup's registry/config loading; run the actual plugin check below.
    checker = getattr(module, class_name).__new__(getattr(module, class_name))
    checker._expected_term_cache = None
    checker._geo_var_cache = "tas"
    checker.project_name = module_name.split(".")[1]
    checker.variable_id_to_branded_variable = {"Amon.tas": "tas_tavg-h2m-hxy-u"}
    checker.config = WCRPConfig.model_validate({
        "project_name": checker.project_name, "project_version": "test",
        "variable": {"attributes": {"units": {
            "value_type": "str", "cv_source_term_key": "cf_units",
        }}},
    })

    with Dataset(tmp_path / "tas.nc", "w") as ds:
        ds.variable_id = "tas"
        ds.table_id = "Amon"
        ds.branded_variable = "tas_tavg-h2m-hxy-u"
        var = ds.createVariable("tas", "f4")
        var.units = expected_units if expected_units is not None else "K"
        results = checker.check_Geophysical_Variable(ds)
        registry_results = [r for r in results if "[ATTR004]" in r.name]
        assert len(registry_results) == 1
        result = registry_results[0]
        assert checker._expected_term_cache.cf_units == expected_units
        if expected_units is None or expected_units == "":
            assert result.value[0] < result.value[1]
            assert "Registry has no value for key 'cf_units'." in result.msgs
        else:
            assert all(r.value[0] == r.value[1] for r in results), results
            # A mismatching file still fails; registry resolution stays cached.
            var.units = "wrong-unit"
            mismatch = checker.check_Geophysical_Variable(ds)
            assert any(r.value[0] < r.value[1] for r in mismatch if "[ATTR004]" in r.name)
        assert calls == ["known_branded_variable", "variable"]
