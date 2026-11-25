"""Identity metadata for esgf-qa; no live vocabulary database is needed."""

import json
from hashlib import md5

import pytest
from netCDF4 import Dataset

from checks.coordinate_checks.consistency import (
    coordinate_ids_for_summary,
    dimension_metadata_for_summary,
)
from checks.coordinate_checks.model import Catalog
from plugins import wcrp_base


@pytest.mark.parametrize("catalogue_state", ["absent", "unavailable", "empty"])
def test_cmip6_summary_without_coordinate_metadata(tmp_path, monkeypatch, catalogue_state):
    """Legacy projects still produce usable summaries without coordinate IDs."""
    from plugins.cmip6.cmip6 import Cmip6ProjectCheck

    # Keep the test independent of the installed ESGVoc databases.
    monkeypatch.setattr(wcrp_base, "ESG_VOCAB_AVAILABLE", False)
    output = tmp_path / "cmip6-consistency.json"
    checker = Cmip6ProjectCheck({"consistency_output": str(output)})
    if catalogue_state == "unavailable":
        checker._coordinate_catalog = None
    elif catalogue_state == "empty":
        checker._coordinate_catalog = Catalog("cmip6", "tas", {}, ())
    else:
        assert not hasattr(checker, "_coordinate_catalog")

    with Dataset(tmp_path / "tas.nc", "w") as ds:
        ds.variable_id = "tas"
        ds.frequency = "fx"
        ds.createDimension("lat", 2)
        lat = ds.createVariable("lat", "f8", ("lat",))
        lat.standard_name = "latitude"
        lat.units = "degrees_north"
        lat[:] = [-10.0, 10.0]
        tas = ds.createVariable("tas", "f4", ("lat",))
        tas.standard_name = "air_temperature"
        tas.units = "K"
        tas[:] = [280.0, 281.0]
        expected_checksum = md5(str(lat[:].tobytes()).encode("utf-8")).hexdigest()

        checker.setup(ds)

        assert checker.consistency_output_error is None
        assert checker.check_consistency_output(ds)[0].value == (1, 1)

    payload = json.loads(output.read_text())
    assert payload["coordinate_ids"] == {}
    assert payload["coordinates"] == {"lat": expected_checksum}
    assert payload["dimensions"] == {"lat": 2}
    assert payload["variable_attributes"]["tas"]["units"] == "K"
    assert payload["variable_dtypes"]["tas"] == "float32"
    assert payload["time_info"] == {}


def catalog(**kwargs):
    return Catalog("cmip7", "tas", {}, **kwargs)


@pytest.mark.parametrize("text_id", ["soilpools", "basin"])
def test_coordinate_ids_use_requested_identity(tmp_path, text_id):
    cv = catalog(
        coordinate_ids=("height2m", text_id),
        data_coordinates={
            "height2m": {"coordinate_type": "scalar", "out_name": "height"},
            text_id: {"coordinate_type": "auxiliary", "out_name": "type"},
            "height10m": {"coordinate_type": "scalar", "out_name": "height"},
        },
        grid_variables={"latitude": {"out_name": "nav_lat"}},
        grid_axes={"x": {"out_name": "x"}},
    )
    with Dataset(tmp_path / "test.nc", "w") as ds:
        ds.createDimension("type", 2)
        ds.createDimension("strlen", 8)
        ds.createVariable("sector", "S1", ("type", "strlen"))
        assert coordinate_ids_for_summary(
            ds, cv, ["height", "sector", "nav_lat", "unresolved"]
        ) == {"height": "height2m", "sector": text_id, "nav_lat": "latitude"}
        assert coordinate_ids_for_summary(ds, None, ["height"]) == {}


def test_ambiguous_coordinate_identity_is_omitted(tmp_path):
    cv = catalog(
        coordinate_ids=("basin", "soilpools"),
        data_coordinates={key: {"coordinate_type": "auxiliary"}
                          for key in ("basin", "soilpools")},
    )
    with Dataset(tmp_path / "test.nc", "w") as ds:
        ds.createDimension("type", 2)
        ds.createDimension("strlen", 8)
        ds.createVariable("sector", "S1", ("type", "strlen"))
        assert coordinate_ids_for_summary(ds, cv, ["sector"]) == {}


def test_string_length_dimension_also_used_as_data_axis_is_not_ignored(tmp_path):
    with Dataset(tmp_path / "shared.nc", "w") as ds:
        ds.createDimension("type", 2)
        ds.createDimension("width", 8)
        ds.createVariable("sector", "S1", ("type", "width"))
        ds.createVariable("data", "f4", ("width",))
        assert dimension_metadata_for_summary(ds, {}) == ({}, [])


@pytest.mark.parametrize("standard_name,expected", [("depth", {"lev": "ocean_sigma"}), ("unknown", {})])
def test_model_level_identity(tmp_path, standard_name, expected):
    cv = catalog(
        coordinate_ids=("olevel",),
        data_coordinates={"olevel": {"coordinate_type": "generic_vertical", "out_name": "lev"}},
        model_levels={"ocean_sigma": {"generic_level_name": "olevel", "cf_standard_name": "depth"}},
    )
    with Dataset(tmp_path / "test.nc", "w") as ds:
        ds.createVariable("lev", "f8").standard_name = standard_name
        assert coordinate_ids_for_summary(ds, cv, ["lev"]) == expected


@pytest.mark.parametrize("variable,dimension", [("sector", "type"), ("labels", "type"), ("sector", "categories")])
def test_text_dimension_metadata_uses_actual_dimensions(tmp_path, variable, dimension):
    cv = catalog(
        coordinate_ids=("soilpools",),
        data_coordinates={"soilpools": {"coordinate_type": "auxiliary", "out_name": "type"}},
    )
    with Dataset(tmp_path / "text.nc", "w") as ds:
        ds.createDimension(dimension, 3)
        ds.createDimension("label_width", 32)
        ds.createVariable(variable, "S1", (dimension, "label_width"))
        identities = coordinate_ids_for_summary(ds, cv, [variable])
        assert identities == {variable: "soilpools"}
        assert dimension_metadata_for_summary(ds, identities) == (
            {dimension: "soilpools"}, ["label_width"]
        )


@pytest.mark.parametrize("dimensions,dtype", [((), "S1"), (("a", "b", "c"), "S1"), (("a", "b"), "f8")])
def test_malformed_text_coordinate_does_not_break_summary(tmp_path, monkeypatch, dimensions, dtype):
    from plugins.cmip6.cmip6 import Cmip6ProjectCheck

    monkeypatch.setattr(wcrp_base, "ESG_VOCAB_AVAILABLE", False)
    output = tmp_path / "summary.json"
    with Dataset(tmp_path / "malformed.nc", "w") as ds:
        for dimension in dimensions:
            ds.createDimension(dimension, 2)
        ds.createVariable("sector", dtype, dimensions)
        ds.createVariable("tas", "f4")
        ds.variable_id = "tas"
        ds.frequency = "fx"
        checker = Cmip6ProjectCheck({"consistency_output": str(output)})
        checker._coordinate_catalog = catalog(
            coordinate_ids=("soilpools",),
            data_coordinates={"soilpools": {"coordinate_type": "auxiliary", "out_name": "type"}},
        )
        checker.setup(ds)
        assert checker.consistency_output_error is None
    payload = json.loads(output.read_text())
    assert payload["dimension_ids"] == {}
    assert "sector" in payload["variable_attributes"]


@pytest.mark.parametrize("project", ["cmip7", "cordex_cmip6"])
def test_summary_written_after_catalogue_setup(tmp_path, monkeypatch, project):
    from plugins.cmip7 import cmip7
    from plugins.cordex_cmip6 import cordex_cmip6

    module, checker_class = (
        (cmip7, cmip7.Cmip7ProjectCheck) if project == "cmip7"
        else (cordex_cmip6, cordex_cmip6.CordexCmip6ProjectCheck)
    )

    cv = catalog(
        coordinate_ids=("height2m",),
        data_coordinates={"height2m": {"coordinate_type": "scalar", "out_name": "height"}},
    )
    monkeypatch.setattr(wcrp_base, "ESG_VOCAB_AVAILABLE", False)
    monkeypatch.setattr(module, "load_catalog", lambda *a, **kw: cv)
    monkeypatch.setattr(cmip7, "load_grid_metadata", lambda *a: {})
    monkeypatch.setattr(
        cordex_cmip6.CordexCmip6ProjectCheck, "_mapped_branded_variable",
        lambda *a: "tas",
    )
    output = tmp_path / "summary.json"
    with Dataset(tmp_path / "test.nc", "w") as ds:
        ds.branded_variable = "tas"
        ds.variable_id = "tas"
        ds.createVariable("height", "f8").assignValue(2)
        ds.createVariable("tas", "f4").assignValue(280)
        checker = checker_class({"consistency_output": str(output)})
        checker.setup(ds)
        assert checker.consistency_output_error is None
    payload = json.loads(output.read_text())
    assert payload["coordinate_ids"] == {"height": "height2m"}
    assert "height" in payload["coordinates"]
