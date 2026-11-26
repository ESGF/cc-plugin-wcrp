"""Model-level catalogue tests

The suite reads every model-level entry and formula-term descriptor from the
active ESGVoc **CMIP7 database**.

Each entry gets metadata/bounds, combined vertical-check, numerical-profile and
negative metadata tests. Non-parametric entries explicitly have no formula terms.
Expected profiles are hand-calculated.
New formula families need explicit numerical examples; they will not silently pass.
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from checks.coordinate_checks.formulas import formula_vertical_profile
from checks.coordinate_checks.esgvoc import (
    _get_api, _all, MODEL_LEVEL_FIELDS, FORMULA_TERM_FIELDS,
)
from checks.coordinate_checks.model import Catalog, records_by_id, reference_id, reference_ids
from checks.coordinate_checks.validation import Findings
from checks.coordinate_checks.vertical import validate_model_level


# Use the same project collections, fields and compatibility gate as the plugin.
API = _get_api()
LEVELS = records_by_id(_all(API, "cmip7", "model_level_coordinate", MODEL_LEVEL_FIELDS))
TERMS = records_by_id(_all(API, "cmip7", "formula_term", FORMULA_TERM_FIELDS))
for entry in LEVELS.values():
    entry["generic_level_name"] = reference_id(entry.get("generic_level_name"))
FAMILIES = dict.fromkeys(("identity", "attributes", "recommendations", "direction",
                          "valid_range", "bounds", "bounds_name", "formula"), 3)


def family(identifier):
    return identifier.removesuffix("_half")


def attributes(var, entry):
    for key in ("axis", "units", "positive", "long_name", "formula", "computed_standard_name"):
        if entry.get(key):
            var.setncattr(key, entry[key])
    if entry.get("cf_standard_name"):
        var.standard_name = entry["cf_standard_name"]


def build_file(ds, entry):
    """Three levels, two bounds per level, and a tiny horizontal/time sample."""
    generic = reference_id(entry["generic_level_name"])
    name = entry["out_name"]
    for dim, size in ((name, 3), ("bnds", 2), ("time", 1), ("latitude", 1), ("longitude", 1)):
        ds.createDimension(dim, size)
    values = np.array([0.9, 0.5, 0.1])
    kind = family(entry["id"])
    if kind.startswith("depth_coord"):
        values = np.array([10., 20., 30.])
    elif kind.startswith("modified_sleve"):
        values = np.array([3., 2., 1.])
    elif kind.startswith("ocean_sigma"):
        values = np.array([-0.1, -0.5, -0.9])
        if kind == "ocean_sigma_z":
            values = np.array([-2., -50., -70.])
    elif kind == "hybrid_height":
        values = np.array([10., 20., 30.])
    coordinate = ds.createVariable(name, "f8", (name,))
    coordinate[:] = values
    attributes(coordinate, entry)
    direction = 1 if values[-1] > values[0] else -1
    bounds = ds.createVariable(name + "_bnds", "f8", (name, "bnds"))
    bounds[:] = np.column_stack((values - direction * 0.04, values + direction * 0.04))
    if entry.get("bounds_required"):
        coordinate.bounds = bounds.name
    mappings = []
    for is_bounds, field in ((False, "z_factors"), (True, "z_bounds_factors")):
        mapping = {}
        if is_bounds and not entry.get("bounds_required"):
            mappings.append(mapping)
            continue
        for identifier in reference_ids(entry.get(field)):
            descriptor = TERMS[identifier]
            term_name = descriptor["out_name"]
            symbol = term_name.removesuffix("_bnds").removesuffix("_half")
            mapping[symbol] = term_name
            if term_name in ds.variables:
                continue
            dims = tuple(name if d == generic else d for d in reversed(reference_ids(descriptor.get("dimensions"))))
            if is_bounds and name in dims:
                dims += ("bnds",)
            # The descriptor uses its own generic level (including half levels).
            dims = tuple(name if d in {"alevel", "alevhalf", "olevel", "olevhalf"} else d for d in dims)
            for dim in dims:
                if dim not in ds.dimensions:
                    ds.createDimension(dim, 1)
            var = ds.createVariable(term_name, {"real": "f4", "double": "f8", "integer": "i4"}[descriptor["data_type"]], dims)
            attributes(var, descriptor)
            constants = {"p0": 100000., "ps": 100000., "ptop": 10000., "orog": 100., "eta": 0., "depth": 100., "depth_c": 20.}
            if name in dims:
                vector = values if symbol in {"b", "sigma"} else np.array([3., 2., 1.])
                if kind == "hybrid_height" and symbol == "b":
                    vector = np.array([0.1, 0.2, 0.3])
                if symbol == "ap":
                    vector = np.array([3000., 2000., 1000.])
                if symbol == "zlev":
                    vector = np.array([-30., -50., -70.])
                if kind == "ocean_sigma_z" and symbol in {"sigma", "zlev"}:
                    vector = np.ma.array(
                        [-0.1, 0., 0.] if symbol == "sigma" else [-0., -50., -70.],
                        mask=[False, True, True] if symbol == "sigma" else [True, False, False],
                    )
                var[:] = vector[:, None] if len(dims) == 2 else vector
            else:
                var[...] = constants.get(symbol, 1.)
        if kind in {"standard_sigma", "ocean_sigma"}:
            mapping["sigma"] = bounds.name if is_bounds else name
        if kind == "hybrid_height":
            mapping["a"] = bounds.name if is_bounds else name
        if kind == "hybrid_sigma_ln_pressure":
            mapping["eta"] = bounds.name if is_bounds else name
        if mapping:
            (bounds if is_bounds else coordinate).formula_terms = " ".join(f"{k}: {v}" for k, v in mapping.items())
        mappings.append(mapping)
    catalog = Catalog(project_id="cmip7", branded_variable_id="test", branded_variable={},
                      coordinate_ids=(generic,), data_coordinates={generic: {"out_name": name}},
                      model_levels=LEVELS, formula_terms=TERMS)
    return coordinate, catalog, mappings[0]


@pytest.fixture(params=sorted(LEVELS))
def model_file(request, tmp_path):
    entry = LEVELS[request.param]
    with Dataset(tmp_path / (request.param + ".nc"), "w") as ds:
        coordinate, catalog, mapping = build_file(ds, entry)
        yield ds, entry, coordinate, catalog, mapping


def test_model_level_metadata_and_bounds(model_file):
    ds, entry, coordinate, catalog, mapping = model_file
    findings = Findings(FAMILIES)
    validate_model_level(findings, ds, catalog, entry["generic_level_name"], (), check_formula_derived_profile=False)
    assert not any(findings.messages.values()), findings.messages


def test_model_level_checks_together(model_file):
    ds, entry, coordinate, catalog, mapping = model_file
    findings = Findings(FAMILIES)
    validate_model_level(findings, ds, catalog, entry["generic_level_name"], ())
    assert not any(findings.messages.values()), findings.messages


def test_model_level_physical_profile(model_file):
    ds, entry, coordinate, catalog, mapping = model_file
    if not entry.get("formula"):
        assert not mapping
        assert "formula_terms" not in coordinate.ncattrs()
        return
    expected = {
        "standard_sigma": [91000., 55000., 19000.],
        "standard_hybrid_sigma": [390000., 250000., 110000.],
        "alternate_hybrid_sigma": [93000., 52000., 11000.],
        "hybrid_height": [20., 40., 60.],
        "hybrid_sigma_ln_pressure": [90000., 50000., 10000.],
        "ocean_sigma": [-10., -50., -90.],
        "ocean_sigma_z": [-2., -50., -70.],
    }[family(entry["id"])]
    profile, _ = formula_vertical_profile(ds, coordinate, entry, mapping)
    np.testing.assert_allclose(profile, expected)


def test_model_level_wrong_standard_name_is_reported(model_file):
    ds, entry, coordinate, catalog, _ = model_file
    coordinate.standard_name = "not_a_registered_vertical_coordinate"
    findings = Findings(FAMILIES)
    validate_model_level(findings, ds, catalog, entry["generic_level_name"], ())
    assert any(findings.messages.values())


def test_model_level_wrong_units_are_reported(model_file):
    ds, entry, coordinate, catalog, _ = model_file
    coordinate.units = "seconds"
    findings = Findings(FAMILIES)
    validate_model_level(findings, ds, catalog, entry["generic_level_name"], (), check_formula_derived_profile=False)
    assert findings.messages["attributes"]


def test_model_level_required_bounds_are_reported(model_file):
    ds, entry, coordinate, catalog, _ = model_file
    if not entry.get("bounds_required"):
        assert "bounds" not in coordinate.ncattrs()
        return
    coordinate.bounds = "absent_bounds"
    findings = Findings(FAMILIES)
    validate_model_level(findings, ds, catalog, entry["generic_level_name"], (), check_formula_derived_profile=False)
    assert findings.messages["bounds"]


def test_model_level_formula_mutation_is_reported(model_file):
    ds, entry, coordinate, catalog, _ = model_file
    coordinate.formula = "p = incorrect"
    findings = Findings(FAMILIES)
    validate_model_level(findings, ds, catalog, entry["generic_level_name"], ())
    assert any(findings.messages.values())


@pytest.mark.parametrize("identifier", ["ocean_sigma_z", "ocean_sigma_z_half"])
@pytest.mark.parametrize("with_nsigma", [False, True], ids=["without-nsigma", "with-nsigma"])
@pytest.mark.parametrize(
    "sigma_mask,expected",
    [
        ([False, False, False], [-2., -10., -18.]),
        ([True, True, True], [-30., -50., -70.]),
        ([False, True, True], [-2., -50., -70.]),
    ],
    ids=["all-sigma", "all-fixed-depth", "mixed"],
)
def test_ocean_sigma_z_profile_variants(
    tmp_path, identifier, with_nsigma, sigma_mask, expected
):
    """Exercise the real NetCDF mask reader as well as the physical evaluator.

    nsigma is optional in CF, including at the endpoints zero and nlevels.
    The catalogue comparison also permits this CF-defined optional term.
    """
    entry = LEVELS[identifier]
    with Dataset(tmp_path / "sigma_z.nc", "w") as ds:
        coordinate, catalog, mapping = build_file(ds, entry)
        sigma_mask = np.array(sigma_mask)
        sigma = np.ma.array([-0.1, -0.5, -0.9], mask=sigma_mask)
        zlev = np.ma.array([-30., -50., -70.], mask=~sigma_mask)
        ds.variables[mapping["sigma"]][:] = sigma
        ds.variables[mapping["zlev"]][:] = zlev
        coordinate[:] = expected
        # Keep the coordinate bounds consistent with the new centre values.
        if "bounds" in coordinate.ncattrs():
            for i, value in enumerate(expected):
                ds.variables[coordinate.bounds][i, :] = [value + 0.04, value - 0.04]
        if with_nsigma:
            ds.createVariable("nsigma", "i4").assignValue(int(np.count_nonzero(~sigma_mask)))
            mapping["nsigma"] = "nsigma"
            coordinate.formula_terms += " nsigma: nsigma"
            if "bounds" in coordinate.ncattrs():
                ds.variables[coordinate.bounds].formula_terms += " nsigma: nsigma"
        findings = Findings(FAMILIES)
        validate_model_level(findings, ds, catalog, entry["generic_level_name"], ())
        assert not any(findings.messages.values()), findings.messages
        profile, _ = formula_vertical_profile(ds, coordinate, entry, mapping)
        np.testing.assert_allclose(profile, expected)


@pytest.mark.parametrize("problem", ["missing", "wrong-count", "nonscalar", "unknown-term"])
def test_optional_nsigma_does_not_hide_bad_terms(tmp_path, problem):
    entry = LEVELS["ocean_sigma_z"]
    with Dataset(tmp_path / "bad_count.nc", "w") as ds:
        coordinate, catalog, _ = build_file(ds, entry)
        coordinate.formula_terms += " nsigma: count"
        if problem != "missing":
            count = ds.createVariable("count", "i4", (coordinate.name,) if problem == "nonscalar" else ())
            count[...] = 2 if problem == "wrong-count" else 1
        if problem == "unknown-term":
            coordinate.formula_terms += " unexpected: count"
        findings = Findings(FAMILIES)
        validate_model_level(findings, ds, catalog, entry["generic_level_name"], (), check_formula_derived_profile=False)
        assert findings.messages["formula"]
