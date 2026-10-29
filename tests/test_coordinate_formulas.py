"""Focused unit tests for safe vertical-formula evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest

from checks.coordinate_checks.formulas import (
    _evaluate_formula,
    _sample_formula_term,
    formula_vertical_profile,
)


class _Variable:
    def __init__(self, name, values, dimensions):
        self.name = name
        self.values = np.ma.asarray(values)
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)

    def __getitem__(self, key):
        return self.values[key]

    def __len__(self):
        return len(self.values)


def test_formula_evaluator_supports_arithmetic_functions_and_fortran_power():
    terms = {"a": np.array([1.0, 2.0]), "b": 2.0}
    result = _evaluate_formula("p = max(a^2, b) + exp(0)", "", terms, 2)
    np.testing.assert_allclose(result, [3.0, 5.0])


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("a + b", "no '='"),
        ("p = missing + 1", "not declared"),
        ("p = abs(a)", "unsupported formula function"),
        ("p = exp(a, a)", "expects one argument"),
        ("p = a // 2", "unsupported operator"),
    ],
)
def test_formula_evaluator_rejects_unsupported_or_incomplete_expressions(
    formula, message
):
    with pytest.raises((KeyError, ValueError), match=message):
        _evaluate_formula(formula, "", {"a": np.array([1.0, 2.0])}, 2)


def test_ocean_sigma_z_formula_and_required_terms():
    terms = {
        "eta": 1.0,
        "sigma": np.array([-1.0, -0.5, 0.0]),
        "depth": 100.0,
        "depth_c": 20.0,
        "nsigma": 2,
        "zlev": np.array([-50.0, -25.0, -10.0]),
    }
    result = _evaluate_formula("unused", "ocean_sigma_z_coordinate", terms, 3)
    np.testing.assert_allclose(result, [-20.0, -9.5, -10.0])

    del terms["eta"]
    with pytest.raises(KeyError, match="eta"):
        _evaluate_formula("unused", "ocean_sigma_z_coordinate", terms, 3)


def test_sample_formula_term_handles_vertical_scalar_and_invalid_samples():
    vertical = _Variable("a", [1.0, 2.0], ("lev",))
    np.testing.assert_allclose(_sample_formula_term(vertical, "lev", {}), [1.0, 2.0])

    scalar = _Variable("p0", 100000.0, ())
    assert _sample_formula_term(scalar, "lev", {}) == 100000.0

    nonscalar = _Variable("bad", [1.0, 2.0], ())
    with pytest.raises(ValueError, match="did not reduce to a scalar"):
        _sample_formula_term(nonscalar, "lev", {})

    masked = _Variable("masked", np.ma.array([1.0, 2.0], mask=[False, True]), ("lev",))
    with pytest.raises(ValueError, match="masked"):
        _sample_formula_term(masked, "lev", {})


def test_vertical_profile_skips_unusable_horizontal_point_and_reports_location():
    coordinate = _Variable("lev", [0.0, 1.0], ("lev",))
    term = _Variable(
        "a",
        np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[True, True], [False, False]]),
        ("x", "lev"),
    )
    dataset = SimpleNamespace(
        variables={"a": term}, dimensions={"lev": range(2), "x": range(2)}
    )
    profile, sample = formula_vertical_profile(
        dataset,
        coordinate,
        {"formula": "z = a", "cf_standard_name": ""},
        {"a": "a"},
    )
    np.testing.assert_allclose(profile, [3.0, 4.0])
    assert sample == "sampled point x=1"


def test_vertical_profile_reports_structural_and_lookup_failures():
    coordinate = _Variable("lev", [0.0, 1.0], ("lev",))
    dataset = SimpleNamespace(variables={}, dimensions={"lev": range(2)})

    with pytest.raises(ValueError, match="mapping is missing"):
        formula_vertical_profile(dataset, coordinate, {}, {})
    with pytest.raises(KeyError, match="missing 'a'"):
        formula_vertical_profile(dataset, coordinate, {}, {"a": "a"})

    two_dimensional = _Variable("lev", [[0.0], [1.0]], ("lev", "x"))
    with pytest.raises(ValueError, match="must be one-dimensional"):
        formula_vertical_profile(dataset, two_dimensional, {}, {"a": "a"})
