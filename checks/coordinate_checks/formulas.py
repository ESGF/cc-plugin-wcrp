"""Safe evaluation of a model-level formula at one horizontal point."""

from __future__ import annotations

import ast
import re

import numpy as np


def _sample_formula_term(var, vertical_dimension: str, point: dict, *, allow_masked=False):
    key = tuple(
        slice(None) if dimension == vertical_dimension else point.get(dimension, 0)
        for dimension in var.dimensions
    )
    masked = np.ma.asarray(var[key], dtype="float64")
    if np.any(np.ma.getmaskarray(masked)) and not allow_masked:
        raise ValueError(f"formula term '{var.name}' is masked at the sampled point")
    sampled = masked.squeeze()
    if not np.all(np.isfinite(sampled.compressed())):
        raise ValueError(
            f"formula term '{var.name}' is non-finite at the sampled point"
        )
    if vertical_dimension in var.dimensions:
        return sampled.reshape(-1) if allow_masked else np.asarray(sampled).reshape(-1)
    if allow_masked:
        raise ValueError(f"formula term '{var.name}' must include the vertical dimension")
    if np.asarray(sampled).size != 1:
        raise ValueError(
            f"formula term '{var.name}' did not reduce to a scalar at the "
            "sampled point"
        )
    return float(np.asarray(sampled).reshape(-1)[0])


def _evaluate_node(node, terms):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, terms)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in terms:
            raise KeyError(f"formula term '{node.id}' is not declared")
        return terms[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand, terms)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, terms)
        right = _evaluate_node(node.right, terms)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left**right,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError("formula contains an unsupported operator")
        return operation()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_evaluate_node(argument, terms) for argument in node.args]
        functions = {
            "min": np.minimum,
            "max": np.maximum,
            "exp": np.exp,
            "log": np.log,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
        }
        function = functions.get(node.func.id)
        if function is None or not arguments:
            raise ValueError(f"unsupported formula function '{node.func.id}'")
        if node.func.id in {"min", "max"}:
            result = arguments[0]
            for argument in arguments[1:]:
                result = function(result, argument)
            return result
        if len(arguments) != 1:
            raise ValueError(
                f"formula function '{node.func.id}' expects one argument"
            )
        return function(arguments[0])
    raise ValueError("formula contains unsupported syntax")


def _evaluate_formula(formula: str, standard_name: str, terms, level_count: int):
    if standard_name == "ocean_sigma_z_coordinate":
        required = {"eta", "sigma", "depth", "depth_c", "zlev"}
        missing = sorted(required - terms.keys())
        if missing:
            raise KeyError(f"formula term(s) {missing} are not declared")
        sigma = np.ma.asarray(terms["sigma"], dtype="float64")
        zlev = np.ma.asarray(terms["zlev"], dtype="float64")
        if sigma.shape != (level_count,) or zlev.shape != (level_count,):
            raise ValueError("sigma and zlev must each contain one value per level")
        sigma_missing = np.ma.getmaskarray(sigma)
        zlev_missing = np.ma.getmaskarray(zlev)
        if np.any(sigma_missing == zlev_missing):
            raise ValueError("exactly one of sigma and zlev must be defined at each level")
        if not all(np.all(np.isfinite(v.compressed())) for v in (sigma, zlev)):
            raise ValueError("sigma or zlev contains non-finite unmasked values")
        # CF identifies the two regions through complementary missing values.
        # nsigma is deprecated; when present it only confirms that split.
        if "nsigma" in terms:
            count = np.ma.asarray(terms["nsigma"])
            if (count.size != 1 or np.any(np.ma.getmaskarray(count))
                    or not np.isfinite(count.item())
                    or count.item() != np.count_nonzero(zlev_missing)):
                raise ValueError("nsigma must equal the number of missing zlev values")
        profile = np.array(zlev.data, dtype="float64", copy=True)
        profile[zlev_missing] = terms["eta"] + sigma.data[zlev_missing] * (
            min(terms["depth_c"], terms["depth"]) + terms["eta"]
        )
        return profile

    if "=" not in formula:
        raise ValueError("formula has no '=' expression")
    expression = formula.split("=", 1)[1].strip().replace("^", "**")
    for term in sorted(terms, key=len, reverse=True):
        expression = re.sub(rf"\b{re.escape(term)}\s*\([^()]*\)", term, expression)
    parsed = ast.parse(expression, mode="eval")
    result = _evaluate_node(parsed, terms)
    try:
        return np.asarray(np.broadcast_to(result, (level_count,)), dtype="float64")
    except ValueError as exc:
        raise ValueError(
            f"formula result has shape {np.shape(result)}, expected "
            f"{level_count} levels"
        ) from exc


def formula_vertical_profile(ds, coordinate, entry: dict, formula_terms: dict):
    """Return one finite formula-derived profile and its sampled location."""
    if coordinate.ndim != 1:
        raise ValueError(
            f"coordinate '{coordinate.name}' must be one-dimensional; found "
            f"dimensions {list(coordinate.dimensions)}"
        )
    vertical_dimension = coordinate.dimensions[0]
    if not formula_terms:
        raise ValueError("formula_terms mapping is missing or empty")

    variables = {}
    for term, variable_name in formula_terms.items():
        if variable_name not in ds.variables:
            raise KeyError(
                f"formula term '{term}' references missing '{variable_name}'"
            )
        variables[term] = ds.variables[variable_name]

    sample_dimensions = []
    for variable in variables.values():
        for dimension in variable.dimensions:
            if dimension != vertical_dimension and dimension not in sample_dimensions:
                sample_dimensions.append(dimension)
    sample_shape = tuple(len(ds.dimensions[name]) for name in sample_dimensions)
    sample_count = int(np.prod(sample_shape)) if sample_shape else 1

    last_error = None
    for flat_index in range(min(sample_count, 10000)):
        indices = np.unravel_index(flat_index, sample_shape) if sample_shape else ()
        point = dict(zip(sample_dimensions, indices))
        try:
            terms = {
                term: _sample_formula_term(
                    variable, vertical_dimension, point,
                    allow_masked=(entry.get("cf_standard_name") == "ocean_sigma_z_coordinate"
                                  and term in {"sigma", "zlev"}),
                )
                for term, variable in variables.items()
            }
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue
        try:
            profile = _evaluate_formula(
                str(entry.get("formula") or ""),
                str(entry.get("cf_standard_name") or ""),
                terms,
                len(coordinate),
            )
        except (KeyError, TypeError, ValueError, SyntaxError):
            # Formula syntax and term declarations do not depend on the point.
            raise
        try:
            profile = np.asarray(profile, dtype="float64").reshape(-1)
            if not np.all(np.isfinite(profile)):
                raise ValueError("formula-derived profile contains non-finite values")
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue

        sample = (
            "sampled point "
            + ", ".join(f"{name}={point[name]}" for name in sample_dimensions)
            if sample_dimensions
            else "vertical-only formula terms"
        )
        return profile, sample

    detail = f": {last_error}" if last_error else ""
    raise ValueError(f"no usable sampled point was found{detail}")
