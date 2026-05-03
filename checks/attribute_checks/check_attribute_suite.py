#!/usr/bin/env python
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

import numpy as np
from compliance_checker.base import TestCtx

# ESGVOC
try:
    from esgvoc import api as voc  # type: ignore

    _ESGVOC_AVAILABLE = True
except Exception:
    voc = None
    _ESGVOC_AVAILABLE = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _to_list(value: Any) -> list[str]:
    """Convert attribute value to list of tokens."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [t for t in str(value).split() if t.strip()]


def _to_float(value: Any) -> Optional[float]:
    """Try converting to float (supports numpy scalar types)."""
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def _ci_attr_name_lookup(obj, attribute_name: str) -> str:
    """Case-insensitive lookup of NetCDF attribute name."""
    try:
        nc_attrs = obj.ncattrs()
    except Exception:
        return attribute_name
    matches = [a for a in nc_attrs if a.lower() == attribute_name.lower()]
    return matches[0] if matches else attribute_name


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def check_attribute_suite(
    ds,
    attribute_name: str,
    severity: int,
    value_type: Optional[str] = None,
    is_required: bool = True,
    na_value: Optional[Any] = None,
    pattern: Optional[str] = None,
    constant: Any = None,
    enum: Optional[Iterable[Any]] = None,
    as_variable: Optional[bool] = None,
    is_positive: Optional[bool] = None,
    # ESGVOC vocab rule (key is OPTIONAL param of the same rule)
    cv_source_collection: Optional[str] = None,
    cv_source_collection_key: Optional[str] = None,
    project_name: Optional[str] = None,
    # Registry expected-term rule (exclusive with everything else in ATTR004)
    cv_source_term_key: Optional[str] = None,
    expected_term: Any = None,
    # Location
    var_name: Optional[str] = None,
    attribute_nc_name: Optional[str] = None,
    # Optional label prefix for nicer report grouping
    context: Optional[str] = None,
):
    """
    Attribute suite checks.

    ATTR004 rule selection (STRICT):
      Exactly ONE of the following rules can be active:
        - pattern
        - constant
        - enum
        - as_variable
        - is_positive
        - vocabulary: cv_source_collection (+ optional cv_source_collection_key)
        - registry expected-term: cv_source_term_key (+ expected_term)

      The ONLY allowed combination is:
        cv_source_collection + cv_source_collection_key

    na_value behavior:
      - if the attribute is missing -> normal ATTR001 logic
      - if the attribute exists and its value equals na_value -> stop after ATTR001
      - otherwise continue with normal validation
    """

    results = []

    # -------------------------------------------------------------------------
    # Resolve attribute source (global vs variable)
    # -------------------------------------------------------------------------
    if var_name:
        if var_name not in ds.variables:
            return []
        obj = ds.variables[var_name]
        where = f"Variable '{var_name}' attribute"
    else:
        obj = ds
        where = "Global attribute"

    # Resolve actual netCDF attribute key (case-insensitive)
    nc_key = attribute_nc_name or _ci_attr_name_lookup(obj, attribute_name)

    prefix = f"{context} " if context else ""
    label = f"{prefix}{where} '{attribute_name}'"

    # -------------------------------------------------------------------------
    # ATTR001 - Existence
    # -------------------------------------------------------------------------
    existence_ctx = TestCtx(severity, f"[ATTR001] {label} existence")
    try:
        attr_value = obj.getncattr(nc_key)
    except AttributeError:
        if is_required:
            existence_ctx.add_failure(
                f"Required {where.lower()} '{attribute_name}' is missing."
            )
            results.append(existence_ctx.to_result())
        return results  # stop here if missing

    # -------------------------------------------------------------------------
    # Short-circuit on sentinel / not-applicable value
    # -------------------------------------------------------------------------
    if na_value is not None:
        if str(attr_value).strip().lower() == str(na_value).strip().lower():
            return results

    existence_ctx.add_pass()
    results.append(existence_ctx.to_result())

    # -------------------------------------------------------------------------
    # ATTR002 - Type check
    # -------------------------------------------------------------------------
    if value_type:
        type_ctx = TestCtx(severity, f"[ATTR002] {label} type")

        vt = str(value_type).lower()
        if vt in ("double", "real"):
            vt = "float"

        type_map = {
            "str": str,
            "int": (int, np.integer),
            "float": (float, np.floating),
            "double": np.float64,
            "simple": np.float32,
            "bool": (bool, np.bool_),
            "str_array": list,
        }
        expected = type_map.get(vt)

        if expected is None:
            type_ctx.add_failure(f"Unsupported value_type '{value_type}'.")
        else:
            val_for_type = _to_list(attr_value) if vt == "str_array" else attr_value
            if isinstance(val_for_type, expected):
                type_ctx.add_pass()
            else:
                type_ctx.add_failure(
                    f"Type mismatch: expected {expected}, got {type(attr_value)}"
                )

        results.append(type_ctx.to_result())

    # -------------------------------------------------------------------------
    # ATTR003 - UTF-8 check
    # -------------------------------------------------------------------------
    if isinstance(attr_value, str):
        utf8_ctx = TestCtx(severity, f"[ATTR003] {label} UTF-8 encoding")
        try:
            attr_value.encode("utf-8")
            utf8_ctx.add_pass()
        except UnicodeEncodeError:
            utf8_ctx.add_failure("Non UTF-8 characters detected.")
        results.append(utf8_ctx.to_result())

    # -------------------------------------------------------------------------
    # ATTR004 - ONE exclusive rule (except vocab key parameter)
    # -------------------------------------------------------------------------

    vocab_rule_active = cv_source_collection is not None
    registry_rule_active = cv_source_term_key is not None

    rule_flags = [
        pattern is not None,
        constant is not None,
        enum is not None,
        bool(as_variable),
        bool(is_positive),
        vocab_rule_active,
        registry_rule_active,
    ]

    # If collection_key is provided without collection => config error
    if cv_source_collection_key and not cv_source_collection:
        cfg_ctx = TestCtx(severity, f"[ATTR004] {label} rule configuration")
        cfg_ctx.add_failure(
            "cv_source_collection_key provided without cv_source_collection."
        )
        results.append(cfg_ctx.to_result())
        return results

    # Enforce strict exclusivity: only ONE rule among the 7 flags above
    if sum(rule_flags) > 1:
        cfg_ctx = TestCtx(severity, f"[ATTR004] {label} rule configuration")
        cfg_ctx.add_failure(
            "Multiple mutually exclusive ATTR004 rules defined. "
            "Allowed: exactly one of (pattern|constant|enum|as_variable|is_positive|cv_source_collection|cv_source_term_key). "
            "Note: cv_source_collection_key is only a parameter of cv_source_collection."
        )
        results.append(cfg_ctx.to_result())
        return results

    # ---------------- PATTERN ----------------
    if pattern is not None:
        ctx = TestCtx(severity, f"[ATTR004] {label} pattern check")
        try:
            if re.fullmatch(str(pattern), str(attr_value)):
                ctx.add_pass()
            else:
                ctx.add_failure(
                    f"Value '{attr_value}' does not match pattern '{pattern}'."
                )
        except re.error:
            ctx.add_failure(f"Invalid regex '{pattern}'.")
        results.append(ctx.to_result())
        return results

    # ---------------- CONSTANT ----------------
    if constant is not None:
        ctx = TestCtx(severity, f"[ATTR004] {label} constant check")
        if str(attr_value).strip() == str(constant).strip():
            ctx.add_pass()
        else:
            ctx.add_failure(f"Expected '{constant}', got '{attr_value}'.")
        results.append(ctx.to_result())
        return results

    # ---------------- ENUM ----------------
    if enum is not None:
        ctx = TestCtx(severity, f"[ATTR004] {label} enum check")
        allowed = [str(x) for x in enum]
        if str(attr_value) in allowed:
            ctx.add_pass()
            # Non-blocking advisory for time:calendar="standard"
            if (
            var_name == "time"
            and str(attribute_name).lower() == "calendar"
            and str(attr_value).strip().lower() == "gregorian"
        ):
                ctx.messages.append(
                "Value 'gregorian' is accepted for variable 'time' attribute "
                "'calendar', but 'proleptic_gregorian' is recommended."
                )
        else:
            ctx.add_failure(f"Value '{attr_value}' not in allowed values {allowed}.")
        results.append(ctx.to_result())
        return results

    # ---------------- AS VARIABLE ----------------
    if as_variable:
        ctx = TestCtx(severity, f"[ATTR004] {label} as-variable check")
        tokens = _to_list(attr_value)
        missing = [t for t in tokens if t not in ds.variables]
        if missing:
            ctx.add_failure(f"Referenced variables not found: {missing}")
        else:
            ctx.add_pass()
        results.append(ctx.to_result())
        return results

    # ---------------- IS POSITIVE ----------------
    if is_positive:
        ctx = TestCtx(severity, f"[ATTR004] {label} positive check")
        val = _to_float(attr_value)
        if val is None:
            ctx.add_failure(f"Value '{attr_value}' is not numeric.")
        elif val > 0:
            ctx.add_pass()
        else:
            ctx.add_failure(f"Expected positive value (>0), got {val}.")
        results.append(ctx.to_result())
        return results

    # ---------------- REGISTRY expected-term ----------------
    if registry_rule_active:
        ctx = TestCtx(severity, f"[ATTR004] {label} registry expected-term check")
        if expected_term is None:
            ctx.add_failure(
                "Registry rule enabled but expected_term is None (registry not resolved)."
            )
        else:
            expected_val = getattr(expected_term, str(cv_source_term_key), None)
            if expected_val is None or str(expected_val).strip() == "":
                ctx.add_failure(
                    f"Registry has no value for key '{cv_source_term_key}'."
                )
            elif str(attr_value).strip() == str(expected_val).strip():
                ctx.add_pass()
            else:
                ctx.add_failure(
                    f"Expected '{expected_val}' from registry key '{cv_source_term_key}', got '{attr_value}'."
                )
        results.append(ctx.to_result())
        return results

    # ---------------- ESGVOC vocabulary ----------------
    if vocab_rule_active:
        ctx = TestCtx(severity, f"[ATTR004] {label} vocabulary check")

        if not _ESGVOC_AVAILABLE:
            ctx.add_failure("ESGVOC library not available.")
            results.append(ctx.to_result())
            return results

        values = _to_list(attr_value) if value_type == "str_array" else [attr_value]
        invalid: list[Any] = []

        try:
            if cv_source_collection_key:
                terms = voc.get_all_terms_in_collection(
                    project_id=project_name,
                    collection_id=cv_source_collection,
                )

                if not terms:
                    ctx.add_failure(
                        f"CV collection '{cv_source_collection}' is empty or could not be retrieved."
                    )
                    results.append(ctx.to_result())
                    return results

                for val in values:
                    found = False
                    val_norm = " ".join(str(val).strip().lower().split())

                    for term in terms:
                        candidate = getattr(term, str(cv_source_collection_key), None)
                        if candidate is None:
                            continue

                        candidate_norm = " ".join(str(candidate).strip().lower().split())

                        if val_norm == candidate_norm or val_norm in candidate_norm:
                            found = True
                            break

                    if not found:
                        invalid.append(val)

                if invalid:
                    ctx.add_failure(
                        f"Value(s) {invalid} not found in field '{cv_source_collection_key}' "
                        f"of any term in CV collection '{cv_source_collection}'."
                    )
                else:
                    ctx.add_pass()

                results.append(ctx.to_result())
                return results

            for val in values:
                if not voc.valid_term_in_collection(
                    value=val,
                    project_id=project_name,
                    collection_id=cv_source_collection,
                ):
                    invalid.append(val)

            if invalid:
                ctx.add_failure(
                    f"Invalid value(s) {invalid} for CV collection '{cv_source_collection}'."
                )
            else:
                ctx.add_pass()

        except Exception as e:
            ctx.add_failure(f"Vocabulary lookup error: {e}")

        results.append(ctx.to_result())
        return results

    # If no ATTR004 rule configured, we simply return existence/type/utf8 results.
    return results