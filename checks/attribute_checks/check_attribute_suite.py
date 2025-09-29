#!/usr/bin/env python


from compliance_checker.base import BaseCheck, Result, TestCtx
import numpy as np
from esgvoc import api as voc
import re

def check_attribute_suite(
    ds,
    attribute_name,
    severity,
    expected_type=None,
    constraint=None,
    var_name=None,
    project_name=None
):
    """
    Runs a basic suite of checks on a single attribute:
    1. Existence (ATTR001)
    2. Type (ATTR002)
    3. UTF-8 Encoding (ATTR003)
    4. Vocabulary or Pattern Match (ATTR004)
    """

    def label(code, desc):
        prefix = f"[{code}]"
        if var_name:
            return f"{prefix} {var_name} | {desc}: Variable Attribute '{attribute_name}'"
        else:
            return f"{prefix} {desc}: Global Attribute '{attribute_name}'"

    all_results = []
    attr_value = None

    # --- Check 1: Attribute Existence (ATTR001) ---
    existence_ctx = TestCtx(severity, label("ATTR001", "Existence"))
    try:
        if var_name:
            if var_name not in ds.variables:
                existence_ctx.add_failure(f"Cannot check attribute '{attribute_name}' because variable '{var_name}' does not exist.")
                all_results.append(existence_ctx.to_result())
                return all_results
            attr_value = ds.variables[var_name].getncattr(attribute_name)
        else:
            attr_value = ds.getncattr(attribute_name)
        existence_ctx.add_pass()
    except AttributeError:
        existence_ctx.add_failure(f"Attribute '{attribute_name}' is missing.")
    all_results.append(existence_ctx.to_result())

    if attr_value is None:
        return all_results

    # --- Check 2: Attribute Type (ATTR002) ---
    if expected_type:
        type_ctx = TestCtx(severity, label("ATTR002", "Type"))
        type_map = {
            "str": str,
            "int": (int, np.integer),
            "float": (float, np.floating),
        }
        py_type = type_map.get(str(expected_type).lower())

        if py_type and isinstance(attr_value, py_type):
            type_ctx.add_pass()
        elif not py_type:
            type_ctx.add_failure(f"Configuration error: unknown expected_type '{expected_type}'.")
        else:
            type_ctx.add_failure(f"Value has type {type(attr_value).__name__}, expected {py_type.__name__}.")
        all_results.append(type_ctx.to_result())

    # --- Check 3: UTF-8 Encoding (ATTR003) ---
    if isinstance(attr_value, str):
        utf8_ctx = TestCtx(severity, label("ATTR003", "UTF-8 Encoding"))
        try:
            attr_value.encode('utf-8')
            utf8_ctx.add_pass()
        except UnicodeEncodeError:
            utf8_ctx.add_failure("Attribute contains non-UTF-8 characters.")
        all_results.append(utf8_ctx.to_result())

    # --- Check 4: Vocabulary ( Esgvoc/Universe ) or Regex Check(ATTR004) ---
    MULTI_TERM_ATTRIBUTES = {"source_type", "realm", "activity_id", "Conventions"}
    if constraint is None and isinstance(attr_value, str) and expected_type == "str" and project_name:
        vocab_ctx = TestCtx(severity, label("ATTR004", "ESGVOC Vocabulary Check"))
        try:
            if attribute_name in MULTI_TERM_ATTRIBUTES:
                values_to_check = attr_value.strip().split()
            else:
                values_to_check = [attr_value]
            invalid_values = []                       

            for val in values_to_check:
                if not voc.valid_term_in_collection(       
                    value=val,
                    project_id=project_name,
                    collection_id=attribute_name.lower()
                ):
                    invalid_values.append(val)   
            if not invalid_values:
                vocab_ctx.add_pass()
            else:
                vocab_ctx.add_failure(f"Value(s) {invalid_values} are not valid in collection '{attribute_name}' for project '{project_name}'.")
        
        except Exception as e:
            vocab_ctx.add_failure(f"Vocabulary validation failed: {str(e)}")
        all_results.append(vocab_ctx.to_result())
    
    elif constraint is not None and constraint.strip():
        pattern_ctx = TestCtx(severity, label("ATTR004", "Pattern/Universe Match Check"))
        try:
            if re.fullmatch(constraint, attr_value):
                pattern_ctx.add_pass()
            else:
                pattern_ctx.add_failure(f"Value '{attr_value}' does not match expected pattern: '{constraint}'.")
        except re.error:
            pattern_ctx.add_failure(f"Invalid regex pattern in configuration: '{constraint}'.")
        all_results.append(pattern_ctx.to_result())

    return all_results
