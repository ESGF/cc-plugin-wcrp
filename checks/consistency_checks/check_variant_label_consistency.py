#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atomic variant_label consistency checks (ATTR006a-d).

Split from the monolithic ATTR006 into one function per index so each has its
own severity and Result:

  ATTR006a  variant_label vs realization_index
  ATTR006b  variant_label vs initialization_index
  ATTR006c  variant_label vs physics_index
  ATTR006d  variant_label vs forcing_index

CMIP6 stores indices as numeric ("1"/1). CMIP7 stores them prefixed
("r1"/"i1"/"p1"/"f3"). Both are supported.
"""

import re
from compliance_checker.base import TestCtx


def _is_cmip7(ds) -> bool:
    try:
        if str(ds.getncattr("mip_era")).upper() == "CMIP7":
            return True
    except Exception:
        pass
    try:
        if str(ds.getncattr("drs_specs")) == "MIP-DRS7":
            return True
    except Exception:
        pass
    return False


def _to_int_index(attr_value, prefix: str, is_cmip7: bool):
    if attr_value is None:
        return None
    if isinstance(attr_value, int):
        return attr_value
    s = str(attr_value).strip()
    if not s:
        return None
    if is_cmip7 and s.lower().startswith(prefix.lower()):
        s = s[len(prefix):].strip()
    try:
        return int(s)
    except Exception:
        return None


def _parsed_variant_indices(ds):
    """
    Return (indices_dict, error_message).
    indices_dict maps each *_index attr name to the int implied by variant_label.
    """
    if "variant_label" not in ds.ncattrs():
        return None, "Missing required global attribute: 'variant_label'."
    variant_label = str(ds.getncattr("variant_label")).strip()
    m = re.match(r"^r(\d+)i(\d+)p(\d+)f(\d+)$", variant_label)
    if not m:
        return None, (
            f"The format of 'variant_label' ('{variant_label}') is invalid. "
            "Expected 'r<k>i<l>p<m>f<n>'."
        )
    return {
        "realization_index": int(m.group(1)),
        "initialization_index": int(m.group(2)),
        "physics_index": int(m.group(3)),
        "forcing_index": int(m.group(4)),
    }, None


def _check_one_index(ds, severity, check_id, index_name, prefix):
    label = f"Consistency: variant_label vs {index_name}"
    ctx = TestCtx(severity, f"[{check_id}] {label}")

    parsed, err = _parsed_variant_indices(ds)
    if err:
        ctx.add_failure(err)
        return [ctx.to_result()]

    if index_name not in ds.ncattrs():
        ctx.add_failure(f"Missing required global attribute: '{index_name}'.")
        return [ctx.to_result()]

    raw = ds.getncattr(index_name)
    is7 = _is_cmip7(ds)
    attr_int = _to_int_index(raw, prefix, is7)

    if attr_int is None:
        ctx.add_failure(
            f"Could not interpret '{index_name}' value '{raw}' as an integer."
        )
        return [ctx.to_result()]

    implied = parsed[index_name]
    if implied == attr_int:
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"Inconsistency for '{index_name}': variant_label implies "
            f"'{implied}', but attribute is '{raw}'."
        )
    return [ctx.to_result()]


def check_variant_vs_realization_index(ds, severity):
    return _check_one_index(ds, severity, "ATTR006a", "realization_index", "r")


def check_variant_vs_initialization_index(ds, severity):
    return _check_one_index(ds, severity, "ATTR006b", "initialization_index", "i")


def check_variant_vs_physics_index(ds, severity):
    return _check_one_index(ds, severity, "ATTR006c", "physics_index", "p")


def check_variant_vs_forcing_index(ds, severity):
    return _check_one_index(ds, severity, "ATTR006d", "forcing_index", "f")
