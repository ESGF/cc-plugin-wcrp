#!/usr/bin/env python

import re

from compliance_checker.base import TestCtx


def _is_cmip7(ds) -> bool:
    """Detect CMIP7 from global attrs when available."""
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
    """
    Convert index attribute to int.

    CMIP6:
      often "1" or int 1

    CMIP7:
      often "r1", "i1", "p1", "f3"
    """
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


def _parse_variant_label(variant_label):
    """
    Parse variant_label r<i>i<j>p<k>f<l>.

    Return:
      dict or None
    """
    match = re.match(r"^r(\d+)i(\d+)p(\d+)f(\d+)$", str(variant_label))

    if not match:
        return None

    return {
        "realization_index": int(match.group(1)),
        "initialization_index": int(match.group(2)),
        "physics_index": int(match.group(3)),
        "forcing_index": int(match.group(4)),
    }


def _check_variant_label_vs_index(ds, severity, index_attr, prefix):
    """
    Generic single-comparison check:
      variant_label implied index vs global attribute index
    """
    fixed_check_id = "ATTR006"
    description = f"[{fixed_check_id}] Consistency: variant_label vs {index_attr}"
    ctx = TestCtx(severity, description)

    try:
        variant_label = str(ds.getncattr("variant_label"))
        parsed_indices = _parse_variant_label(variant_label)

        if parsed_indices is None:
            ctx.add_failure(
                f"The format of 'variant_label' ('{variant_label}') is invalid. "
                "Expected format is 'r<k>i<l>p<m>f<n>'."
            )
            return [ctx.to_result()]

        attr_raw = ds.getncattr(index_attr)
        attr_int = _to_int_index(attr_raw, prefix, _is_cmip7(ds))

        if attr_int is None:
            ctx.add_failure(
                f"Could not interpret '{index_attr}' attribute value '{attr_raw}' as an integer."
            )
            return [ctx.to_result()]

        expected_from_variant = parsed_indices[index_attr]

        if expected_from_variant == attr_int:
            ctx.add_pass()
        else:
            ctx.add_failure(
                f"Inconsistency: variant_label implies {index_attr}='{expected_from_variant}', "
                f"but global attribute '{index_attr}' is '{attr_raw}'."
            )

    except AttributeError as e:
        ctx.add_failure(f"Missing required global attribute for this check: {e}")
    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]


# ==============================================================================
# Atomic ATTR006 checks
# ==============================================================================

def check_variant_label_vs_realization_index(ds, severity):
    return _check_variant_label_vs_index(
        ds,
        severity,
        index_attr="realization_index",
        prefix="r",
    )


def check_variant_label_vs_initialization_index(ds, severity):
    return _check_variant_label_vs_index(
        ds,
        severity,
        index_attr="initialization_index",
        prefix="i",
    )


def check_variant_label_vs_physics_index(ds, severity):
    return _check_variant_label_vs_index(
        ds,
        severity,
        index_attr="physics_index",
        prefix="p",
    )


def check_variant_label_vs_forcing_index(ds, severity):
    return _check_variant_label_vs_index(
        ds,
        severity,
        index_attr="forcing_index",
        prefix="f",
    )

