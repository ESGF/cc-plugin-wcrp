#!/usr/bin/env python

import re
from compliance_checker.base import TestCtx

def check_variant_label_consistency(ds, severity):
    """
    [ATTR009] Checks if the variant_label attribute is consistent with the individual
    r,i,p,f index attributes.
    
    """
    fixed_check_id = "ATTR006"
    description = f"[{fixed_check_id}] Consistency: variant_label vs index attributes"
    ctx = TestCtx(severity, description)

    # List of attributes required for this check
    required_attrs = [
        "variant_label",
        "realization_index",
        "initialization_index",
        "physics_index",
        "forcing_index"
    ]

    try:
        # ---  Read all required attributes from the NetCDF file ---
        attributes = {attr: ds.getncattr(attr) for attr in required_attrs}
        variant_label = attributes["variant_label"]

        # ---  Parse the variant_label string using regex ---
        match = re.match(r"r(\d+)i(\d+)p(\d+)f(\d+)", variant_label)
        
        if not match:
            ctx.add_failure(f"The format of 'variant_label' ('{variant_label}') is invalid. Expected format is 'r<k>i<l>p<m>f<n>'.")
            return [ctx.to_result()]

        # Extract integer values from the parsed string
        parsed_indices = {
            "realization_index": int(match.group(1)),
            "initialization_index": int(match.group(2)),
            "physics_index": int(match.group(3)),
            "forcing_index": int(match.group(4))
        }

        # --- Compare parsed values with attribute values ---
        failures = []
        for key, parsed_value in parsed_indices.items():
            attr_value = attributes[key]
            if parsed_value != attr_value:
                failures.append(f"Inconsistency for '{key}': variant_label implies '{parsed_value}', but attribute is '{attr_value}'.")

        if not failures:
            ctx.add_pass()
        else:
            for f in failures:
                ctx.add_failure(f)

    except AttributeError as e:
        # If any required attribute is missing, we skip the check.
        # Its existence should be caught by the attribute_suite check.
        ctx.messages.append(f"Missing a required attribute for this check: {e}. Check skipped.")
    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]
