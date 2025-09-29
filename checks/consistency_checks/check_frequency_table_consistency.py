#!/usr/bin/env python



from compliance_checker.base import TestCtx

def check_frequency_table_id_consistency(ds, mapping, severity):
    """
    Checks if the global attributes 'frequency' and 'table_id' are consistent
    based on the mapping in TOML config
    """
    fixed_check_id = "ATTR008" 
    description = f"[{fixed_check_id}] Consistency: Frequency vs Table ID"
    ctx = TestCtx(severity, description)

    try:
        table_id = ds.getncattr("table_id")
        frequency = ds.getncattr("frequency")

        if table_id in mapping:
            allowed_frequencies = mapping[table_id]
            if frequency in allowed_frequencies:
                ctx.add_pass()
            else:
                msg = (f"Inconsistency found. For table_id '{table_id}', "
                       f"frequency should be one of {allowed_frequencies}, "
                       f"but found '{frequency}'.")
                ctx.add_failure(msg)
        else:
            
            # If the table_id is not in our mapping, we can't check it.
            ctx.messages.append(f"No frequency mapping found for table_id '{table_id}'. Check skipped.")
            ctx.add_pass()

    except AttributeError as e:
        ctx.add_failure(f"Missing required attribute for check: {e}")
    except Exception as e:
        ctx.add_failure(f"An unexpected error occurred: {e}")

    return [ctx.to_result()]
