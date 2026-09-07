#!/usr/bin/env python

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx

from checks.utils import dtypesdict


def check_coord_data_types(
    CheckerObject, ctype=np.float64, auxtype=np.float32, severity=BaseCheck.MEDIUM
):
    """
    Checks if the variable and coordinate data types are correct.

    This check ensures that all coordinate variables are of the expected data type.
    The expected type for auxiliary coordinates (eg. ps, orog for vertical axes) can differ
    from the expected type for coordinates.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    ctype : numpy.dtype
        The expected data type for coordinate variables. Default: np.float64.
    auxtype : numpy.dtype
        The expected data type for auxiliary coordinate variables. Default: np.float32.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "VAR011"
    desc = f"[{check_id}] Data Type"
    testctx = TestCtx(severity, desc)
    failure_registered = False

    # Verify arguments
    if ctype not in dtypesdict or auxtype not in dtypesdict:
        testctx.add_failure(
            "Invalid checker configuration. At least one of the requested data types is not "
            f"supported ({ctype}, {auxtype}). Supported types: {', '.join(dtypesdict.keys())}"
        )
        return [testctx.to_result()]

    # Verify coordinate data types
    for c in set(CheckerObject.coords) | set(CheckerObject.bounds):
        if c not in CheckerObject.ds.variables:
            continue
        coord = CheckerObject.ds.variables[c]
        if (
            ctype == "character"
            and coord.dtype.kind != np.dtype(dtypesdict[ctype]).kind
        ) or coord.dtype != dtypesdict[ctype]:
            if len(CheckerObject.varname) > 0 and c == getattr(
                CheckerObject.ds.variables[CheckerObject.varname[0]],
                "grid_mapping",
                None,
            ):
                pass
            elif c in {
                value
                for terms in CheckerObject.formula_terms.values()
                for key, value in terms.items()
                if key in ("ps", "orog")
            }:
                pass
            else:
                try:
                    testctx.add_failure(
                        f"The coordinate variable '{c}' must be of the data type '{dtypesdict[ctype].__name__}', is of type '{coord.dtype}'."
                    )
                except AttributeError:
                    testctx.add_failure(
                        f"The coordinate variable '{c}' must be of the data type '{dtypesdict[ctype]}', is of type '{coord.dtype}'."
                    )
                failure_registered = True
        elif (
            c
            in {
                value
                for terms in CheckerObject.formula_terms.values()
                for key, value in terms.items()
                if key in ("ps", "orog")
            }
            and coord.dtype != dtypesdict[auxtype]
            or (
                auxtype == "character"
                and coord.dtype.kind != np.dtype(dtypesdict[auxtype]).kind
            )
        ):
            try:
                testctx.add_failure(
                    f"The auxiliary coordinate variable '{c}' must be of the data type '{dtypesdict[auxtype].__name__}', is of type '{coord.dtype}'."
                )
            except AttributeError:
                testctx.add_failure(
                    f"The auxiliary coordinate variable '{c}' must be of the data type '{dtypesdict[auxtype]}', is of type '{coord.dtype}'."
                )
            failure_registered = True

    if not failure_registered:
        testctx.add_pass()

    return [testctx.to_result()]


def check_var_data_type(
    CheckerObject, var="main", vartype=np.float32, severity=BaseCheck.MEDIUM
):
    """
    Checks data type of a specified variable.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    var : str
        The name of the variable to check. Default: "main", i.e. the main variable is being inferred and checked.
    vartype : numpy.dtype
        The expected data type for the main variable. Default: np.float32.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "VAR011"
    desc = f"[{check_id}] Data Type"
    testctx = TestCtx(severity, desc)

    # Verify arguments
    if vartype not in dtypesdict:
        testctx.add_failure(
            "Invalid checker configuration. The requested data type '{vartype}' is "
            f"not supported. Supported types: {', '.join(dtypesdict.keys())}"
        )
        return [testctx.to_result()]
    if var == "main" or var is None:
        if len(CheckerObject.varname) > 0:
            var = CheckerObject.varname[0]
        else:
            testctx.add_failure("No main variable found.")
            return [testctx.to_result()]
    elif var not in CheckerObject.ds.variables:
        testctx.add_failure(f"Variable '{var}' not found.")
        return [testctx.to_result()]

    # Verify variable data type
    variable = CheckerObject.ds.variables[var]
    if variable.dtype != dtypesdict[vartype] or (
        vartype == "character"
        and variable.dtype.kind != np.dtype(dtypesdict[vartype]).kind
    ):
        try:
            testctx.add_failure(
                f"The variable '{var}' must be of data type '{dtypesdict[vartype].__name__}', is of type '{variable.dtype.name}'."
            )
        except AttributeError:
            testctx.add_failure(
                f"The variable '{var}' must be of data type '{dtypesdict[vartype]}', is of type '{variable.dtype}'."
            )
    else:
        testctx.add_pass()

    return [testctx.to_result()]
