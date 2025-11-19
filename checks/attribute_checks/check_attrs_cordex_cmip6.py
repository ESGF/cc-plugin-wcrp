#!/usr/bin/env python


import re

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx

from checks.utils import severity_word


def check_grid_mapping(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if the grid_mapping label is compliant with the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] grid_mapping"
    testctx = TestCtx(severity, desc)

    # The allowed grid_mapping_name attribute values in CORDEX-CMIP6
    gmallowed = ["lambert_conformal_conic", "rotated_latitude_longitude"]
    # One of the following attributes needs to be specified for the grid_mapping variable
    # assuming that means that the Earth is specified/described as requested
    # (the checking of the validity of the description is left to CF checks)
    gmoptattrs = ["earth_radius", "semi_major_axis"]

    grid_mapping_name = False
    if len(CheckerObject.varname) > 0:
        crs = getattr(
            CheckerObject.ds.variables[CheckerObject.varname[0]], "grid_mapping", False
        )
        if crs:
            grid_mapping_name = getattr(
                CheckerObject.ds.variables[crs], "grid_mapping_name", False
            )
            # Check grid_mapping label
            if grid_mapping_name and crs in ["crs", grid_mapping_name]:
                testctx.add_pass()
            else:
                testctx.add_failure(
                    f"The grid_mapping label '{crs}' needs to be either 'crs'"
                    " or equal to the grid_mapping_name (eg. 'rotated_latitude_longitude')."
                )
            # Check grid_mapping_name
            if grid_mapping_name and grid_mapping_name in gmallowed:
                testctx.add_pass()
            else:
                testctx.add_failure(
                    f"The grid_mapping_name '{grid_mapping_name}' must be one of:"
                    f""" {", ".join(["'" + gm  + "'" for gm in gmallowed])}."""
                )
            # Check presence of description of spherical / ellipsoid Earth
            # - leave actual checking of the validity of that info to CF
            if any(
                [
                    getattr(CheckerObject.ds.variables[crs], attr, False)
                    for attr in gmoptattrs
                ]
            ):
                testctx.add_pass()
            else:
                testctx.add_failure(
                    f"The grid_mapping variable '{crs}' needs to include information regarding"
                    " the shape and size of the Earth used for the model grid. See 'CF-1.11 Appendix F'"
                    " of the CF-Conventions for further information."
                )
            # Check data type of grid_mapping variable (int or char)
            if (
                CheckerObject.ds[crs].dtype == np.int32
                or CheckerObject.ds[crs].dtype.kind == "S"
            ):
                testctx.add_pass()
            else:
                testctx.add_failure(
                    f"The grid_mapping variable '{crs}' needs to be of type 'int' or 'char', "
                    f"but is of type '{CheckerObject.ds[crs].dtype} ({CheckerObject.ds[crs].dtype.kind})'."
                )
        else:
            testctx.add_failure(
                f"The grid_mapping variable '{crs}', describing the coordinate reference system,"
                " could not be found in the file."
            )

    else:
        testctx.add_pass()

    # rlat, rlon or y, x must be present in file, depending on the grid_mapping_name
    if grid_mapping_name and grid_mapping_name in gmallowed:
        if grid_mapping_name == "lambert_conformal_conic":
            if "y" not in CheckerObject.xrds or "x" not in CheckerObject.xrds:
                testctx.add_failure(
                    "The grid_mapping_name 'lambert_conformal_conic' requires the variables"
                    " 'y' and 'x' to be present in the file defining the native coordinate"
                    " system used by the RCM."
                )
            else:
                testctx.add_pass()
        else:
            if "rlat" not in CheckerObject.xrds or "rlon" not in CheckerObject.xrds:
                testctx.add_failure(
                    "The grid_mapping_name 'rotated_latitude_longitude' requires the variables"
                    " 'rlat' and 'rlon' to be present in the file defining the native"
                    " coordinate system used by the RCM."
                )
            else:
                testctx.add_pass()
    else:
        if ("rlat" in CheckerObject.xrds and "rlon" in CheckerObject.xrds) or (
            "y" in CheckerObject.xrds and "x" in CheckerObject.xrds
        ):
            testctx.add_pass()
        else:
            testctx.add_failure(
                "The variables 'rlat', 'rlon' or 'y' and 'x' need to be present in the"
                " file defining the native coordinate system used by the RCM."
            )
    return [testctx.to_result()]


def check_domain_id(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if the domain_id is compliant with the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] domain_id"
    testctx = TestCtx(severity, desc)

    # Get domain_id from global attributes
    domain_id = CheckerObject._get_attr("domain_id", default=False)

    # Do not give a result if not defined
    if not domain_id:
        testctx.add_pass()
        return [testctx.to_result()]

    # If the grid is rectilinear, the domain_id needs to include the suffix "i"
    try:
        lat = CheckerObject.xrds.cf.coordinates["latitude"][0]
        lon = CheckerObject.xrds.cf.coordinates["longitude"][0]
    except KeyError:
        testctx.add_failure(
            "Cannot check 'domain_id' as latitude and longitude coordinate variables could not be identified."
        )
        return [testctx.to_result()]

    # Rectilinear case: lat and lon must be 1D
    #  (would also be the case for unstructured grids, but those are not allowed in CORDEX-CMIP6)
    if CheckerObject.xrds[lat].ndim == 1 and CheckerObject.xrds[lon].ndim == 1:
        if not domain_id.endswith("i"):
            testctx.add_failure(
                "The global attribute 'domain_id' is not compliant with the archive specifications "
                f"('domain_id' should get the suffix 'i' in case of a rectilinear grid): '{domain_id}'."
            )
        else:
            testctx.add_pass()
            domain_id = domain_id[:-1]
    else:
        testctx.add_pass()

    # Check if domain_id is in the CV
    if domain_id not in CheckerObject.CV["domain_id"]:
        testctx.add_failure(
            f"The global attribute 'domain_id' is not compliant with the CV: '{domain_id}'."
        )
    else:
        testctx.add_pass()

    return [testctx.to_result()]


def check_institution(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if the institution is compliant with the CV.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] institution"
    testctx = TestCtx(severity, desc)

    # Get institution from global attributes
    institution = CheckerObject._get_attr("institution", default=False)
    institution_id = CheckerObject._get_attr("institution_id", default=False)

    # If check cannot be conducted, rely on basic global attr. checks to raise the failure
    if (
        not institution_id
        or not institution
        or institution_id not in CheckerObject.CV["institution_id"]
    ):
        testctx.add_pass()
        return [testctx.to_result()]

    # Check institution against CV
    if institution != CheckerObject.CV["institution_id"][institution_id]:
        testctx.add_failure(
            f"The global attribute 'institution' is not compliant with the CV:"
            f" '{institution}' instead of '{CheckerObject.CV['institution_id'][institution_id]}'."
        )
    else:
        testctx.add_pass()

    return [testctx.to_result()]


def check_references(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if references is defined as recommended in the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] references"
    testctx = TestCtx(severity, desc)

    references = CheckerObject._get_attr("references", default=False)
    if not references:
        testctx.add_failure(
            f"The {severity_word(severity)} global attribute 'references' is not specified. "
            "It should include published or web-based references that describe "
            "the data, model or methods used."
        )
    else:
        testctx.add_pass()

    return [testctx.to_result()]


def check_version_realization_info(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if version_realization_info is defined when and as recommended in the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] version_realization_info"
    testctx = TestCtx(severity, desc)

    if (
        any(
            [
                x != "v1-r1"
                for x in [
                    CheckerObject.drs_fn["version_realization"],
                    CheckerObject.drs_dir["version_realization"],
                    CheckerObject.drs_gatts["version_realization"],
                ]
            ]
        )
        and CheckerObject._get_attr("version_realization_info", default="") == ""
    ):
        testctx.add_failure(
            f"The global attribute 'version_realization_info' is missing. It is however {severity_word(severity)}, "
            "if 'version_realization' deviates from 'v1-r1'. The attribute 'version_realization_info' "
            "provides information on how reruns (eg. v2, v3) and/or realizations (eg. r2, r3) are generated."
        )
    else:
        testctx.add_pass()

    return [testctx.to_result()]


def check_grid(CheckerObject, severity=BaseCheck.LOW):
    """
    Checks if the global attribute grid is defined as suggested in the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.LOW.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] grid"
    testctx = TestCtx(severity, desc)

    # Get grid from global attributes - if not defined, another check will throw the error
    grid = CheckerObject._get_attr("grid", default=False)
    if grid:
        # Check if grid description is following the examples
        if re.fullmatch(r"^.* with .* grid spacing.*$", grid):
            testctx.add_pass()
        else:
            testctx.add_failure(
                f"The global attribute 'grid' has no standard form, but it is {severity_word(severity)} to include a brief description "
                "of the native grid and resolution. If the data have been regridded, the regridding procedure and a "
                "description of the target grid should be provided as well. "
                "For example: 'Rotated-pole latitude-longitude with 0.22 degree grid spacing'. For a full set of"
                " examples, please have a look at the CORDEX-CMIP6 Archive Specifications."
            )
    else:
        testctx.add_pass()

    return [testctx.to_result()]


def check_version_realization(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if version_realization is defined as required in the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    # This check is required as the version_realization pattern is not defined in the CORDEX-CMIP6 CV.
    #  The existence of the attributes will be checked elsewhere, this only checks the correct values, if defined.
    check_id = "CDXA001"
    desc = f"[{check_id}] version_realization"
    testctx = TestCtx(severity, desc)

    # Filename
    expected = r"Expected pattern: 'v[1-9]\d*-r[1-9]\d*', eg. 'v1-r3'."
    if CheckerObject.drs_fn["version_realization"]:
        if not bool(
            re.fullmatch(
                r"v[1-9]\d*-r[1-9]\d*",
                CheckerObject.drs_fn["version_realization"],
                flags=re.ASCII,
            )
        ):
            testctx.add_failure(
                f"DRS filename building block 'version_realization' does not comply with the CORDEX-CMIP6 Archive Specifications: '{CheckerObject.drs_fn['version_realization']}'. {expected}"
            )
        else:
            testctx.add_pass()
    else:
        testctx.add_pass()

    # Folder structure
    if CheckerObject.drs_dir["version_realization"]:
        if not bool(
            re.fullmatch(
                r"v[1-9]\d*-r[1-9]\d*",
                CheckerObject.drs_dir["version_realization"],
                flags=re.ASCII,
            )
        ):
            testctx.add_failure(
                f"DRS path building block 'version_realization' does not comply with the CORDEX-CMIP6 Archive Specifications: '{CheckerObject.drs_dir['version_realization']}'. {expected}"
            )
        else:
            testctx.add_pass()
    else:
        testctx.add_pass()

    # Global attribute
    if CheckerObject.drs_gatts["version_realization"]:
        if not bool(
            re.fullmatch(
                r"v[1-9]\d*-r[1-9]\d*",
                CheckerObject.drs_gatts["version_realization"],
                flags=re.ASCII,
            )
        ):
            testctx.add_failure(
                f"Global attribute 'version_realization' does not comply with the CORDEX-CMIP6 Archive Specifications: '{CheckerObject.drs_gatts['version_realization']}'. {expected}"
            )
        else:
            testctx.add_pass()
    else:
        testctx.add_pass()

    return [testctx.to_result()]


def check_driving_attributes(CheckerObject, severity=BaseCheck.MEDIUM):
    """
    Checks if all driving attributes are defined as required by the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] Driving Attributes"
    testctx = TestCtx(severity, desc)

    dei = CheckerObject._get_attr("driving_experiment_id", False)
    dvl = CheckerObject._get_attr("driving_variant_label", False)
    dsi = CheckerObject._get_attr("driving_source_id", False)

    if dvl and dvl == "r0i0p0f0":
        testctx.add_failure(
            "The global attribute 'driving_variant_label' is not compliant with the CORDEX-CMIP6 Archive Specifications "
            f"('r1i1p1f1' is the minimum 'driving_variant_label'): '{dvl}'."
        )
    else:
        testctx.add_pass()

    if dei and dei == "evaluation":
        if dvl and dvl != "r1i1p1f1":
            testctx.add_failure(
                "The global attribute 'driving_variant_label' is not compliant with the CORDEX-CMIP6 Archive Specifications "
                f"('r1i1p1f1'): '{dei}'."
            )
        else:
            testctx.add_pass()
        if dsi and dsi != "ERA5":
            testctx.add_failure(
                "The global attribute 'driving_source_id' is not compliant with the CORDEX-CMIP6 Archive Specifications "
                f"('ERA5'): '{dei}'."
            )
        else:
            testctx.add_pass()
    else:
        testctx.add_pass()

    # According to the Archive Specifications, 'driving_source' is recommended, but not required.
    # If it is defined however, it is required to be compliant with the CV.
    drivs = CheckerObject._get_attr("driving_source", False)
    if not drivs:
        testctx.add_pass()
    else:
        # Abort if driving_source_id undefined or unknown (will cause a failed check elsewhere)
        if not dsi or dsi not in CheckerObject.CV["driving_source_id"]:
            testctx.add_pass()
        # Else compare with CV
        elif drivs != CheckerObject.CV["driving_source_id"][dsi]["driving_source"]:
            testctx.add_failure(
                "The global attribute 'driving_source' does not comply with the CV."
            )
        else:
            testctx.add_pass()

    return [testctx.to_result()]
