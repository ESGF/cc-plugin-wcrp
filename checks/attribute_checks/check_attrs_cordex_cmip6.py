#!/usr/bin/env python


import re

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx
from compliance_checker.cf import util as cfutil

from checks.utils import severity_word

# Compliance Checker 6 moved these helpers from compliance_checker.cfutil.
if not hasattr(cfutil, "get_geophysical_variables"):
    from compliance_checker import cfutil


def check_grid_mapping(
    CheckerObject,
    severity=BaseCheck.MEDIUM,
    missing_severity=BaseCheck.MEDIUM,
    horizontal_topology=None,
    topology_error=None,
):
    """
    Checks if the grid_mapping label is compliant with the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of invalid grid metadata. Default: BaseCheck.MEDIUM.
    missing_severity : str
        The severity of recommending Earth-size metadata when grid_mapping is absent.
        Default: BaseCheck.MEDIUM.
    horizontal_topology : str, optional
        Independently inferred topology, used only to decide whether omission is
        permitted for a rectilinear latitude-longitude grid.
    topology_error : str, optional
        Reason topology could not be inferred. COORD011 owns that failure, so it
        is not duplicated here.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] grid_mapping"
    testctx = TestCtx(severity, desc)
    missingctx = TestCtx(
        missing_severity,
        f"[{check_id}] grid_mapping Earth description",
    )

    omission_marker = "(no grid_mapping)"
    grid_description = str(getattr(CheckerObject.ds, "grid", "") or "").lower()
    ocean_omission = omission_marker in grid_description
    # One of the following attributes needs to be specified for the grid_mapping variable
    # assuming that means that the Earth is specified/described as requested
    # (the checking of the validity of the description is left to CF checks)
    gmoptattrs = ["earth_radius", "semi_major_axis"]

    if CheckerObject.varname:
        variable_name = CheckerObject.varname[0]
        if variable_name not in CheckerObject.ds.variables:
            testctx.add_failure(
                f"The geophysical variable {variable_name!r} is absent, so its "
                "grid_mapping attribute could not be checked."
            )
            return [testctx.to_result()]
        crs = getattr(CheckerObject.ds.variables[variable_name], "grid_mapping", "")
        if crs and crs not in CheckerObject.ds.variables:
            testctx.add_failure(
                f"The grid_mapping attribute names {crs!r}, but that variable is "
                "absent from the file."
            )
        elif crs:
            mapping = CheckerObject.ds.variables[crs]
            grid_mapping_name = getattr(mapping, "grid_mapping_name", "")
            # Check grid_mapping label
            if grid_mapping_name and crs in ["crs", grid_mapping_name]:
                testctx.add_pass()
            else:
                testctx.add_failure(
                    f"The grid_mapping label '{crs}' needs to be either 'crs'"
                    " or equal to the grid_mapping_name (eg. 'rotated_latitude_longitude')."
                )
            # CF owns validation of the grid-mapping name itself. CORDEX only
            # requires the attribute here; topology resolution is handled by
            # the coordinate checks using their separate mapping configuration.
            if grid_mapping_name:
                testctx.add_pass()
            else:
                testctx.add_failure(
                    f"The grid_mapping variable '{crs}' does not define a "
                    "grid_mapping_name. Valid names and their required "
                    "attributes are checked by the CF checker."
                )
            # Check presence of description of spherical / ellipsoid Earth
            # - leave actual checking of the validity of that info to CF
            if any(getattr(mapping, attr, False) for attr in gmoptattrs):
                testctx.add_pass()
            else:
                testctx.add_failure(
                    f"The grid_mapping variable '{crs}' needs to include information regarding"
                    " the shape and size of the Earth used for the model grid. See 'CF-1.11 Appendix F'"
                    " of the CF-Conventions for further information."
                )
            # Check data type of grid_mapping variable (int or char)
            if mapping.dtype == np.int32 or mapping.dtype.kind == "S":
                testctx.add_pass()
            else:
                testctx.add_failure(
                    f"The grid_mapping variable '{crs}' needs to be of type 'int' or 'char', "
                    f"but is of type '{mapping.dtype} ({mapping.dtype.kind})'."
                )
        else:
            missingctx.add_failure(
                "No grid_mapping variable was found. It is "
                f"{severity_word(missing_severity)} to define one with information "
                "about the shape and size of the Earth used for the model grid, "
                "even for latitude-longitude grids (e.g., regular grids and "
                "curvilinear ocean grids)."
            )
            if horizontal_topology == "rectilinear" or ocean_omission:
                testctx.add_pass()
            elif topology_error:
                # COORD011 owns the topology failure; do not repeat its cause here.
                testctx.add_pass()
            else:
                testctx.add_failure(
                    "No grid_mapping variable describing the coordinate reference "
                    "system was found. It may be omitted for a rectilinear "
                    "latitude-longitude grid, or for a supported ocean grid when "
                    f"the global 'grid' attribute contains {omission_marker!r}."
                )

    else:
        testctx.add_pass()

    results = [testctx.to_result()]
    if missingctx.messages:
        results.append(missingctx.to_result())
    return results


def check_domain_id(CheckerObject, severity=BaseCheck.MEDIUM, use_esgvoc=False):
    """
    Checks if the domain_id is compliant with the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.
    use_esgvoc : bool
        If True, skip the parts of the check that rely on the CORDEX-CMIP6 CV tables.

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
    latitudes = cfutil.get_true_latitude_variables(CheckerObject.ds)
    longitudes = cfutil.get_true_longitude_variables(CheckerObject.ds)
    if not latitudes or not longitudes:
        testctx.add_failure(
            "Cannot check 'domain_id' as latitude and longitude coordinate variables could not be identified."
        )
        return [testctx.to_result()]
    lat = latitudes[0]
    lon = longitudes[0]

    # If the domain_id ends in "i" (interpolated grid), we expect 1D lat and lon coordinates
    if domain_id.endswith("i"):
        if (
            CheckerObject.ds.variables[lat].ndim != 1
            or CheckerObject.ds.variables[lon].ndim != 1
        ):
            testctx.add_failure(
                "The global attribute 'domain_id' indicates an interpolated grid (suffix 'i'), "
                f"which requires 1D latitude and longitude coordinate variables. "
                f"Found lat ndim={CheckerObject.ds.variables[lat].ndim}, "
                f"lon ndim={CheckerObject.ds.variables[lon].ndim}."
            )
        else:
            testctx.add_pass()
    else:
        # Non-i domains are allowed to have 1D or 2D lat/lons.
        testctx.add_pass()

    # Do not run comparison against CV if esgvoc is used
    if use_esgvoc:
        testctx.add_pass()
        return [testctx.to_result()]

    # Check if domain_id is in the CV
    if domain_id not in CheckerObject.CV["domain_id"]:
        testctx.add_failure(
            f"The global attribute 'domain_id' is not compliant with the CV: '{domain_id}'."
        )
    else:
        testctx.add_pass()

    return [testctx.to_result()]


def check_institution(CheckerObject, severity=BaseCheck.MEDIUM, use_esgvoc=False):
    """
    Checks if the institution is compliant with the CV.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.
    use_esgvoc : bool
        If True, skip the parts of the check that rely on the CORDEX-CMIP6 CV tables.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    check_id = "CDXA001"
    desc = f"[{check_id}] institution"
    testctx = TestCtx(severity, desc)

    # Do not run comparison against CV if esgvoc is used
    if use_esgvoc:
        testctx.add_pass()
        return [testctx.to_result()]

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

    version_realization = str(
        CheckerObject._get_attr("version_realization", default="") or ""
    ).strip()
    version_realization_info = str(
        CheckerObject._get_attr("version_realization_info", default="") or ""
    ).strip()
    if (
        version_realization
        and version_realization != "v1-r1"
        and not version_realization_info
    ):
        testctx.add_failure(
            "The global attribute 'version_realization_info' is missing. It is "
            f"{severity_word(severity)} when 'version_realization' differs from "
            "'v1-r1', and should describe why a new version was produced or how "
            "the realization differs from 'v1-r1'."
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


def check_version_realization(
    CheckerObject, severity=BaseCheck.MEDIUM, use_esgvoc=False
):
    """
    Checks if version_realization is defined as required in the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.
    use_esgvoc : bool
        If True, skip the parts of the check that rely on the CORDEX-CMIP6 CV tables.

    Returns
    -------
    List of compliance_checker.base.Result
    """
    # This check is required as the version_realization pattern is not defined in the CORDEX-CMIP6 CV.
    #  The existence of the attributes will be checked elsewhere, this only checks the correct values, if defined.
    check_id = "CDXA001"
    desc = f"[{check_id}] version_realization"
    testctx = TestCtx(severity, desc)

    # Do not run comparison with CV if esgvoc is used
    if use_esgvoc:
        testctx.add_pass()
        return [testctx.to_result()]

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


def check_driving_attributes(
    CheckerObject, severity=BaseCheck.MEDIUM, use_esgvoc=False
):
    """
    Checks if all driving attributes are defined as required by the CORDEX-CMIP6 archive specifications.

    Parameters
    ----------
    CheckerObject : WCRPBaseCheck object
        The initialized WCRPBaseCheck object for the project/dataset being checked.
    severity : str
        The severity of the check. Default: BaseCheck.MEDIUM.
    use_esgvoc : bool
        If True, skip the parts of the check that rely on the CORDEX-CMIP6 CV tables.

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
        # Do not run comparison with CV if esgvoc is used
        if use_esgvoc:
            testctx.add_pass()
            return [testctx.to_result()]
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
