"""CMIP7 expectations for the mini-esgf-data curvilinear ocean file."""

from pathlib import PurePosixPath

from tests.remote.model import (
    ExpectedCheck,
    ExpectedFailure,
    RemoteDataset,
    RemoteFile,
    RemoteProject,
)

_BASE_URL = "https://raw.githubusercontent.com/roocs/mini-esgf-data"
_REVISION = "master"

_TIME_SQUARENESS_FAILURE = ExpectedFailure(
    name="[TIME001] Check Time Squareness ",
    severity=3,
    messages=("Cannot parse filename time range start",),
)

_TIME_RANGE_FAILURE = ExpectedFailure(
    name="[TIME003] Check Time Range vs Filename",
    severity=3,
    messages=("No time range token found at the end of the filename",),
)

_FILENAME_FAILURE = ExpectedFailure(
    name="[FILE001] DRS Filename Vocabulary Check",
    severity=3,
    messages=("extra term 202201 invalidated by the optional collection time_range",),
)

_PATH_ATTRIBUTES_FAILURE = ExpectedFailure(
    name="[PATH001] Consistency: Directory Structure vs Global Attributes",
    severity=3,
    messages=(
        (
            "DRS path component 'variant_label' ('r1i1p1f1') does not match global "
            "attribute 'variant_label' ('r8i1p1f1'"
        ),
        (
            "DRS path component 'variable_id' ('tos') does not match global attribute "
            "'variable_id' ('zos'"
        ),
    ),
)

_PATH_FILENAME_FAILURE = ExpectedFailure(
    name="[PATH002] Consistency: Directory Structure vs Filename",
    severity=3,
    messages=(
        "Token 'variable_id' is inconsistent: path has 'tos', filename has 'zos'",
        (
            "Token 'variant_label' is inconsistent: path has 'r1i1p1f1', filename "
            "has 'r8i1p1f1'"
        ),
    ),
)

_COMMENT_FAILURE = ExpectedFailure(
    name="[ATTR004] Variable 'zos' attribute 'comment' registry expected-term check",
    severity=1,
    messages=("No variable 'zos' attribute 'comment' is defined",),
)


CURVILINEAR_OCEAN = RemoteDataset(
    id="zos_curvilinear_ocean",
    file=RemoteFile(
        repository="mini-esgf-data",
        base_url=_BASE_URL,
        revision=_REVISION,
        relative_path=PurePosixPath(
            "test_data/badc/MIP-DRS7/CMIP7/ScenarioMIP/UKNCSP/UKESM1-3-LL/"
            "esm-scen7-vl/r1i1p1f1/glb/mon/tos/tavg-u-hxy-sea/g126/v20260825/"
            "zos_tavg-u-hxy-sea_mon_glb_g126_UKESM1-3-LL_esm-scen7-vl_"
            "r8i1p1f1_202201.nc"
        ),
        sha256="1f9fea29bfaf817aa7217ba60d12ae05047478289f9795b3058afd47bf606aad",
    ),
    checks={
        "check_Coordinate_Metadata_Setup": ExpectedCheck(1),
        # The current development vocabulary permits -180..360 for grid
        # longitude and its vertices, so the real curvilinear grid passes.
        "check_Coordinate_Standard": ExpectedCheck(17),
        "check_Coordinates": ExpectedCheck(
            5, (_TIME_SQUARENESS_FAILURE, _TIME_RANGE_FAILURE)
        ),
        "check_DRS": ExpectedCheck(
            4,
            (
                _FILENAME_FAILURE,
                _PATH_ATTRIBUTES_FAILURE,
                _PATH_FILENAME_FAILURE,
            ),
        ),
        "check_File_Compression": ExpectedCheck(0),
        "check_File_Format": ExpectedCheck(1),
        "check_File_Internal_Packing_Data": ExpectedCheck(1),
        "check_File_Internal_Packing_Metadata": ExpectedCheck(1),
        "check_File_Internal_Packing_Time": ExpectedCheck(2),
        "check_Geophysical_Variable": ExpectedCheck(39, (_COMMENT_FAILURE,)),
        "check_Global_Attributes": ExpectedCheck(81),
        "check_Global_Consistency": ExpectedCheck(9),
        "check_consistency_output": ExpectedCheck(1),
        "check_setup_warnings": ExpectedCheck(1),
    },
)


CMIP7 = RemoteProject(
    id="cmip7",
    checker="wcrp_cmip7",
    datasets=(CURVILINEAR_OCEAN,),
)
