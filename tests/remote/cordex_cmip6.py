"""CORDEX-CMIP6 expectations for the py-cordex-data reference files."""

from pathlib import PurePosixPath

from tests.remote.model import (
    ExpectedCheck,
    ExpectedFailure,
    RemoteDataset,
    RemoteFile,
    RemoteProject,
)

_BASE_URL = "https://raw.githubusercontent.com/euro-cordex/py-cordex-data"
_REVISION = "91cf4b5b9a082c823c5f8e87555d4aaac25613fa"

_COMPRESSION_FAILURE = ExpectedFailure(
    name="[FILE003] Compression",
    severity=2,
    messages=(
        "It is recommended that data be compressed with a 'deflate level' of "
        "'1' and with the 'shuffle' option enabled. The 'shuffle' option is "
        "disabled.",
    ),
)

_HORIZONTAL_AXES_BOUNDS_FAILURE = ExpectedFailure(
    name="[CDXV002] Existence of horizontal axes bounds",
    severity=2,
    messages=(
        "It is recommended for the variables 'rlat' and 'rlon' or 'x' and "
        "'y' to have bounds defined.",
    ),
)


_GRID_MAPPING_FAILURE = ExpectedFailure(
    name="[CDXA001] grid_mapping",
    severity=3,
    messages=(
        "The grid_mapping variable 'rotated_latitude_longitude' needs to "
        "include information regarding the shape and size of the Earth used "
        "for the model grid.",
    ),
)


def _checks(
    *,
    coordinate_results: int,
    standard_results: int,
    variable_results: int,
    variable_name: str,
):
    """Return a complete independent expectation mapping for one dataset."""
    return {
        "check_Coordinate_Metadata_Setup": ExpectedCheck(1),
        "check_Coordinate_Standard": ExpectedCheck(standard_results),
        "check_Coordinates": ExpectedCheck(coordinate_results),
        "check_DRS": ExpectedCheck(4),
        "check_File_Compression": ExpectedCheck(1, (_COMPRESSION_FAILURE,)),
        "check_File_Format": ExpectedCheck(1),
        "check_Geophysical_Variable": ExpectedCheck(
            variable_results,
            (
                ExpectedFailure(
                    name=(
                        f"[ATTR004] Variable '{variable_name}' attribute "
                        "'comment' registry expected-term check"
                    ),
                    severity=1,
                    messages=(
                        f"No variable '{variable_name}' attribute 'comment' is "
                        "defined",
                    ),
                ),
            ),
        ),
        "check_Global_Attributes": ExpectedCheck(58),
        "check_Global_Consistency": ExpectedCheck(4),
        "check_attributes_cordex": ExpectedCheck(8, (_GRID_MAPPING_FAILURE,)),
        "check_calendar": ExpectedCheck(1),
        "check_consistency_output": ExpectedCheck(1),
        "check_data_types": ExpectedCheck(1),
        "check_horizontal_axes_bounds": ExpectedCheck(
            1, (_HORIZONTAL_AXES_BOUNDS_FAILURE,)
        ),
        "check_lat_lon_bounds": ExpectedCheck(1),
        "check_lon_value_range": ExpectedCheck(1),
        "check_setup_warnings": ExpectedCheck(1),
        "check_time_chunking": ExpectedCheck(1),
        "check_time_units": ExpectedCheck(1),
    }


TAS_REMO = RemoteDataset(
    id="tas_remo",
    file=RemoteFile(
        repository="py-cordex-data",
        base_url=_BASE_URL,
        revision=_REVISION,
        relative_path=PurePosixPath(
            "CORDEX-CMIP6/DD/EUR-12/GERICS/ERA5/evaluation/r1i1p1f1/"
            "REMO2020-2-2/v1-r1/mon/tas/v20241120/"
            "tas_EUR-12_ERA5_evaluation_r1i1p1f1_GERICS_"
            "REMO2020-2-2_v1-r1_mon_201901-202012.nc"
        ),
        sha256="e2605fc7d28ea2396880df1e41e221114bd1b465ec0d9de83c4af9c6026de047",
    ),
    checks=_checks(
        coordinate_results=5,
        standard_results=14,
        variable_results=43,
        variable_name="tas",
    ),
)

OROG_REMO = RemoteDataset(
    id="orog_remo",
    file=RemoteFile(
        repository="py-cordex-data",
        base_url=_BASE_URL,
        revision=_REVISION,
        relative_path=PurePosixPath(
            "CORDEX-CMIP6/DD/EUR-12/GERICS/ERA5/evaluation/r1i1p1f1/"
            "REMO2020-2-2/v1-r1/fx/orog/v20241120/"
            "orog_EUR-12_ERA5_evaluation_r1i1p1f1_GERICS_"
            "REMO2020-2-2_v1-r1_fx.nc"
        ),
        sha256="29e5da1eb8a6a86853b6f8f981f9480b509472993e984bc51f0553de24238418",
    ),
    checks=_checks(
        coordinate_results=1,
        standard_results=13,
        variable_results=37,
        variable_name="orog",
    ),
)

CORDEX_CMIP6 = RemoteProject(
    id="cordex_cmip6",
    checker="wcrp_cordex_cmip6",
    datasets=(TAS_REMO, OROG_REMO),
)
