"""Tests that cf-xarray still provides the vocabulary we depend on.

The coordinate classifier uses cf-xarray's criteria tables to recognize
coordinate names (rlon/rlat as grid axes, lev/depth as vertical, and so
on). If a future cf-xarray version changes or removes these entries, the
classifier would quietly stop recognizing those coordinates.
"""
import pytest

from cf_xarray.criteria import coordinate_criteria, regex

from plugins import coordinate_standard as cs
from plugins.coordinate_standard import Role, VERTICAL_STANDARD_NAMES


@pytest.mark.parametrize("key, member", [
    ("time", "time"), ("T", "time"),
    ("X", "rlon"), ("X", "x"), ("X", "nlon"),
    ("Y", "rlat"), ("Y", "y"), ("Y", "nlat"),
    ("Z", "lev"), ("Z", "depth"), ("Z", "sigma"),
    ("longitude", "lon"), ("latitude", "lat"),
])
def test_cf_xarray_name_regex(key, member):
    """The regex table still has the key, and it still matches the spellings
    the weak-signal classifier counts on."""
    assert key in regex, f"cf_xarray.criteria.regex lost key {key!r}"
    assert regex[key].fullmatch(member), (
        f"cf_xarray regex[{key!r}] no longer matches {member!r}; "
        "the upstream vocabulary changed.")


def test_vertical_standard_names():
    """The vertical criteria still exist and carry the CMOR-relevant names;
    model_level_number stays as the local CMOR addition."""
    upstream = coordinate_criteria["vertical"]["standard_name"]
    for sn in ("air_pressure", "depth", "height", "altitude",
               "geopotential_height"):
        assert sn in upstream, f"cf-xarray vertical criteria lost {sn!r}"
    assert "model_level_number" in VERTICAL_STANDARD_NAMES


@pytest.mark.parametrize("name, units, dtype, expected", [
    ("rlon", "degrees", "double", Role.GRID_X),
    ("rlat", "degrees", "double", Role.GRID_Y),
    ("x", "m", "double", Role.GRID_X),
    ("lev", "", "double", Role.VERTICAL),
    ("lon", "", "double", Role.LONGITUDE),
    ("lat", "", "double", Role.LATITUDE),
    ("time", "", "double", Role.TIME),
    # integer/character variables are never name-guessed
    ("i", "1", "integer", None),
    ("j", "1", "integer", None),
    ("basin", "", "character", None),
])
def test_weak_signal_roles(name, units, dtype, expected):
    """End-to-end: the classifier reaches the intended role via the cf-xarray
    tables (or abstains for index/label axes)."""
    assert cs._role_from_weak_signals(units, name, "", dtype=dtype) is expected
