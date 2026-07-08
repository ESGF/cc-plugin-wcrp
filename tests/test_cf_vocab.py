"""Guard test: coordinate_standard._role_from_weak_signals relies on
cf-xarray's public criteria tables (``cf_xarray.criteria.coordinate_criteria``
and ``cf_xarray.criteria.regex``).

These are public API, so they should be stable -- but the weak-signal
classifier depends on specific keys and specific spellings inside them
(rlon/rlat in X/Y, lev/depth in Z, air_pressure in the vertical standard
names). If a cf-xarray release reshapes or renames any of these, this test
fails LOUDLY at CI time so we notice on a dependency bump and can react
deliberately, instead of silently losing classification coverage.

Historical note: this file previously guarded private ``_possiblet``-style
name sets inside compliance_checker.cf.util; that dependency was replaced by
cf-xarray's public tables, which is what is guarded now.
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
