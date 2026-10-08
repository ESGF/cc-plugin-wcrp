import re
from pathlib import Path

import pytest
import toml


ROOT = Path(__file__).parents[1]
# CMIP6Plus and CMIP7 obtain this rule from the ESGVoc time_unit collection;
# these are the project configurations that retain the local fallback regex.
CONFIG_PATHS = (
    ROOT / "plugins/cmip6/config/wcrp/global_attributes.toml",
    ROOT / "plugins/c3scmip6/c3scmip6/config/wcrp/global_attributes.toml",
)
CALENDARS = (
    "gregorian",
    "standard",
    "proleptic_gregorian",
    "julian",
    "noleap",
    "365_day",
    "all_leap",
    "366_day",
    "360_day",
    "utc",
    "tai",
)


def _parent_time_units_pattern(config_path):
    config = toml.load(config_path)
    return config["global"]["attributes"]["parent_time_units"]["pattern"]


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
@pytest.mark.parametrize("calendar", CALENDARS)
def test_parent_time_units_pattern_accepts_parenthesized_calendars(
    config_path,
    calendar,
):
    pattern = _parent_time_units_pattern(config_path)

    assert re.fullmatch(
        pattern,
        f"days since 1850-01-01 00:00:00 ({calendar})",
    )


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_parent_time_units_pattern_retains_plain_units(config_path):
    pattern = _parent_time_units_pattern(config_path)

    assert re.fullmatch(pattern, "days since 1850-01-01")
    assert re.fullmatch(pattern, "days since 1850-01-01 00:00:00")


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_parent_time_units_pattern_rejects_unknown_calendar(config_path):
    pattern = _parent_time_units_pattern(config_path)

    assert not re.fullmatch(
        pattern,
        "days since 1850-01-01 00:00:00 (unknown_calendar)",
    )

