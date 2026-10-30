"""Unit tests for configurable NetCDF compression recommendations."""

from types import SimpleNamespace

from compliance_checker.base import BaseCheck

from checks.format_checks.check_compression import check_compression
from tests.wcrp_cmip6_test_checks.conftest import result_passed


class _Variable:
    def __init__(self, filters):
        self._filters = filters

    def filters(self):
        return self._filters


class _Dataset:
    def __init__(self, variables, dimensions=()):
        self.variables = variables
        self.dimensions = {name: SimpleNamespace() for name in dimensions}

    def __getitem__(self, name):
        return self.variables[name]


def test_compression_selects_first_non_dimension_variable_and_passes():
    dataset = _Dataset(
        {"time": _Variable({}), "tas": _Variable({"complevel": 1, "shuffle": True})},
        dimensions=("time",),
    )
    result = check_compression(dataset)[0]
    assert result_passed(result)


def test_compression_handles_empty_and_uncompressed_datasets():
    result = check_compression(_Dataset({}))[0]
    assert result_passed(result)
    assert result.msgs == []

    result = check_compression(_Dataset({"tas": _Variable(None)}), variable_name="tas")[
        0
    ]
    assert not result_passed(result)
    assert "appears uncompressed" in result.msgs[0]


def test_compression_reports_level_and_shuffle_mismatches_at_configured_severity():
    too_high = _Dataset({"tas": _Variable({"complevel": 4, "shuffle": False})})
    result = check_compression(
        too_high,
        variable_name="tas",
        expected_complevel=1,
        severity=BaseCheck.HIGH,
    )[0]
    assert result.weight == BaseCheck.HIGH
    assert not result_passed(result)
    assert "higher 'deflate level': '4'" in result.msgs[0]
    assert "shuffle' option is disabled" in result.msgs[0]

    uncompressed = _Dataset({"tas": _Variable({"complevel": 0, "shuffle": True})})
    result = check_compression(uncompressed, variable_name="tas")[0]
    assert "data is uncompressed" in result.msgs[0]
