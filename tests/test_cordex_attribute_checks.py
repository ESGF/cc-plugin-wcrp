"""Unit tests for CORDEX-CMIP6-specific metadata policy."""

from types import SimpleNamespace

import pytest

from checks.attribute_checks import check_attrs_cordex_cmip6 as checks
from checks.variable_checks import check_coords_cordex_cmip6 as coordinate_checks
from tests.wcrp_cmip6_test_checks.conftest import result_passed


class _Checker:
    def __init__(self, attrs=None, **overrides):
        self.attrs = attrs or {}
        self.ds = SimpleNamespace(variables={})
        self.CV = {
            "institution_id": {"INST": "Expected Institution"},
            "domain_id": {"EUR-11": {}},
            "driving_source_id": {
                "MODEL": {"driving_source": "Expected driving source"}
            },
        }
        self.drs_fn = {"version_realization": "v1-r1"}
        self.drs_dir = {"version_realization": "v1-r1"}
        self.drs_gatts = {"version_realization": "v1-r1"}
        self.__dict__.update(overrides)

    def _get_attr(self, name, default=None):
        return self.attrs.get(name, default)


def test_cordex_institution_and_references_policy():
    checker = _Checker({"institution_id": "INST", "institution": "Wrong Institution"})
    assert not result_passed(checks.check_institution(checker)[0])
    assert result_passed(checks.check_institution(checker, use_esgvoc=True)[0])

    assert not result_passed(checks.check_references(checker)[0])
    checker.attrs["references"] = "https://example.org/model"
    assert result_passed(checks.check_references(checker)[0])


@pytest.mark.parametrize(
    ("version", "info", "passes"),
    [("v1-r1", "", True), ("v2-r1", "why rerun", True), ("v2-r1", "", False)],
)
def test_cordex_version_realization_info_policy(version, info, passes):
    checker = _Checker(
        {"version_realization": version, "version_realization_info": info}
    )
    result = checks.check_version_realization_info(checker)[0]
    assert result_passed(result) is passes


@pytest.mark.parametrize(
    ("description", "passes"),
    [
        ("Rotated-pole latitude-longitude with 0.22 degree grid spacing", True),
        ("a grid", False),
        (None, True),
    ],
)
def test_cordex_grid_description_policy(description, passes):
    attrs = {} if description is None else {"grid": description}
    results = checks.check_grid(_Checker(attrs))
    if description is None:
        assert results == []
    else:
        assert result_passed(results[0]) is passes


def test_grid_mapping_reports_unowned_topology_error():
    data = SimpleNamespace()
    checker = _Checker(varname=["tas"])
    checker.ds.variables = {"tas": data}

    delegated = checks.check_grid_mapping(
        checker,
        horizontal_topology=None,
        topology_error="latitude and longitude could not be identified",
        topology_error_delegated=True,
    )
    fallback = checks.check_grid_mapping(
        checker,
        horizontal_topology=None,
        topology_error="latitude and longitude could not be identified",
        topology_error_delegated=False,
    )

    assert len(delegated) == 1
    assert "shape and size of the Earth" in delegated[0].msgs[0]
    assert not result_passed(fallback[0])
    assert "topology could not be established" in fallback[0].msgs[0]


def test_cordex_version_realization_validates_all_three_locations():
    checker = _Checker(
        drs_fn={"version_realization": "v20-r10112"},
        drs_dir={"version_realization": "v0-r1"},
        drs_gatts={"version_realization": "v1-r0"},
    )
    result = checks.check_version_realization(checker)[0]
    assert not result_passed(result)
    assert len(result.msgs) == 2
    assert "DRS path" in result.msgs[0]
    assert "Global attribute" in result.msgs[1]
    assert result_passed(checks.check_version_realization(checker, use_esgvoc=True)[0])


def test_cordex_driving_attributes_cover_evaluation_and_optional_source():
    checker = _Checker(
        {
            "driving_experiment_id": "evaluation",
            "driving_variant_label": "r0i0p0f0",
            "driving_source_id": "OTHER",
        }
    )
    result = checks.check_driving_attributes(checker)[0]
    assert not result_passed(result)
    assert len(result.msgs) == 3

    checker = _Checker(
        {
            "driving_experiment_id": "historical",
            "driving_variant_label": "r1i1p1f1",
            "driving_source_id": "MODEL",
            "driving_source": "Wrong source",
        }
    )
    result = checks.check_driving_attributes(checker)[0]
    assert not result_passed(result)
    assert "does not comply with the CV" in result.msgs[0]


def test_cordex_domain_id_checks_discovery_interpolation_and_legacy_cv(monkeypatch):
    one_dimensional = SimpleNamespace(ndim=1)
    checker = _Checker({"domain_id": "EUR-11i"})
    checker.ds.variables = {"lat": one_dimensional, "lon": one_dimensional}
    monkeypatch.setattr(
        checks.cfutil, "get_true_latitude_variables", lambda _ds: ["lat"]
    )
    monkeypatch.setattr(
        checks.cfutil, "get_true_longitude_variables", lambda _ds: ["lon"]
    )

    assert result_passed(checks.check_domain_id(checker, use_esgvoc=True)[0])

    checker.ds.variables["lat"] = SimpleNamespace(ndim=2)
    result = checks.check_domain_id(checker, use_esgvoc=True)[0]
    assert not result_passed(result)
    assert "requires 1D" in result.msgs[0]

    checker.attrs["domain_id"] = "UNKNOWN"
    result = checks.check_domain_id(checker, use_esgvoc=False)[0]
    assert not result_passed(result)
    assert "not compliant with the CV" in result.msgs[-1]


def test_cordex_metadata_checks_report_unowned_missing_prerequisites():
    checker = _Checker({})

    domain = checks.check_domain_id(
        checker,
        use_esgvoc=True,
        missing_attribute_delegated=False,
    )
    grid = checks.check_grid(checker, missing_attribute_delegated=False)

    assert "domain_id' is missing" in domain[0].msgs[0]
    assert "attribute 'grid' is missing" in grid[0].msgs[0]


def test_cordex_longitude_policy_reports_unowned_discovery_failure(monkeypatch):
    checker = _Checker(varname=[])
    monkeypatch.setattr(
        coordinate_checks.cfutil,
        "get_true_latitude_variables",
        lambda _dataset: [],
    )
    monkeypatch.setattr(
        coordinate_checks.cfutil,
        "get_true_longitude_variables",
        lambda _dataset: [],
    )

    delegated = coordinate_checks.check_lon_value_range(
        checker, topology_error_delegated=True
    )
    fallback = coordinate_checks.check_lon_value_range(
        checker, topology_error_delegated=False
    )

    assert delegated == []
    assert "longitude coordinate could not be identified" in fallback[0].msgs[0]
