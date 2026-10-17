"""Unit tests for CORDEX-CMIP6-specific metadata policy."""

from types import SimpleNamespace

import pytest

from checks.attribute_checks import check_attrs_cordex_cmip6 as checks
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
    assert result_passed(checks.check_grid(_Checker(attrs))[0]) is passes


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
