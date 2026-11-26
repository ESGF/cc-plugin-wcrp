"""Preserve nested CMOR CV relationships in the ESGVoc validation path."""

from types import SimpleNamespace

import pytest
from netCDF4 import Dataset

from checks.consistency_checks import check_institution_source_consistency as checks
from plugins.cordex_cmip6.cordex_cmip6 import CordexCmip6ProjectCheck
from plugins.wcrp_schema import WCRPConfig

RELATIONSHIPS = (
    ("source_id", "institution_id", "CLMcom-DWD", "CLMcom-WEGC", "ATTR010"),
    ("driving_source_id", "driving_institution_id", "ECMWF", "IPSL", "ATTR010"),
    ("source_id", "source_type", "ARCM", "AOGCM", "ATTR011"),
    ("source_id", "activity_id", "DD", "ESD", "ATTR011"),
    (
        "source_id",
        "license",
        "https://example.org/license",
        "https://example.org/other",
        "ATTR011",
    ),
    ("domain_id", "domain", "South America", "Europe", "ATTR011"),
    (
        "driving_experiment_id",
        "driving_experiment",
        "Reanalysis",
        "Historical",
        "ATTR011",
    ),
    (
        "driving_source_id",
        "driving_experiment_id",
        "evaluation",
        "historical",
        "ATTR011",
    ),
)


@pytest.fixture
def catalogue(monkeypatch):
    terms = {
        ("source_id", "model"): SimpleNamespace(
            id="model",
            organisation_id=["clmcom-dwd"],
            source_type="ARCM",
            activity_participation=["DD"],
            license={"id": "terms"},
        ),
        ("driving_source_id", "era5"): SimpleNamespace(
            id="era5",
            organisation_id=["ecmwf"],
            driving_experiment_id=["evaluation"],
        ),
        ("domain_id", "sam-50"): SimpleNamespace(
            id="sam-50", description="South America"
        ),
        ("driving_experiment_id", "evaluation"): SimpleNamespace(
            id="evaluation", experiment="Reanalysis"
        ),
        ("driving_experiment_id", "historical"): SimpleNamespace(
            id="historical", experiment="Historical"
        ),
        ("institution_id", "clmcom-dwd"): SimpleNamespace(id="clmcom-dwd"),
        ("driving_institution_id", "ecmwf"): SimpleNamespace(id="ecmwf"),
        ("license", "terms"): SimpleNamespace(id="terms"),
        ("activity_id", "dd"): SimpleNamespace(id="dd"),
        ("activity_id", "esd"): SimpleNamespace(id="esd"),
    }

    def resolve(project_id, collection_id, term_id):
        assert project_id == "cordex-cmip6"
        return terms.get((collection_id, term_id))

    def valid(value, project, collection, identifier):
        assert project == "cordex-cmip6"
        allowed = {
            "clmcom-dwd": ["CLMcom-DWD"],
            "ecmwf": ["ECMWF"],
            "terms": ["https://example.org/license", "https://example.org/license/"],
            "evaluation": ["evaluation"],
            "historical": ["historical"],
            "dd": ["DD"],
            "esd": ["ESD"],
        }
        return value in allowed[identifier]

    monkeypatch.setattr(checks.voc, "get_term_in_collection", resolve)
    monkeypatch.setattr(checks.voc, "valid_term", valid)
    monkeypatch.setattr(checks, "ESG_VOCAB_AVAILABLE", True)
    return terms


def checker_for(selector, attribute, *, attribute_checks=True, enabled=True):
    checker = CordexCmip6ProjectCheck()
    checker.config = WCRPConfig.model_validate(
        {
            "project_name": "cordex-cmip6",
            "project_version": "test",
            "global": {
                "attributes": {
                    name: {"severity": "H"} for name in (selector, attribute)
                }
                if attribute_checks
                else {},
                "consistency": {f"{selector}_vs_{attribute}": {"severity": "H"}}
                if enabled
                else {},
            },
        }
    )
    return checker


def populate(ds):
    ds.source_id = "MODEL"
    ds.driving_source_id = "ERA5"
    ds.domain_id = "SAM-50"
    ds.driving_experiment_id = "evaluation"


@pytest.mark.parametrize("selector,attribute,good,bad,check_id", RELATIONSHIPS)
@pytest.mark.parametrize("legacy", [False, True])
def test_relationships_through_plugin(
    tmp_path, catalogue, selector, attribute, good, bad, check_id, legacy
):
    checker = checker_for(selector, attribute)
    checker.verification_against_tables = legacy
    with Dataset(tmp_path / "relationship.nc", "w") as ds:
        populate(ds)
        ds.setncattr(attribute, good)
        results = checker.check_Global_Consistency(ds)
        assert len(results) == 1
        assert results[0].weight == 3
        assert not results[0].msgs
        ds.setncattr(attribute, bad)
        results = checker.check_Global_Consistency(ds)
        assert len(results) == 1
        assert results[0].weight == 3
        assert results[0].name.startswith(f"[{check_id}]")
        assert "inconsistent" in results[0].msgs[0]


@pytest.mark.parametrize("selector,attribute,good,bad,check_id", RELATIONSHIPS)
@pytest.mark.parametrize("delegated", [False, True])
def test_missing_attributes_have_fallback(
    tmp_path, catalogue, selector, attribute, good, bad, check_id, delegated
):
    checker = checker_for(selector, attribute, attribute_checks=delegated)
    with Dataset(tmp_path / "missing.nc", "w") as ds:
        results = checker.check_Global_Consistency(ds)
    assert len(results) == (0 if delegated else 1)
    if results:
        assert results[0].weight == 3
        assert len(results[0].msgs) == 2


@pytest.mark.parametrize(
    "state,actual,passes",
    [
        ("absent", "historical", True),
        (["evaluation"], "evaluation", True),
        (["evaluation"], "historical", False),
        (["evaluation", "historical"], "historical", True),
        ([], "evaluation", False),
        (None, "evaluation", False),
    ],
)
def test_driving_experiment_restriction(tmp_path, catalogue, state, actual, passes):
    term = catalogue[("driving_source_id", "era5")]
    if state == "absent":
        del term.driving_experiment_id
    else:
        term.driving_experiment_id = state
    checker = checker_for("driving_source_id", "driving_experiment_id")
    with Dataset(tmp_path / "restriction.nc", "w") as ds:
        populate(ds)
        ds.driving_experiment_id = actual
        (result,) = checker.check_Global_Consistency(ds)
    assert bool(result.msgs) is not passes
    if state is None or state == []:
        assert "no value for field 'driving_experiment_id'" in result.msgs[0]


@pytest.mark.parametrize("delegated", [False, True])
def test_unknown_selector_delegates_to_vocabulary_check(tmp_path, catalogue, delegated):
    checker = checker_for("source_id", "source_type", attribute_checks=delegated)
    with Dataset(tmp_path / "unknown.nc", "w") as ds:
        ds.source_id = "unknown"
        ds.source_type = "ARCM"
        results = checker.check_Global_Consistency(ds)
    assert len(results) == (0 if delegated else 1)


@pytest.mark.parametrize("field", ["source_type", "license"])
def test_missing_catalogue_fields_are_not_unrestricted(tmp_path, catalogue, field):
    delattr(catalogue[("source_id", "model")], field)
    checker = checker_for("source_id", field)
    with Dataset(tmp_path / "incomplete.nc", "w") as ds:
        populate(ds)
        ds.setncattr(field, "value")
        (result,) = checker.check_Global_Consistency(ds)
    assert result.weight == 3
    assert "no value for field" in result.msgs[0]


@pytest.mark.parametrize(
    "registered,actual,passes",
    [
        (["DD", "esd"], "DD ESD", True),
        (["dd"], "DD ESD", False),
        (["DD"], "", False),
        ([], "DD", False),
        (None, "DD", False),
    ],
)
def test_source_activity_participation(tmp_path, catalogue, registered, actual, passes):
    source = catalogue[("source_id", "model")]
    if registered is None:
        del source.activity_participation
    else:
        source.activity_participation = registered
    checker = checker_for("source_id", "activity_id")
    with Dataset(tmp_path / "activity.nc", "w") as ds:
        populate(ds)
        ds.activity_id = actual
        (result,) = checker.check_Global_Consistency(ds)
    assert result.weight == 3
    assert bool(result.msgs) is not passes


def test_license_uses_the_selected_term_pattern(tmp_path, catalogue):
    checker = checker_for("source_id", "license")
    with Dataset(tmp_path / "license.nc", "w") as ds:
        populate(ds)
        ds.license = "https://example.org/license/"
        (result,) = checker.check_Global_Consistency(ds)
    assert not result.msgs


@pytest.mark.parametrize("failure", ["missing_reference", "database_error"])
def test_reference_lookup_failure_is_reported(
    tmp_path, catalogue, monkeypatch, failure
):
    if failure == "missing_reference":
        del catalogue[("license", "terms")]
    else:

        def unavailable(*args, **kwargs):
            raise RuntimeError("database unavailable")

        monkeypatch.setattr(checks.voc, "valid_term", unavailable)
    checker = checker_for("source_id", "license")
    with Dataset(tmp_path / "reference-error.nc", "w") as ds:
        populate(ds)
        ds.license = "https://example.org/license"
        (result,) = checker.check_Global_Consistency(ds)
    assert result.weight == 3
    assert "could not be verified" in result.msgs[0]
    assert (
        "is absent" if failure == "missing_reference" else "database unavailable"
    ) in result.msgs[0]


@pytest.mark.parametrize("selector,attribute,good,bad,check_id", RELATIONSHIPS)
def test_shipped_configuration_and_disabling(
    tmp_path, catalogue, selector, attribute, good, bad, check_id
):
    checker = CordexCmip6ProjectCheck()
    checker._load_split_config()
    assert (
        getattr(
            checker.config.global_.consistency, f"{selector}_vs_{attribute}"
        ).severity
        == "H"
    )
    checker = checker_for(selector, attribute, enabled=False)
    with Dataset(tmp_path / "disabled.nc", "w") as ds:
        populate(ds)
        ds.setncattr(attribute, bad)
        assert checker.check_Global_Consistency(ds) == []
