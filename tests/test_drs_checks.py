"""Local tests for DRS vocabulary and consistency orchestration."""

from types import SimpleNamespace

from compliance_checker.base import BaseCheck

from checks.consistency_checks import check_drs_consistency as consistency
from checks.consistency_checks import check_drs_filename_cv as vocabulary
from tests.wcrp_cmip6_test_checks.conftest import result_passed


class _Dataset:
    def __init__(self, path, attrs=None):
        self.path = path
        self.attrs = attrs or {}

    def filepath(self):
        return self.path

    def getncattr(self, name):
        return self.attrs[name]

    def ncattrs(self):
        return list(self.attrs)


class _Report:
    def __init__(self, errors=()):
        self.errors = list(errors)


class _Validator:
    filename_errors = ()
    directory_errors = ()

    def __init__(self, project_id):
        self.project_id = project_id

    def validate_file_name(self, _filename):
        return _Report(self.filename_errors)

    def validate_directory(self, _directory):
        return _Report(self.directory_errors)


def test_drs_filename_vocabulary_passes_and_reports_validator_errors(monkeypatch):
    monkeypatch.setattr(vocabulary, "ESG_VOCAB_AVAILABLE", True)
    monkeypatch.setattr(vocabulary, "DrsValidator", _Validator)
    dataset = _Dataset("/tmp/tas_Amon_MODEL.nc")

    assert result_passed(vocabulary.check_drs_filename(dataset, BaseCheck.HIGH)[0])

    _Validator.filename_errors = ("bad table", "extra token")
    result = vocabulary.check_drs_filename(dataset, BaseCheck.HIGH)[0]
    assert not result_passed(result)
    assert "bad table; extra token" in result.msgs[0]
    _Validator.filename_errors = ()


def test_drs_vocabulary_handles_missing_library_path_and_validator_exception(
    monkeypatch,
):
    monkeypatch.setattr(vocabulary, "ESG_VOCAB_AVAILABLE", False)
    result = vocabulary.check_drs_filename(_Dataset("/tmp/file.nc"), BaseCheck.HIGH)[0]
    assert "required but not installed" in result.msgs[0]

    monkeypatch.setattr(vocabulary, "ESG_VOCAB_AVAILABLE", True)
    result = vocabulary.check_drs_filename(_Dataset(None), BaseCheck.HIGH)[0]
    assert result.msgs == ["File path could not be determined."]

    class BrokenValidator:
        def __init__(self, project_id):
            raise RuntimeError("bad project specs")

    monkeypatch.setattr(vocabulary, "DrsValidator", BrokenValidator)
    result = vocabulary.check_drs_filename(_Dataset("/tmp/file.nc"), BaseCheck.HIGH)[0]
    assert "bad project specs" in result.msgs[0]


def test_cmip7_directory_vocabulary_uses_mip_drs7_root_and_can_suppress_structure_error(
    monkeypatch,
):
    monkeypatch.setattr(vocabulary, "ESG_VOCAB_AVAILABLE", True)
    monkeypatch.setattr(vocabulary, "DrsValidator", _Validator)
    valid = _Dataset("/archive/MIP-DRS7/CMIP7/activity/file.nc")
    assert result_passed(
        vocabulary.check_drs_directory(valid, BaseCheck.HIGH, project_id="cmip7")[0]
    )

    missing = _Dataset("/archive/cmip7/file.nc")
    assert (
        vocabulary.check_drs_directory(
            missing,
            BaseCheck.HIGH,
            project_id="cmip7",
            report_directory_structure_error=False,
        )
        == []
    )
    result = vocabulary.check_drs_directory(
        missing, BaseCheck.HIGH, project_id="cmip7"
    )[0]
    assert "MIP-DRS7" in result.msgs[0]


def test_legacy_drs_vocabulary_reports_suffix_messages_and_unchecked_terms(monkeypatch):
    checker = SimpleNamespace(
        filepath="/tmp/tas_Amon.nc",
        drs_suffix="nc4",
        drs_fn={"variable_id": "tas", "time_range": "200001-200012"},
        drs_dir={"variable_id": "tas", "version": "v1"},
    )
    monkeypatch.setattr(
        vocabulary,
        "_compare_CV",
        lambda _checker, facets, _prefix: (
            {key: key == "variable_id" for key in facets},
            ["bad controlled term"],
        ),
    )
    result = vocabulary.check_drs_filename_cv(checker)[0]
    assert not result_passed(result)
    assert any("File suffix differs" in message for message in result.msgs)
    assert any("time_range" in message for message in result.msgs)

    result = vocabulary.check_drs_directory_cv(checker, ["version"])[0]
    assert not result_passed(result)
    assert result.msgs == ["bad controlled term"]
    assert result_passed(vocabulary.check_drs_filename_cv(checker, use_esgvoc=True)[0])


def test_cmip7_drs_consistency_compares_attributes_path_and_filename():
    directory = (
        "/MIP-DRS7/CMIP7/CMIP/INST/MODEL/experiment/r1i1p1f1/"
        "glb/mon/tas/tavg-h2m-hxy-atm/g001/v20260101"
    )
    filename = (
        "tas_tavg-h2m-hxy-atm_mon_glb_g001_MODEL_experiment_r1i1p1f1_200001-200012.nc"
    )
    attrs = {
        "drs_specs": "MIP-DRS7",
        "mip_era": "CMIP7",
        "activity_id": "CMIP",
        "institution_id": "INST",
        "source_id": "MODEL",
        "experiment_id": "experiment",
        "variant_label": "r1i1p1f1",
        "region": "glb",
        "frequency": "mon",
        "variable_id": "tas",
        "branding_suffix": "tavg-h2m-hxy-atm",
        "grid_label": "g001",
    }
    dataset = _Dataset(f"{directory}/{filename}", attrs)
    assert result_passed(
        consistency.check_attributes_match_directory_structure(
            dataset, BaseCheck.HIGH, project_id="cmip7"
        )[0]
    )
    assert result_passed(
        consistency.check_filename_matches_directory_structure(
            dataset, BaseCheck.HIGH, project_id="cmip7"
        )[0]
    )

    dataset.attrs["grid_label"] = "wrong"
    result = consistency.check_attributes_match_directory_structure(
        dataset, BaseCheck.HIGH, project_id="cmip7"
    )[0]
    assert "grid_label" in result.msgs[0]


def test_drs_consistency_structure_errors_can_be_owned_elsewhere():
    dataset = _Dataset("/tmp/file.nc", {"mip_era": "CMIP7", "drs_specs": "MIP-DRS7"})
    assert (
        consistency.check_attributes_match_directory_structure(
            dataset,
            BaseCheck.HIGH,
            project_id="cmip7",
            report_directory_structure_error=False,
        )
        == []
    )
    assert (
        consistency.check_filename_matches_directory_structure(
            dataset,
            BaseCheck.HIGH,
            project_id="cmip7",
            report_directory_structure_error=False,
        )
        == []
    )


def test_cmip6_drs_consistency_uses_member_id_and_shared_filename_facets(monkeypatch):
    dataset = _Dataset(
        "/archive/CMIP6/file.nc",
        {"activity_id": "ScenarioMIP AerChemMIP", "variant_label": "r1i1p1f1"},
    )
    directory = {
        "activity_id": "ScenarioMIP",
        "variant_label": "s1960-r1i1p1f1",
        "version": "v1",
        "source_id": "MODEL",
    }
    filename = {"activity_id": "ScenarioMIP", "source_id": "OTHER"}
    monkeypatch.setattr(
        consistency,
        "_get_drs_facets",
        lambda *_args: (directory, filename, None),
    )
    monkeypatch.setattr(consistency, "resolve_member_id", lambda _ds: "s1960-r1i1p1f1")

    attribute_result = consistency.check_attributes_match_directory_structure(
        dataset, BaseCheck.HIGH, project_id="cmip6"
    )[0]
    assert result_passed(attribute_result)
    assert any("source_id" in message for message in attribute_result.msgs)

    filename_result = consistency.check_filename_matches_directory_structure(
        dataset,
        BaseCheck.HIGH,
        project_id="cmip6",
        dir_template_keys=("activity_id", "source_id"),
        filename_template_keys=("activity_id", "source_id"),
    )[0]
    assert not result_passed(filename_result)
    assert filename_result.msgs == [
        "Token 'source_id' is inconsistent: path has 'MODEL', filename has 'OTHER'."
    ]
