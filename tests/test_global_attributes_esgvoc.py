from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from compliance_checker.base import BaseCheck
from esgvoc.apps.ncattvalid import AttributeResult, GAReport

from checks.attribute_checks.check_global_attributes_esgvoc import (
    check_global_attributes_esgvoc,
    normalize_global_attributes,
)


@dataclass
class FakeSpec:
    source_collection: str | None
    attr_field_value_type: str
    attr_field_name: str | None = None


class FakeDataset:
    def __init__(self, attributes):
        self.attributes = attributes

    def ncattrs(self):
        return list(self.attributes)

    def getncattr(self, name):
        return self.attributes[name]

    def filepath(self):
        return "/tmp/example.nc"


class FakeValidator:
    def __init__(self, report=None):
        self._specs = [
            FakeSpec("activity", "string_array", "activity_id"),
            FakeSpec(None, "string", "title"),
        ]
        self.report = report
        self.received = None

    def validate(self, attributes, filename=None):
        self.received = (attributes, filename)
        return self.report


def test_normalize_global_attributes_handles_netcdf_string_arrays():
    validator = FakeValidator()
    dataset = FakeDataset(
        {
            "activity_id": np.asarray(["CMIP", "ScenarioMIP"]),
            "title": "Keep   free-text spacing",
            "realization_index": np.int32(1),
        }
    )

    found = normalize_global_attributes(dataset, validator)

    assert found == {
        "activity_id": "CMIP ScenarioMIP",
        "title": "Keep   free-text spacing",
        "realization_index": 1,
    }


def test_esgvoc_report_uses_toml_severity_and_ignores_extra_attributes():
    report = GAReport(
        project_id="cmip7",
        filename="example.nc",
        missing=["source_id"],
        extra=["history"],
        results=[
            AttributeResult(
                name="activity_id",
                is_valid=False,
                message="'WRONG' not found in collection 'activity'",
                value="WRONG",
                collection="activity",
            ),
            AttributeResult(
                name="title",
                is_valid=True,
                message="free-text attribute",
                value="Example",
                collection=None,
            ),
        ],
    )
    validator = FakeValidator(report)

    results = check_global_attributes_esgvoc(
        FakeDataset({"activity_id": "WRONG", "title": "Example"}),
        "cmip7",
        severity_by_attribute={"activity_id": BaseCheck.MEDIUM},
        validator=validator,
    )

    failures = [result for result in results if result.msgs]
    assert len(failures) == 2
    assert failures[0].name == "[ATTR001] Global attribute 'source_id' existence"
    assert failures[0].weight == BaseCheck.HIGH
    assert failures[1].name == (
        "[ATTR004] Global attribute 'activity_id' ESGVoc vocabulary check"
    )
    assert failures[1].weight == BaseCheck.MEDIUM
    assert all("history" not in result.name for result in results)


def test_esgvoc_string_array_results_are_grouped_by_attribute():
    report = GAReport(
        project_id="cmip6",
        filename="example.nc",
        results=[
            AttributeResult("activity_id", True, "valid", "CMIP", "activity_id"),
            AttributeResult(
                "activity_id",
                True,
                "valid",
                "ScenarioMIP",
                "activity_id",
            ),
        ],
    )

    results = check_global_attributes_esgvoc(
        FakeDataset({"activity_id": "CMIP  ScenarioMIP"}),
        "cmip6",
        validator=FakeValidator(report),
    )

    assert len(results) == 2
    assert all(not result.msgs for result in results)


def test_esgvoc_failure_is_reported_once_as_setup_error():
    class BrokenValidator(FakeValidator):
        def validate(self, attributes, filename=None):
            raise RuntimeError("database unavailable")

    results = check_global_attributes_esgvoc(
        FakeDataset({"activity_id": "CMIP"}),
        "cmip7",
        validator=BrokenValidator(),
    )

    assert len(results) == 1
    assert results[0].name == "[ATTR000] ESGVoc global attribute validation setup"
    assert "RuntimeError: database unavailable" in results[0].msgs[0]
