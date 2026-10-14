"""End-to-end plugin regressions against checksum-verified remote NetCDF files."""

from __future__ import annotations

from collections import defaultdict

import pytest
from compliance_checker.suite import CheckSuite

from tests.remote import PROJECT_CASES
from tests.remote.fetch import resolve_remote_file


def _all_check_cases():
    for project in PROJECT_CASES:
        for dataset in project.datasets:
            for check_name, expectation in dataset.checks.items():
                yield pytest.param(
                    project,
                    dataset,
                    check_name,
                    expectation,
                    id=f"{project.id}-{dataset.id}-{check_name}",
                )


def _checker_names(suite, checker):
    checks = suite._get_checks(
        suite.checkers[checker],
        [],
        defaultdict(lambda: None),
    )
    return {entry[0].__name__ for entry in checks}


def _result_passed(result):
    value = result.value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0] == value[1]
    return bool(value)



@pytest.fixture(scope="session")
def remote_check_suite():
    suite = CheckSuite()
    suite.load_all_available_checkers()
    return suite


@pytest.mark.parametrize(
    "project",
    [pytest.param(project, id=project.id) for project in PROJECT_CASES],
)
def test_remote_expectations_cover_every_plugin_check(remote_check_suite, project):
    """Require every exposed checker method to have an explicit baseline."""
    discovered = _checker_names(remote_check_suite, project.checker)
    for dataset in project.datasets:
        expected = set(dataset.checks)
        assert expected == discovered, (
            f"{project.id}/{dataset.id} remote expectations do not match the "
            f"plugin inventory. Missing={sorted(discovered - expected)}; "
            f"obsolete={sorted(expected - discovered)}"
        )


@pytest.mark.remote_data
@pytest.mark.parametrize(
    "project,dataset,check_name,expectation",
    list(_all_check_cases()),
)
def test_remote_plugin_check(
    remote_check_suite,
    project,
    dataset,
    check_name,
    expectation,
):
    """Run one plugin check and compare all failures with its explicit baseline."""
    path = resolve_remote_file(dataset.file)
    loaded = remote_check_suite.load_dataset(path)
    try:
        results, errors = remote_check_suite.run_all(
            loaded,
            [project.checker],
            include_checks=[check_name],
            skip_checks=[],
        )[project.checker]
    finally:
        close = getattr(loaded, "close", None)
        if close is not None:
            close()

    assert not errors, f"{project.id}/{dataset.id}/{check_name}: {errors}"
    assert len(results) == expectation.result_count, (
        f"{project.id}/{dataset.id}/{check_name}: expected "
        f"{expectation.result_count} Result objects, got {len(results)}"
    )

    for result in results:
        assert _result_passed(result) == (not result.msgs), (
            f"{project.id}/{dataset.id}/{check_name}: Result {result.name!r} "
            f"has score {result.value!r} and messages {result.msgs!r}"
        )

    actual_failures = [result for result in results if not _result_passed(result)]
    assert len(actual_failures) == len(expectation.failures), (
        f"{project.id}/{dataset.id}/{check_name}: expected "
        f"{len(expectation.failures)} failing Results, got "
        f"{[(result.name, result.weight, result.msgs) for result in actual_failures]}"
    )

    for actual, expected in zip(actual_failures, expectation.failures, strict=True):
        assert actual.name == expected.name
        assert actual.weight == expected.severity
        assert len(actual.msgs) == len(expected.messages)
        for message, substring in zip(actual.msgs, expected.messages, strict=True):
            assert substring in message
