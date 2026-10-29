from types import SimpleNamespace

import numpy as np
import pytest
from compliance_checker.base import BaseCheck

from checks.format_checks import check_internal_packing as packing
from plugins.cmip7.cmip7 import Cmip7ProjectCheck
from plugins.cordex_cmip6.cordex_cmip6 import CordexCmip6ProjectCheck
from plugins.wcrp_schema import FileInternalPackingDataRule


class FakeDataset:
    def __init__(self, path="/tmp/packing.nc"):
        self.path = path

    def filepath(self):
        return self.path


class FakeChunkId:
    def __init__(self, count):
        self.count = count

    def get_num_chunks(self):
        return self.count


class FakeVariable:
    def __init__(self, chunks, *, chunk_count=2, dtype="f4", attrs=None):
        self.chunks = chunks
        self.id = FakeChunkId(chunk_count)
        self.dtype = np.dtype(dtype)
        self.attrs = attrs or {}


class FakePackingFile(dict):
    def __init__(self, variables, *, consolidated_metadata, variable_id="tas"):
        super().__init__(variables)
        self.consolidated_metadata = consolidated_metadata
        self.attrs = {"variable_id": variable_id}
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def clear_packing_sessions():
    packing._INTERNAL_PACKING_FILE_SESSIONS.clear()
    packing._INTERNAL_PACKING_FINALIZE_COUNTERS.clear()
    yield
    for session in packing._INTERNAL_PACKING_FILE_SESSIONS.values():
        session["file"].close()
    packing._INTERNAL_PACKING_FILE_SESSIONS.clear()
    packing._INTERNAL_PACKING_FINALIZE_COUNTERS.clear()


def messages(results):
    return [message for result in results for message in result.msgs]


def test_internal_packing_sections_reuse_and_close_pyfive_file(monkeypatch):
    packing_file = FakePackingFile(
        {
            "time": FakeVariable(
                (2,), chunk_count=2, dtype="f8", attrs={"bounds": "time_bnds"}
            ),
            "time_bnds": FakeVariable(None, chunk_count=1, dtype="f8"),
            "tas": FakeVariable((6, 1), chunk_count=2),
        },
        consolidated_metadata=False,
    )
    opened = []

    def open_file(path):
        opened.append(path)
        return packing_file

    monkeypatch.setattr(packing, "_PYFIVE_OK", True)
    monkeypatch.setattr(packing, "pyfive", SimpleNamespace(File=open_file))
    dataset = FakeDataset()

    metadata = packing.check_internal_packing(
        dataset, severity=BaseCheck.MEDIUM, run_time=False, run_data=False
    )
    packing.finalize_internal_packing_session(dataset)
    time = packing.check_internal_packing(
        dataset, severity=BaseCheck.MEDIUM, run_metadata=False, run_data=False
    )
    packing.finalize_internal_packing_session(dataset)
    data = packing.check_internal_packing(
        dataset,
        severity=BaseCheck.MEDIUM,
        min_chunk_size_bytes=4 * (2**20),
        frequency="mon",
        frequency_min_timesteps={"mon": 6},
        run_metadata=False,
        run_time=False,
    )
    packing.finalize_internal_packing_session(dataset)

    assert opened == [dataset.path]
    assert "consolidated internal metadata" in messages(metadata)[0]
    assert len(time) == 2
    assert "2 chunks" in messages(time)[0]
    assert time[1].msgs == []
    assert data[0].msgs == []
    assert packing_file.closed is True
    assert packing._INTERNAL_PACKING_FILE_SESSIONS == {}
    assert packing._INTERNAL_PACKING_FINALIZE_COUNTERS == {}


def test_data_chunk_frequency_exception_requires_time_dependency():
    variable = FakeVariable((6, 1), chunk_count=2)

    accepted, detail = packing._check_data_variable(
        variable,
        min_chunk_size_bytes=4 * (2**20),
        frequency="mon",
        frequency_min_timesteps={"mon": 6},
        has_time=True,
    )
    assert accepted is True
    assert "frequency exception" in detail

    accepted, detail = packing._check_data_variable(
        variable,
        min_chunk_size_bytes=4 * (2**20),
        frequency="mon",
        frequency_min_timesteps={"mon": 6},
        has_time=False,
    )
    assert accepted is False
    assert "frequency exception" not in detail


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_internal_packing_frequency_steps_must_be_positive_integers(value):
    with pytest.raises(ValueError, match="positive integer"):
        FileInternalPackingDataRule(frequency_min_timesteps={"mon": value})


def test_project_packing_configuration_is_loaded():
    cmip7 = Cmip7ProjectCheck()
    cmip7._load_split_config()
    cmip7_rule = cmip7.config.file.internal_packing

    assert cmip7_rule.metadata.severity == "H"
    assert cmip7_rule.time.severity == "H"
    assert cmip7_rule.data.severity == "H"
    assert cmip7_rule.data.frequency_min_timesteps is None

    cordex = CordexCmip6ProjectCheck()
    cordex._load_split_config()
    cordex_rule = cordex.config.file.internal_packing

    assert cordex_rule.metadata.severity == "M"
    assert cordex_rule.time.severity == "M"
    assert cordex_rule.data.severity == "M"
    assert cordex_rule.data.frequency_min_timesteps == {
        "mon": 6,
        "day": 1,
        "6hr": 4,
        "3hr": 1,
        "1hr": 6,
    }


class BrokenChunkVariable:
    dtype = np.dtype("f4")

    def __init__(self, *, fail_chunks=False, fail_count=False):
        self.fail_chunks = fail_chunks
        self.id = self if fail_count else FakeChunkId(2)

    @property
    def chunks(self):
        if self.fail_chunks:
            raise RuntimeError("broken chunks")
        return (2, 2)

    def get_num_chunks(self):
        raise RuntimeError("broken count")


@pytest.mark.parametrize(
    ("variable", "fragment"),
    [
        (BrokenChunkVariable(fail_chunks=True), "unable to read chunk metadata"),
        (BrokenChunkVariable(fail_count=True), "unable to read number of chunks"),
    ],
)
def test_packing_helpers_turn_chunk_metadata_errors_into_details(variable, fragment):
    assert fragment in packing._is_single_chunk_or_contiguous(variable)[1]
    assert fragment in packing._check_data_variable(variable, 1024)[1]


def test_data_chunk_helper_covers_contiguous_single_threshold_and_frequency_failures():
    assert packing._check_data_variable(FakeVariable(None), 1024)[0]
    assert packing._check_data_variable(FakeVariable((2, 2), chunk_count=1), 1024)[0]
    assert packing._check_data_variable(FakeVariable((128, 2)), 1024)[0]

    failed, detail = packing._check_data_variable(
        FakeVariable((2, 2)),
        4096,
        frequency="unknown",
        frequency_min_timesteps={"mon": 6},
        has_time=True,
    )
    assert failed is False
    assert "frequency is unavailable" in detail

    failed, detail = packing._check_data_variable(
        FakeVariable((2, 2)),
        4096,
        frequency="mon",
        frequency_min_timesteps={"mon": 6},
        has_time=True,
    )
    assert failed is False
    assert "at least 6 time steps" in detail


def test_finalize_rejects_inconsistent_expected_section_count(monkeypatch):
    dataset = FakeDataset()
    monkeypatch.setattr(packing, "_PYFIVE_OK", True)
    packing._INTERNAL_PACKING_FINALIZE_COUNTERS[dataset.path] = {
        "count": 1,
        "total": 3,
    }
    with pytest.raises(ValueError, match="same total_packing_checks"):
        packing.finalize_internal_packing_session(dataset, total_packing_checks=2)


def test_internal_packing_reports_dependency_path_and_open_failures(monkeypatch):
    monkeypatch.setattr(packing, "_PYFIVE_OK", False)
    monkeypatch.setattr(packing, "_PYFIVE_ERR", "not installed")
    assert "not installed" in messages(packing.check_internal_packing(FakeDataset()))[0]

    monkeypatch.setattr(packing, "_PYFIVE_OK", True)
    no_path = SimpleNamespace(filepath=lambda: None)
    assert (
        "Could not retrieve dataset file path"
        in messages(packing.check_internal_packing(no_path))[0]
    )

    monkeypatch.setattr(
        packing,
        "pyfive",
        SimpleNamespace(File=lambda _path: (_ for _ in ()).throw(OSError("bad file"))),
    )
    assert (
        "Could not open file with pyfive: bad file"
        in messages(packing.check_internal_packing(FakeDataset()))[0]
    )


def test_internal_packing_can_disable_every_section():
    assert (
        packing.check_internal_packing(
            FakeDataset(), run_metadata=False, run_time=False, run_data=False
        )
        == []
    )


def test_internal_packing_reports_only_unowned_missing_prerequisites(monkeypatch):
    packing_file = FakePackingFile({}, consolidated_metadata=True)
    packing_file.attrs = {}
    monkeypatch.setattr(packing, "_PYFIVE_OK", True)
    monkeypatch.setattr(
        packing, "pyfive", SimpleNamespace(File=lambda _path: packing_file)
    )
    dataset = FakeDataset()

    time = packing.check_internal_packing(
        dataset,
        run_metadata=False,
        run_time=True,
        run_data=False,
        time_presence_delegated=False,
    )
    packing.finalize_internal_packing_session(dataset)
    data = packing.check_internal_packing(
        dataset,
        run_metadata=False,
        run_time=False,
        run_data=True,
        variable_id_presence_delegated=False,
    )
    packing.finalize_internal_packing_session(dataset)
    delegated = packing.check_internal_packing(
        dataset,
        run_metadata=False,
        run_time=True,
        run_data=True,
        time_presence_delegated=True,
        variable_id_presence_delegated=True,
    )
    packing.finalize_internal_packing_session(dataset)

    assert "Coordinate variable 'time' is missing" in messages(time)[0]
    assert "Global attribute 'variable_id' is missing" in messages(data)[0]
    assert delegated == []


@pytest.mark.parametrize(
    ("variable_id", "expected"),
    [
        ("tas", "Data variable 'tas' named by global attribute"),
        ("", "Global attribute 'variable_id' is empty"),
    ],
)
def test_internal_packing_reports_unowned_unusable_data_selector(
    monkeypatch, variable_id, expected
):
    packing_file = FakePackingFile(
        {}, consolidated_metadata=True, variable_id=variable_id
    )
    monkeypatch.setattr(packing, "_PYFIVE_OK", True)
    monkeypatch.setattr(
        packing, "pyfive", SimpleNamespace(File=lambda _path: packing_file)
    )
    dataset = FakeDataset()

    results = packing.check_internal_packing(
        dataset,
        run_metadata=False,
        run_time=False,
        run_data=True,
        variable_id_presence_delegated=False,
        data_variable_presence_delegated=False,
    )
    packing.finalize_internal_packing_session(dataset)

    assert expected in messages(results)[0]
