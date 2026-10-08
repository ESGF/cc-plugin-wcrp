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
