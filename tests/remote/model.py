"""Data model shared by remote plugin regression suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping


@dataclass(frozen=True)
class RemoteFile:
    """One remote test file with optional checksum verification."""

    repository: str
    base_url: str
    revision: str
    relative_path: PurePosixPath
    sha256: str | None = None

    @property
    def url(self) -> str:
        return (
            f"{self.base_url.rstrip('/')}/{self.revision}/"
            f"{self.relative_path.as_posix()}"
        )


@dataclass(frozen=True)
class ExpectedFailure:
    """One expected failing Result and identifying message substrings."""

    name: str
    severity: int
    messages: tuple[str, ...]


@dataclass(frozen=True)
class ExpectedCheck:
    """Expected output from running one checker method in isolation."""

    result_count: int
    failures: tuple[ExpectedFailure, ...] = ()


@dataclass(frozen=True)
class RemoteDataset:
    """Remote file and complete checker-method baseline for that file."""

    id: str
    file: RemoteFile
    checks: Mapping[str, ExpectedCheck]


@dataclass(frozen=True)
class RemoteProject:
    """One Compliance Checker plugin and its remote regression datasets."""

    id: str
    checker: str
    datasets: tuple[RemoteDataset, ...]
