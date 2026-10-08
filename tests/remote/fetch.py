"""Resolve remote NetCDF files used by integration tests."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pooch
import pytest

from tests.remote.model import RemoteFile


@lru_cache
def resolve_remote_file(spec: RemoteFile) -> Path:
    """Resolve an override file or download it to the persistent test cache."""
    override = os.environ.get("WCRP_REMOTE_TEST_DATA_DIR")
    if override:
        path = Path(override).expanduser() / spec.relative_path
        if not path.is_file():
            pytest.fail(
                f"WCRP_REMOTE_TEST_DATA_DIR does not contain "
                f"{spec.relative_path.as_posix()}"
            )
    else:
        cache_root = Path(
            os.environ.get(
                "WCRP_REMOTE_TEST_DATA_CACHE",
                pooch.os_cache("cc-plugin-wcrp"),
            )
        ).expanduser()
        path = Path(
            pooch.retrieve(
                url=spec.url,
                known_hash=(f"sha256:{spec.sha256}" if spec.sha256 else None),
                path=(
                    cache_root
                    / "remote-tests"
                    / spec.repository
                    / spec.revision
                    / spec.relative_path.parent
                ),
                fname=spec.relative_path.name,
                progressbar=False,
            )
        )

    if spec.sha256:
        actual_hash = pooch.file_hash(path, alg="sha256")
        if actual_hash != spec.sha256:
            pytest.fail(
                f"Remote test file {path} has SHA-256 {actual_hash}; "
                f"expected {spec.sha256}."
            )
    return path
