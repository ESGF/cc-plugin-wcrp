#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[FILE004] Configurable internal packing checks.

Faithfully ports the logic of the official `check_cmip7_packing` script
(https://github.com/NCAS-CMS/cmip7repack)

Four sub-checks, each producing an independent Result:
  FILE004a — Consolidated internal metadata
  FILE004b — Time coordinate variable: single chunk or contiguous
  FILE004c — Time bounds variable: single chunk or contiguous
  FILE004d — Data variable: single chunk / contiguous, or chunk >= 4 MiB

Requires: pyfive >= 1.1.2 (pure-Python HDF5 reader, no libhdf5 needed)
"""

from math import prod

import numpy as np
from compliance_checker.base import BaseCheck, TestCtx

from checks.utils import severity_word

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:
    from packaging.version import Version
    import pyfive

    _PYFIVE_MIN = Version("1.1.2")
    _pyfive_version = Version(
        __import__("importlib.metadata", fromlist=["version"]).version("pyfive")
    )
    if _pyfive_version < _PYFIVE_MIN:
        raise RuntimeError(
            f"pyfive >= {_PYFIVE_MIN} required, got {_pyfive_version}"
        )
    _PYFIVE_OK = True
    _PYFIVE_ERR = None
except Exception as e:
    pyfive = None
    _PYFIVE_OK = False
    _PYFIVE_ERR = str(e)

_DEFAULT_MIN_CHUNK_SIZE_BYTES = 4 * (2**20)
_INTERNAL_PACKING_FILE_SESSIONS = {}
_INTERNAL_PACKING_FINALIZE_COUNTERS = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_file_path(ds) -> str | None:
    """Return the local file path from a netCDF4.Dataset."""
    try:
        return ds.filepath()
    except Exception:
        return None


def _attr_to_str(attr) -> str:
    """Convert a pyfive attribute value to a plain Python string."""
    return str(np.array(attr).astype("U"))


def _session_key(ds, file_path: str | None) -> str:
    return file_path or f"id:{id(ds)}"


def _acquire_internal_packing_file(ds, file_path: str):
    key = _session_key(ds, file_path)
    session = _INTERNAL_PACKING_FILE_SESSIONS.get(key)
    if session and session.get("file") is not None:
        return session["file"]

    packing_file = pyfive.File(file_path)
    _INTERNAL_PACKING_FILE_SESSIONS[key] = {"file": packing_file}
    return packing_file


def finalize_internal_packing_session(ds, total_packing_checks: int = 3) -> None:
    """Close the shared pyfive handle after all packing sections have run."""
    file_path = _get_file_path(ds)
    key = _session_key(ds, file_path)
    entry = _INTERNAL_PACKING_FINALIZE_COUNTERS.get(key)
    if entry is None:
        entry = {"count": 1, "total": total_packing_checks}
    else:
        if entry["total"] != total_packing_checks:
            raise ValueError(
                "All internal-packing checks must use the same "
                "total_packing_checks value."
            )
        entry = {"count": entry["count"] + 1, "total": entry["total"]}

    if entry["count"] >= entry["total"]:
        session = _INTERNAL_PACKING_FILE_SESSIONS.pop(key, None)
        if session:
            try:
                session["file"].close()
            except Exception:
                pass
        _INTERNAL_PACKING_FINALIZE_COUNTERS.pop(key, None)
    else:
        _INTERNAL_PACKING_FINALIZE_COUNTERS[key] = entry


def _is_single_chunk_or_contiguous(var) -> tuple[bool, str]:
    """
    True  if the variable is contiguous (chunks is None)
              or has exactly one chunk.
    Mirrors the logic in the official check_cmip7_packing script.
    """
    try:
        chunks = var.chunks
    except Exception as exc:
        return False, f"unable to read chunk metadata ({exc})"
    if chunks is None:
        return True, "contiguous"
    try:
        n = var.id.get_num_chunks()
    except Exception as exc:
        return False, f"unable to read number of chunks ({exc})"
    if n <= 1:
        return True, f"1 chunk of shape {tuple(chunks)}"
    return False, f"{n} chunks (expected 1 chunk or contiguous)"


def _check_data_variable(
    var,
    min_chunk_size_bytes: int,
    frequency: str | None = None,
    frequency_min_timesteps: dict[str, int] | None = None,
    has_time: bool = False,
) -> tuple[bool, str]:
    """

    Pass conditions (any one sufficient):
      • contiguous (chunks is None)
      • exactly 1 chunk
      • uncompressed chunk size >= the configured threshold
      • adding one element along the leading dimension would reach the threshold
        (the "lee_way" rule from the official script)
      • an optional frequency-specific minimum number of time steps
    """
    try:
        chunks = var.chunks
    except Exception as exc:
        return False, f"unable to read chunk metadata ({exc})"
    if chunks is None:
        return True, "contiguous"

    try:
        n = var.id.get_num_chunks()
    except Exception as exc:
        return False, f"unable to read number of chunks ({exc})"
    if n <= 1:
        return True, "1 chunk"

    try:
        wordsize = var.dtype.itemsize
        chunksize = prod(chunks) * wordsize
    except Exception as exc:
        return False, f"unable to compute chunk byte size ({exc})"

    # Adding one element along leading dim gives this extra size
    try:
        lee_way = prod(chunks[1:]) * wordsize if len(chunks) > 1 else 0
    except Exception as exc:
        return False, f"unable to compute chunk threshold margin ({exc})"

    if chunksize + lee_way >= min_chunk_size_bytes:
        return True, (
            f"chunk size {chunksize} B "
            f"(>= {min_chunk_size_bytes - lee_way} B threshold)"
        )

    if frequency_min_timesteps and has_time:
        normalized_frequency = str(frequency or "").strip()
        if not normalized_frequency or normalized_frequency.lower() == "unknown":
            known = ", ".join(sorted(frequency_min_timesteps))
            return False, (
                f"uncompressed chunk size {chunksize} B "
                f"(expected at least {min_chunk_size_bytes - lee_way} B, "
                f"or 1 chunk, or contiguous). Frequency-specific exceptions "
                f"are configured for {known}, but the frequency is unavailable"
            )
        if normalized_frequency in frequency_min_timesteps:
            required_steps = int(frequency_min_timesteps[normalized_frequency])
            chunk_time_steps = int(chunks[0]) if chunks else None
            if chunk_time_steps is not None and chunk_time_steps >= required_steps:
                return True, (
                    f"chunk size {chunksize} B below threshold, but frequency "
                    f"exception for '{normalized_frequency}' is met "
                    f"({chunk_time_steps} >= {required_steps} time steps per chunk)"
                )
            return False, (
                f"uncompressed chunk size {chunksize} B "
                f"(expected at least {min_chunk_size_bytes - lee_way} B, "
                f"or at least {required_steps} time steps per chunk for "
                f"frequency '{normalized_frequency}', or 1 chunk, or contiguous)"
            )

    return False, (
        f"uncompressed chunk size {chunksize} B "
        f"(expected at least {min_chunk_size_bytes - lee_way} B, or 1 chunk, "
        "or contiguous)"
    )


# ---------------------------------------------------------------------------
# Public check function
# ---------------------------------------------------------------------------

def check_internal_packing(
    ds,
    severity=BaseCheck.HIGH,
    severity_metadata=None,
    severity_time=None,
    severity_data=None,
    min_chunk_size_bytes: int = _DEFAULT_MIN_CHUNK_SIZE_BYTES,
    frequency: str | None = None,
    frequency_min_timesteps: dict[str, int] | None = None,
    run_metadata: bool = True,
    run_time: bool = True,
    run_data: bool = True,
) -> list:
    """Run selected FILE004 internal-packing checks."""
    results = []

    try:
        min_chunk_size_bytes = int(min_chunk_size_bytes)
        if min_chunk_size_bytes <= 0:
            raise ValueError
    except (TypeError, ValueError):
        min_chunk_size_bytes = _DEFAULT_MIN_CHUNK_SIZE_BYTES

    default_severity = severity
    sev_metadata = severity_metadata or default_severity
    sev_time = severity_time or default_severity
    sev_data = severity_data or default_severity
    if not (run_metadata or run_time or run_data):
        return results

    # -- pyfive available? ---------------------------------------------------
    if not _PYFIVE_OK:
        ctx = TestCtx(BaseCheck.HIGH, "[FILE004] Internal packing")
        ctx.add_failure(
            f"Optional dependency 'pyfive >= {_PYFIVE_MIN}' is not installed or "
            f"incompatible — FILE004 skipped. ({_PYFIVE_ERR})"
        )
        return [ctx.to_result()]

    # -- get file path -------------------------------------------------------
    file_path = _get_file_path(ds)
    if not file_path:
        ctx = TestCtx(default_severity, "[FILE004] Internal packing")
        ctx.add_failure(
            "Could not retrieve dataset file path — FILE004 skipped. It is "
            f"{severity_word(default_severity)} to run internal packing checks "
            "on an accessible local file."
        )
        return [ctx.to_result()]

    # -- open/reuse pyfive file handle ---------------------------------------
    try:
        packing_file = _acquire_internal_packing_file(ds, file_path)
    except Exception as exc:
        ctx = TestCtx(default_severity, "[FILE004] Internal packing")
        ctx.add_failure(f"Could not open file with pyfive: {exc}")
        return [ctx.to_result()]

    # FILE004a — Consolidated internal metadata
    if run_metadata:
        ctx_a = TestCtx(
            sev_metadata,
            "[FILE004a] Internal packing : Consolidated internal metadata",
        )
        try:
            if packing_file.consolidated_metadata:
                ctx_a.add_pass()
            else:
                ctx_a.add_failure(
                    "File does not have consolidated internal metadata. "
                    f"It is {severity_word(sev_metadata)} to consolidate internal "
                    "metadata using a project repacking tool."
                )
        except Exception as exc:
            ctx_a.add_failure(f"Unable to inspect consolidated metadata: {exc}")
        results.append(ctx_a.to_result())

    # FILE004b-c — Time and bounds variables
    has_time = False
    if run_time or run_data:
        try:
            has_time = "time" in packing_file
        except Exception as exc:
            if run_time:
                ctx_b = TestCtx(
                    sev_time,
                    "[FILE004b] Internal packing : Time coordinate chunking",
                )
                ctx_b.add_failure(
                    f"Unable to inspect time coordinate presence: {exc}"
                )
                results.append(ctx_b.to_result())

    if run_time and has_time:
        try:
            time_var = packing_file["time"]
        except Exception as exc:
            time_var = None
            ctx_b = TestCtx(
                sev_time,
                "[FILE004b] Internal packing : Time coordinate chunking",
            )
            ctx_b.add_failure(f"Unable to access time coordinate variable: {exc}")
            results.append(ctx_b.to_result())

        if time_var is not None:
            ctx_b = TestCtx(
                sev_time,
                "[FILE004b] Internal packing : Time coordinate chunking",
            )
            ok, detail = _is_single_chunk_or_contiguous(time_var)
            if ok:
                ctx_b.add_pass()
            else:
                ctx_b.add_failure(
                    f"Time coordinate variable 'time' has {detail}. It is "
                    f"{severity_word(sev_time)} to repack this variable using a "
                    "project repacking tool."
                )
            results.append(ctx_b.to_result())

            try:
                if "bounds" in time_var.attrs:
                    bounds_name = _attr_to_str(time_var.attrs["bounds"])
                    if bounds_name in packing_file:
                        bounds_var = packing_file[bounds_name]
                        ctx_c = TestCtx(
                            sev_time,
                            "[FILE004c] Internal packing : Time bounds chunking "
                            f"('{bounds_name}')",
                        )
                        ok, detail = _is_single_chunk_or_contiguous(bounds_var)
                        if ok:
                            ctx_c.add_pass()
                        else:
                            ctx_c.add_failure(
                                f"Time bounds variable '{bounds_name}' has {detail}. "
                                f"It is {severity_word(sev_time)} to repack this "
                                "variable using a project repacking tool."
                            )
                        results.append(ctx_c.to_result())
            except Exception as exc:
                ctx_c = TestCtx(sev_time, "[FILE004c] Time bounds chunking")
                ctx_c.add_failure(f"Unable to inspect time bounds chunking: {exc}")
                results.append(ctx_c.to_result())

    # FILE004d — Data-variable chunk size
    if run_data:
        try:
            has_variable_id = "variable_id" in packing_file.attrs
        except Exception as exc:
            has_variable_id = False
            ctx_d = TestCtx(
                sev_data,
                "[FILE004d] Internal packing : Data variable chunking",
            )
            ctx_d.add_failure(
                f"Unable to inspect 'variable_id' attribute presence: {exc}"
            )
            results.append(ctx_d.to_result())

        if has_variable_id:
            try:
                variable_id = _attr_to_str(packing_file.attrs["variable_id"])
            except Exception:
                variable_id = None

            try:
                has_variable = bool(variable_id) and variable_id in packing_file
            except Exception as exc:
                has_variable = False
                ctx_d = TestCtx(
                    sev_data,
                    "[FILE004d] Internal packing : Data variable chunking",
                )
                ctx_d.add_failure(f"Unable to inspect data variable presence: {exc}")
                results.append(ctx_d.to_result())

            if has_variable:
                try:
                    data_var = packing_file[variable_id]
                except Exception as exc:
                    data_var = None
                    ctx_d = TestCtx(
                        sev_data,
                        "[FILE004d] Internal packing : Data variable chunking "
                        f"('{variable_id}')",
                    )
                    ctx_d.add_failure(
                        f"Unable to access data variable '{variable_id}': {exc}"
                    )
                    results.append(ctx_d.to_result())

                if data_var is not None:
                    ctx_d = TestCtx(
                        sev_data,
                        "[FILE004d] Internal packing : Data variable chunking "
                        f"('{variable_id}')",
                    )
                    ok, detail = _check_data_variable(
                        data_var,
                        min_chunk_size_bytes=min_chunk_size_bytes,
                        frequency=frequency,
                        frequency_min_timesteps=frequency_min_timesteps,
                        has_time=has_time,
                    )
                    if ok:
                        ctx_d.add_pass()
                    else:
                        ctx_d.add_failure(
                            f"Data variable '{variable_id}': {detail}. It is "
                            f"{severity_word(sev_data)} to repack its chunking "
                            "using a project repacking tool."
                        )
                    results.append(ctx_d.to_result())

    return results
