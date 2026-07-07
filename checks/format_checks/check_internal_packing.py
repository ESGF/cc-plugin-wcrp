#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[FILE004] CMIP7 internal packing checks.

Faithfully ports the logic of the official `check_cmip7_packing` script
(https://github.com/NCAS-CMS/cmip7repack)

Four sub-checks, each producing an independent Result:
  FILE004a — Consolidated internal metadata
  FILE004b — Time coordinate variable: single chunk or contiguous
  FILE004c — Time bounds variable: single chunk or contiguous
  FILE004d — Data variable: single chunk / contiguous, or chunk >= 4 MiB

Requires: pyfive >= 1.1.1  (pure-Python HDF5 reader, no libhdf5 needed)
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

    _PYFIVE_MIN = Version("1.1.1")
    _pyfive_version = Version(__import__("importlib.metadata", fromlist=["version"]).version("pyfive"))
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

_DEFAULT_MIN_CHUNK_SIZE_BYTES = 4 * (2**20)  # 4 194 304 bytes


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


def _extract_time_chunk_steps(var) -> int | None:
    """Return chunk length along the time dimension, if available."""
    try:
        chunks = var.chunks
    except Exception:
        return None
    if chunks is None:
        return None

    try:
        dims = tuple(str(d) for d in var.dimensions)
    except Exception:
        return None

    if "time" not in dims:
        return None

    time_dim_index = dims.index("time")
    if time_dim_index >= len(chunks):
        return None

    return int(chunks[time_dim_index])


def _is_single_chunk_or_contiguous(var) -> tuple[bool, str]:
    """
    True  if the variable is contiguous (chunks is None)
              or has exactly one chunk.
    Mirrors the logic in the official check_cmip7_packing script.
    """
    try:
        chunks = var.chunks          # None -> contiguous
    except Exception as e:
        return False, f"unable to read chunk metadata ({e})"
    if chunks is None:
        return True, "contiguous"
    try:
        n = var.id.get_num_chunks()
    except Exception as e:
        return False, f"unable to read number of chunks ({e})"
    if n <= 1:
        return True, f"1 chunk of shape {tuple(chunks)}"
    return False, f"{n} chunks (expected 1 chunk or contiguous)"


def _check_data_variable(
    var,
    min_chunk_size_bytes: int,
    frequency: str | None = None,
    frequency_min_timesteps: dict[str, int] | None = None,
) -> tuple[bool, str]:
    """

    Pass conditions (any one sufficient):
      • contiguous (chunks is None)
      • exactly 1 chunk
      • uncompressed chunk size >= configurable threshold (4 MiB by default)
      • adding one element along the leading dimension would reach threshold
        (the "lee_way" rule from the official script)
      • optional frequency-specific fallback: minimum timesteps per chunk
    """
    try:
        chunks = var.chunks
    except Exception as e:
        return False, f"unable to read chunk metadata ({e})"
    if chunks is None:
        return True, "contiguous"

    try:
        n = var.id.get_num_chunks()
    except Exception as e:
        return False, f"unable to read number of chunks ({e})"
    if n <= 1:
        return True, f"1 chunk"

    try:
        wordsize = var.dtype.itemsize
        chunksize = prod(chunks) * wordsize
    except Exception as e:
        return False, f"unable to compute chunk byte size ({e})"

    # Adding one element along leading dim gives this extra size
    try:
        lee_way = prod(chunks[1:]) * wordsize if len(chunks) > 1 else 0
    except Exception as e:
        return False, f"unable to compute chunk threshold margin ({e})"

    if chunksize + lee_way >= min_chunk_size_bytes:
        return True, (
            f"chunk size {chunksize} B "
            f"(>= {min_chunk_size_bytes - lee_way} B threshold)"
        )

    if frequency_min_timesteps:
        if not frequency or str(frequency).strip().lower() == "unknown":
            known = ", ".join(sorted(str(k) for k in frequency_min_timesteps))
            return False, (
                f"uncompressed chunk size {chunksize} B "
                f"(expected at least {min_chunk_size_bytes - lee_way} B, "
                f"or 1 chunk, or contiguous). "
                f"Warning: frequency-specific exceptions are configured for {known}, "
                "but frequency cannot be inferred from the file metadata."
            )

        if frequency in frequency_min_timesteps:
            required_steps = int(frequency_min_timesteps[frequency])
            chunk_time_steps = _extract_time_chunk_steps(var)

            if chunk_time_steps is not None and chunk_time_steps >= required_steps:
                return True, (
                    f"chunk size {chunksize} B below threshold, but frequency "
                    f"exception for '{frequency}' is met "
                    f"({chunk_time_steps} >= {required_steps} timesteps per time-chunk)"
                )

            return False, (
                f"uncompressed chunk size {chunksize} B "
                f"(expected at least {min_chunk_size_bytes - lee_way} B, "
                f"or at least {required_steps} timesteps per time-chunk for "
                f"frequency '{frequency}', or 1 chunk, or contiguous)"
            )

        known = ", ".join(sorted(str(k) for k in frequency_min_timesteps))
        return False, (
            f"uncompressed chunk size {chunksize} B "
            f"(expected at least {min_chunk_size_bytes - lee_way} B, "
            f"or 1 chunk, or contiguous)."
        )

    return False, (
        f"uncompressed chunk size {chunksize} B "
        f"(expected at least {min_chunk_size_bytes - lee_way} B, "
        f"or 1 chunk, or contiguous)"
    )


# ---------------------------------------------------------------------------
# Public check function
# ---------------------------------------------------------------------------

def check_internal_packing(
    ds,
    severity=BaseCheck.HIGH,
    severity_a=None,
    severity_b=None,
    severity_c=None,
    severity_d=None,
    min_chunk_size_bytes: int = _DEFAULT_MIN_CHUNK_SIZE_BYTES,
    frequency: str | None = None,
    frequency_min_timesteps: dict[str, int] | None = None,
) -> list:
    """
    [FILE004] Internal packing checks for CMIP7 and similar workflows.
    """
    results = []

    try:
        min_chunk_size_bytes = int(min_chunk_size_bytes)
        if min_chunk_size_bytes <= 0:
            raise ValueError
    except Exception:
        min_chunk_size_bytes = _DEFAULT_MIN_CHUNK_SIZE_BYTES

    default_severity = severity
    sev_a = severity_a or default_severity
    sev_b = severity_b or default_severity
    sev_c = severity_c or default_severity
    sev_d = severity_d or default_severity

    # -- pyfive available? ---------------------------------------------------
    if not _PYFIVE_OK:
        ctx = TestCtx(BaseCheck.HIGH, "[FILE004] CMIP7 internal packing")
        ctx.add_failure(
            f"Optional dependency 'pyfive >= 1.1.1' is not installed or "
            f"incompatible — FILE004 skipped. ({_PYFIVE_ERR})"
        )
        return [ctx.to_result()]

    # -- get file path -------------------------------------------------------
    file_path = _get_file_path(ds)
    if not file_path:
        ctx = TestCtx(default_severity, "[FILE004] Internal packing")
        qualifier = severity_word(default_severity)
        ctx.add_failure(
            "Could not retrieve dataset file path — FILE004 skipped. "
            f"It is {qualifier} to run internal packing checks on an accessible local file."
        )
        return [ctx.to_result()]

    # -- open with pyfive  ---------------------------
    try:
        f = pyfive.File(file_path)
    except Exception as e:
        ctx = TestCtx(default_severity, "[FILE004] Internal packing")
        ctx.add_failure(f"Could not open file with pyfive: {e}")
        return [ctx.to_result()]

    try:
        # ----------------------------------------------------------------
        # FILE004a — Consolidated internal metadata
        # ----------------------------------------------------------------
        ctx_a = TestCtx(sev_a, "[FILE004a] Internal packing : Consolidated internal metadata")
        try:
            if f.consolidated_metadata:
                ctx_a.add_pass()
            else:
                qualifier = severity_word(sev_a)
                ctx_a.add_failure(
                    "File does not have consolidated internal metadata. "
                    f"It is {qualifier} to consolidate internal metadata "
                    "using cmip7repack or comparable tools."
                )
        except Exception as e:
            ctx_a.add_failure(f"Unable to inspect consolidated metadata: {e}")
        results.append(ctx_a.to_result())

        # ----------------------------------------------------------------
        # FILE004b — Time coordinate: single chunk or contiguous
        # ----------------------------------------------------------------
        try:
            has_time = "time" in f
        except Exception as e:
            has_time = False
            ctx_b = TestCtx(sev_b, "[FILE004b] Internal packing : Time coordinate chunking")
            ctx_b.add_failure(f"Unable to inspect time coordinate presence: {e}")
            results.append(ctx_b.to_result())

        if has_time:
            try:
                t = f["time"]
            except Exception as e:
                ctx_b = TestCtx(sev_b, "[FILE004b] Internal packing : Time coordinate chunking")
                ctx_b.add_failure(f"Unable to access time coordinate variable: {e}")
                results.append(ctx_b.to_result())
                t = None

            if t is not None:
                ctx_b = TestCtx(sev_b, "[FILE004b] Internal packing : Time coordinate chunking")
                ok, detail = _is_single_chunk_or_contiguous(t)
                if ok:
                    ctx_b.add_pass()
                else:
                    qualifier = severity_word(sev_b)
                    ctx_b.add_failure(
                        f"Time coordinate variable 'time' has {detail}. "
                        f"It is {qualifier} to repack this variable "
                        "using cmip7repack or comparable tools."
                    )
                results.append(ctx_b.to_result())

                # ------------------------------------------------------------
                # FILE004c — Time bounds: single chunk or contiguous
                # ------------------------------------------------------------
                try:
                    if "bounds" in t.attrs:
                        bounds_name = _attr_to_str(t.attrs["bounds"])
                        try:
                            has_bounds_var = bounds_name in f
                        except Exception as e:
                            has_bounds_var = False
                            ctx_c = TestCtx(sev_c, "[FILE004c] Time bounds chunking")
                            ctx_c.add_failure(f"Unable to inspect time bounds variable presence: {e}")
                            results.append(ctx_c.to_result())

                        if has_bounds_var:
                            b = f[bounds_name]
                            ctx_c = TestCtx(
                                sev_c,
                                f"[FILE004c] Internal packing : Time bounds chunking ('{bounds_name}')",
                            )
                            ok, detail = _is_single_chunk_or_contiguous(b)
                            if ok:
                                ctx_c.add_pass()
                            else:
                                qualifier = severity_word(sev_c)
                                ctx_c.add_failure(
                                    f"Time bounds variable '{bounds_name}' has {detail}. "
                                    f"It is {qualifier} to repack this variable "
                                    "using cmip7repack or comparable tools."
                                )
                            results.append(ctx_c.to_result())
                except Exception as e:
                    ctx_c = TestCtx(sev_c, "[FILE004c] Time bounds chunking")
                    ctx_c.add_failure(f"Unable to inspect time bounds chunking: {e}")
                    results.append(ctx_c.to_result())

        # ----------------------------------------------------------------
        # FILE004d — Data variable chunk size
        # ----------------------------------------------------------------
        try:
            has_variable_id_attr = "variable_id" in f.attrs
        except Exception as e:
            has_variable_id_attr = False
            ctx_d = TestCtx(sev_d, "[FILE004d] Internal packing : Data variable chunking")
            ctx_d.add_failure(f"Unable to inspect 'variable_id' attribute presence: {e}")
            results.append(ctx_d.to_result())

        if has_variable_id_attr:
            try:
                variable_id = _attr_to_str(f.attrs["variable_id"])
            except Exception:
                variable_id = None

            try:
                has_variable = bool(variable_id) and (variable_id in f)
            except Exception as e:
                has_variable = False
                ctx_d = TestCtx(sev_d, "[FILE004d] Internal packing : Data variable chunking")
                ctx_d.add_failure(f"Unable to inspect data variable presence: {e}")
                results.append(ctx_d.to_result())

            if has_variable:
                try:
                    d = f[variable_id]
                except Exception as e:
                    d = None
                    ctx_d = TestCtx(
                        sev_d,
                        f"[FILE004d] Internal packing : Data variable chunking ('{variable_id}')",
                    )
                    ctx_d.add_failure(f"Unable to access data variable '{variable_id}': {e}")
                    results.append(ctx_d.to_result())

            if has_variable and d is not None:
                ctx_d = TestCtx(
                    sev_d,
                    f"[FILE004d] Internal packing : Data variable chunking ('{variable_id}')",
                )
                ok, detail = _check_data_variable(
                    d,
                    min_chunk_size_bytes=min_chunk_size_bytes,
                    frequency=frequency,
                    frequency_min_timesteps=frequency_min_timesteps,
                )
                if ok:
                    ctx_d.add_pass()
                else:
                    qualifier = severity_word(sev_d)
                    ctx_d.add_failure(
                        f"Data variable '{variable_id}': {detail}. "
                        f"It is {qualifier} to repack chunking "
                        "using cmip7repack or comparable tools."
                    )
                results.append(ctx_d.to_result())

    finally:
        try:
            f.close()
        except Exception:
            pass

    return results
