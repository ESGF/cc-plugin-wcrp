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

_FOUR_MiB = 4 * (2**20)  # 4 194 304 bytes


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


def _is_single_chunk_or_contiguous(var) -> tuple[bool, str]:
    """
    True  if the variable is contiguous (chunks is None)
              or has exactly one chunk.
    Mirrors the logic in the official check_cmip7_packing script.
    """
    chunks = var.chunks          # None → contiguous
    if chunks is None:
        return True, "contiguous"
    n = var.id.get_num_chunks()
    if n <= 1:
        return True, f"1 chunk of shape {tuple(chunks)}"
    return False, f"{n} chunks (expected 1 chunk or contiguous)"


def _check_data_variable(var, variable_id: str) -> tuple[bool, str]:
    """

    Pass conditions (any one sufficient):
      • contiguous (chunks is None)
      • exactly 1 chunk
      • uncompressed chunk size >= 4 MiB
      • adding one element along the leading dimension would reach >= 4 MiB
        (the "lee_way" rule from the official script)
    """
    chunks = var.chunks
    if chunks is None:
        return True, "contiguous"

    n = var.id.get_num_chunks()
    if n <= 1:
        return True, f"1 chunk"

    wordsize  = var.dtype.itemsize
    chunksize = prod(chunks) * wordsize

    # Adding one element along leading dim gives this extra size
    lee_way = prod(chunks[1:]) * wordsize if len(chunks) > 1 else 0

    if chunksize + lee_way >= _FOUR_MiB:
        return True, (
            f"chunk size {chunksize} B "
            f"(>= {_FOUR_MiB - lee_way} B threshold)"
        )

    return False, (
        f"uncompressed chunk size {chunksize} B "
        f"(expected at least {_FOUR_MiB - lee_way} B, "
        f"or 1 chunk, or contiguous)"
    )


# ---------------------------------------------------------------------------
# Public check function
# ---------------------------------------------------------------------------

def check_cmip7_packing(ds, severity=BaseCheck.HIGH) -> list:
    """
    [FILE004] CMIP7 internal packing checks.
    """
    results = []

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
        ctx = TestCtx(severity, "[FILE004] CMIP7 internal packing")
        ctx.add_failure("Could not retrieve dataset file path — FILE004 skipped.")
        return [ctx.to_result()]

    # -- open with pyfive  ---------------------------
    try:
        f = pyfive.File(file_path)
    except Exception as e:
        ctx = TestCtx(severity, "[FILE004] CMIP7 internal packing")
        ctx.add_failure(f"Could not open file with pyfive: {e}")
        return [ctx.to_result()]

    try:
        # ----------------------------------------------------------------
        # FILE004a — Consolidated internal metadata
        # ----------------------------------------------------------------
        ctx_a = TestCtx(severity, "[FILE004a] CMIP7 internal packing : Consolidated internal metadata")
        try:
            if f.consolidated_metadata:
                ctx_a.add_pass()
            else:
                ctx_a.add_failure(
                    "File does not have consolidated internal metadata. "
                    "Run 'cmip7repack' to fix this."
                )
        except Exception as e:
            ctx_a.add_failure(f"Unable to inspect consolidated metadata: {e}")
        results.append(ctx_a.to_result())

        # ----------------------------------------------------------------
        # FILE004b — Time coordinate: single chunk or contiguous
        # ----------------------------------------------------------------
        if "time" in f:
            t = f["time"]
            ctx_b = TestCtx(severity, "[FILE004b] CMIP7 internal packing : Time coordinate chunking")
            ok, detail = _is_single_chunk_or_contiguous(t)
            if ok:
                ctx_b.add_pass()
            else:
                ctx_b.add_failure(
                    f"Time coordinate variable 'time' has {detail}. "
                    f"Run 'cmip7repack' to fix this."
                )
            results.append(ctx_b.to_result())

            # ------------------------------------------------------------
            # FILE004c — Time bounds: single chunk or contiguous
            # ------------------------------------------------------------
            try:
                if "bounds" in t.attrs:
                    bounds_name = _attr_to_str(t.attrs["bounds"])
                    if bounds_name in f:
                        b = f[bounds_name]
                        ctx_c = TestCtx(
                            severity,
                            f"[FILE004c] CMIP7 internal packing : Time bounds chunking ('{bounds_name}')",
                        )
                        ok, detail = _is_single_chunk_or_contiguous(b)
                        if ok:
                            ctx_c.add_pass()
                        else:
                            ctx_c.add_failure(
                                f"Time bounds variable '{bounds_name}' has {detail}. "
                                f"Run 'cmip7repack' to fix this."
                            )
                        results.append(ctx_c.to_result())
            except Exception as e:
                ctx_c = TestCtx(severity, "[FILE004c] Time bounds chunking")
                ctx_c.add_failure(f"Unable to inspect time bounds chunking: {e}")
                results.append(ctx_c.to_result())

        # ----------------------------------------------------------------
        # FILE004d — Data variable chunk size
        # ----------------------------------------------------------------
        if "variable_id" in f.attrs:
            try:
                variable_id = _attr_to_str(f.attrs["variable_id"])
            except Exception:
                variable_id = None

            if variable_id and variable_id in f:
                d = f[variable_id]
                ctx_d = TestCtx(
                    severity,
                    f"[FILE004d] CMIP7 internal packing : Data variable chunking ('{variable_id}')",
                )
                ok, detail = _check_data_variable(d, variable_id)
                if ok:
                    ctx_d.add_pass()
                else:
                    ctx_d.add_failure(
                        f"Data variable '{variable_id}': {detail}. "
                        f"Run 'cmip7repack' to fix this."
                    )
                results.append(ctx_d.to_result())

    finally:
        f.close()

    return results
