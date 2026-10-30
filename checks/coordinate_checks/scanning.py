"""Shared bounded-I/O helpers for coordinate and grid checks."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import product

import numpy as np

# Process at most 2**18 values per logical block. For float64 data,
# 2**18 values at 8 bytes each use exactly 2 MiB. This limits the memory used
# by temporary arrays while avoiding too many small reads.
DEFAULT_TARGET_VALUES = 2**18

# 32 * 2**20 bytes equals 32 MiB. This keeps the HDF5 cache modest for
# normally packed files. It is enlarged below when one physical storage chunk
# would not otherwise fit completely.
DEFAULT_SCAN_CACHE_BYTES = 32 * 2**20


def storage_chunk_working_set_bytes(variable, *, trailing_dimensions=0):
    """Estimate the cache needed for one logical block.

    Trailing dimensions, such as vertices, are always read together. If HDF5
    stores them in several chunks, the cache must hold all of those chunks to
    avoid unpacking the same data repeatedly.
    """
    try:
        chunks = variable.chunking()
        shape = tuple(int(length) for length in variable.shape)
        if not isinstance(chunks, (list, tuple)) or not chunks:
            return 0
        if trailing_dimensions < 0 or trailing_dimensions > len(shape):
            raise ValueError("invalid trailing dimension count")
        chunk_bytes = int(np.prod(chunks)) * int(variable.dtype.itemsize)
        trailing_chunks = 1
        if trailing_dimensions:
            for length, chunk_length in zip(
                shape[-trailing_dimensions:], chunks[-trailing_dimensions:]
            ):
                trailing_chunks *= (int(length) + int(chunk_length) - 1) // int(
                    chunk_length
                )
        return chunk_bytes * trailing_chunks
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0


@contextmanager
def temporary_chunk_cache(
    variable,
    maximum_bytes=DEFAULT_SCAN_CACHE_BYTES,
    *,
    minimum_working_set_bytes=0,
):
    """Temporarily adjust a variable's HDF5 cache for a full scan.

    The cache normally stays below ``maximum_bytes``. If one storage chunk is
    larger, the cache is enlarged enough to hold it. This prevents each small
    read from unpacking the same large chunk again. The original setting is
    restored after the scan, including when the check raises an exception.
    """
    try:
        size, nelems, preemption = variable.get_var_chunk_cache()
    except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
        yield
        return
    effective_maximum = max(maximum_bytes, minimum_working_set_bytes)
    if size < minimum_working_set_bytes:
        target_size = minimum_working_set_bytes
    elif size > effective_maximum:
        target_size = effective_maximum
    else:
        yield
        return
    try:
        variable.set_var_chunk_cache(target_size, nelems, preemption)
    except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
        yield
        return
    try:
        yield
    finally:
        try:
            # Resetting also releases the scan cache. netCDF4 allocates the
            # restored cache lazily if a later consumer accesses the variable.
            variable.set_var_chunk_cache(size, nelems, preemption)
        except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
            pass


@contextmanager
def scan_chunk_cache(
    variable,
    *,
    trailing_dimensions=0,
    maximum_cache_bytes=DEFAULT_SCAN_CACHE_BYTES,
):
    """Manage cache for a variable read with externally supplied selections."""
    working_set = storage_chunk_working_set_bytes(
        variable,
        trailing_dimensions=trailing_dimensions,
    )
    with temporary_chunk_cache(
        variable,
        maximum_bytes=maximum_cache_bytes,
        minimum_working_set_bytes=working_set,
    ):
        yield


def spatial_selection(origin, spatial_shape):
    """Return slices selecting a spatial block from its origin and shape."""
    return tuple(
        slice(start, start + length) for start, length in zip(origin, spatial_shape)
    )


def global_index(origin, local_shape, flat_index):
    """Translate a flat index within a logical block to the full array index."""
    local = np.unravel_index(flat_index, local_shape)
    return tuple(int(start + offset) for start, offset in zip(origin, local))


def iter_spatial_chunks(
    variable,
    *,
    trailing_dimensions=0,
    target_values=None,
):
    """Yield bounded blocks without loading a complete array.

    ``trailing_dimensions`` keeps dimensions such as vertices complete in each
    block. All other dimensions may be split, including grids with more than
    two spatial dimensions.
    """
    if not variable.size:
        return
    shape = tuple(int(length) for length in variable.shape)
    if trailing_dimensions < 0 or trailing_dimensions > len(shape):
        raise ValueError("invalid trailing dimension count")
    target_values = int(
        DEFAULT_TARGET_VALUES if target_values is None else target_values
    )
    if target_values <= 0:
        raise ValueError("target_values must be positive")

    split_at = len(shape) - trailing_dimensions if trailing_dimensions else len(shape)
    spatial_shape = shape[:split_at]
    trailing_shape = shape[split_at:]
    # Keep complete trailing dimensions in each block while respecting the value
    # limit. For example, 16 vertices per cell reduce the number of cells in a
    # block to target_values // 16.
    trailing_size = int(np.prod(trailing_shape)) if trailing_shape else 1
    target_cells = max(1, target_values // max(1, trailing_size))
    if not spatial_shape:
        yield (), variable[...]
        return

    # Use as much of the fastest-varying dimensions as possible, then split
    # earlier dimensions until the block fits within the value limit.
    remaining = target_cells
    block_shape = [1] * len(spatial_shape)
    for axis in range(len(spatial_shape) - 1, -1, -1):
        block_shape[axis] = min(spatial_shape[axis], max(1, remaining))
        remaining = max(1, remaining // block_shape[axis])

    starts = (
        range(0, length, block) for length, block in zip(spatial_shape, block_shape)
    )
    for origin in product(*starts):
        selection = spatial_selection(origin, block_shape)
        if trailing_dimensions:
            selection += (slice(None),) * trailing_dimensions
        yield tuple(origin), variable[selection]


def iter_bounded_spatial_chunks(
    variable,
    *,
    trailing_dimensions=0,
    target_values=None,
    maximum_cache_bytes=DEFAULT_SCAN_CACHE_BYTES,
):
    """Yield logical blocks while managing the physical HDF5 chunk cache."""
    with scan_chunk_cache(
        variable,
        trailing_dimensions=trailing_dimensions,
        maximum_cache_bytes=maximum_cache_bytes,
    ):
        yield from iter_spatial_chunks(
            variable,
            trailing_dimensions=trailing_dimensions,
            target_values=target_values,
        )
