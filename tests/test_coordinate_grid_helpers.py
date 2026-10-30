from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from compliance_checker.base import BaseCheck

from checks.coordinate_checks.grid import (
    _cell_circular_widths,
    _check_grid_geometry_missing_values,
    _check_grid_variable_valid_range,
    _check_vertex_valid_range,
    _grid_dimensions,
    _grid_mapping_name,
    _identify_coordinate,
)
from checks.coordinate_checks.scanning import (
    DEFAULT_SCAN_CACHE_BYTES,
    global_index,
    iter_bounded_spatial_chunks,
    iter_spatial_chunks,
    storage_chunk_working_set_bytes,
    temporary_chunk_cache,
)
from checks.coordinate_checks.validation import (
    Findings,
    _cyclic_absolute_difference,
    _is_longitude,
    _unwrap_longitude_pairs,
    check_single_longitude_cycle,
    check_valid_range,
    valid_range_slack,
)


def _findings():
    return Findings(
        {
            "valid_range": BaseCheck.HIGH,
            "grid": BaseCheck.HIGH,
            "grid_geometry_missing_values": BaseCheck.HIGH,
            "grid_latitude_valid_range": BaseCheck.MEDIUM,
            "grid_longitude_valid_range": BaseCheck.MEDIUM,
            "grid_longitude_single_cycle": BaseCheck.MEDIUM,
        }
    )


class _AttributeVariable:
    def __init__(self, **attributes):
        self.attributes = attributes

    def getncattr(self, name):
        if name not in self.attributes:
            raise AttributeError(name)
        return self.attributes[name]


class _UnreadableArray:
    def __init__(self, shape):
        self.shape = shape
        self.size = int(np.prod(shape))

    def __getitem__(self, selection):
        raise OSError("simulated read failure")


class _CachedArray:
    def __init__(self, values, cache_size=64 * 2**20, chunks=None):
        self.values = np.asanyarray(values)
        self.shape = self.values.shape
        self.size = self.values.size
        self.dtype = self.values.dtype
        self.chunks = chunks
        self.cache = (cache_size, 1000, 0.75)
        self.cache_changes = []
        self.selections = []

    def __getitem__(self, selection):
        self.selections.append(selection)
        return self.values[selection]

    def get_var_chunk_cache(self):
        return self.cache

    def set_var_chunk_cache(self, size, nelems, preemption):
        self.cache = (size, nelems, preemption)
        self.cache_changes.append(self.cache)

    def chunking(self):
        return self.chunks if self.chunks is not None else "contiguous"


class _FailingCacheArray(_CachedArray):
    def __init__(self, values, *, fail_get=False, fail_set_calls=()):
        super().__init__(values)
        self.fail_get = fail_get
        self.fail_set_calls = set(fail_set_calls)
        self.set_calls = 0

    def get_var_chunk_cache(self):
        if self.fail_get:
            raise RuntimeError("simulated cache read failure")
        return super().get_var_chunk_cache()

    def set_var_chunk_cache(self, size, nelems, preemption):
        self.set_calls += 1
        if self.set_calls in self.fail_set_calls:
            raise RuntimeError("simulated cache update failure")
        super().set_var_chunk_cache(size, nelems, preemption)


def test_spatial_chunks_splits_one_dimensional_values(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.scanning.DEFAULT_TARGET_VALUES", 4)
    values = np.arange(10)

    chunks = list(iter_spatial_chunks(values))

    assert [origin for origin, _ in chunks] == [(0,), (4,), (8,)]
    np.testing.assert_array_equal(
        np.concatenate([chunk for _, chunk in chunks]), values
    )


def test_spatial_chunks_handles_empty_and_exact_chunk_boundaries(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.scanning.DEFAULT_TARGET_VALUES", 4)

    assert list(iter_spatial_chunks(np.asarray([]))) == []
    chunks = list(iter_spatial_chunks(np.arange(8)))
    assert [origin for origin, _ in chunks] == [(0,), (4,)]
    assert [chunk.shape for _, chunk in chunks] == [(4,), (4,)]


def test_spatial_chunks_bounds_two_dimensional_blocks(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.scanning.DEFAULT_TARGET_VALUES", 4)
    values = np.arange(15).reshape(3, 5)

    chunks = list(iter_spatial_chunks(values))

    assert [origin for origin, _ in chunks] == [
        (0, 0),
        (0, 4),
        (1, 0),
        (1, 4),
        (2, 0),
        (2, 4),
    ]
    assert all(chunk.size <= 4 for _, chunk in chunks)


def test_spatial_chunks_accounts_for_vertex_dimension(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.scanning.DEFAULT_TARGET_VALUES", 8)
    vertices = np.arange(20).reshape(5, 4)

    chunks = list(iter_spatial_chunks(vertices, trailing_dimensions=1))

    assert [origin for origin, _ in chunks] == [(0,), (2,), (4,)]
    assert [chunk.shape for _, chunk in chunks] == [(2, 4), (2, 4), (1, 4)]


def test_spatial_chunks_handles_scalar_and_higher_dimensional_grids(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.scanning.DEFAULT_TARGET_VALUES", 4)

    scalar_chunks = list(iter_spatial_chunks(np.asarray(7.0)))
    volume_chunks = list(iter_spatial_chunks(np.arange(12).reshape(3, 2, 2)))

    assert scalar_chunks[0][0] == ()
    assert scalar_chunks[0][1] == 7.0
    assert [origin for origin, _ in volume_chunks] == [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    ]


def test_spatial_chunks_bound_every_dimension_of_an_arbitrary_grid():
    values = np.arange(2 * 5 * 7).reshape(2, 5, 7)

    chunks = list(iter_spatial_chunks(values, target_values=6))

    assert all(chunk.size <= 6 for _, chunk in chunks)
    assert {origin[0] for origin, _ in chunks} == {0, 1}
    assert {origin[1] for origin, _ in chunks} == {0, 1, 2, 3, 4}
    assert {origin[2] for origin, _ in chunks} == {0, 6}


def test_bounded_variable_chunk_cache_caps_and_restores_after_errors():
    variable = _CachedArray(np.arange(4))
    original = variable.cache

    with (
        pytest.raises(RuntimeError, match="simulated check failure"),
        temporary_chunk_cache(variable),
    ):
        assert variable.cache[0] == DEFAULT_SCAN_CACHE_BYTES
        raise RuntimeError("simulated check failure")

    assert variable.cache == original
    assert [change[0] for change in variable.cache_changes] == [
        DEFAULT_SCAN_CACHE_BYTES,
        original[0],
    ]


def test_bounded_variable_chunk_cache_leaves_small_and_plain_arrays_unchanged():
    variable = _CachedArray(np.arange(4), cache_size=8 * 2**20)

    with temporary_chunk_cache(variable):
        assert variable.cache[0] == 8 * 2**20
    with temporary_chunk_cache(np.arange(4)):
        pass

    assert variable.cache_changes == []


@pytest.mark.parametrize(
    ("fail_get", "fail_set_calls"),
    [
        (True, ()),
        (False, (1,)),
        (False, (2,)),
    ],
)
def test_bounded_scan_survives_cache_api_failures(fail_get, fail_set_calls):
    variable = _FailingCacheArray(
        np.arange(7),
        fail_get=fail_get,
        fail_set_calls=fail_set_calls,
    )

    chunks = list(iter_bounded_spatial_chunks(variable, target_values=3))

    np.testing.assert_array_equal(
        np.concatenate([chunk for _, chunk in chunks]),
        variable.values,
    )
    assert [origin for origin, _ in chunks] == [(0,), (3,), (6,)]


def test_vertex_chunk_working_set_includes_all_trailing_storage_chunks():
    vertices = _CachedArray(
        np.zeros((100, 16), dtype="float64"),
        chunks=(25, 1),
    )

    assert storage_chunk_working_set_bytes(
        vertices, trailing_dimensions=1
    ) == 25 * 16 * 8

    with temporary_chunk_cache(
        vertices,
        maximum_bytes=128,
        minimum_working_set_bytes=storage_chunk_working_set_bytes(
            vertices, trailing_dimensions=1
        ),
    ):
        assert vertices.cache[0] == 25 * 16 * 8

    assert vertices.cache[0] == 64 * 2**20


def test_vertex_chunk_cache_grows_to_retain_one_oversized_storage_chunk():
    vertices = _CachedArray(
        np.zeros((100, 16), dtype="float64"),
        cache_size=128,
        chunks=(25, 16),
    )
    working_set = storage_chunk_working_set_bytes(
        vertices, trailing_dimensions=1
    )

    with temporary_chunk_cache(
        vertices,
        maximum_bytes=128,
        minimum_working_set_bytes=working_set,
    ):
        assert vertices.cache[0] == working_set

    assert vertices.cache[0] == 128


def test_global_cell_index_translates_flat_chunk_index():
    assert global_index((10, 20), (2, 3), 4) == (11, 21)


def test_identify_coordinate_prefers_name_and_controls_standard_name_fallback():
    exact = _AttributeVariable(standard_name="not_longitude")
    fallback = _AttributeVariable(standard_name="longitude")
    dataset = SimpleNamespace(
        variables={"candidate": fallback, "longitude": exact}
    )

    assert _identify_coordinate(dataset, "longitude", "longitude", True) is exact

    dataset.variables.pop("longitude")
    assert _identify_coordinate(dataset, "longitude", "longitude", True) is fallback
    assert _identify_coordinate(dataset, "longitude", "longitude", False) is None


def test_grid_mapping_name_parses_simple_and_extended_cf_syntax():
    mapping = _AttributeVariable(grid_mapping_name="rotated_latitude_longitude")
    data = _AttributeVariable(grid_mapping="rotated_pole")
    dataset = SimpleNamespace(variables={"rotated_pole": mapping})

    assert _grid_mapping_name(dataset, data) == "rotated_latitude_longitude"

    data.attributes["grid_mapping"] = "rotated_pole: rlat rlon"
    assert _grid_mapping_name(dataset, data) == "rotated_latitude_longitude"
    assert _grid_mapping_name(dataset, None) == ""

    data.attributes["grid_mapping"] = "missing_mapping"
    assert _grid_mapping_name(dataset, data) == ""


@pytest.mark.parametrize(
    ("dimensions", "entry_dimensions", "vertex_dimension", "expected"),
    [
        (("y", "x"), ["longitude", "latitude"], None, ["y", "x"]),
        (
            ("y", "x"),
            ["vertices", "longitude", "latitude"],
            "nv",
            ["y", "x", "nv"],
        ),
        (("cell",), ["longitude", "latitude"], None, ["cell"]),
        (("j", "x"), ["longitude", "latitude"], None, ["j", "x"]),
        (("y", "x"), [], "nv", ["y", "x", "nv"]),
    ],
)
def test_grid_dimensions_resolve_axes_shared_cells_and_vertices(
    dimensions, entry_dimensions, vertex_dimension, expected
):
    catalogue = SimpleNamespace(
        grid_axes={
            "x": {"out_name": "x", "axis": "X"},
            "y": {"out_name": "y", "axis": "Y"},
        }
    )

    actual = _grid_dimensions(
        catalogue,
        {"dimensions": entry_dimensions},
        dimensions,
        vertex_dimension,
    )

    assert actual == expected


def test_cell_circular_widths_use_shortest_arc_and_ignore_padding():
    cells = np.asarray(
        [
            [359.0, 0.0, 1.0, np.nan],
            [10.0, 10.0, 10.0, np.nan],
            [-180.0, 0.0, 180.0, np.nan],
        ]
    )

    np.testing.assert_allclose(_cell_circular_widths(cells), [2.0, 0.0, 180.0])


def test_grid_variable_valid_range_counts_cells_and_reports_extreme_indices():
    variable = np.ma.masked_invalid(
        [[1.0, -2.0, 3.0], [4.0, 12.0, np.nan]]
    )
    findings = _findings()

    _check_grid_variable_valid_range(
        findings,
        variable,
        "longitude",
        {"valid_min": 0.0, "valid_max": 10.0},
        family="grid_longitude_valid_range",
    )

    messages = findings.messages["grid_longitude_valid_range"]
    assert any(
        "1 of 6 grid cells" in msg
        and "Lowest value: -2.0 at index (0, 1)" in msg
        for msg in messages
    )
    assert any(
        "1 of 6 grid cells" in msg
        and "Highest value: 12.0 at index (1, 1)" in msg
        for msg in messages
    )


def test_grid_variable_valid_range_reports_nonfinite_and_read_failures():
    nonfinite_findings = _findings()
    _check_grid_variable_valid_range(
        nonfinite_findings,
        np.asarray([np.nan, np.inf]),
        "longitude",
        {"valid_min": 0.0, "valid_max": 360.0},
        family="grid_longitude_valid_range",
    )
    assert nonfinite_findings.messages["grid_longitude_valid_range"] == []

    unreadable_findings = _findings()
    _check_grid_variable_valid_range(
        unreadable_findings,
        _UnreadableArray((2,)),
        "longitude",
        {"valid_min": 0.0, "valid_max": 360.0},
        family="grid_longitude_valid_range",
    )
    assert "simulated read failure" in (
        unreadable_findings.messages["grid_longitude_valid_range"][0]
    )


def test_grid_geometry_allows_fully_void_curvilinear_positions():
    latitude = np.ma.masked_invalid([[10.0, np.nan]])
    longitude = np.ma.masked_invalid([[20.0, np.nan]])
    latitude_vertices = np.ma.masked_invalid(
        [[[9.0, 9.0, 11.0, 11.0], [np.nan] * 4]]
    )
    longitude_vertices = np.ma.masked_invalid(
        [[[19.0, 21.0, 21.0, 19.0], [np.nan] * 4]]
    )
    findings = _findings()

    _check_grid_geometry_missing_values(
        findings,
        latitude,
        longitude,
        latitude_vertices,
        longitude_vertices,
        topology="curvilinear",
    )

    assert findings.messages["grid_geometry_missing_values"] == []


def test_grid_geometry_rejects_unmasked_nan_as_missing_value():
    findings = _findings()

    _check_grid_geometry_missing_values(
        findings,
        np.asarray([[10.0, np.nan]]),
        np.asarray([[20.0, np.nan]]),
        np.asarray([[[9.0, 9.0, 11.0, 11.0], [np.nan] * 4]]),
        np.asarray([[[19.0, 21.0, 21.0, 19.0], [np.nan] * 4]]),
        topology="curvilinear",
    )

    message = findings.messages["grid_geometry_missing_values"][0]
    assert "unmasked NaN or infinite values" in message
    assert "1 of 2 grid positions" in message
    assert "index (0, 1)" in message
    assert "configured finite fill value" in message


@pytest.mark.parametrize(
    ("latitude", "longitude", "latitude_vertices", "longitude_vertices"),
    [
        (
            [[10.0, 11.0]],
            [[20.0, np.nan]],
            [[[9.0, 9.0, 11.0, 11.0], [10.0, 10.0, 12.0, 12.0]]],
            [[[19.0, 21.0, 21.0, 19.0], [20.0, 22.0, 22.0, 20.0]]],
        ),
        (
            [[10.0, 11.0]],
            [[20.0, 21.0]],
            [[[9.0, 9.0, 11.0, 11.0], [10.0, 10.0, 12.0, np.nan]]],
            [[[19.0, 21.0, 21.0, 19.0], [20.0, 22.0, 22.0, 20.0]]],
        ),
        (
            [[10.0, 11.0]],
            [[20.0, 21.0]],
            [[[9.0, 9.0, 11.0, 11.0], [10.0, 10.0, 12.0, np.nan]]],
            [[[19.0, 21.0, 21.0, 19.0], [20.0, 22.0, 22.0, np.nan]]],
        ),
    ],
)
def test_grid_geometry_rejects_partial_curvilinear_positions(
    latitude, longitude, latitude_vertices, longitude_vertices
):
    findings = _findings()

    _check_grid_geometry_missing_values(
        findings,
        np.ma.masked_invalid(latitude),
        np.ma.masked_invalid(longitude),
        np.ma.masked_invalid(latitude_vertices),
        np.ma.masked_invalid(longitude_vertices),
        topology="curvilinear",
    )

    assert len(findings.messages["grid_geometry_missing_values"]) == 1
    message = findings.messages["grid_geometry_missing_values"][0]
    assert "1 of 2 grid positions" in message
    assert "First incident at index (0, 1)" in message
    assert "fully void or define both centres with all vertex pairs finite" in message


def test_grid_geometry_allows_paired_unstructured_padding_but_requires_three_pairs():
    latitude = np.ma.masked_invalid([10.0, np.nan, 30.0])
    longitude = np.ma.masked_invalid([20.0, np.nan, 40.0])
    latitude_vertices = np.ma.masked_invalid(
        [
            [9.0, 10.0, 11.0, np.nan, np.nan],
            [np.nan] * 5,
            [29.0, 31.0, np.nan, np.nan, np.nan],
        ]
    )
    longitude_vertices = np.ma.masked_invalid(
        [
            [19.0, 21.0, 20.0, np.nan, np.nan],
            [np.nan] * 5,
            [39.0, 41.0, np.nan, np.nan, np.nan],
        ]
    )
    findings = _findings()

    _check_grid_geometry_missing_values(
        findings,
        latitude,
        longitude,
        latitude_vertices,
        longitude_vertices,
        topology="unstructured",
    )

    message = findings.messages["grid_geometry_missing_values"][0]
    assert "1 of 3 grid positions" in message
    assert "First incident at index (2,)" in message
    assert "2 of 5 vertex pairs are finite" in message
    assert "at least 3 paired finite vertices" in message


def test_grid_geometry_reuses_compact_vertex_scan_summaries():
    latitude_vertices = _CachedArray(
        [[9.0, 10.0, 11.0, 10.0], [19.0, 20.0, 21.0, 20.0]]
    )
    longitude_vertices = _CachedArray(
        [[19.0, 20.0, 21.0, 20.0], [29.0, 30.0, 31.0, 30.0]]
    )
    findings = _findings()
    latitude_summary = _check_vertex_valid_range(
        findings,
        latitude_vertices,
        "vertices_latitude",
        {"valid_min": -90.0, "valid_max": 90.0},
        longitude=False,
        family="grid_latitude_valid_range",
    )
    longitude_summary = _check_vertex_valid_range(
        findings,
        longitude_vertices,
        "vertices_longitude",
        {"valid_min": 0.0, "valid_max": 360.0},
        longitude=True,
        family="grid_longitude_valid_range",
    )
    reads = (len(latitude_vertices.selections), len(longitude_vertices.selections))

    _check_grid_geometry_missing_values(
        findings,
        np.asarray([10.0, 20.0]),
        np.asarray([20.0, 30.0]),
        _UnreadableArray((2, 4)),
        _UnreadableArray((2, 4)),
        topology="unstructured",
        latitude_vertex_summary=latitude_summary,
        longitude_vertex_summary=longitude_summary,
    )

    assert reads == (
        len(latitude_vertices.selections),
        len(longitude_vertices.selections),
    )
    assert latitude_summary.finite_slots.nbytes == 2
    assert longitude_summary.finite_slots.nbytes == 2
    assert findings.messages["grid_geometry_missing_values"] == []


def test_grid_geometry_reuses_vertex_summaries_across_scan_chunks(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.scanning.DEFAULT_TARGET_VALUES", 8)
    latitude_values = np.tile([9.0, 10.0, 11.0, 10.0], (5, 1))
    longitude_values = np.tile([19.0, 20.0, 21.0, 20.0], (5, 1))
    latitude_values = np.ma.asarray(latitude_values)
    latitude_values[4, 3] = np.ma.masked
    latitude_vertices = _CachedArray(latitude_values)
    longitude_vertices = _CachedArray(longitude_values)
    findings = _findings()

    latitude_summary = _check_vertex_valid_range(
        findings,
        latitude_vertices,
        "vertices_latitude",
        {},
        longitude=False,
        family="grid_latitude_valid_range",
    )
    longitude_summary = _check_vertex_valid_range(
        findings,
        longitude_vertices,
        "vertices_longitude",
        {},
        longitude=True,
        family="grid_longitude_valid_range",
    )

    _check_grid_geometry_missing_values(
        findings,
        np.arange(5.0),
        np.arange(5.0),
        _UnreadableArray((5, 4)),
        _UnreadableArray((5, 4)),
        topology="curvilinear",
        latitude_vertex_summary=latitude_summary,
        longitude_vertex_summary=longitude_summary,
    )

    assert latitude_vertices.selections == [
        (slice(0, 2), slice(None)),
        (slice(2, 4), slice(None)),
        (slice(4, 6), slice(None)),
    ]
    assert longitude_vertices.selections == latitude_vertices.selections
    message = findings.messages["grid_geometry_missing_values"][0]
    assert "1 of 5 grid positions" in message
    assert "First incident at index (4,)" in message


def test_grid_geometry_unpacks_vertex_summaries_larger_than_one_byte():
    latitude_values = np.ma.asarray(
        [
            np.arange(10.0),
            [20.0, 21.0, 22.0] + [np.nan] * 7,
        ]
    )
    longitude_values = np.ma.asarray(
        [
            np.arange(10.0) + 30.0,
            [40.0, 41.0, 42.0] + [np.nan] * 7,
        ]
    )
    latitude_vertices = _CachedArray(np.ma.masked_invalid(latitude_values))
    longitude_vertices = _CachedArray(np.ma.masked_invalid(longitude_values))
    findings = _findings()

    latitude_summary = _check_vertex_valid_range(
        findings,
        latitude_vertices,
        "vertices_latitude",
        {},
        longitude=False,
        family="grid_latitude_valid_range",
    )
    longitude_summary = _check_vertex_valid_range(
        findings,
        longitude_vertices,
        "vertices_longitude",
        {},
        longitude=True,
        family="grid_longitude_valid_range",
    )

    assert latitude_summary.vertex_count == 10
    assert latitude_summary.finite_slots.shape == (2, 2)
    assert longitude_summary.finite_slots.shape == (2, 2)

    _check_grid_geometry_missing_values(
        findings,
        np.asarray([5.0, 21.0]),
        np.asarray([35.0, 41.0]),
        _UnreadableArray((2, 10)),
        _UnreadableArray((2, 10)),
        topology="unstructured",
        latitude_vertex_summary=latitude_summary,
        longitude_vertex_summary=longitude_summary,
    )

    assert findings.messages["grid_geometry_missing_values"] == []


def test_vertex_valid_range_ignores_padding_and_checks_finite_values():
    vertices = np.ma.masked_invalid(
        [
            [-1.0, 0.0, 1.0, np.nan, np.nan],
            [12.0, 12.0, 12.0, np.nan, np.nan],
            [2.0, 3.0, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    findings = _findings()

    _check_vertex_valid_range(
        findings,
        vertices,
        "vertices_longitude",
        {"valid_min": 0.0, "valid_max": 10.0},
        longitude=True,
        family="grid_longitude_valid_range",
    )

    assert any(
        "Highest value 12.0" in msg
        and "circular longitude width=0.0" in msg
        and "index (1,)" in msg
        for msg in findings.messages["grid_longitude_valid_range"]
    )
    assert not any(
        "Lowest value" in msg
        for msg in findings.messages["grid_longitude_valid_range"]
    )


def test_vertex_valid_range_reports_nonfinite_and_read_failures():
    nonfinite_findings = _findings()
    _check_vertex_valid_range(
        nonfinite_findings,
        np.asarray([[np.nan, np.nan, np.nan]]),
        "vertices_longitude",
        {"valid_min": 0.0, "valid_max": 360.0},
        longitude=True,
        family="grid_longitude_valid_range",
    )
    assert nonfinite_findings.messages["grid_longitude_valid_range"] == []

    unreadable_findings = _findings()
    _check_vertex_valid_range(
        unreadable_findings,
        _UnreadableArray((1, 4)),
        "vertices_longitude",
        {"valid_min": 0.0, "valid_max": 360.0},
        longitude=True,
        family="grid_longitude_valid_range",
    )
    assert "simulated read failure" in (
        unreadable_findings.messages["grid_longitude_valid_range"][0]
    )


def test_vertex_valid_range_bounds_and_releases_netcdf_chunk_cache():
    vertices = _CachedArray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])

    _check_vertex_valid_range(
        _findings(),
        vertices,
        "vertices_longitude",
        {"valid_min": 0.0, "valid_max": 360.0},
        longitude=True,
        family="grid_longitude_valid_range",
    )

    assert [change[0] for change in vertices.cache_changes] == [
        DEFAULT_SCAN_CACHE_BYTES,
        64 * 2**20,
    ]


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (None, None, 1.0e-12),
        (0.0, 0.0, 1.0e-12),
        (0.0, 1.0e-4, 1.0e-10),
        (0.0, 360.0, 3.6e-4),
        (-90.0, 90.0, 9.0e-5),
    ],
)
def test_valid_range_slack_is_symmetric_and_never_zero(lower, upper, expected):
    assert valid_range_slack(lower, upper) == pytest.approx(expected)


def test_check_valid_range_uses_slack_and_requested_result_family():
    findings = _findings()
    check_valid_range(
        findings,
        np.asarray([-0.0003, 360.0003]),
        "longitude",
        {"valid_min": 0.0, "valid_max": 360.0},
        family="grid_longitude_valid_range",
    )
    assert findings.messages["grid_longitude_valid_range"] == []

    check_valid_range(
        findings,
        np.asarray([-0.001, 360.001]),
        "longitude",
        {"valid_min": 0.0, "valid_max": 360.0},
        family="grid_longitude_valid_range",
    )
    assert len(findings.messages["grid_longitude_valid_range"]) == 2
    assert findings.messages["valid_range"] == []


def test_single_longitude_cycle_uses_numerical_slack_and_configured_family():
    findings = _findings()
    check_single_longitude_cycle(
        findings,
        np.asarray([-180.0, 180.0003]),
        "longitude",
        family="grid_longitude_single_cycle",
    )
    assert findings.messages["grid_longitude_single_cycle"] == []

    check_single_longitude_cycle(
        findings,
        np.asarray([-180.0, 180.001]),
        "longitude",
        family="grid_longitude_single_cycle",
    )
    assert len(findings.messages["grid_longitude_single_cycle"]) == 1
    assert "recommended to span no more than one 360-degree cycle" in (
        findings.messages["grid_longitude_single_cycle"][0]
    )


def test_single_longitude_cycle_uses_shared_streaming_by_default(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.scanning.DEFAULT_TARGET_VALUES", 2)
    longitude = _CachedArray([-180.0, -90.0, 0.0, 180.001])
    findings = _findings()

    check_single_longitude_cycle(
        findings,
        longitude,
        "longitude",
        family="grid_longitude_single_cycle",
    )

    assert longitude.selections == [(slice(0, 2),), (slice(2, 4),)]
    assert len(findings.messages["grid_longitude_single_cycle"]) == 1


def test_is_longitude_uses_catalogue_value_then_file_standard_name():
    latitude = _AttributeVariable(standard_name="latitude")
    longitude = _AttributeVariable(standard_name="longitude")

    assert _is_longitude(latitude, {"cf_standard_name": "grid_longitude"})
    assert _is_longitude(longitude, {})
    assert not _is_longitude(latitude, {})


def test_unwrap_longitude_pairs_uses_parent_or_first_bound_as_reference():
    pairs = np.asarray([[359.5, 0.5], [0.5, 1.5]])

    with_parents = _unwrap_longitude_pairs(pairs, np.asarray([0.0, 1.0]))
    without_parents = _unwrap_longitude_pairs(pairs, None)

    np.testing.assert_allclose(with_parents, [[-0.5, 0.5], [0.5, 1.5]])
    np.testing.assert_allclose(without_parents, [[359.5, 360.5], [0.5, 1.5]])


@pytest.mark.parametrize(
    ("pair", "parent", "expected"),
    [
        ([0.0, 360.0], 180.0, [0.0, 360.0]),
        ([360.0, 0.0], 180.0, [360.0, 0.0]),
        ([-180.0, 180.0], 0.0, [-180.0, 180.0]),
    ],
)
def test_unwrap_longitude_pairs_preserves_full_cycle_direction(
    pair, parent, expected
):
    actual = _unwrap_longitude_pairs(
        np.asarray([pair]), np.asarray([parent])
    )

    np.testing.assert_allclose(actual, [expected])


def test_cyclic_absolute_difference_uses_shortest_longitude_distance():
    difference = _cyclic_absolute_difference([359.0, 1.0, 180.0], [1.0, 359.0, 0.0])

    np.testing.assert_allclose(difference, [2.0, 2.0, 180.0])
