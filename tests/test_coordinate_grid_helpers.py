from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from compliance_checker.base import BaseCheck

from checks.coordinate_checks.grid import (
    _cell_circular_widths,
    _check_grid_variable_valid_range,
    _check_vertex_valid_range,
    _global_cell_index,
    _grid_dimensions,
    _grid_mapping_name,
    _identify_coordinate,
    _spatial_chunks,
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


def test_spatial_chunks_splits_one_dimensional_values(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.grid._VERTEX_CHUNK_VALUES", 4)
    values = np.arange(10)

    chunks = list(_spatial_chunks(values))

    assert [origin for origin, _ in chunks] == [(0,), (4,), (8,)]
    np.testing.assert_array_equal(
        np.concatenate([chunk for _, chunk in chunks]), values
    )


def test_spatial_chunks_handles_empty_and_exact_chunk_boundaries(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.grid._VERTEX_CHUNK_VALUES", 4)

    assert list(_spatial_chunks(np.asarray([]))) == []
    chunks = list(_spatial_chunks(np.arange(8)))
    assert [origin for origin, _ in chunks] == [(0,), (4,)]
    assert [chunk.shape for _, chunk in chunks] == [(4,), (4,)]


def test_spatial_chunks_bounds_two_dimensional_blocks(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.grid._VERTEX_CHUNK_VALUES", 4)
    values = np.arange(15).reshape(3, 5)

    chunks = list(_spatial_chunks(values))

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
    monkeypatch.setattr("checks.coordinate_checks.grid._VERTEX_CHUNK_VALUES", 8)
    vertices = np.arange(20).reshape(5, 4)

    chunks = list(_spatial_chunks(vertices, vertices=True))

    assert [origin for origin, _ in chunks] == [(0,), (2,), (4,)]
    assert [chunk.shape for _, chunk in chunks] == [(2, 4), (2, 4), (1, 4)]


def test_spatial_chunks_handles_scalar_and_higher_dimensional_grids(monkeypatch):
    monkeypatch.setattr("checks.coordinate_checks.grid._VERTEX_CHUNK_VALUES", 4)

    scalar_chunks = list(_spatial_chunks(np.asarray(7.0)))
    volume_chunks = list(_spatial_chunks(np.arange(12).reshape(3, 2, 2)))

    assert scalar_chunks[0][0] == ()
    assert scalar_chunks[0][1] == 7.0
    assert [origin for origin, _ in volume_chunks] == [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    ]


def test_global_cell_index_translates_flat_chunk_index():
    assert _global_cell_index((10, 20), (2, 3), 4) == (11, 21)


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
    assert any("missing or non-finite grid-cell values" in msg for msg in messages)
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
    assert nonfinite_findings.messages["grid_longitude_valid_range"] == [
        "'longitude' has no finite numeric values to check."
    ]

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


def test_vertex_valid_range_accepts_padding_but_rejects_invalid_cells():
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
        allow_missing_padding=True,
    )

    assert any(
        "all vertex values missing" in msg and "index (3,)" in msg
        for msg in findings.messages["grid"]
    )
    assert any(
        "fewer than 3 finite vertex values" in msg
        and "index (2,) has 2 finite vertex values" in msg
        for msg in findings.messages["grid"]
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
    assert nonfinite_findings.messages["grid_longitude_valid_range"] == [
        "'vertices_longitude' has no finite numeric values to check."
    ]

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


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (None, None, 1.0e-6),
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


def test_cyclic_absolute_difference_uses_shortest_longitude_distance():
    difference = _cyclic_absolute_difference([359.0, 1.0, 180.0], [1.0, 359.0, 0.0])

    np.testing.assert_allclose(difference, [2.0, 2.0, 180.0])
