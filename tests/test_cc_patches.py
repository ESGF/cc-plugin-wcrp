"""Tests for the temporary compliance-checker fixes in plugins.cc_patches.

Which fix is active depends on plugins/cc_patches.toml and on the installed
compliance-checker. For that installation, the "upstream" tests check that

* an active fix replaces reviewed upstream code that still shows the bug
  (otherwise: review the new upstream code, or lower max_version), and
* where a fix is not active, upstream does not show the bug
  (otherwise: the fix is missing for this compliance-checker release).
"""

import ast
import inspect
import logging
import re
import shutil
import subprocess
import tracemalloc
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from compliance_checker.cf import CF1_6Check, CF1_7Check
from compliance_checker.cf.cf_1_11 import CF1_11Check
from netCDF4 import Dataset
from packaging.version import Version

import plugins.wcrp_base  # noqa: F401  (applies the patches on import)
from plugins import cc_patches

GEOGRAPHIC_REGION = "geographic_region"
CELL_BOUNDARIES_INTERVAL = "cell_boundaries_interval"
MAIN_VARIABLE_VERTICAL_POSITIVE = "main_variable_vertical_positive"
PATCHES_BY_NAME = {patch.name: patch for patch in cc_patches.PATCHES}

CONFIG = cc_patches.load_config()
CC_VERSION = cc_patches.installed_compliance_checker_version()


def _is_active(name):
    return cc_patches.is_active(CONFIG[name], CC_VERSION)


def _original(name):
    """The installed upstream implementation, whether or not it was patched."""
    if name in cc_patches.ORIGINALS:
        return cc_patches.ORIGINALS[name]
    patch = PATCHES_BY_NAME[name]
    return patch.owner.__dict__[patch.attribute]


def _upstream_reviewed(name):
    """Whether the installed upstream method is one a patch was written for."""
    return cc_patches.source_sha256(_original(name)) in PATCHES_BY_NAME[name].reviewed_upstream_sha256


def _result_summary(results):
    return [(result.value, sorted(result.msgs)) for result in results]


@pytest.fixture
def restore_patches():
    """Re-apply the shipped configuration after a test changed it."""
    yield
    cc_patches.apply_patches()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_configuration_is_package_data():
    """The configuration is read from the plugins package itself."""
    resource = files("plugins") / "cc_patches.toml"
    assert resource.is_file()
    assert Path(str(resource)) == cc_patches.CONFIG_PATH


def test_configuration_is_tracked_by_git():
    """The wheel includes cc_patches.toml only if git tracks it.

    setuptools_scm adds git-tracked files inside packages as package data, as
    for all other configuration files. If cc_patches.toml is missing from a
    build, no patch is applied, without any message.
    """
    repository = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None or not (repository / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "plugins/cc_patches.toml"],
        cwd=repository,
        capture_output=True,
    )
    assert tracked.returncode == 0, "plugins/cc_patches.toml is not tracked by git; run 'git add' on it"


def test_every_patch_is_configured():
    assert sorted(CONFIG) == sorted(PATCHES_BY_NAME)


@pytest.mark.parametrize("name", sorted(PATCHES_BY_NAME))
def test_patch_configuration_is_valid(name):
    settings = CONFIG[name]
    allowed = {"enabled", "min_version", "max_version", *PATCHES_BY_NAME[name].option_keys}
    assert set(settings) <= allowed
    assert isinstance(settings.get("enabled"), bool)
    if "min_version" in settings and "max_version" in settings:
        assert Version(settings["min_version"]) <= Version(settings["max_version"])
    else:
        for key in ("min_version", "max_version"):
            if key in settings:
                Version(settings[key])


def test_pinned_compliance_checker_is_covered_by_configuration():
    """Each fix covers the pinned release or was deliberately limited."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pins = re.findall(r'"compliance-checker==([^"]+)"', pyproject.read_text())
    assert len(pins) == 1, "compliance-checker must be pinned to one release"
    pinned = Version(pins[0])
    for name, settings in CONFIG.items():
        if settings["enabled"]:
            assert cc_patches.is_active(settings, pinned), (
                f"patch '{name}' is enabled but not active for the pinned "
                f"compliance-checker {pinned}; adjust its version range"
            )


@pytest.mark.parametrize(
    ("settings", "version", "expected"),
    [
        ({"enabled": True}, "6.1.0", True),
        ({"enabled": False}, "6.1.0", False),
        ({}, "6.1.0", False),
        ({"enabled": True, "min_version": "6.1.0"}, "6.0.2", False),
        ({"enabled": True, "min_version": "6.1.0"}, "6.1.0", True),
        ({"enabled": True, "max_version": "6.2.0"}, "6.2.0", True),
        ({"enabled": True, "max_version": "6.2.0"}, "6.2.1", False),
        ({"enabled": True, "min_version": "6.1.0", "max_version": "6.1.0"}, "6.1.1.dev125", False),
        ({"enabled": True, "min_version": "6.1.0", "max_version": "6.2.0"}, "6.1.1.dev125", True),
    ],
)
def test_is_active(settings, version, expected):
    assert cc_patches.is_active(settings, Version(version)) is expected


# --------------------------------------------------------------------------
# Applying patches
# --------------------------------------------------------------------------


@pytest.mark.parametrize("patch", cc_patches.PATCHES, ids=lambda patch: patch.name)
def test_status_follows_configuration(patch):
    current = patch.owner.__dict__[patch.attribute]
    if _is_active(patch.name):
        assert cc_patches.STATUS[patch.name] == "applied"
        assert current is patch.replacement
    else:
        assert cc_patches.STATUS[patch.name] != "applied"
        assert current is not patch.replacement


@pytest.mark.parametrize("patch", cc_patches.PATCHES, ids=lambda patch: patch.name)
def test_active_patch_replaces_reviewed_upstream_code(patch):
    if not _is_active(patch.name):
        pytest.skip(f"not active for compliance-checker {CC_VERSION}")
    assert cc_patches.source_sha256(_original(patch.name)) in patch.reviewed_upstream_sha256, (
        f"compliance-checker {CC_VERSION} changed {patch.owner.__name__}.{patch.attribute}. "
        f"Review the new upstream code, then add its fingerprint to the '{patch.name}' "
        "patch or exclude this release in plugins/cc_patches.toml."
    )


def test_patches_reach_newer_cf_versions(restore_patches):
    """Later CF checker classes inherit the patched methods."""
    config = {name: {"enabled": True} for name in PATCHES_BY_NAME}
    cc_patches.apply_patches(config, CC_VERSION)
    assert CF1_11Check.check_geographic_region is cc_patches.check_geographic_region
    assert CF1_11Check.check_cell_boundaries_interval is cc_patches.check_cell_boundaries_interval
    assert CF1_11Check.check_dimensional_vertical_coordinate is cc_patches.check_dimensional_vertical_coordinate


def test_apply_and_revert(restore_patches):
    patch = PATCHES_BY_NAME[GEOGRAPHIC_REGION]
    upstream = _original(GEOGRAPHIC_REGION)
    version = Version("6.1.0")

    status = cc_patches.apply_patches({GEOGRAPHIC_REGION: {"enabled": True, "max_version": "6.1.0"}}, version)
    assert status[GEOGRAPHIC_REGION] == "applied"
    assert status[CELL_BOUNDARIES_INTERVAL] == "disabled"
    assert CF1_6Check.check_geographic_region is patch.replacement

    # Calling again is harmless and keeps the upstream original.
    cc_patches.apply_patches({GEOGRAPHIC_REGION: {"enabled": True}}, version)
    assert cc_patches.ORIGINALS[GEOGRAPHIC_REGION] is upstream

    status = cc_patches.apply_patches({GEOGRAPHIC_REGION: {"enabled": True, "max_version": "6.0.2"}}, version)
    assert status[GEOGRAPHIC_REGION] == "not applicable to compliance-checker 6.1.0"
    assert CF1_6Check.check_geographic_region is upstream

    status = cc_patches.apply_patches({GEOGRAPHIC_REGION: {"enabled": False}}, version)
    assert status[GEOGRAPHIC_REGION] == "disabled"
    assert CF1_6Check.check_geographic_region is upstream


def test_unreadable_configuration_applies_nothing(tmp_path, monkeypatch, restore_patches):
    config_path = tmp_path / "cc_patches.toml"
    config_path.write_text("[patches.geographic_region\nenabled = yes\n")
    monkeypatch.setattr(cc_patches, "CONFIG_PATH", config_path)

    status = cc_patches.apply_patches()

    assert set(status.values()) == {"disabled"}
    for patch in cc_patches.PATCHES:
        assert patch.owner.__dict__[patch.attribute] is not patch.replacement


def test_applying_patches_is_silent(capsys, caplog, restore_patches):
    """Nothing reaches stderr, stdout or logs above debug level."""
    configs = [
        None,
        {},
        {name: {"enabled": True} for name in PATCHES_BY_NAME},
        {name: {"enabled": True, "max_version": "1.0"} for name in PATCHES_BY_NAME},
        {name: {"enabled": True, "max_version": "not a version"} for name in PATCHES_BY_NAME},
    ]
    with caplog.at_level(logging.INFO):
        for config in configs:
            cc_patches.apply_patches(config)

    assert capsys.readouterr() == ("", "")
    assert [record for record in caplog.records if record.name == cc_patches.__name__] == []


# --------------------------------------------------------------------------
# CF 6.1.1: check_geographic_region
# --------------------------------------------------------------------------


def _write_region_dataset(path, labels, *, storage="char", padding=""):
    """Write a region variable holding ``labels`` (str or list of str)."""
    multiple = isinstance(labels, list)
    names = labels if multiple else [labels]
    with Dataset(path, "w") as ds:
        dimensions = ()
        if multiple:
            ds.createDimension("basin", len(names))
            dimensions = ("basin",)
        if storage == "char":
            # Use exactly enough space for the longest label, or add padding.
            width = max(map(len, names)) + len(padding)
            ds.createDimension("strlen", width)
            variable = ds.createVariable("sector", "S1", dimensions + ("strlen",))
            fill = padding[0] if padding else "\x00"
            characters = np.array([list(name.ljust(width, fill)) for name in names], dtype="S1")
            variable[:] = characters if multiple else characters[0]
        else:
            variable = ds.createVariable("sector", str, dimensions)
            values = [name + padding for name in names]
            if multiple:
                variable[:] = np.array(values, dtype=object)
            else:
                variable[()] = values[0]
        variable.standard_name = "region"
    return path


@pytest.mark.parametrize("storage", ["char", "string"])
@pytest.mark.parametrize("multiple", [False, True], ids=["single", "multiple"])
@pytest.mark.parametrize("padding", ["", "\x00\x00", "  "], ids=["none", "nul", "space"])
@pytest.mark.parametrize("invalid", [False, True], ids=["valid", "invalid"])
def test_geographic_region_labels(tmp_path, storage, multiple, padding, invalid):
    labels = ["Neverland" if invalid else "atlantic_ocean"]
    if multiple:
        labels.insert(0, "global_ocean")
    path = _write_region_dataset(
        tmp_path / "regions.nc", labels if multiple else labels[0], storage=storage, padding=padding
    )

    # Read from disk to exercise netCDF's character masking and string handling.
    with Dataset(path) as ds:
        results = cc_patches.check_geographic_region(CF1_6Check(), ds)

    assert len(results) == 1
    assert results[0].value == (len(labels) - int(invalid), len(labels))
    assert results[0].msgs == (
        ["6.1.1 'Neverland' specified by 'sector' is not a valid region"] if invalid else []
    )


def test_geographic_region_ocean_basins(tmp_path):
    """Basin axis as used by hfbasin, htovgyre, sltovgyre, ... in CMIP."""
    basins = ["atlantic_arctic_ocean", "indian_pacific_ocean", "global_ocean"]
    path = _write_region_dataset(tmp_path / "basins.nc", basins)

    with Dataset(path) as ds:
        results = cc_patches.check_geographic_region(CF1_11Check(), ds)

    assert len(results) == 1
    assert results[0].value == (3, 3)
    assert results[0].msgs == []


# Names of the CF standardized region list (version 5, 12 November 2024) that
# the list in compliance-checker 6.1.0 lacks.
CF_REGION_NAMES_ADDED = {
    "atlantic_arctic_ocean",
    "davis_strait",
    "gibraltar_strait",
    "indian_pacific_ocean",
    "northern_hemisphere",
    "southern_hemisphere",
}


def _upstream_region_names():
    """The region names in the installed upstream check_geographic_region."""
    tree = ast.parse(inspect.getsource(_original(GEOGRAPHIC_REGION)).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "region_list" for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    pytest.fail("compliance-checker's check_geographic_region no longer defines region_list")


def test_region_names_are_cf_standardized_region_list():
    assert len(cc_patches._CF_REGION_NAMES) == 74
    assert all(name == name.lower() for name in cc_patches._CF_REGION_NAMES)
    assert CF_REGION_NAMES_ADDED <= cc_patches._CF_REGION_NAMES


def test_region_names_extend_upstream_list():
    """Upstream's names are all kept; only CF's newer names are added."""
    upstream = _upstream_region_names()
    assert upstream <= cc_patches._CF_REGION_NAMES
    if _upstream_reviewed(GEOGRAPHIC_REGION):
        assert cc_patches._CF_REGION_NAMES - upstream == CF_REGION_NAMES_ADDED


@pytest.mark.parametrize("label", sorted(CF_REGION_NAMES_ADDED))
def test_geographic_region_current_cf_names(tmp_path, label):
    path = _write_region_dataset(tmp_path / "region.nc", label)

    with Dataset(path) as ds:
        results = cc_patches.check_geographic_region(CF1_6Check(), ds)

    assert results[0].value == (1, 1)
    assert results[0].msgs == []


@pytest.mark.parametrize("label", ["atlantic_ocean", "Atlantic_Ocean", "Neverland"])
def test_geographic_region_single_label_same_as_upstream(tmp_path, label):
    """For the single-name case upstream works; results must not change."""
    path = _write_region_dataset(tmp_path / "region.nc", label)

    with Dataset(path) as ds:
        patched = cc_patches.check_geographic_region(CF1_6Check(), ds)
        original = _original(GEOGRAPHIC_REGION)(CF1_6Check(), ds)

    assert _result_summary(patched) == _result_summary(original)


def test_upstream_geographic_region_multiple_labels(tmp_path):
    """Upstream must fail where the fix is active, and work where it is not.

    Uses a CMIP basin axis, so upstream counts as fixed only if it handles
    several names and also knows the current CF region names.
    """
    path = _write_region_dataset(
        tmp_path / "regions.nc",
        ["atlantic_arctic_ocean", "indian_pacific_ocean", "global_ocean", "Neverland"],
    )

    with Dataset(path) as ds:
        expected = _result_summary(cc_patches.check_geographic_region(CF1_6Check(), ds))
        try:
            upstream = _result_summary(_original(GEOGRAPHIC_REGION)(CF1_6Check(), ds))
        except Exception as error:
            upstream = error

    assert expected == [((3, 4), ["6.1.1 'Neverland' specified by 'sector' is not a valid region"])]
    if _is_active(GEOGRAPHIC_REGION):
        assert upstream != expected, (
            f"compliance-checker {CC_VERSION} handles several region names and the "
            "current CF region names. Exclude this release from the "
            f"'{GEOGRAPHIC_REGION}' patch in plugins/cc_patches.toml."
        )
    else:
        assert upstream == expected, (
            f"compliance-checker {CC_VERSION} still fails for a CMIP basin axis "
            f"({upstream!r}), but the '{GEOGRAPHIC_REGION}' patch is not active "
            "for it in plugins/cc_patches.toml."
        )


# --------------------------------------------------------------------------
# CF 7.1: check_cell_boundaries_interval
# --------------------------------------------------------------------------


def _write_bounds_dataset(path, centers, bnds, *, longitude=False, chunksizes=None, mask=None):
    centers = np.asarray(centers, dtype="f8")
    bnds = np.asarray(bnds, dtype="f8")
    dims = tuple(f"d{axis}" for axis in range(centers.ndim))
    with Dataset(path, "w") as ds:
        for name, size in zip(dims, centers.shape):
            ds.createDimension(name, size)
        ds.createDimension("nv", bnds.shape[-1])
        name = "lon" if longitude else "lat"
        variable = ds.createVariable(name, "f8", dims)
        variable.standard_name = "longitude" if longitude else "latitude"
        variable.units = "degrees_east" if longitude else "degrees_north"
        variable.bounds = f"{name}_bnds"
        boundary = ds.createVariable(f"{name}_bnds", "f8", dims + ("nv",), chunksizes=chunksizes)
        variable[:] = np.ma.masked_array(centers, mask=mask) if mask is not None else centers
        boundary[:] = bnds
        # Make the coordinate a dimension coordinate where possible so that
        # compliance-checker's boundary map picks it up.
        if centers.ndim > 1:
            data = ds.createVariable("data", "f4", dims)
            data.coordinates = name
    return path


BOUNDS_CASES = {
    "intervals_all_inside": ([0.5, 1.5, 2.5], [[0, 1], [1, 2], [2, 3]], False),
    "intervals_some_outside": ([0.5, 2.0, 2.5, 4.0, 4.5], [[0, 1], [0, 1], [2, 3], [2, 3], [4, 5]], False),
    "descending_bounds": ([0.5, 2.0], [[1, 0], [1, 0]], False),
    "on_boundary": ([0.0, 2.0], [[0, 1], [1, 2]], False),
    "polygon_vertices": ([0.5, 5.0], [[0, 1, 1, 0], [0, 1, 1, 0]], False),
    "lon_antimeridian": ([179.5, -179.5, 0.0], [[179, -180], [180, -179], [10, 20]], True),
    "curvilinear_2d": (
        [[0.5, 1.5], [2.5, 9.0]],
        [[[0, 1], [1, 2]], [[2, 3], [3, 4]]],
        False,
    ),
}


@pytest.mark.parametrize("case", BOUNDS_CASES, ids=str)
@pytest.mark.parametrize("chunk_values", [cc_patches._CELL_BOUNDARY_CHUNK_VALUES, 2], ids=["default", "tiny"])
def test_cell_boundaries_interval_same_as_upstream(tmp_path, monkeypatch, case, chunk_values):
    """Block-wise processing must give the same results as upstream."""
    monkeypatch.setattr(cc_patches, "_CELL_BOUNDARY_CHUNK_VALUES", chunk_values)
    centers, bnds, longitude = BOUNDS_CASES[case]
    path = _write_bounds_dataset(tmp_path / "bounds.nc", centers, bnds, longitude=longitude)

    with Dataset(path) as ds:
        patched = cc_patches.check_cell_boundaries_interval(CF1_7Check(), ds)
        original = _original(CELL_BOUNDARIES_INTERVAL)(CF1_7Check(), ds)

    assert patched, "dataset did not exercise the check"
    assert _result_summary(patched) == _result_summary(original)


def test_cell_boundaries_interval_masked_centers(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_patches, "_CELL_BOUNDARY_CHUNK_VALUES", 2)
    path = _write_bounds_dataset(
        tmp_path / "bounds.nc",
        [0.5, 9.0, 2.5],
        [[0, 1], [1, 2], [2, 3]],
        mask=[False, True, False],
    )

    with Dataset(path) as ds:
        patched = cc_patches.check_cell_boundaries_interval(CF1_7Check(), ds)
        original = _original(CELL_BOUNDARIES_INTERVAL)(CF1_7Check(), ds)

    assert _result_summary(patched) == _result_summary(original) == [(True, [])]


def test_cell_boundaries_interval_aggregates_blocks(tmp_path, monkeypatch):
    """Failures from separate blocks are reported in one result."""
    monkeypatch.setattr(cc_patches, "_CELL_BOUNDARY_CHUNK_VALUES", 4)
    centers, bnds, _ = BOUNDS_CASES["intervals_some_outside"]
    path = _write_bounds_dataset(tmp_path / "bounds.nc", centers, bnds)

    with Dataset(path) as ds:
        results = cc_patches.check_cell_boundaries_interval(CF1_7Check(), ds)

    assert len(results) == 1
    assert results[0].value is False
    assert results[0].msgs == [
        "2 point(s) specified by the coordinate variable 'lat' lie outside "
        "the bounding box of the associated boundary variable 'lat_bnds'"
    ]


def test_cell_boundaries_interval_restores_chunk_cache(tmp_path):
    centers = np.arange(100) + 0.5
    bnds = np.stack([np.arange(100), np.arange(100) + 1], axis=-1)
    path = _write_bounds_dataset(tmp_path / "bounds.nc", centers, bnds, chunksizes=(100, 2))

    with Dataset(path) as ds:
        boundary = ds.variables["lat_bnds"]
        boundary.set_var_chunk_cache(1, 1, 0.5)
        before = boundary.get_var_chunk_cache()
        results = cc_patches.check_cell_boundaries_interval(CF1_7Check(), ds)
        after = boundary.get_var_chunk_cache()

    assert results[0].value is True
    assert after == before


def test_cell_boundary_slices(monkeypatch):
    monkeypatch.setattr(cc_patches, "_CELL_BOUNDARY_CHUNK_VALUES", 4)

    assert list(cc_patches._cell_boundary_slices((), 2)) == [()]
    assert list(cc_patches._cell_boundary_slices((5,), 2)) == [
        (slice(0, 2),),
        (slice(2, 4),),
        (slice(4, 5),),
    ]
    assert list(cc_patches._cell_boundary_slices((2, 3), 2)) == [
        (slice(0, 1), slice(0, 2)),
        (slice(0, 1), slice(2, 3)),
        (slice(1, 2), slice(0, 2)),
        (slice(1, 2), slice(2, 3)),
    ]


@pytest.mark.parametrize("shape", [(7,), (3, 5), (2, 3, 4)])
@pytest.mark.parametrize("chunk_values", [1, 3, 8, 1000])
def test_cell_boundary_slices_cover_every_cell_once(monkeypatch, shape, chunk_values):
    monkeypatch.setattr(cc_patches, "_CELL_BOUNDARY_CHUNK_VALUES", chunk_values)
    visits = np.zeros(shape, dtype=int)
    for selection in cc_patches._cell_boundary_slices(shape, 2):
        visits[selection] += 1
    assert (visits == 1).all()


def _peak_traced_memory(function, *args):
    tracemalloc.start()
    try:
        function(*args)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_upstream_cell_boundaries_interval_memory(tmp_path):
    """Upstream must exhaust memory where the fix is active, not where it is not.

    1,000,000 cells with 4 vertices; the bounds alone take 32 MB. Upstream
    6.1.0 reads them completely and needs well above that, the patched
    version processes blocks of about 2 MB.
    """
    cells, vertices = 1_000_000, 4
    centers = np.arange(cells, dtype="f8") + 0.5
    bnds = centers[:, None] + np.linspace(-0.5, 0.5, vertices)
    path = _write_bounds_dataset(tmp_path / "large.nc", centers, bnds)

    with Dataset(path) as ds:
        patched_peak = _peak_traced_memory(cc_patches.check_cell_boundaries_interval, CF1_7Check(), ds)
        upstream_peak = _peak_traced_memory(_original(CELL_BOUNDARIES_INTERVAL), CF1_7Check(), ds)

    bounds_size = cells * vertices * 8
    assert patched_peak < bounds_size / 2, "the patch no longer limits memory use"
    if _is_active(CELL_BOUNDARIES_INTERVAL):
        assert upstream_peak >= bounds_size, (
            f"compliance-checker {CC_VERSION} no longer reads the complete bounds at "
            f"once (peak {upstream_peak / 2**20:.1f} MiB). If its results are correct, "
            f"exclude this release from the '{CELL_BOUNDARIES_INTERVAL}' patch in "
            "plugins/cc_patches.toml."
        )
    else:
        assert upstream_peak < bounds_size / 2, (
            f"compliance-checker {CC_VERSION} still reads the complete bounds at once "
            f"(peak {upstream_peak / 2**20:.1f} MiB), but the '{CELL_BOUNDARIES_INTERVAL}' "
            "patch is not active for it in plugins/cc_patches.toml."
        )


# --------------------------------------------------------------------------
# CF 4.3: check_dimensional_vertical_coordinate
# --------------------------------------------------------------------------

POSITIVE_MESSAGE = (
    "{}: vertical coordinates not defining pressure must include a positive "
    "attribute that is either 'up' or 'down'"
)


@pytest.fixture
def configured_main_variables(monkeypatch):
    """Use the shipped options, also where the patch is not active."""
    _reset_main_variable_options(monkeypatch)
    cc_patches._configure_main_variables_not_vertical_coordinates(CONFIG[MAIN_VARIABLE_VERTICAL_POSITIVE])


def _reset_main_variable_options(monkeypatch):
    monkeypatch.setattr(cc_patches, "_MAIN_VARIABLES_NOT_VERTICAL_COORDINATES", frozenset())
    monkeypatch.setattr(cc_patches, "_MAIN_VARIABLES_REQUIRED_GLOBAL_ATTRIBUTES", {})


def _write_depth_dataset(
    path,
    *,
    variable_id="depthsl",
    mip_era="CMIP7",
    name="depthsl",
    dims=("lat", "lon"),
    units="m",
    referenced_by=None,
    lev=None,
):
    """A CMIP-like file whose ``name`` has standard_name depth.

    ``referenced_by``: attribute ("coordinates", "bounds", "formula_terms")
    through which another variable references ``name``.
    ``lev``: attributes of an additional depth coordinate variable "lev".
    """
    with Dataset(path, "w") as ds:
        ds.Conventions = "CF-1.11"
        ds.variable_id = variable_id
        if mip_era:
            ds.mip_era = mip_era
        for dim, standard_name, dim_units in (
            ("lat", "latitude", "degrees_north"),
            ("lon", "longitude", "degrees_east"),
        ):
            ds.createDimension(dim, 2)
            coordinate = ds.createVariable(dim, "f8", (dim,))
            coordinate.standard_name = standard_name
            coordinate.units = dim_units
            coordinate[:] = [0, 1]
        if name in dims:
            ds.createDimension(name, 2)
        variable = ds.createVariable(name, "f4", dims)
        variable.standard_name = "depth"
        if units:
            variable.units = units
        variable[:] = 1
        if referenced_by:
            other = ds.createVariable("other", "f4", ("lat", "lon"))
            value = f"a: {name} b: lat" if referenced_by == "formula_terms" else name
            setattr(other, referenced_by, value)
        if lev is not None:
            ds.createDimension("lev", 2)
            coordinate = ds.createVariable("lev", "f8", ("lev",))
            coordinate.setncatts(lev)
            coordinate[:] = [5, 15]
    return path


def _vertical_results(path, function):
    with Dataset(path) as ds:
        return {
            tuple(result.msgs): result.value
            for result in function(CF1_11Check(), ds)
        }


def _vertical_messages(path, function):
    with Dataset(path) as ds:
        return sorted(msg for result in function(CF1_11Check(), ds) for msg in result.msgs)


@pytest.mark.usefixtures("configured_main_variables")
@pytest.mark.parametrize("dims", [("lat", "lon"), ("lat",)], ids=["2d", "1d"])
def test_vertical_main_variable_needs_no_positive(tmp_path, dims):
    path = _write_depth_dataset(tmp_path / "depthsl.nc", dims=dims)

    results = _vertical_results(path, cc_patches.check_dimensional_vertical_coordinate)

    assert results == {(): (1, 1)}


@pytest.mark.usefixtures("configured_main_variables")
def test_vertical_main_variable_still_needs_units(tmp_path):
    path = _write_depth_dataset(tmp_path / "depthsl.nc", units=None)

    messages = _vertical_messages(path, cc_patches.check_dimensional_vertical_coordinate)

    assert messages == [
        "§4.3.1 depthsl's units must be defined for vertical coordinates, there is no default"
    ]


@pytest.mark.usefixtures("configured_main_variables")
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"variable_id": "tos"}, id="not-the-main-variable"),
        pytest.param({"mip_era": "CMIP6"}, id="not-cmip7"),
        pytest.param({"mip_era": None}, id="no-mip-era"),
        pytest.param({"variable_id": "deptho", "name": "deptho"}, id="not-listed"),
        pytest.param({"referenced_by": "coordinates"}, id="used-as-auxiliary-coordinate"),
        pytest.param({"referenced_by": "bounds"}, id="used-as-bounds"),
        pytest.param({"referenced_by": "formula_terms"}, id="used-as-formula-term"),
        pytest.param({"dims": ("depthsl",)}, id="coordinate-variable"),
    ],
)
def test_vertical_positive_still_required_otherwise(tmp_path, kwargs):
    """Everything except the main data variable is checked as upstream does."""
    path = _write_depth_dataset(tmp_path / "depth.nc", **kwargs)
    name = kwargs.get("name", "depthsl")

    patched = _vertical_messages(path, cc_patches.check_dimensional_vertical_coordinate)
    upstream = _vertical_messages(path, _original(MAIN_VARIABLE_VERTICAL_POSITIVE))

    assert POSITIVE_MESSAGE.format(name) in patched
    if _upstream_reviewed(MAIN_VARIABLE_VERTICAL_POSITIVE):
        assert patched == upstream


@pytest.mark.usefixtures("configured_main_variables")
@pytest.mark.parametrize(
    "lev",
    [
        {"standard_name": "depth", "units": "m", "positive": "down", "axis": "Z"},
        {"standard_name": "depth", "units": "m", "axis": "Z"},
        {"standard_name": "air_pressure", "units": "Pa", "axis": "Z"},
    ],
    ids=["with-positive", "without-positive", "pressure"],
)
def test_vertical_coordinates_next_to_main_variable_unchanged(tmp_path, lev):
    path = _write_depth_dataset(tmp_path / "depth.nc", variable_id="tos", lev=lev)

    patched = _vertical_messages(path, cc_patches.check_dimensional_vertical_coordinate)

    # depthsl is not the main variable here, so it is checked as a coordinate.
    expected = [POSITIVE_MESSAGE.format("depthsl")]
    if "positive" not in lev and lev["units"] != "Pa":
        expected.append(POSITIVE_MESSAGE.format("lev"))
    assert patched == sorted(expected)
    if _upstream_reviewed(MAIN_VARIABLE_VERTICAL_POSITIVE):
        assert _vertical_results(path, cc_patches.check_dimensional_vertical_coordinate) == _vertical_results(
            path, _original(MAIN_VARIABLE_VERTICAL_POSITIVE)
        )


def test_vertical_exception_needs_configured_variable(tmp_path, monkeypatch):
    _reset_main_variable_options(monkeypatch)
    path = _write_depth_dataset(tmp_path / "depthsl.nc")

    messages = _vertical_messages(path, cc_patches.check_dimensional_vertical_coordinate)

    assert messages == [POSITIVE_MESSAGE.format("depthsl")]


@pytest.mark.parametrize(
    ("settings", "variables", "attributes"),
    [
        ({"variables": ["depthsl"]}, {"depthsl"}, {}),
        (
            {"variables": ["depthsl"], "global_attributes": {"mip_era": "CMIP7"}},
            {"depthsl"},
            {"mip_era": "CMIP7"},
        ),
        ({"variables": "depthsl"}, set(), {}),
        ({"variables": [1]}, set(), {}),
        ({"variables": ["depthsl"], "global_attributes": {"mip_era": 7}}, set(), {}),
        ({"variables": ["depthsl"], "global_attributes": "CMIP7"}, set(), {}),
        ({}, set(), {}),
    ],
)
def test_vertical_options(monkeypatch, settings, variables, attributes):
    """Invalid options disable the exception instead of widening it."""
    _reset_main_variable_options(monkeypatch)
    cc_patches._configure_main_variables_not_vertical_coordinates(settings)
    assert cc_patches._MAIN_VARIABLES_NOT_VERTICAL_COORDINATES == variables
    assert cc_patches._MAIN_VARIABLES_REQUIRED_GLOBAL_ATTRIBUTES == attributes


def test_vertical_options_shipped_configuration():
    settings = CONFIG[MAIN_VARIABLE_VERTICAL_POSITIVE]
    assert settings["variables"] == ["depthsl"]
    assert settings["global_attributes"] == {"mip_era": "CMIP7"}


def test_vertical_patch_changes_only_the_positive_condition():
    """The replacement is the reviewed upstream method plus one condition."""
    if not _is_active(MAIN_VARIABLE_VERTICAL_POSITIVE):
        pytest.skip(f"not active for compliance-checker {CC_VERSION}")

    def body(function):
        source = inspect.getsource(function)
        code = source[source.index("ret_val = []") :]
        return [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith("#")]

    upstream = body(_original(MAIN_VARIABLE_VERTICAL_POSITIVE))
    patched = body(cc_patches.check_dimensional_vertical_coordinate)
    changed = [(a, b) for a, b in zip(upstream, patched) if a != b]
    assert len(upstream) == len(patched)
    assert changed == [
        (
            'if not cfutil.units_convertible("bar", units):',
            'if not cfutil.units_convertible("bar", units) and not _is_main_variable_not_coordinate(ds, name):',
        )
    ]


def _write_cmor_like_depthsl(path):
    """A CMIP7 depthsl file as CMOR writes it (CMIP7_land.json, depthsl_ti-u-hxy-u).

    Kept close to real output so that upstream's coordinate/data variable
    inference sees what it would see in production files.
    """
    with Dataset(path, "w") as ds:
        ds.setncatts(
            {
                "Conventions": "CF-1.11",
                "mip_era": "CMIP7",
                "activity_id": "CMIP",
                "experiment_id": "piControl",
                "source_id": "EXAMPLE-ESM",
                "variable_id": "depthsl",
                "branded_variable": "depthsl_ti-u-hxy-u",
                "branding_suffix": "ti-u-hxy-u",
                "frequency": "fx",
                "realm": "land",
                "region": "glb",
                "grid_label": "gn",
            }
        )
        ds.createDimension("lat", 3)
        ds.createDimension("lon", 4)
        ds.createDimension("bnds", 2)
        for name, standard_name, units, axis, values, step in (
            ("lat", "latitude", "degrees_north", "Y", [-60.0, 0.0, 60.0], 60.0),
            ("lon", "longitude", "degrees_east", "X", [45.0, 135.0, 225.0, 315.0], 90.0),
        ):
            coordinate = ds.createVariable(name, "f8", (name,))
            coordinate.setncatts(
                {
                    "bounds": f"{name}_bnds",
                    "units": units,
                    "axis": axis,
                    "long_name": standard_name.capitalize(),
                    "standard_name": standard_name,
                }
            )
            coordinate[:] = values
            bounds = ds.createVariable(f"{name}_bnds", "f8", (name, "bnds"))
            bounds[:] = np.stack([np.array(values) - step / 2, np.array(values) + step / 2], axis=-1)
        depthsl = ds.createVariable("depthsl", "f4", ("lat", "lon"), fill_value=np.float32(1e20))
        depthsl.setncatts(
            {
                "standard_name": "depth",
                "long_name": "Total (Cumulative) Thickness of All Soil Layers",
                "units": "m",
                "cell_methods": "area: mean",
                "missing_value": np.float32(1e20),
            }
        )
        depthsl[:] = 2.5
    return path


def _upstream_cf_suite_messages(path):
    """All messages of the cf:1.11 suite, run with compliance-checker's own code
    for the vertical coordinate check (the other patches stay as configured)."""
    from compliance_checker.suite import CheckSuite

    patch = PATCHES_BY_NAME[MAIN_VARIABLE_VERTICAL_POSITIVE]
    patched = patch.owner.__dict__[patch.attribute]
    setattr(patch.owner, patch.attribute, _original(MAIN_VARIABLE_VERTICAL_POSITIVE))
    try:
        suite = CheckSuite()
        suite.load_all_available_checkers()
        ds = suite.load_dataset(str(path))
        try:
            groups, errors = suite.run_all(ds, ["cf:1.11"], skip_checks=[])["cf:1.11"]
        finally:
            ds.close()
    finally:
        setattr(patch.owner, patch.attribute, patched)
    assert not errors, errors

    messages = []

    def collect(result):
        messages.extend(result.msgs)
        for child in getattr(result, "children", None) or []:
            collect(child)

    for result in groups:
        collect(result)
    return messages


@pytest.mark.usefixtures("configured_main_variables")
def test_cmor_like_depthsl_passes_vertical_check(tmp_path):
    path = _write_cmor_like_depthsl(tmp_path / "depthsl.nc")

    assert _vertical_messages(path, cc_patches.check_dimensional_vertical_coordinate) == []


def test_upstream_vertical_main_variable_positive(tmp_path):
    """Upstream must require positive where the fix is active, not elsewhere.

    Runs the complete upstream cf suite on a CMOR-like file, so an upstream
    fix is noticed wherever it is made: in the vertical coordinate check
    itself, or in how cfutil tells coordinates from data variables.
    """
    path = _write_cmor_like_depthsl(tmp_path / "depthsl.nc")

    findings = [
        msg
        for msg in _upstream_cf_suite_messages(path)
        if "depthsl" in msg and "positive" in msg
    ]

    if _is_active(MAIN_VARIABLE_VERTICAL_POSITIVE):
        assert findings, (
            f"compliance-checker {CC_VERSION} no longer requires 'positive' for a "
            "depthsl main variable. Exclude this release from the "
            f"'{MAIN_VARIABLE_VERTICAL_POSITIVE}' patch in plugins/cc_patches.toml."
        )
    else:
        assert not findings, (
            f"compliance-checker {CC_VERSION} still requires 'positive' for a depthsl "
            f"main variable ({findings}), but the '{MAIN_VARIABLE_VERTICAL_POSITIVE}' "
            "patch is not active for it in plugins/cc_patches.toml."
        )
