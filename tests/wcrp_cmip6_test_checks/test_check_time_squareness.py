"""
Tests for TIME001 (check_time_squareness), focused on the unit-aware
tolerance comparison introduced to replace exact-equality-after-rounding.

Background: comparing a computed theoretical time axis (built as
``first + i * step`` in float64) against the axis stored in a file used to
be done by rounding both to 6 decimal places and requiring exact equality.
That is fragile: two float64 values that agree to well below any physically
meaningful precision can still round to different 1e-6 buckets when they
sit a few ulps to either side of a x.xxxxxx5 boundary, producing a "phantom"
mismatch (see the reported 47028.354166 vs 47028.354167 case in 
https://github.com/ESGF/cc-plugin-wcrp/issues/77). The fix compares the raw
values with an absolute tolerance (``TOL_SECONDS``,expressed in the file's 
own time units via ``_tolerance``) instead of bucketing them.

NOTE: sibling test modules in this package import the checker with a
relative import (``from ...checks... import ...``), but that import is
currently broken in this checkout (relative import beyond top-level
package) regardless of these changes. Absolute imports are used here so
the tests actually collect; align with the relative style if/when that's
fixed repo-wide.
"""

import os

import cftime
import numpy as np
import pytest
from netCDF4 import Dataset

from compliance_checker.base import BaseCheck
from compliance_checker.tests import BaseTestCase

from checks.time_checks import check_time_squareness as checker


def _make_time_file(
    path,
    start,
    values,
    units,
    calendar,
    freq_id,
    var="pr",
    cell_methods=None,
):
    """Write a minimal single-variable time-series file the checker can read.

    ``start`` is a (Y, M, D, hh, mm, ss) tuple used only to build the
    filename's time-range token (what ``_parse_filename_start`` reads);
    ``values`` are the numeric time-axis values actually stored.
    """
    ds = Dataset(path, "w")
    try:
        ds.createDimension("time", None)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = units
        time_var.calendar = calendar
        time_var[:] = np.asarray(values, dtype="f8")

        data_var = ds.createVariable(var, "f4", ("time",))
        data_var[:] = np.zeros(len(values), dtype="f4")
        if cell_methods:
            data_var.cell_methods = cell_methods

        ds.frequency = freq_id
        ds.variable_id = var
    finally:
        ds.close()
    return path


def _theoretical_axis(start, n, inc_val, inc_unit, units, calendar):
    """Reproduce exactly what check_time_squareness builds internally,
    for constructing golden (exactly-correct) axes in tests."""
    d0 = cftime.datetime(*start, calendar=calendar)
    d1 = checker._add_time_increment(d0, inc_val, inc_unit, calendar)
    n0 = float(cftime.date2num(d0, units=units, calendar=calendar))
    n1 = float(cftime.date2num(d1, units=units, calendar=calendar))
    step = n1 - n0
    return n0 + np.arange(n, dtype="f8") * step, step


def _filename(tmp_path, start, n_steps, freq_token, var="pr"):
    y, mo, d, hh, mm, ss = start
    start_tok = f"{y:04d}{mo:02d}{d:02d}{hh:02d}{mm:02d}"
    end_tok = f"{y + 20:04d}{mo:02d}{d:02d}{hh:02d}{mm:02d}"
    fname = f"{var}_{freq_token}_test-model_r1i1p1f1_gn_{start_tok}-{end_tok}.nc"
    return os.path.join(tmp_path, fname)


class TestCheckTimeSquareness(BaseTestCase):
    START = (1970, 1, 1, 0, 0, 0)
    UNITS = "days since 1970-01-01"
    CALENDAR = "standard"

    def _run(self, path):
        ds = Dataset(path, mode="r")
        try:
            return checker.check_time_squareness(ds, severity=BaseCheck.HIGH)
        finally:
            ds.close()

    # -- golden path -----------------------------------------------------

    def test_correct_hourly_axis_passes(self, tmp_path):
        n = 10 * 8760  # 10 years of hourly data, matching the reported chunking
        theo, _ = _theoretical_axis(self.START, n, 1, "hours", self.UNITS, self.CALENDAR)
        path = _filename(str(tmp_path), self.START, n, "1hr")
        _make_time_file(path, self.START, theo, self.UNITS, self.CALENDAR, "1hr")
        results = self._run(path)
        assert len(results) == 1
        self.assert_result_is_good(results[0])

    def test_scales_to_several_hundred_years(self, tmp_path):
        n = 300 * 8766  # ~300 years hourly; vectorized axis build, no per-index loop
        theo, _ = _theoretical_axis(self.START, n, 1, "hours", self.UNITS, self.CALENDAR)
        path = _filename(str(tmp_path), self.START, n, "1hr")
        _make_time_file(path, self.START, theo, self.UNITS, self.CALENDAR, "1hr")
        results = self._run(path)
        self.assert_result_is_good(results[0])

    # -- the reported bug: sub-tolerance FP noise must NOT fail ----------

    def test_phantom_subtolerance_offset_does_not_fail(self, tmp_path):
        n = 10 * 8760
        theo, _ = _theoretical_axis(self.START, n, 1, "hours", self.UNITS, self.CALENDAR)
        actual = theo.copy()
        # 1e-6 days ~= 0.0864s, the exact magnitude from the reported
        # 47028.354166 vs 47028.354167 mismatch; TOL_SECONDS=0.1s (~1.157e-6
        # days) comfortably covers it. Mirrors both indices from the report.
        for i in (68720, 34360):
            actual[i] += 1e-6
        path = _filename(str(tmp_path), self.START, n, "1hr")
        _make_time_file(path, self.START, actual, self.UNITS, self.CALENDAR, "1hr")
        results = self._run(path)
        self.assert_result_is_good(results[0])

    # -- real drift must still be caught ----------------------------------

    def test_offset_beyond_tolerance_is_still_caught(self, tmp_path):
        n = 1000
        theo, _ = _theoretical_axis(self.START, n, 1, "hours", self.UNITS, self.CALENDAR)
        actual = theo.copy()
        actual[500] += 5.0 / 86400.0  # 5 real seconds, well past the 0.1s tolerance
        path = _filename(str(tmp_path), self.START, n, "1hr")
        _make_time_file(path, self.START, actual, self.UNITS, self.CALENDAR, "1hr")
        results = self._run(path)
        self.assert_result_is_bad(results[0])
        assert "index 500" in results[0].msgs[0]

    def test_wrong_step_fails_at_first_index(self, tmp_path):
        n = 100
        # File claims frequency=1hr but the stored axis actually steps by 2h.
        d0 = cftime.datetime(*self.START, calendar=self.CALENDAR)
        actual = np.array(
            [float(cftime.date2num(d0 + __import__("datetime").timedelta(hours=2 * i),
                                    units=self.UNITS, calendar=self.CALENDAR))
             for i in range(n)],
            dtype="f8",
        )
        path = _filename(str(tmp_path), self.START, n, "1hr")
        _make_time_file(path, self.START, actual, self.UNITS, self.CALENDAR, "1hr")
        results = self._run(path)
        self.assert_result_is_bad(results[0])
        assert "index 1" in results[0].msgs[0]

    # -- tolerance must be computed in the file's own units --------------

    @pytest.mark.parametrize(
        "units,offset,expect_pass",
        [
            ("seconds since 1970-01-01", 0.05, True),   # under 0.1s tol
            ("seconds since 1970-01-01", 0.5, False),   # over 0.1s tol
            ("hours since 1970-01-01", 0.1 / 3600 * 0.5, True),
            ("hours since 1970-01-01", 0.1 / 3600 * 5.0, False),
        ],
    )
    def test_tolerance_scales_with_time_units(self, tmp_path, units, offset, expect_pass):
        n = 50
        theo, _ = _theoretical_axis(self.START, n, 1, "hours", units, self.CALENDAR)
        actual = theo.copy()
        actual[10] += offset
        path = _filename(str(tmp_path), self.START, n, "1hr")
        _make_time_file(path, self.START, actual, units, self.CALENDAR, "1hr")
        results = self._run(path)
        if expect_pass:
            self.assert_result_is_good(results[0])
        else:
            self.assert_result_is_bad(results[0])

    def test_tolerance_helper_unit_conversion(self):
        ref = cftime.datetime(*self.START, calendar=self.CALENDAR)
        assert checker._tolerance(ref, "days since 1970-01-01", self.CALENDAR) == pytest.approx(0.1 / 86400.0)
        assert checker._tolerance(ref, "hours since 1970-01-01", self.CALENDAR) == pytest.approx(0.1 / 3600.0)
        assert checker._tolerance(ref, "seconds since 1970-01-01", self.CALENDAR) == pytest.approx(0.1)
        # Unit synonyms a hand-rolled table would miss are still correct,
        # because this asks cftime itself rather than re-parsing the word.
        assert checker._tolerance(ref, "secs since 1970-01-01", self.CALENDAR) == pytest.approx(0.1)
        assert checker._tolerance(ref, "sec since 1970-01-01", self.CALENDAR) == pytest.approx(0.1)

    def test_tolerance_correct_for_months_since_360_day(self):
        # "months since" is only valid CF units on a 360_day calendar, where
        # a month is a fixed 30-day duration -- cftime.date2num handles this
        # correctly, and so should a tolerance derived from it.
        ref = cftime.datetime(2000, 1, 1, calendar="360_day")
        tol = checker._tolerance(ref, "months since 2000-01-01", "360_day")
        assert tol == pytest.approx(0.1 / (30 * 86400.0))

    # -- monthly "point" convention branch untouched by the refactor -----

    def test_monthly_point_single_consistent_convention_passes(self, tmp_path):
        n = 24
        cur = cftime.datetime(*self.START, calendar=self.CALENDAR)
        values = []
        for _ in range(n):
            mid = cftime.datetime(cur.year, cur.month, 15, 0, 0, 0, calendar=self.CALENDAR)
            values.append(float(cftime.date2num(mid, units=self.UNITS, calendar=self.CALENDAR)))
            cur = checker._add_time_increment(cur, 1, "months", self.CALENDAR)
        path = _filename(str(tmp_path), self.START, n, "monPt")
        _make_time_file(
            path, self.START, values, self.UNITS, self.CALENDAR, "monPt",
            cell_methods="time: point",
        )
        results = self._run(path)
        self.assert_result_is_good(results[0])

    def test_monthly_point_mixed_convention_fails(self, tmp_path):
        n = 24
        cur = cftime.datetime(*self.START, calendar=self.CALENDAR)
        values = []
        for i in range(n):
            if i < n // 2:
                stamp = cur  # month-start convention
            else:
                stamp = cftime.datetime(cur.year, cur.month, 15, 0, 0, 0, calendar=self.CALENDAR)
            values.append(float(cftime.date2num(stamp, units=self.UNITS, calendar=self.CALENDAR)))
            cur = checker._add_time_increment(cur, 1, "months", self.CALENDAR)
        path = _filename(str(tmp_path), self.START, n, "monPt")
        _make_time_file(
            path, self.START, values, self.UNITS, self.CALENDAR, "monPt",
            cell_methods="time: point",
        )
        results = self._run(path)
        self.assert_result_is_bad(results[0])
