"""Coordinate checks: validate a file's coordinates against the CMOR-table
standard (via the ``plugins.coordinate_standard`` engine)."""

from checks.coordinate_checks.check_coordinates import check_coordinates

__all__ = [
    "check_coordinates",
]
