"""Classify a netCDF file's coordinates and check them against the STANDARD.

Shim over ``plugins/coordinate_standard.py`` (the engine, built on
``compliance_checker.cf.util``) -- no logic lives here. ``classify_file`` is
kept as an alias of the engine's ``classify_dataset`` for old callers.

Run
---
    python classify_file.py FILE.nc                  # default: CMIP7 standard
    python classify_file.py FILE.nc --prefix CMIP6

Exit code = number of coordinates with FAIL findings (0 = all clean), so it is
usable in a folder sweep / CI loop.

Needs the repo checkout (bootstrapped below), netCDF4 and compliance-checker.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

from plugins.coordinate_standard import (  # noqa: E402, F401  (re-exports)
    Role, Representation, VariableKind, Coordinate, Standard,
    CmorTableProvider, classify_dataset, match, diff, read_1d_values, report,
    GOOD_OUTCOMES, WARN_OUTCOMES,
)

classify_file = classify_dataset            # old name for the previous API

TABLES_DIR = Path(__file__).parent / "tables"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("netcdf", help="path to the netCDF file")
    ap.add_argument("--tables", default=str(TABLES_DIR),
                    help="directory of CMOR table JSON")
    ap.add_argument("--prefix", default="CMIP7",
                    help="table prefix: CMIP6 | CMIP7 | CORDEX-CMIP6")
    args = ap.parse_args()

    try:
        from netCDF4 import Dataset
    except ImportError:
        sys.exit("netCDF4 not found. Activate an environment with netCDF4 "
                 "installed.")

    std = CmorTableProvider(args.tables, args.prefix).build()
    print(f"Standard: {std.source}"
          + (f" (version {std.version})" if std.version else "")
          + f", {len(std.coordinates)} coords")
    print(f"File:     {args.netcdf}")

    ds = Dataset(args.netcdf)
    try:
        fails = report(ds, std)
    finally:
        ds.close()
    sys.exit(min(fails, 255))


if __name__ == "__main__":
    main()
