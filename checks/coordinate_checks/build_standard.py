"""Inspect the STANDARD built from the CMOR tables.

Shim over ``plugins/coordinate_standard.py`` (the engine, built on
``compliance_checker.cf.util``) -- no logic lives here. The CLI summarises the
built standard: how many coordinate objects, and the role/representation
coverage.

Run
---
    python build_standard.py                 # default: CMIP7 tables in ./tables
    python build_standard.py --prefix CMIP6

Needs the repo checkout (bootstrapped below) and compliance-checker.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root only when run as a script; as a module import the
# root is already on sys.path, and imports must not mutate it.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.coordinate_standard import (  # noqa: E402, F401  (re-exports)
    Role, Representation, VariableKind, Coordinate, Standard,
    StandardProvider, CmorTableProvider, EsgvocProvider,
    classify_role, classify_representation,
    VERTICAL_STANDARD_NAMES, PARAMETRIC_STANDARD_NAMES, ROLE_STANDARD_NAMES,
)

TABLES_DIR = Path(__file__).parent / "tables"


def summarise(std: Standard) -> None:
    """Print the standard's provenance and classification coverage."""
    print(f"Standard: {std.source}"
          + (f"  (version {std.version})" if std.version else ""))
    print(f"  {len(std.coordinates)} coordinate objects "
          f"| skipped: {len(std.formula_terms)} formula terms, "
          f"{len(std.bounds_vars)} bounds, {len(std.grid_mappings)} grid mapping(s)\n")
    for title, key in (("Roles", lambda c: c.role.value),
                       ("Representations", lambda c: c.representation.value)):
        print(f"  {title}")
        for value, n in Counter(map(key, std.coordinates)).most_common():
            print(f"    {value:24} {n}")
    unknown = [c.name for c in std.coordinates if c.role is Role.UNKNOWN]
    print(f"  Unknown roles: {len(unknown)}"
          + (f" -> {', '.join(unknown)}" if unknown else ""))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables", default=str(TABLES_DIR),
                    help="directory of CMOR table JSON")
    ap.add_argument("--prefix", default="CMIP7",
                    help="table prefix: CMIP6 | CMIP7 | CORDEX-CMIP6")
    args = ap.parse_args()

    summarise(CmorTableProvider(args.tables, args.prefix).build())


if __name__ == "__main__":
    main()
