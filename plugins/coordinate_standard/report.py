"""Plain-text report (the CLI lives in checks/coordinate_checks/classify_file.py)."""
from __future__ import annotations

from plugins.coordinate_standard.classify import classify_dataset, read_1d_values
from plugins.coordinate_standard.matching import match, missing_coordinates
from plugins.coordinate_standard.model import Standard
from plugins.coordinate_standard.providers import esgvoc_name_findings

_STATUS = {                       # outcome prefix -> (label, is_fail)
    "MATCH": ("PASS", False), "PINNED": ("PASS", False),
    "MATCH_WITH_WARNINGS": ("WARN", False),
    "MISMATCH": ("FAIL", True), "NO_VALUE_MATCH": ("FAIL", True),
    "NO_STANDARD_ENTRY": ("FAIL", True),
    "AMBIGUOUS_UNPINNABLE": ("WARN", False), "AMBIGUOUS": ("WARN", False),
}


def report(ds, std: Standard) -> int:
    """Classify ds, match against std, print a report, return the FAIL count."""
    kinds, candidates = classify_dataset(ds)

    by_kind: dict[str, list[str]] = {}
    for name, kind in kinds.items():
        by_kind.setdefault(kind.value, []).append(name)
    print(f"\nVariables ({len(kinds)}):")
    for kind, names in sorted(by_kind.items()):
        print(f"  {kind:22} {', '.join(names)}")

    print("\nCoordinate checks:")
    fails = warns = passes = 0
    for name, cand in candidates.items():
        outcome, detail = match(cand, read_1d_values(ds.variables[name]), std)
        label, is_fail = _STATUS.get(outcome.split(":", 1)[0], ("WARN", False))
        fails += is_fail
        warns += label == "WARN"
        passes += label == "PASS"
        var = ds.variables[name]
        cf_missing = ("standard_name" not in var.ncattrs()
                      and "axis" not in var.ncattrs())
        note = "  (no standard_name/axis - role inferred)" if cf_missing else ""
        detail_str = f" [{detail}]" if detail else ""
        print(f"  {label:4} {name:18} {cand.role.value:13} "
              f"{cand.representation.value:22} {outcome}{detail_str}{note}")

    missing = missing_coordinates(ds, kinds, std)
    if missing:
        print("\nMissing coordinates:")
        for name, finding in missing.items():
            print(f"  FAIL {name:18} {finding}")
        fails += len(missing)

    vocab = esgvoc_name_findings(ds, kinds, candidates)
    if vocab:
        print("\nesgvoc vocabulary findings:")
        for name, finding in vocab.items():
            print(f"  WARN {name:18} {finding}")

    print(f"\nSummary: {len(candidates)} coordinates  |  "
          f"PASS {passes}  WARN {warns}  FAIL {fails}"
          + (f"  |  missing {len(missing)}" if missing else "")
          + (f"  |  esgvoc vocab WARN {len(vocab)}" if vocab else ""))
    return fails
