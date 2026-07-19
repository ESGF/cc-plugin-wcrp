#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atomic experiment-consistency checks (ATTR007a-d).

Split from the former monolithic ATTR007 into one function per comparison, so
each has its own severity and its own Result:

  ATTR007a  experiment_id_vs_activity_id
  ATTR007b  experiment_id_vs_experiment
  ATTR007c  experiment_id_vs_parent_experiment_id
  ATTR007d  experiment_id_vs_sub_experiment_id   (CMIP6 / cmip6plus only)

Precedence rule for parent/sub:
  If the file's parent_experiment_id / sub_experiment_id attribute is absent or
  equals a "no value" token ('no parent' / 'none'), the consistency check SKIPS
  silently (returns []). Whether the attribute *should* be present is the job of
  the attribute-suite has_parent_experiment()/has_sub_experiment() rule, not of the consistency check.
  This prevents a single missing-parent situation from producing two errors.
"""

from compliance_checker.base import TestCtx

from checks.utils import (
    resolve_experiment_term,
    _as_list,
    _lower_str_list,
    NO_VALUE_TOKENS as _NO_VALUE_TOKENS,
    _ESG_VOCAB_PROJECT_API as ESG_VOCAB_AVAILABLE,
)


def _get_attr(ds, name):
    if name not in ds.ncattrs():
        return None
    return str(ds.getncattr(name)).strip()


def _no_vocab_result(check_id, label, severity):
    ctx = TestCtx(severity, f"[{check_id}] {label}")
    ctx.add_failure("The 'esgvoc' library is not installed.")
    return [ctx.to_result()]


# ---------------------------------------------------------------------------
# ATTR007a  experiment_id vs activity_id
# ---------------------------------------------------------------------------
def check_experiment_id_vs_activity_id(ds, severity, project_id="cmip6"):
    check_id, label = "ATTR007a", "Consistency: experiment_id vs activity_id"
    if not ESG_VOCAB_AVAILABLE:
        return _no_vocab_result(check_id, label, severity)
    ctx = TestCtx(severity, f"[{check_id}] {label}")

    actual = _get_attr(ds, "activity_id")
    if actual is None:
        ctx.add_failure("Missing required global attribute: 'activity_id'.")
        return [ctx.to_result()]

    term = resolve_experiment_term(ds, project_id)
    if term is None:
        ctx.add_failure("Could not resolve experiment_id in the ESGF vocabulary.")
        return [ctx.to_result()]

    expected = getattr(term, "activity_id", None)
    if not expected:
        ctx.add_pass()
        return [ctx.to_result()]

    if actual.lower() in _lower_str_list(expected):
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"Inconsistency for 'activity_id': CV expects one of "
            f"{list(_as_list(expected))}, file has '{actual}'."
        )
    return [ctx.to_result()]


# ---------------------------------------------------------------------------
# ATTR007b  experiment_id vs experiment
# ---------------------------------------------------------------------------
def check_experiment_id_vs_experiment(ds, severity, project_id="cmip6"):
    check_id, label = "ATTR007b", "Consistency: experiment_id vs experiment"
    if not ESG_VOCAB_AVAILABLE:
        return _no_vocab_result(check_id, label, severity)
    ctx = TestCtx(severity, f"[{check_id}] {label}")

    actual = _get_attr(ds, "experiment")
    if actual is None:
        ctx.add_failure("Missing required global attribute: 'experiment'.")
        return [ctx.to_result()]

    term = resolve_experiment_term(ds, project_id)
    if term is None:
        ctx.add_failure("Could not resolve experiment_id in the ESGF vocabulary.")
        return [ctx.to_result()]

    expected = getattr(term, "experiment", None) or getattr(term, "description", None)
    if not expected:
        ctx.add_pass()
        return [ctx.to_result()]

    if actual == str(expected).strip():
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"Inconsistency for 'experiment': CV expects '{expected}', "
            f"file has '{actual}'."
        )
    return [ctx.to_result()]


# ---------------------------------------------------------------------------
# ATTR007c  experiment_id vs parent_experiment_id  (precedence-aware)
# ---------------------------------------------------------------------------
def check_experiment_id_vs_parent_experiment_id(ds, severity, project_id="cmip6"):
    check_id, label = "ATTR007c", "Consistency: experiment_id vs parent_experiment_id"
    if not ESG_VOCAB_AVAILABLE:
        return _no_vocab_result(check_id, label, severity)
    ctx = TestCtx(severity, f"[{check_id}] {label}")

    actual = _get_attr(ds, "parent_experiment_id")

    # Precedence: absent or "no parent" -> not this check's responsibility.
    if actual is None or actual.strip().lower() in _NO_VALUE_TOKENS:
        return []

    term = resolve_experiment_term(ds, project_id)
    if term is None:
        ctx.add_failure("Could not resolve experiment_id in the ESGF vocabulary.")
        return [ctx.to_result()]

    expected = getattr(term, "parent_experiment_id", None)
    # CMIP7 exposes the parent as a nested object under 'parent_experiment'
    if not expected:
        parent_obj = getattr(term, "parent_experiment", None)
        if parent_obj is not None:
            expected = [getattr(parent_obj, "drs_name", None)
                        or getattr(parent_obj, "id", None)]

    if not expected:
        # File declares a parent but the CV declares none -> inconsistency.
        ctx.add_failure(
            f"Inconsistency for 'parent_experiment_id': file declares '{actual}' "
            f"but the CV declares no parent for this experiment."
        )
        return [ctx.to_result()]

    if actual.lower() in _lower_str_list(expected):
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"Inconsistency for 'parent_experiment_id': CV expects one of "
            f"{list(_as_list(expected))}, file has '{actual}'."
        )
    return [ctx.to_result()]


# ---------------------------------------------------------------------------
# ATTR007d  experiment_id vs sub_experiment_id  (CMIP6/plus, precedence-aware)
# ---------------------------------------------------------------------------
def check_experiment_id_vs_sub_experiment_id(ds, severity, project_id="cmip6"):
    check_id, label = "ATTR007d", "Consistency: experiment_id vs sub_experiment_id"
    if not ESG_VOCAB_AVAILABLE:
        return _no_vocab_result(check_id, label, severity)
    ctx = TestCtx(severity, f"[{check_id}] {label}")

    actual = _get_attr(ds, "sub_experiment_id")

    # Precedence: absent or 'none' -> not this check's responsibility.
    if actual is None or actual.strip().lower() in _NO_VALUE_TOKENS:
        return []

    term = resolve_experiment_term(ds, project_id)
    if term is None:
        ctx.add_failure("Could not resolve experiment_id in the ESGF vocabulary.")
        return [ctx.to_result()]

    expected = getattr(term, "sub_experiment_id", None)
    expected_norm = [s for s in _lower_str_list(expected) if s not in _NO_VALUE_TOKENS]

    if not expected_norm:
        ctx.add_failure(
            f"Inconsistency for 'sub_experiment_id': file declares '{actual}' "
            f"but the CV declares no sub-experiment for this experiment."
        )
        return [ctx.to_result()]

    if actual.lower() in expected_norm:
        ctx.add_pass()
    else:
        ctx.add_failure(
            f"Inconsistency for 'sub_experiment_id': CV expects one of "
            f"{list(_as_list(expected))}, file has '{actual}'."
        )
    return [ctx.to_result()]
