"""Pareto-aware mutation logging policy.

This module decides whether a :class:`~src.rag.bookkeeping.MutationEvent` is
*Pareto-eligible* within the context of the current generation's population.
The result is stored as the ``is_pareto_eligible`` flag on the event — it is
**additive** metadata that downstream analysis can use to filter; every event
is always appended to the ledger regardless of eligibility.

Design decisions
----------------
- **Per-generation window**: eligibility is computed relative to the current
  generation's population only.  This avoids early-generation low-quality
  events dominating later generations, and avoids cross-experiment pollution
  (per-run Pareto would mix different conditions in ablation studies).
- **Percentile OR eligibility**: an event is eligible if it is in the top
  ``RAG_LOG_TOP_ACCURACY_PCT`` percent by ``test_accuracy`` OR in the bottom
  ``RAG_LOG_BOTTOM_PARAMS_PCT`` percent by ``total_params``.  The OR logic
  is inclusive — any improvement along either objective is noteworthy.
- **Rounding**: ``math.ceil`` is used when converting a percentile to a count
  so that a 10% cut of 7 elements gives ceil(0.7) = 1 rather than 0.
  This guarantees at least one event is eligible in any non-empty population.
- **Missing eval_outputs**: events without ``eval_outputs`` (e.g. augment-only
  events from RagService) are always ineligible and never crash the caller.

Relationship to ``scripts/pareto_front.py``
--------------------------------------------
``scripts/pareto_front.py:is_pareto_efficient`` operates on a raw numpy array
of minimisation objectives loaded from results files — it is a plotting helper
and does not know about :class:`MutationEvent`.  The ``compute_pareto_front``
function here operates on :class:`MutationEvent` sequences and uses the same
dominance criterion (weakly better in both objectives, strictly better in at
least one) translated to the ``(test_accuracy, -total_params)`` space so both
are maximised.

TODO: once PR 6 lands, consider making ``scripts/pareto_front.py`` delegate to
``src.rag.pareto_policy.compute_pareto_front`` to avoid duplication of the
dominance logic.
"""

from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING, Sequence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.rag.bookkeeping import MutationEvent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Return True if point *a* (weakly) Pareto-dominates point *b*.

    Both points are in the space ``(test_accuracy, -total_params)`` where
    **higher is better** for both objectives.

    A dominates B iff:
    - a[0] >= b[0] and a[1] >= b[1]  (weakly better in all objectives)
    - a[0] >  b[0] or  a[1] >  b[1]  (strictly better in at least one)

    Args:
        a: ``(test_accuracy_a, neg_params_a)`` — candidate dominator.
        b: ``(test_accuracy_b, neg_params_b)`` — candidate dominated point.

    Returns:
        True if *a* dominates *b*; False otherwise (including the case where
        a == b, which is not dominated by definition).
    """
    weakly_better = a[0] >= b[0] and a[1] >= b[1]
    strictly_better = a[0] > b[0] or a[1] > b[1]
    return weakly_better and strictly_better


def _event_to_point(event: "MutationEvent") -> tuple[float, float] | None:
    """Extract the Pareto objective point from *event*.

    Returns ``None`` when ``eval_outputs`` is missing or incomplete.
    The point is in ``(test_accuracy, -total_params)`` space (both maximised).
    """
    eo = event.eval_outputs
    if not eo:
        return None
    # Support both key variants emitted by the codebase:
    # - run_improved.py writes "test_acc" and "total_params"
    # - bookkeeping schema uses "test_accuracy" and "total_params"
    acc = eo.get("test_accuracy") if eo.get("test_accuracy") is not None else eo.get("test_acc")
    params = eo.get("total_params")
    if acc is None or params is None:
        return None
    try:
        return (float(acc), -float(params))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pareto_front(events: Sequence["MutationEvent"]) -> list["MutationEvent"]:
    """Return the non-dominated subset of *events* (the Pareto front).

    Events without ``eval_outputs`` are excluded from dominance calculations
    (they cannot be on the Pareto front) and are also excluded from the return
    value.

    The dominance space is ``(test_accuracy, -total_params)`` so both
    objectives are maximised.  An event is non-dominated if no other event
    with valid ``eval_outputs`` dominates it.

    Args:
        events: Sequence of :class:`~src.rag.bookkeeping.MutationEvent` from
            a single generation (or any window).

    Returns:
        List of non-dominated :class:`~src.rag.bookkeeping.MutationEvent`
        instances in the order they appear in *events*.
    """
    # Build (index, point) pairs for events that have valid eval_outputs.
    valid: list[tuple[int, tuple[float, float]]] = []
    for i, ev in enumerate(events):
        pt = _event_to_point(ev)
        if pt is not None:
            valid.append((i, pt))

    if not valid:
        return []

    # Mark each valid point as non-dominated.
    n = len(valid)
    is_non_dominated = [True] * n

    for i in range(n):
        if not is_non_dominated[i]:
            continue
        for j in range(n):
            if i == j or not is_non_dominated[j]:
                continue
            # If j dominates i, i is dominated.
            if _dominates(valid[j][1], valid[i][1]):
                is_non_dominated[i] = False
                break

    return [events[valid[k][0]] for k, non_dominated in enumerate(is_non_dominated) if non_dominated]


def is_pareto_eligible(
    event: "MutationEvent",
    generation_population: Sequence["MutationEvent"],
    top_accuracy_pct: float | None = None,
    bottom_params_pct: float | None = None,
) -> bool:
    """Return True if *event* is Pareto-eligible within *generation_population*.

    Eligibility is an OR of two percentile conditions:

    1. The event is in the top ``top_accuracy_pct`` percent of the population
       by ``eval_outputs["test_accuracy"]`` (or ``"test_acc"``).
    2. The event is in the bottom ``bottom_params_pct`` percent of the
       population by ``eval_outputs["total_params"]``.

    Both conditions use *inclusive* rounding (``math.ceil``) so that at least
    one event is always eligible in any non-empty population with valid data.

    If *event* has no ``eval_outputs`` (or the required keys are absent),
    this returns ``False`` immediately.  An event without metrics cannot be
    classified and is never eligible.

    Args:
        event: The :class:`~src.rag.bookkeeping.MutationEvent` to classify.
        generation_population: All events in the current generation window
            (including *event* itself).
        top_accuracy_pct: Override for ``RAG_LOG_TOP_ACCURACY_PCT``. When
            ``None``, the value is read from ``cfg.constants`` (lazy import).
        bottom_params_pct: Override for ``RAG_LOG_BOTTOM_PARAMS_PCT``. When
            ``None``, the value is read from ``cfg.constants`` (lazy import).

    Returns:
        ``True`` if the event meets either percentile criterion; ``False``
        otherwise.
    """
    # Guard: event must have valid eval_outputs.
    event_pt = _event_to_point(event)
    if event_pt is None:
        return False

    # Resolve percentile knobs (lazy import to stay torch-free at module load).
    if top_accuracy_pct is None or bottom_params_pct is None:
        try:
            from cfg.constants import (  # noqa: PLC0415
                RAG_LOG_TOP_ACCURACY_PCT,
                RAG_LOG_BOTTOM_PARAMS_PCT,
            )
            if top_accuracy_pct is None:
                top_accuracy_pct = float(RAG_LOG_TOP_ACCURACY_PCT)
            if bottom_params_pct is None:
                bottom_params_pct = float(RAG_LOG_BOTTOM_PARAMS_PCT)
        except ImportError:
            top_accuracy_pct = top_accuracy_pct if top_accuracy_pct is not None else 10.0
            bottom_params_pct = bottom_params_pct if bottom_params_pct is not None else 10.0

    event_acc = event_pt[0]         # test_accuracy (higher is better)
    event_params = -event_pt[1]     # total_params (lower is better; point stored as negative)

    # Collect acc and params for all events with valid eval_outputs.
    all_accs: list[float] = []
    all_params: list[float] = []
    for ev in generation_population:
        pt = _event_to_point(ev)
        if pt is not None:
            all_accs.append(pt[0])
            all_params.append(-pt[1])  # convert back from negated storage

    if not all_accs:
        # No comparable events — trivially eligible.
        return True

    n = len(all_accs)

    # Top accuracy threshold: top top_accuracy_pct% means highest accuracy values.
    top_count = max(1, math.ceil(n * top_accuracy_pct / 100.0))
    # Sort descending by accuracy; threshold is the accuracy of the top_count-th item.
    sorted_accs_desc = sorted(all_accs, reverse=True)
    acc_threshold = sorted_accs_desc[top_count - 1]

    # Bottom params threshold: bottom bottom_params_pct% means lowest param counts.
    bottom_count = max(1, math.ceil(n * bottom_params_pct / 100.0))
    sorted_params_asc = sorted(all_params)
    params_threshold = sorted_params_asc[bottom_count - 1]

    in_top_accuracy = event_acc >= acc_threshold
    in_bottom_params = event_params <= params_threshold

    return in_top_accuracy or in_bottom_params
