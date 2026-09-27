"""EXP-002: Reusable evaluation harness for business entity resolution.

Metric: macro F0.5 averaged across all Source 1 entities (including singletons).

Zero-case conventions (documented here, tested in test_evaluation.py):
  - truth={}, pred={}  → P=1, R=1, F0.5=1  (correctly predicted "no match")
  - truth={}, pred={x} → P=0, R=1, F0.5=0  (false positives on a singleton)
    R=1 because there's nothing to miss.  F0.5 is dominated by precision so =0.
  - truth={x}, pred={} → P=1, R=0, F0.5=0  (missed a real match; P=1 vacuously)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Set


GroundTruth = Dict[str, Set[str]]


# ---------------------------------------------------------------------------
# A/B. Parsers
# ---------------------------------------------------------------------------

def load_ground_truth(path: str | Path) -> GroundTruth:
    """Stream-parse train_ground_truth.tsv → {s1_id: {matched_ids}}."""
    gt: GroundTruth = {}
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header[0].strip() == "source1_entity_id", f"Unexpected header: {header}"
        for row in reader:
            s1 = row[0].strip()
            raw = row[1].strip() if len(row) > 1 else ""
            matched = {m.strip() for m in raw.split(",") if m.strip()} if raw else set()
            gt[s1] = matched
    return gt


def parse_predictions(preds: dict) -> GroundTruth:
    """Normalise a prediction dict to {s1_id: set(matched_ids)}.

    Accepts values as set, list, or comma-separated string.
    Deduplicates predicted IDs silently (duplicates don't help score).
    """
    out: GroundTruth = {}
    for s1, val in preds.items():
        if isinstance(val, set):
            out[s1] = val
        elif isinstance(val, (list, tuple)):
            out[s1] = set(val)
        elif isinstance(val, str):
            out[s1] = {m.strip() for m in val.split(",") if m.strip()} if val.strip() else set()
        else:
            raise TypeError(f"Unexpected type for predictions[{s1!r}]: {type(val)}")
    return out


# ---------------------------------------------------------------------------
# C/D/E. Per-S1 metrics
# ---------------------------------------------------------------------------

def per_s1_precision(pred: Set[str], truth: Set[str]) -> float:
    """See module docstring for zero-case behaviour."""
    if not pred:
        return 1.0  # vacuously correct: predicted nothing
    return len(pred & truth) / len(pred)


def per_s1_recall(pred: Set[str], truth: Set[str]) -> float:
    if not truth:
        return 1.0  # nothing to miss
    return len(pred & truth) / len(truth)


def per_s1_f05(pred: Set[str], truth: Set[str]) -> float:
    p = per_s1_precision(pred, truth)
    r = per_s1_recall(pred, truth)
    denom = 0.25 * p + r
    if denom == 0:
        return 0.0
    return 1.25 * p * r / denom


# ---------------------------------------------------------------------------
# F/G. Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: dict,
    ground_truth: GroundTruth,
    *,
    per_entity: bool = False,
) -> dict:
    """Evaluate predictions against ground truth.

    Parameters
    ----------
    predictions : dict
        {s1_id: set/list/str of predicted matched IDs}.
        S1 IDs absent from predictions are treated as predicting ∅.
    ground_truth : GroundTruth
        As returned by load_ground_truth().
    per_entity : bool
        If True, include ``per_s1`` key with per-entity detail.

    Returns
    -------
    dict with keys: macro_f0_5, macro_precision, macro_recall,
    num_s1_entities, num_zero_match_truth, num_nonzero_match_truth.
    """
    preds = parse_predictions(predictions)

    sum_f, sum_p, sum_r = 0.0, 0.0, 0.0
    n_zero = 0
    n_nonzero = 0
    details = {} if per_entity else None

    for s1, truth in ground_truth.items():
        pred = preds.get(s1, set())
        p = per_s1_precision(pred, truth)
        r = per_s1_recall(pred, truth)
        f = per_s1_f05(pred, truth)
        sum_p += p
        sum_r += r
        sum_f += f
        if truth:
            n_nonzero += 1
        else:
            n_zero += 1
        if details is not None:
            details[s1] = {"precision": p, "recall": r, "f0_5": f,
                           "n_truth": len(truth), "n_pred": len(pred)}

    n = len(ground_truth)
    result = {
        "macro_f0_5": sum_f / n if n else 0.0,
        "macro_precision": sum_p / n if n else 0.0,
        "macro_recall": sum_r / n if n else 0.0,
        "num_s1_entities": n,
        "num_zero_match_truth": n_zero,
        "num_nonzero_match_truth": n_nonzero,
    }
    if details is not None:
        result["per_s1"] = details
    return result
