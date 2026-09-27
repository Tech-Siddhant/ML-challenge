"""Unit tests for evaluation.py — synthetic data only."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation import (
    per_s1_precision, per_s1_recall, per_s1_f05,
    evaluate_predictions, parse_predictions,
)


def _f05(p, r):
    d = 0.25 * p + r
    return 1.25 * p * r / d if d else 0.0


# 1. exact 1-to-1 match
def test_exact_1to1():
    assert per_s1_precision({"A"}, {"A"}) == 1.0
    assert per_s1_recall({"A"}, {"A"}) == 1.0
    assert per_s1_f05({"A"}, {"A"}) == 1.0


# 2. correct zero-match prediction (singleton)
def test_correct_singleton():
    assert per_s1_precision(set(), set()) == 1.0
    assert per_s1_recall(set(), set()) == 1.0
    assert per_s1_f05(set(), set()) == 1.0


# 3. false positive
def test_false_positive():
    p = per_s1_precision({"A"}, set())
    assert p == 0.0
    r = per_s1_recall({"A"}, set())
    assert r == 1.0  # nothing to miss
    assert per_s1_f05({"A"}, set()) == 0.0


# 4. false negative
def test_false_negative():
    p = per_s1_precision(set(), {"A"})
    assert p == 1.0  # vacuous
    r = per_s1_recall(set(), {"A"})
    assert r == 0.0
    assert per_s1_f05(set(), {"A"}) == 0.0


# 5. multiple correct matches
def test_multi_correct():
    s = {"A", "B", "C"}
    assert per_s1_f05(s, s) == 1.0


# 6. partially correct multi-match
def test_partial_multi():
    pred = {"A", "B", "X"}
    truth = {"A", "B", "C"}
    p = 2 / 3
    r = 2 / 3
    assert abs(per_s1_precision(pred, truth) - p) < 1e-9
    assert abs(per_s1_recall(pred, truth) - r) < 1e-9
    assert abs(per_s1_f05(pred, truth) - _f05(p, r)) < 1e-9


# 7. duplicate predicted IDs (should be deduplicated)
def test_dup_predicted():
    preds = parse_predictions({"S1-1": ["A", "A", "B"]})
    assert preds["S1-1"] == {"A", "B"}


# 8. empty prediction against non-empty truth
def test_empty_pred():
    assert per_s1_f05(set(), {"A", "B"}) == 0.0


# 9. empty ground truth with non-empty prediction
def test_empty_truth_nonempty_pred():
    assert per_s1_f05({"X"}, set()) == 0.0


# 10. mixture of singleton and multi-match S1 entities
def test_mixture():
    gt = {
        "S1-1": set(),           # singleton
        "S1-2": {"A"},           # 1 match
        "S1-3": {"A", "B", "C"}, # multi
    }
    # Perfect predictions
    preds_perfect = {
        "S1-1": set(),
        "S1-2": {"A"},
        "S1-3": {"A", "B", "C"},
    }
    res = evaluate_predictions(preds_perfect, gt)
    assert abs(res["macro_f0_5"] - 1.0) < 1e-9
    assert res["num_s1_entities"] == 3
    assert res["num_zero_match_truth"] == 1
    assert res["num_nonzero_match_truth"] == 2

    # Partially wrong
    preds_bad = {
        "S1-1": {"X"},           # FP on singleton
        "S1-2": set(),           # FN
        "S1-3": {"A", "B"},     # partial
    }
    res2 = evaluate_predictions(preds_bad, gt, per_entity=True)
    d = res2["per_s1"]
    assert d["S1-1"]["f0_5"] == 0.0          # FP on singleton
    assert d["S1-2"]["f0_5"] == 0.0          # FN
    assert abs(d["S1-3"]["f0_5"] - _f05(1.0, 2/3)) < 1e-9  # pred⊂truth

    expected_macro = (0.0 + 0.0 + _f05(1.0, 2/3)) / 3
    assert abs(res2["macro_f0_5"] - expected_macro) < 1e-9


# Manual F0.5 verification
def test_manual_f05_value():
    # P=0.8, R=0.5 → F0.5 = 1.25*0.8*0.5 / (0.25*0.8+0.5) = 0.5/0.7 ≈ 0.7143
    pred = {"A", "B", "C", "D", "X"}  # 4 correct, 1 wrong out of 5
    truth = {"A", "B", "C", "D", "E", "F", "G", "H"}  # 8 total, 4 found
    p = 4/5  # 0.8
    r = 4/8  # 0.5
    expected = 1.25 * 0.8 * 0.5 / (0.25 * 0.8 + 0.5)
    assert abs(per_s1_f05(pred, truth) - expected) < 1e-9


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  PASS  {name}")
    print("\nAll tests passed.")
