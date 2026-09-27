"""Synthetic unit tests for blocking.py — no real data."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blocking import (
    normalize_name,
    lookup_candidates_name,
    lookup_candidates_name_country,
    compute_candidate_statistics,
    candidate_recall_per_s1,
    macro_candidate_recall,
)


# 1. exact normalised-name match
def test_exact_name():
    assert normalize_name("Acme Corp") == normalize_name("Acme Corp")


# 2. case differences
def test_case():
    assert normalize_name("ACME CORP") == normalize_name("acme corp")
    assert normalize_name("Straße") == normalize_name("STRASSE")  # casefold


# 3. whitespace differences
def test_whitespace():
    assert normalize_name("  Acme   Corp  ") == normalize_name("Acme Corp")
    assert normalize_name("Acme\tCorp") == normalize_name("Acme Corp")


# 4. punctuation differences
def test_punctuation():
    assert normalize_name("Acme, Corp.") == normalize_name("Acme Corp")
    assert normalize_name("O'Reilly's") == normalize_name("OReillys")


# 5. country-specific blocking
def test_country_key():
    index = {
        "acme": [("S2-1", "United States"), ("S3-1", "India")]
    }
    us_cands = lookup_candidates_name_country("acme", "United States", index)
    in_cands = lookup_candidates_name_country("acme", "India", index)
    all_cands = lookup_candidates_name("acme", index)

    assert us_cands == {"S2-1"}
    assert in_cands == {"S3-1"}
    assert all_cands == {"S2-1", "S3-1"}


# 6. multiple S2/S3 candidates sharing a key
def test_multi_candidate_recall():
    cands = {"S2-1", "S2-2", "S3-1"}
    truth = {"S2-1", "S3-1"}
    assert candidate_recall_per_s1(cands, truth) == 1.0


# 7. no candidate
def test_no_candidate():
    assert candidate_recall_per_s1(set(), {"S2-1"}) == 0.0


# 8. union of multiple blocking rules
def test_union_recall():
    gt = {
        "S1-A": {"S2-1", "S3-1"},
        "S1-B": {"S2-2"},
        "S1-C": set(),  # singleton
    }
    cands = {
        "S1-A": {"S2-1"},         # 1/2 recall
        "S1-B": {"S2-2", "S2-9"}, # 1/1 recall
        # S1-C absent → singleton, not applicable
    }
    macro, n_app, n_sing = macro_candidate_recall(cands, gt)
    expected = (0.5 + 1.0) / 2  # 0.75
    assert abs(macro - expected) < 1e-9
    assert n_app == 2
    assert n_sing == 1


# singleton returns None
def test_singleton_recall():
    assert candidate_recall_per_s1(set(), set()) is None
    assert candidate_recall_per_s1({"X"}, set()) is None


# 9. Unicode combining characters / multilingual
def test_multilingual_combining_chars():
    hindi = "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert normalize_name(hindi) == hindi
    french = "Société Générale S.A."
    assert normalize_name(french) == "société générale sa"


# 10. Candidate statistics and reduction ratio
def test_candidate_statistics():
    counts = [0, 1, 2, 3, 4]
    stats = compute_candidate_statistics(counts, total_cartesian_pairs=100)
    assert stats["average_candidates"] == 2.0
    assert stats["median_candidates"] == 2.0
    assert stats["max_candidates"] == 4
    assert stats["total_candidate_pairs"] == 10
    assert stats["reduction_ratio"] == 0.9
    assert stats["reduction_ratio_percent"] == "90.000000%"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  PASS  {name}")
    print("\nAll blocking tests passed.")
