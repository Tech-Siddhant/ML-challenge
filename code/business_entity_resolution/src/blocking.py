"""Blocking / candidate-generation utilities for business entity resolution.

Memory-conscious: builds compact indexes by streaming TSV files row-by-row
with optional target-key filtering. No pandas, no full Cartesian product.
"""

from __future__ import annotations

import csv
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

# Precomputed translation table mapping all Unicode Punctuation ('P*') and
# Symbol ('S*') code points to None.
# Preserves Letters ('L*'), Marks/Combining characters ('M*'), Numbers ('N*'),
# and Whitespace ('Z*'). Crucial for non-Latin scripts (e.g. Hindi matras).
_PUNCT_TABLE = dict.fromkeys(
    i for i in range(0x110000)
    if unicodedata.category(chr(i)).startswith(("P", "S"))
)
_MULTI_WS = re.compile(r"\s+")


def normalize_name(raw: str) -> str:
    """Deterministic business-name normalisation.

    Steps (order matters):
    1. Unicode NFC normalisation (canonical decomposition followed by canonical composition).
    2. Case-folding (locale-independent lowercase, e.g. 'ß' -> 'ss').
    3. Remove Unicode punctuation ('P*') and symbols ('S*') via fast C-level table translation.
       This explicitly preserves combining characters (Unicode 'M*', e.g. Hindi matras).
    4. Collapse runs of whitespace to a single ASCII space and strip leading/trailing whitespace.
    """
    s = unicodedata.normalize("NFC", raw)
    s = s.casefold()
    s = s.translate(_PUNCT_TABLE)
    s = _MULTI_WS.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------
# Index builders  (stream a TSV, emit compact dicts)
# --------------------------------------------------------------------------

IndexEntry = Tuple[str, str]
CandidateIndex = Dict[str, List[IndexEntry]]


def _stream_tsv(path: str | Path):
    """Yield (entity_id, business_name, country) from a source TSV."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        # columns: entity_id, business_name, business_address, country
        for row in reader:
            yield row[0], row[1], row[3]


def update_candidate_index(
    path: str | Path,
    index: CandidateIndex,
    target_names: Optional[Set[str]] = None,
) -> int:
    """Stream a source TSV and add entries to index.

    If target_names is provided, only records whose normalized_name is in
    target_names are indexed (dramatically reducing RAM usage).
    Returns count of indexed records.
    """
    indexed_count = 0
    for eid, name, country in _stream_tsv(path):
        key = normalize_name(name)
        if not key:
            continue
        if target_names is not None and key not in target_names:
            continue
        index.setdefault(key, []).append((eid, country.strip()))
        indexed_count += 1
    return indexed_count



# --------------------------------------------------------------------------
# Candidate generation lookup functions
# --------------------------------------------------------------------------

def lookup_candidates_name(
    s1_name_norm: str,
    index: CandidateIndex,
) -> Set[str]:
    """Rule A: Return S2/S3 candidate IDs sharing the exact normalized name."""
    entries = index.get(s1_name_norm)
    if not entries:
        return set()
    return {eid for eid, _ in entries}


def lookup_candidates_name_country(
    s1_name_norm: str,
    s1_country: str,
    index: CandidateIndex,
) -> Set[str]:
    """Rule B: Return S2/S3 candidate IDs sharing normalized name AND country."""
    entries = index.get(s1_name_norm)
    if not entries:
        return set()
    c_norm = s1_country.strip()
    return {eid for eid, country in entries if country == c_norm}


# --------------------------------------------------------------------------
# Candidate-recall calculation & statistics
# --------------------------------------------------------------------------

def candidate_recall_per_s1(
    candidates: Set[str], truth: Set[str]
) -> Optional[float]:
    """Recall of ground-truth matches within candidate set.

    Returns None for singletons (truth is empty) — not applicable.
    """
    if not truth:
        return None  # singleton: metric undefined
    return len(candidates & truth) / len(truth)


def macro_candidate_recall(
    all_candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, int, int]:
    """Macro-averaged candidate recall across applicable S1 entities.

    Returns (macro_recall, n_applicable, n_singleton).
    """
    total = 0.0
    n_app = 0
    n_sing = 0
    for s1, truth in ground_truth.items():
        if not truth:
            n_sing += 1
            continue
        cands = all_candidates.get(s1, set())
        total += len(cands & truth) / len(truth)
        n_app += 1
    macro = total / n_app if n_app else 0.0
    return macro, n_app, n_sing


def compute_candidate_statistics(
    candidate_counts: List[int],
    total_cartesian_pairs: Optional[int] = None,
) -> Dict[str, object]:
    """Calculate summary statistics for candidate counts across S1 queries."""
    if not candidate_counts:
        return {}
    counts_sorted = sorted(candidate_counts)
    n = len(counts_sorted)
    total_pairs = sum(counts_sorted)

    stats: Dict[str, object] = {
        "average_candidates": round(statistics.mean(counts_sorted), 4),
        "median_candidates": float(statistics.median(counts_sorted)),
        "p95_candidates": counts_sorted[int(n * 0.95)],
        "p99_candidates": counts_sorted[int(n * 0.99)],
        "max_candidates": counts_sorted[-1],
        "total_candidate_pairs": total_pairs,
    }

    if total_cartesian_pairs and total_cartesian_pairs > 0:
        reduction_ratio = 1.0 - (total_pairs / total_cartesian_pairs)
        stats["total_cartesian_pairs"] = total_cartesian_pairs
        stats["reduction_ratio"] = round(reduction_ratio, 8)
        stats["reduction_ratio_percent"] = f"{reduction_ratio * 100:.6f}%"

    return stats

