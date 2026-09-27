#!/usr/bin/env python3
"""EXP-003: Baseline candidate generation via exact-key blocking.

Memory-conscious:
- Streams TSVs row-by-row with target-key filtering.
- Peak RSS observed: ~600 MB during the completed run.
- Evaluates Rules A, B, and C over EXP-002 validation split (441,364 S1 entities).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from blocking import (
    CandidateIndex,
    compute_candidate_statistics,
    lookup_candidates_name,
    lookup_candidates_name_country,
    normalize_name,
    update_candidate_index,
    _stream_tsv,
)

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "train"
REPORTS = REPO / "reports"


def load_validation_ground_truth(
    gt_path: Path, val_ids: set[str]
) -> tuple[dict[str, set[str]], int]:
    """Stream ground truth and return mapping {s1_id: set(s2_s3_ids)} for val_ids.

    Also returns total ground-truth rows read.
    """
    val_gt: dict[str, set[str]] = {s1: set() for s1 in val_ids}
    total_rows = 0
    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            total_rows += 1
            s1_id = row[0]
            if s1_id in val_ids and len(row) > 1 and row[1].strip():
                matches = {m.strip() for m in row[1].split(",") if m.strip()}
                val_gt[s1_id] = matches
    return val_gt, total_rows



def main():
    t_wall = time.time()

    # -- 1. Load validation S1 IDs and ground truth -------------------------
    print("Step 1: Loading validation IDs and filtered ground truth...")
    t0 = time.time()
    val_ids_file = REPORTS / "validation_s1_ids.txt"
    val_ids = set(val_ids_file.read_text().split())
    num_val_s1 = len(val_ids)
    print(f"  Loaded {num_val_s1:,} validation S1 IDs")

    val_gt, _ = load_validation_ground_truth(DATA / "train_ground_truth.tsv", val_ids)
    num_applicable = sum(1 for s in val_gt if val_gt[s])
    num_singletons = num_val_s1 - num_applicable
    print(
        f"  Val ground truth loaded ({time.time()-t0:.1f}s): "
        f"{num_applicable:,} non-singletons, {num_singletons:,} singletons"
    )

    # -- 2. Stream S1 attributes for val entities --------------------------
    print("Step 2: Streaming S1 attributes for validation entities...")
    t0 = time.time()
    s1_attributes: dict[str, tuple[str, str]] = {}
    val_names: set[str] = set()

    for eid, name, country in _stream_tsv(DATA / "train_source1.tsv"):
        if eid in val_ids:
            norm_name = normalize_name(name)
            s1_attributes[eid] = (norm_name, country.strip())
            if norm_name:
                val_names.add(norm_name)

    print(
        f"  Streamed {len(s1_attributes):,} S1 entities ({time.time()-t0:.1f}s), "
        f"{len(val_names):,} unique normalized names"
    )

    # -- 3. Build candidate index from S2 and S3 filtered by val_names ------
    print("Step 3: Building filtered candidate index from S2 & S3...")
    candidate_index: CandidateIndex = {}

    t0 = time.time()
    s2_indexed = update_candidate_index(
        DATA / "train_source2.tsv", candidate_index, target_names=val_names
    )
    print(f"  Source 2: indexed {s2_indexed:,} matching records ({time.time()-t0:.1f}s)")

    t0 = time.time()
    s3_indexed = update_candidate_index(
        DATA / "train_source3.tsv", candidate_index, target_names=val_names
    )
    print(f"  Source 3: indexed {s3_indexed:,} matching records ({time.time()-t0:.1f}s)")
    print(f"  Total index keys: {len(candidate_index):,}")

    # Total S2 + S3 records for reduction ratio:
    total_pool_records = 5034616 + 5285603
    total_cartesian_pairs = num_val_s1 * total_pool_records


    # -- 4. Evaluate strategies --------------------------------------------
    strategies = [
        ("A", "exact_normalized_name", "Baseline blocking: exact normalized name"),
        ("B", "exact_normalized_name_country", "Baseline blocking: exact normalized name + country"),
        ("C", "union_A_B", "Baseline blocking: union A + B"),
    ]

    results = {}

    for label, strategy_key, strategy_desc in strategies:
        print(f"\nEvaluating EXP-003{label} ({strategy_desc})...")
        strat_t0 = time.time()

        recall_sum = 0.0
        candidate_counts: list[int] = []

        for s1_id in val_ids:
            norm_name, country = s1_attributes.get(s1_id, ("", ""))
            truth = val_gt.get(s1_id, set())

            if label == "A":
                cands = lookup_candidates_name(norm_name, candidate_index)
            elif label == "B":
                cands = lookup_candidates_name_country(norm_name, country, candidate_index)
            else:  # "C" = Union of A and B
                cands_a = lookup_candidates_name(norm_name, candidate_index)
                cands_b = lookup_candidates_name_country(norm_name, country, candidate_index)
                cands = cands_a | cands_b

            candidate_counts.append(len(cands))
            if truth:
                recall_sum += len(cands & truth) / len(truth)

        macro_recall = recall_sum / num_applicable if num_applicable > 0 else 0.0
        strat_runtime = round(time.time() - strat_t0, 2)

        stats = compute_candidate_statistics(
            candidate_counts, total_cartesian_pairs=total_cartesian_pairs
        )

        res_entry = {
            "experiment_id": f"EXP-003{label}",
            "strategy": strategy_key,
            "macro_candidate_recall": round(macro_recall, 6),
            "num_s1_evaluated": num_val_s1,
            "num_s1_with_ground_truth": num_applicable,
            "num_singletons_skipped": num_singletons,
            **stats,
            "runtime_seconds": strat_runtime,
        }
        results[f"EXP-003{label}"] = res_entry

        print(f"  Macro Candidate Recall : {macro_recall:.6f} ({macro_recall*100:.2f}%)")
        print(f"  Average Candidates     : {stats['average_candidates']}")
        print(f"  Median Candidates      : {stats['median_candidates']}")
        print(f"  P95 Candidates         : {stats['p95_candidates']}")
        print(f"  P99 Candidates         : {stats['p99_candidates']}")
        print(f"  Max Candidates         : {stats['max_candidates']}")
        print(f"  Total Candidate Pairs  : {stats['total_candidate_pairs']:,}")
        print(f"  Reduction Ratio        : {stats.get('reduction_ratio_percent', 'N/A')}")
        print(f"  Strategy Runtime       : {strat_runtime}s")

    total_wall_time = round(time.time() - t_wall, 2)

    # -- 5. Write reports/exp003_blocking_results.json ---------------------
    report = {
        "experiment": "EXP-003",
        "description": "Baseline candidate generation via exact-key blocking",
        "validation_split": "reports/validation_s1_ids.txt (EXP-002, seed=42)",
        "total_runtime_seconds": total_wall_time,
        "strategies": results,
    }
    report_path = REPORTS / "exp003_blocking_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWritten {report_path}")

    # -- 6. Append to reports/experiment_results.csv -----------------------
    csv_path = REPORTS / "experiment_results.csv"

    existing_lines = []
    if csv_path.exists():
        existing_lines = csv_path.read_text().splitlines()

    header = "experiment_id,experiment_name,status,dataset,runtime_seconds,notes,f05,precision,recall,candidate_recall"
    if not existing_lines:
        existing_lines = [header]

    cleaned_lines = [
        line for line in existing_lines
        if not any(line.startswith(f"EXP-003{s},") for s in ("A", "B", "C"))
    ]

    for label in ("A", "B", "C"):
        r = results[f"EXP-003{label}"]
        notes = (
            f"recall={r['macro_candidate_recall']} "
            f"avg_cands={r['average_candidates']} "
            f"median={r['median_candidates']} "
            f"p99={r['p99_candidates']} "
            f"max={r['max_candidates']} "
            f"reduction_ratio={r.get('reduction_ratio_percent', 'N/A')}"
        )
        row = (
            f"{r['experiment_id']},"
            f"\"{r['strategy']}\","
            f"completed,"
            f"\"val split (441364 S1)\","
            f"{r['runtime_seconds']},"
            f"\"{notes}\","
            f","
            f","
            f","
            f"{r['macro_candidate_recall']}"
        )
        cleaned_lines.append(row)

    csv_path.write_text("\n".join(cleaned_lines) + "\n")
    print(f"Updated {csv_path}")
    print(f"Total EXP-003 execution time: {total_wall_time:.1f}s")


if __name__ == "__main__":
    main()


