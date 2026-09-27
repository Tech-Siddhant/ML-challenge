#!/usr/bin/env python3
"""EXP-002: Create validation split + verify evaluator + generate artifacts.

Reads only train_ground_truth.tsv (121 MB, streamed).
Does NOT touch the multi-GB source TSVs.
"""
import csv, hashlib, json, random, time
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation import load_ground_truth, evaluate_predictions

SEED = 42
VAL_FRAC = 0.20
REPO = Path(__file__).resolve().parents[3]
GT_PATH = REPO / "data" / "train" / "train_ground_truth.tsv"
REPORTS = REPO / "reports"


def main():
    t0 = time.time()
    gt = load_ground_truth(GT_PATH)
    total_s1 = len(gt)
    print(f"Loaded {total_s1:,} S1 entities from ground truth.")

    # Deterministic S1-level split
    s1_ids = sorted(gt.keys())
    rng = random.Random(SEED)
    rng.shuffle(s1_ids)
    n_val = int(total_s1 * VAL_FRAC)
    val_ids = set(s1_ids[:n_val])
    dev_ids = set(s1_ids[n_val:])

    # Verify
    assert len(val_ids & dev_ids) == 0, "LEAK: overlap"
    assert len(val_ids) + len(dev_ids) == total_s1
    val_sing = sum(1 for s in val_ids if not gt[s])
    dev_sing = sum(1 for s in dev_ids if not gt[s])
    print(f"Dev: {len(dev_ids):,} ({dev_sing:,} singletons)")
    print(f"Val: {len(val_ids):,} ({val_sing:,} singletons)")

    # Leakage check: shared S2/S3 IDs across S1 entities
    s2s3_owners: dict[str, list[str]] = {}
    for s1, matched in gt.items():
        for m in matched:
            s2s3_owners.setdefault(m, []).append(s1)
    shared = {m: o for m, o in s2s3_owners.items() if len(o) > 1}
    cross = sum(1 for o in shared.values()
                if any(x in dev_ids for x in o) and any(x in val_ids for x in o))
    leak_note = (f"{len(shared):,} S2/S3 IDs shared across S1s; "
                 f"{cross:,} cross dev/val boundary. Reported, not corrected.")
    print(f"Leakage: {leak_note}")
    del s2s3_owners  # free memory

    # Structural evaluator test
    val_gt = {s: gt[s] for s in val_ids}
    oracle = evaluate_predictions({s: gt[s] for s in val_ids}, val_gt)
    assert abs(oracle["macro_f0_5"] - 1.0) < 1e-9, "Oracle broken"
    empty = evaluate_predictions({}, val_gt)
    exp_empty = val_sing / len(val_ids)
    assert abs(empty["macro_f0_5"] - exp_empty) < 1e-6, "Empty-pred check failed"
    print(f"Evaluator OK: oracle=1.0, empty={exp_empty:.6f}")

    val_match_dist = Counter(len(gt[s]) for s in val_ids)
    elapsed = round(time.time() - t0, 2)

    # Write artifacts
    vs = sorted(val_ids); ds = sorted(dev_ids)
    vh = hashlib.sha256("\n".join(vs).encode()).hexdigest()
    dh = hashlib.sha256("\n".join(ds).encode()).hexdigest()

    split_m = {
        "experiment": "EXP-002", "split_method": "random_shuffle_s1_entity_level",
        "random_seed": SEED, "total_s1_entities": total_s1,
        "num_development_s1": len(dev_ids), "num_validation_s1": len(val_ids),
        "validation_fraction_actual": round(len(val_ids)/total_s1, 6),
        "dev_singletons": dev_sing, "val_singletons": val_sing,
        "overlap_count": 0, "val_id_sha256": vh, "dev_id_sha256": dh,
        "leakage_shared_s2s3_ids": len(shared), "leakage_cross_boundary": cross,
        "leakage_note": leak_note,
        "val_match_count_distribution": {str(k): v for k, v in sorted(val_match_dist.items())},
    }
    (REPORTS/"exp002_validation_split.json").write_text(json.dumps(split_m, indent=2)+"\n")

    eval_spec = {
        "experiment": "EXP-002", "metric": "macro_F0.5",
        "formula": "F0.5 = 1.25*P*R / (0.25*P + R)",
        "scope": "per S1 entity, then macro-averaged", "includes_singletons": True,
        "zero_case_conventions": {
            "truth_empty_pred_empty": {"P":1,"R":1,"F0.5":1},
            "truth_empty_pred_nonempty": {"P":0,"R":1,"F0.5":0},
            "truth_nonempty_pred_empty": {"P":1,"R":0,"F0.5":0},
        },
        "module": "code/business_entity_resolution/src/evaluation.py",
        "api": "evaluate_predictions(predictions, ground_truth, per_entity=False)",
    }
    (REPORTS/"exp002_evaluation_spec.json").write_text(json.dumps(eval_spec, indent=2)+"\n")

    (REPORTS/"validation_s1_ids.txt").write_text("\n".join(vs)+"\n")
    (REPORTS/"development_s1_ids.txt").write_text("\n".join(ds)+"\n")

    with open(REPORTS/"experiment_results.csv", "a", newline="") as f:
        csv.writer(f).writerow([
            "EXP-002","Validation Split & Evaluation Harness","completed",
            "train_ground_truth.tsv", elapsed,
            f"seed={SEED} dev={len(dev_ids)} val={len(val_ids)} "
            f"shared_s2s3={len(shared)} cross_boundary={cross}",
            "","","","",
        ])

    print(f"\nArtifacts written. Runtime: {elapsed}s")
    return split_m, elapsed


if __name__ == "__main__":
    main()

    assert val_sing > 0, "No singletons in validation!"
