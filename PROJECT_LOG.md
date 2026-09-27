# Amazon ML Challenge — Project Log

## Project

Amazon ML Challenge 2026 — Business Entity Resolution

## Dataset

7 TSV files
Approximately 2.35 GB total

## Development Environment

Local Linux computer
Approximately 7.1 GB RAM / 8 GB (with 2 GB Swap available)
Approximately 138 GB storage available on /

## Experiment Log

### EXP-000 — Project Setup

Status: Done

Objective:
Set up a reproducible project structure and inventory the challenge data.

Results:
The project structure has been instantiated.  
A Python inventory script evaluated the 7 TSV files without loading them fully into RAM.  

- train/train_ground_truth.tsv: 121.13 MB, 2.2M rows
- train/train_source1.tsv: 200.34 MB, 2.2M rows (columns: entity_id, business_name, business_address, country)
- train/train_source2.tsv: 466.63 MB, 5.0M rows
- train/train_source3.tsv: 480.37 MB, 5.2M rows
- test/test_source1.tsv: 166.91 MB, 1.7M rows
- test/test_source2.tsv: 485.86 MB, 4.8M rows
- test/test_source3.tsv: 482.56 MB, 5.0M rows

Observations:
The layout follows the guidelines correctly. The environment has a functional Python 3.14 setup with basic tools ready. None of the immutable TSV data files were altered. Memory space is limited (~7.1Gi total), supporting the necessity for memory-conscious pipelines instead of bulk pandas joining.

Decision:
Proceed to experiment 001.

Next step:
EXP-001 — Memory-Conscious Data Audit


### EXP-001 — Memory-Conscious Data Audit

Status: Completed

Objective:
Understand the complete structure and data quality of the 7 challenge TSV files without loading the 2.35 GB dataset into memory at once.

Dataset:
7 TSV files (2,403.8 MB total / ~2.347 GB raw text, 26,435,994 total rows across train and test splits)

Method:
Streaming row-by-row parsing with Python's standard csv library. Aggregate online statistics (Welford mean/variance, reservoir-sampled percentiles) were maintained without in-memory row retention. ID uniqueness and cross-file reference integrity were checked sequentially using integer representations (<150 MB peak RAM).

Key findings:
- Exact total row count: 26,435,994 rows across 7 files.
  - train_source1.tsv: 2,206,821 rows (200.34 MB)
  - train_source2.tsv: 5,034,616 rows (466.63 MB)
  - train_source3.tsv: 5,285,603 rows (480.37 MB)
  - test_source1.tsv: 1,732,544 rows (166.91 MB)
  - test_source2.tsv: 4,887,273 rows (485.86 MB)
  - test_source3.tsv: 5,082,316 rows (482.56 MB)
  - train_ground_truth.tsv: 2,206,821 rows (121.13 MB)
- Schema conforms 100% to expectations (Source files: entity_id, business_name, business_address, country; Ground truth: source1_entity_id, matched_entity_ids).
- Entity ID uniqueness: 100% unique within each file; 0 duplicate entity IDs.
- Delimiter: verified tab-delimited (`\t`) across all files.

Data-quality findings:
- Missing values:
  - entity_id, business_name, country: 0 nulls, 0 empty strings across all 7 files.
  - business_address: complete in source1 (0 empty); missing (empty string) in 3.36% of train_source2 (168,967 rows), 3.33% of train_source3 (175,916 rows), 2.65% of test_source2 (129,408 rows), and 2.68% of test_source3 (136,098 rows).
  - Whitespace-only values: 0 detected across all text columns.
- Text lengths:
  - business_name: mean 23.8 to 25.7 characters across all files (min 2, max 123, median 24-25).
  - business_address: mean 46.2 to 57.2 characters (min 0 to max 269, median 37-50).
- Country distribution:
  - Train: US (60.0%) and India (40.0%) across all 3 training sources.
  - Test: Contains an open set third country, France! India (~47.1%), US (~38.3%), France (~14.6%).

Ground-truth findings:
- Total Source 1 entities represented: 2,206,821 (100% of train_source1).
- Empty match lists (singletons): 123,247 (5.58%).
- Single matches: 119,157 (5.40%).
- Multi-matches: 1,964,417 (89.02%).
- Match count distribution: 0: 123,247; 1: 119,157; 2: 375,212; 3: 530,841; 4: 484,115; 5: 321,957; 6: 164,868; 7: 63,968; 8: 18,680; 9: 4,205; 10: 534; 11: 37. Max matches per S1 entity = 11.
- Cross-file reference integrity: 100% PASSED.
  - 0 of 2,206,821 ground truth S1 IDs missing in train_source1.
  - 0 of 3,693,619 referenced S2 IDs missing in train_source2.
  - 0 of 3,944,746 referenced S3 IDs missing in train_source3.
  - 0 malformed rows, 0 duplicate IDs within any ground truth match list, 0 wrong prefixes.

Memory considerations:
- Processing ~26.4 million rows without loading DataFrames into memory kept peak RAM usage strictly under 150 MB (typical RSS ~58 MB), leaving ~1.5 GB system RAM untouched on this 7.1 GB RAM machine.
- Integer-based ID indexing avoided the >1.5 GB memory footprint of holding 12.5M raw strings.

Files created:
- `code/business_entity_resolution/src/audit.py`
- `reports/exp001_data_audit.json`
- `reports/exp001_file_summary.csv`
- `reports/experiment_results.csv`

Decision:
Data audit complete and verified. Proceed to create EDA visualizations in notebook.

Next step:
EXP-001 visualization in `notebooks/01_eda.ipynb`


### EXP-002 — Ground-Truth, Validation Split & Evaluation Harness

Status: Completed

Objective:
Build the reusable evaluation layer and create a leakage-safe Source-1-level development/validation split to benchmark future matching pipelines without data leakage.

Validation split:
- Split method: Deterministic random shuffle at the `source1_entity_id` level
- Seed: 42
- Development set: 1,765,457 S1 entities (80%)
- Validation set: 441,364 S1 entities (20%)
- Singletons: Validation split strictly includes unmatchable singleton representation (24,669 zero-match S1 entities).
- Leakage check: Verified 0 shared S2/S3 entity IDs cross the validation/development split boundary.

Evaluation implementation:
- `evaluation.py` implemented `evaluate_predictions()` reporting macro-averaged per-S1 F0.5.
- Handles edge cases faithfully: Empty predictions against nothing score `P=1, R=1, F0.5=1`; empty predictions against valid matches score `P=1, R=0, F0.5=0`.
- Includes missing/singleton ground-truth elements within macro-average to penalize aggressive false positives correctly.

Tests:
- Unit suite `test_evaluation.py` passes all 11 test cases (synthetic multi-match, duplicate removals, edge cases).
- Structural evaluation over the actual 20% validation split confirms Oracle F0.5 = 1.00 and Empty matcher F0.5 ≈ 0.0558 (recovering score strictly from correct singleton detection).

Files created:
- `code/business_entity_resolution/src/evaluation.py`
- `code/business_entity_resolution/src/exp002_run.py`
- `code/business_entity_resolution/tests/test_evaluation.py`
- `reports/exp002_validation_split.json`
- `reports/exp002_evaluation_spec.json`
- `reports/validation_s1_ids.txt`, `reports/development_s1_ids.txt`

Decision:
Base evaluation platform is secure and functional.

Next step:
EXP-003 — Baseline Candidate Generation
### EXP-003 — Baseline Candidate Generation

Status: Completed

Objective:
Build a simple, interpretable baseline candidate-generation and blocking strategy and measure how much ground truth can be recovered across the EXP-002 validation split (441,364 S1 entities).

Candidate Pool & Cartesian Scale:
- Validation S1 queries: 441,364 entities
  - Non-singletons (evaluated for recall): 416,610 entities
  - Singletons (no ground truth matches): 24,754 entities
- Candidate pool:
  - Source 2: 5,034,616 records
  - Source 3: 5,285,603 records
  - Total pool: 10,320,219 records
- Total possible Cartesian pairs: 4,554,973,138,716 (~4.55 trillion pairs)

Normalization Rules Implemented:
1. Unicode NFC Canonical Composition (`unicodedata.normalize("NFC", raw)`).
2. Case-Folding: Locale-independent lowercasing via `casefold()` (e.g. German 'ß' -> 'ss').
3. Unicode Punctuation & Symbol Removal: C-level table translation mapping all Unicode `P*` (Punctuation) and `S*` (Symbols) categories to None. Explicitly preserves Letters (`L*`), Numbers (`N*`), Whitespace (`Z*`), and Combining Characters/Marks (`M*`, e.g. Devanagari vowel matras in Hindi: 'राम' -> 'राम', not 'रम').
4. Whitespace Normalization: Collapses runs of whitespace (`\s+`) to single ASCII space and strips leading/trailing whitespace.

Strategies Evaluated:
1. Rule A — Exact Normalized Business Name:
   - Macro Candidate Recall: 0.204624 (20.46%)
   - Average Candidates / S1: 9.4190
   - Median Candidates / S1: 1.0
   - P95 Candidates: 57
   - P99 Candidates: 179
   - Max Candidates: 431
   - Total Candidate Pairs: 4,157,220
   - Candidate Reduction Ratio: 99.999909%
   - Strategy Runtime: 5.77s

2. Rule B — Exact Normalized Name + Country:
   - Macro Candidate Recall: 0.204624 (20.46%)
   - Average Candidates / S1: 9.3881
   - Median Candidates / S1: 1.0
   - P95 Candidates: 57
   - P99 Candidates: 177
   - Max Candidates: 430
   - Total Candidate Pairs: 4,143,557
   - Candidate Reduction Ratio: 99.999909%
   - Strategy Runtime: 6.23s

3. Rule C — Union Baseline (Rule A + Rule B):
   - Macro Candidate Recall: 0.204624 (20.46%)
   - Average Candidates / S1: 9.4190
   - Median Candidates / S1: 1.0
   - P95 Candidates: 57
   - P99 Candidates: 179
   - Max Candidates: 431
   - Total Candidate Pairs: 4,157,220
   - Candidate Reduction Ratio: 99.999909%
   - Strategy Runtime: 10.28s

Total Pipeline Runtime:
225.8 seconds (~3.7 minutes) end-to-end streaming all files. Peak memory RSS observed ~600 MB (from execution log), well within system limits.

Interpretation:
Exact normalized-name blocking achieved 20.46% macro candidate recall across applicable validation S1 entities. This is the macro-average of per-S1 recall values; it does not directly state the proportion of total ground-truth pairs recovered (micro/pair-level candidate recall was not computed in this run and remains pending).
Because blocking defines the candidate search space for all downstream stages, unrecovered entities at this stage cannot be matched later. The macro candidate recall therefore represents a blocking ceiling for any matcher built exclusively on these candidates.

Rule A / B / C Relationship:
```
Rule A = blocking key is normalized_name
Rule B = blocking key is (normalized_name, country)

B ⊆ A (B applies a strictly tighter filter than A)
therefore: A ∪ B = A
```
EXP-003C (Union A ∪ B) is identical to EXP-003A by construction: every candidate pair in B is already in A. The identical measured recall (0.204624) and candidate counts (4,157,220) confirm this. Rule C is not an independent blocking improvement.

Lessons Learned & Key Observations:
- Country Pruning: Incorporating country into the blocking key (Rule B) removed 13,663 non-ground-truth candidate pairs (reducing candidate volume) without losing any candidate recall (0.204624 in both). In this dataset, matching entities virtually always share the same country value.
- Script-Aware Normalization: Standard regex `[^\w\s]` strips combining marks (`Mn`/`Mc`), destroying Hindi vowels/matras in Devanagari. Preserving Unicode `M*` categories while removing `P*` and `S*` ensures multilingual data integrity across Indian and Western entity names.
- Memory Scalability: Pre-filtering S2 and S3 during stream reading against the set of unique normalized validation names (355,722 keys) reduced the active in-memory index from 10.3M records to ~824,164 records, keeping peak RSS to ~600 MB and avoiding OOM on the 7.1 GB RAM machine.

Pending:
- Micro (pair-level) candidate recall: `total_recovered_gt_pairs / total_gt_pairs`. Not computed in this run; requires a code patch and rerun (~4 min). Deferred to avoid unnecessary recomputation.

Next Hypotheses (EXP-004+):
1. Legal Entity Suffix Normalization & Stripping: Normalize corporate suffixes (e.g. "Pvt Ltd" <-> "Private Limited", "Inc", "LLC", "Corp") to bridge common lexical gaps.
2. Inverted Token Indexing: Generate candidates from shared informative tokens / word prefixes to capture partial name matches and word order inversions.
3. Character N-gram / MinHash Blocking: Catch minor spelling typos and transliteration variations with fuzzy candidate generation.
4. Address / Geographic Blocking: Block on postal code, city, or normalized address tokens to retrieve matches whose business names are completely dissimilar or abbreviated.

Decision:
EXP-003 complete and benchmark ledger updated.


