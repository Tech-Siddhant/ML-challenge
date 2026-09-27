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

