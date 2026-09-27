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

