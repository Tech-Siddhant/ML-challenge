# Amazon ML Challenge 2026 — Business Entity Resolution

## What this is

This repository is the development workspace for the **Amazon ML Challenge 2026**.  
The task is **business entity resolution**: given business records from 3 independent
data sources with noisy and inconsistent fields, determine which records refer to the
same real-world business entity.

## Workflow

- **Script-first.** Core pipeline logic lives in Python scripts under
  `code/business_entity_resolution/src/`. Notebooks are for visualization and error
  analysis only — never for pipeline logic.
- **Immutable data.** The original TSV files in `data/` are never modified or copied
  unnecessarily.
- **Memory-conscious.** The dataset is ~2.35 GB total across 7 TSV files. The machine
  has ~8 GB RAM. Files are processed with streaming/chunked IO; never loaded all at once.
- **Reproducible.** Every non-trivial step is a numbered experiment (`EXP-000`,
  `EXP-001`, …) logged in [`PROJECT_LOG.md`](PROJECT_LOG.md).

## Layout

```
amazon-ml-challenge/
├── data/
│   ├── train/          # 4 TSV files — source1, source2, source3, ground_truth
│   └── test/           # 3 TSV files — source1, source2, source3
├── code/
│   └── business_entity_resolution/
│       ├── src/            # pipeline scripts (added per experiment)
│       ├── experiments/    # per-experiment artefacts
│       ├── README.md       # challenge problem statement
│       └── requirements.txt
├── notebooks/          # visualization / error analysis only
├── reports/
├── output/             # submission files land here
├── Documentation_template.md   # official challenge template (preserved as-is)
└── PROJECT_LOG.md      # experiment log
```

## Do not use

- External business databases, entity-resolution APIs, geocoding APIs
- External LLM / data-enrichment services
- Any external data augmentation — the challenge prohibits it

## Evaluation metric

Macro-averaged **F₀.₅** (precision-weighted 2× over recall) across all Source 1
entities on the test set. Singletons (no matches) count too — predict an empty list
there and Score = 1.0.

## Current status

**EXP-000 — Project Setup** (see [`PROJECT_LOG.md`](PROJECT_LOG.md))
