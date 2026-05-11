# Benchmark Metadata

This public repository ships a lightweight description of the benchmark used in
the Phase 7 balanced experiments without shipping the full curated artifact set.

## Scope

- Dataset family: CGT + DAppSCAN runtime-bytecode benchmark
- Active variant: `clean_default`
- Contracts: 2,186
- Evaluated SWCs: 101, 103, 104, 107, 113, 114, 115, 120, 128, 135
- Split counts: train 1,749 / val 219 / test 218
- Source breakdown: 1,814 CGT-only / 363 DAppSCAN-only / 9 shared

## What is intentionally omitted

The full curated graphs, feature matrices, checkpoints, and generated reports
are not included in this first public repository push. This repository is meant
to show the architecture pipeline and the active experiment path clearly.

## Public run path

Use `configs/phase7_balanced.yaml` as the active experiment configuration and
`scripts/level4_analysis.py` as the retained aggregate-analysis script.
