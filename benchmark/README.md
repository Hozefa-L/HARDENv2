# Benchmark Metadata

This public repository ships the benchmark inputs required for the retained Phase 7 rerun path.

## Scope

- Dataset family: CGT + DAppSCAN runtime-bytecode benchmark
- Active variant: `clean_default`
- Contracts: 2,186
- Evaluated SWCs: 101, 103, 104, 107, 113, 114, 115, 120, 128, 135
- Split counts: train 1,749 / val 219 / test 218
- Source breakdown: 1,814 CGT-only / 363 DAppSCAN-only / 9 shared

## Included public inputs

- the retained split files under `data/splits/main_benchmark/`
- the retained `clean_default` variant manifest used by Phase 7
- the retained feature tables under `data/features/main_benchmark/`
- the retained graph artifact directory used by graph-based models
- the minimal Phase 2 and Phase 3 manifests needed to resolve those assets

## Not included

- generated Phase 7 checkpoints
- generated Phase 7 reports
- unrelated historical artifacts, paper files, and local workspace material

## Public run path

Use `configs/phase7_balanced.yaml` with:

```bash
python -m src.training.run_experiments --config configs/phase7_balanced.yaml
```
