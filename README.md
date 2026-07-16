# HARDENv2 / CGT-DAppSCAN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21316588.svg)](https://doi.org/10.5281/zenodo.21316588)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-green.svg)](LICENSE-DATA)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](requirements.txt)

This repository accompanies the paper **"A Reproducible Bytecode Benchmark and
Empirical Study of Smart-Contract Vulnerability Detectors."** It contains two
things.

1. **CGT-DAppSCAN**, a bytecode-level, multi-label benchmark of **2,186
   Ethereum contracts** across ten SWC weakness classes, shipped as ready-to-use
   graph tensors, feature tables, labels with assessment masks, and fixed
   train/validation/test splits.
2. **HARDENv2**, the heterogeneous-graph model family (HMPGT with enriched
   opcode features) and the evaluation pipeline used to compare it against tree
   ensembles, neural sequence and MLP baselines, flat GNNs, and Mythril.

The benchmark data lives directly in this repository, so a clone is all you
need to reproduce the paper's headline numbers. Project page:
https://hozefa-l.github.io/HARDENv2/

---

## Quick start

```bash
git clone https://github.com/Hozefa-L/HARDENv2
cd HARDENv2
pip install -r requirements.txt

# Confirm the benchmark data is intact (about 30 seconds, CPU only).
python scripts/verify_data.py
```

The checker prints one PASS line per property (2,186 graphs, split sizes,
split disjointness, source breakdown, feature-table alignment) and exits
nonzero if anything is off. The same checks run under pytest via
`pytest tests/`.

For a first end-to-end run, train a single configuration on one seed. This
takes a few minutes and works on CPU:

```bash
python -m src.training.run_experiments \
    --config configs/phase7_balanced.yaml \
    --model-variant classical_xgboost --seed 42
```

The full evaluation behind the paper trains 13 configurations over five seeds
(65 runs) and then aggregates them into the paper's tables. The sweep config
also defines a `codebert_classifier` variant that is not part of the paper's
tables: it requires frozen-encoder embeddings
(`data/features/main_benchmark/codebert_features.parquet`, generated with
`src/features/codebert_features.py`), and without that file the run is marked
unavailable and the sweep continues. Earlier releases wired this variant to the
opcode features by mistake, so its historical metrics did not measure CodeBERT
and have been removed:

```bash
python -m src.training.run_experiments --config configs/phase7_balanced.yaml
python scripts/level4_analysis.py \
    --manifest reports/phase7_balanced/phase7_run_manifest.json
```

Results land in a local `reports/phase7_balanced/` directory created by the
run. The runner checkpoints its progress, so an interrupted sweep resumes
where it stopped.

### Environment and runtime

The experiments were run on Python 3.12 with the package versions pinned in
`requirements.txt` (PyTorch 2.12.0 built against CUDA 13.0, torch-geometric
2.7.0). The classical baselines (Random Forest, XGBoost, LightGBM, logistic
regression) train on CPU in minutes. The neural configurations need a CUDA
GPU; peak training memory is about 8 GB, so a single 12 GB card is enough.
Each neural run is capped at one hour, which puts the full 70-run sweep at
roughly a day on one GPU.

---

## What is in this repository

| Path | Contents |
| --- | --- |
| `data/curated/graphs/` | The benchmark dataset itself, 2,186 serialized CFG/DFG graph tensors |
| `data/splits/main_benchmark/` | Fixed train / validation / test split files and manifest |
| `data/features/main_benchmark/` | Opcode, TF-IDF, and expert-pattern feature tables |
| `data/synthetic/` | Ten minimal synthetic e-government contracts (EG-1 to EG-10) |
| `benchmark/` | Benchmark summary and the per-contract identifier manifest |
| `configs/` | Experiment configs (`phase2.yaml` through `phase7_balanced.yaml`) |
| `src/` | Curation, graph lifting, features, models, training, evaluation |
| `scripts/` | Data checks, reconstruction, manifest, and analysis entrypoints |
| `tests/` | Pytest wrapper around the data integrity checks |
| `checkpoints/phase7_balanced/metrics/` | Per-seed metric files for the headline run |

The split files carry contract fingerprints, source metadata, labels, and
assessment masks. Raw contract bytecode is deliberately absent from every
committed file; the section on provenance below explains how to rebuild it
from the upstream sources when you need it.

---

## The benchmark at a glance

| Property | Value |
| --- | --- |
| Contracts | 2,186 |
| Evaluated SWCs | 101, 103, 104, 107, 113, 114, 115, 120, 128, 135 |
| Splits | train 1,749 / val 219 / test 218 (80 / 10 / 10, multi-label stratified) |
| Source mix | 1,814 CGT-only, 363 DAppSCAN-only, 9 shared |
| Input modality | runtime bytecode only (a deliberate scope choice, see the paper) |

A machine-readable summary sits in `benchmark/clean_default_summary.json`.

CGT-DAppSCAN is a derived benchmark. It merges the Consolidated Ground Truth
and DAppSCAN corpora into one masked, multi-label, graph-ready dataset under a
single fixed protocol, and `DATA_PROVENANCE.md` records exactly what is
redistributed and under which terms.

---

## Reproducing the study's numbers

The full sweep writes its outputs to the local `reports/phase7_balanced/`
directory, with the headline metrics in `results_summary.json`, the per-SWC F1
matrix in `per_swc_metrics.parquet`, and the held-out paired tests in
`statistical_tests.json`. The numbers to expect match the paper. XGBoost
(0.805) and Random Forest (0.802) lead on Macro-F1, HARDENv2-Graph reaches
0.641, the held-out paired permutation test for the graph-family comparison
gives p = 0.0002, and Mythril v0.24.8 scores 0.396 on the same split.

Every learned result is averaged over seeds `{42, 123, 456, 789, 2024}`.
Per-class thresholds are tuned on validation only and frozen for test; the
tuning budget, search space, and threshold protocol are implemented in
`src/training/` (`run_experiments.py`, `tune_opcodegt.py`,
`threshold_tuning.py`).

---

## Provenance and rebuilding from source

The data committed here consists of artifacts this project authored (see the
licensing table below). Upstream contract bytecode is referenced by identifier
only. DAppSCAN ships no license, so instead of redistributing those files the
repository publishes a per-contract identifier manifest plus a reconstruction
script, which lets anyone rebuild the corpus bit-for-bit from the original
sources and verify it against the committed artifacts.

```bash
# Clone the upstream sources (not redistributed here).
git clone https://github.com/gsalzer/CGT        data/raw/cgt-main
git clone https://github.com/InPlusLab/DAppSCAN  data/raw/dappscan

# Reconstruct runtime bytecode for all 2,186 contracts and verify each one
# against its canonical fingerprint (a full run reports "2186/2186 verified").
python scripts/reconstruct_bytecode.py \
    --manifest benchmark/contract_identifiers.csv \
    --cgt-root data/raw/cgt-main \
    --dappscan-root data/raw/dappscan \
    --out data/reconstructed
```

Re-lifting the CFG/DFG graphs from reconstructed bytecode additionally needs
EtherSolve:

```bash
wget https://github.com/SeUniVr/EtherSolve/releases/download/v1.0/EtherSolve.jar
mkdir -p tools/EtherSolve && mv EtherSolve.jar tools/EtherSolve/
```

Full provenance, the two-layer release model, and per-source licensing are in
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

---

## Licensing at a glance

| Component | License | File |
| --- | --- | --- |
| Source code (`src/`, `scripts/`) | MIT | `LICENSE` |
| Derived data artifacts (graphs, features, labels/masks, splits) | CC BY 4.0 | `LICENSE-DATA` |
| Upstream contract bytecode | original terms, neither relicensed nor re-hosted | `DATA_PROVENANCE.md` |

---

## Citation

```bibtex
@software{lakadawala2026hardenv2,
  author    = {Lakadawala, Hozefa and Dzigbede, Komla and Chen, Yu},
  title     = {HARDENv2 / CGT-DAppSCAN: A Reproducible Bytecode Benchmark and
               Empirical Study of Smart-Contract Vulnerability Detectors},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.2},
  doi       = {10.5281/zenodo.21316588},
  url       = {https://github.com/Hozefa-L/HARDENv2}
}
```

Machine-readable citation metadata is in `CITATION.cff`. Tagged releases are
archived on Zenodo under the concept DOI
[10.5281/zenodo.21316588](https://doi.org/10.5281/zenodo.21316588), which
always resolves to the latest version.
