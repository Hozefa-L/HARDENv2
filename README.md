# HARDENv2 / CGT-DAppSCAN

This is the public research artifact for the paper
**“A Reproducible Bytecode Benchmark and Empirical Study of Smart-Contract
Vulnerability Detectors.”**

It contains two things:

1. **CGT-DAppSCAN** — a reproducible, bytecode-level, multi-label benchmark of
   **2,186 Ethereum contracts** across ten SWC weakness classes, shipped here as
   ready-to-use graph tensors, feature tables, labels/masks, and fixed
   train/validation/test splits.
2. **HARDENv2** — the heterogeneous-graph model family (HMPGT + enriched opcode
   features) and the full evaluation pipeline used to compare it against tree
   ensembles, neural sequence/MLP baselines, flat GNNs, and Mythril.

- **Repository:** https://github.com/Hozefa-L/HARDENv2
- **Project page:** https://hozefa-l.github.io/HARDENv2/

The benchmark data lives **directly in this repository** — there is nothing
external to download. You can clone it and reproduce the paper's headline
numbers without rebuilding anything from upstream sources.

---

## Quick start

```bash
git clone https://github.com/Hozefa-L/HARDENv2
cd HARDENv2
pip install -r requirements.txt

# Reproduce the headline evaluation (uses the committed benchmark data).
python -m src.training.run_experiments --config configs/phase7_balanced.yaml

# Summarize the results into the paper's tables.
python scripts/level4_analysis.py \
    --manifest reports/phase7_balanced/phase7_run_manifest.json
```

That run reads the 2,186 graph tensors and splits already in `data/`, trains the
learned configurations over five seeds, and writes results to
`reports/phase7_balanced/`. No CGT/DAppSCAN checkout, RPC node, or EtherSolve
install is needed for this path.

---

## What is in this repository

| Path | Contents |
| --- | --- |
| `data/curated/graphs/` | **The benchmark dataset** — 2,186 serialized CFG/DFG graph tensors |
| `data/splits/main_benchmark/` | Fixed train / validation / test split files and manifest |
| `data/features/main_benchmark/` | Opcode, TF-IDF, and expert-pattern feature tables |
| `data/synthetic/` | Ten minimal synthetic e-government contracts (EG-1 … EG-10) |
| `benchmark/` | Benchmark summary + the Layer-2 contract-identifier manifest |
| `configs/` | Experiment configs (`phase2.yaml` … `phase7_balanced.yaml`) |
| `src/` | Curation, graph lifting, features, models, training, evaluation |
| `scripts/` | Reconstruction, manifest, and analysis entrypoints |
| `reports/` | Per-phase evaluation summaries, tables, and figures |
| `checkpoints/phase7_balanced/metrics/` | Per-seed metric files for the headline run |
| `docs/ARTIFACT.md` | Artifact-evaluation guide: claim → how-to-check |

The benchmark data (graphs, splits, features) is **committed here directly**, so
the repository is self-contained. Upstream contract bytecode is the **only**
thing not included — see [Provenance and rebuilding from source](#provenance-and-rebuilding-from-source).

---

## The benchmark at a glance

| Property | Value |
| --- | --- |
| Contracts | 2,186 |
| Evaluated SWCs | 101, 103, 104, 107, 113, 114, 115, 120, 128, 135 |
| Splits | train 1,749 / val 219 / test 218 (80 / 10 / 10, multi-label stratified) |
| Source mix | 1,814 CGT-only · 363 DAppSCAN-only · 9 shared |
| Input modality | runtime bytecode only (a deliberate scope choice — see the paper) |

Machine-readable summary: `benchmark/clean_default_summary.json`.

CGT-DAppSCAN is a *derived* benchmark: it merges the Consolidated Ground Truth
and DAppSCAN corpora into one masked, multi-label, graph-ready dataset under a
single fixed protocol. It is a re-purposing, not a re-release — see
`DATA_PROVENANCE.md`.

---

## Reproducing the paper's results

After `run_experiments`, the outputs map to the paper as follows:

| Paper claim | Where to look |
| --- | --- |
| Headline Macro-F1 (XGBoost 0.805, RF 0.802, HARDENv2-Graph 0.641) | `reports/phase7_balanced/results_summary.json` |
| Per-SWC F1 matrix | `reports/phase7_balanced/per_swc_metrics.parquet` |
| Held-out paired tests (permutation p = 0.0002) | `reports/phase7_balanced/statistical_tests.json` |
| Publication tables | `reports/phase7_balanced/publication_tables.md` |
| Mythril v0.24.8 external baseline (0.396) | `reports/mythril_v0_24_8/` |

Every learned result is averaged over seeds `{42, 123, 456, 789, 2024}`; per-class
thresholds are tuned on validation only and frozen for test. The tuning budget,
search space, and threshold protocol are documented in the paper and implemented
in `src/training/` (`run_experiments.py`, `tune_opcodegt.py`,
`threshold_tuning.py`).

A fuller claim-by-claim checklist for artifact evaluation is in
[`docs/ARTIFACT.md`](docs/ARTIFACT.md).

---

## Provenance and rebuilding from source

The benchmark **data** committed here is fully ours to release (see licensing
below). The **upstream contract bytecode** it was derived from is not re-hosted:
DAppSCAN ships no license, so instead of redistributing those files we publish
contract *identifiers* plus a *reconstruction script*. This lets anyone rebuild
the corpus bit-for-bit from the original sources and verify it against the
committed artifacts.

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

To re-lift the CFG/DFG graphs from reconstructed bytecode, install EtherSolve:

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
| Upstream contract bytecode | original terms — **not relicensed, not re-hosted** | `DATA_PROVENANCE.md` |

---

## Citation

```bibtex
@software{lakadawala2026hardenv2,
  author = {Lakadawala, Hozefa and Dzigbede, Komla and Chen, Yu},
  title  = {HARDENv2 / CGT-DAppSCAN: A Reproducible Bytecode Benchmark and
            Empirical Study of Smart-Contract Vulnerability Detectors},
  year   = {2026},
  url    = {https://github.com/Hozefa-L/HARDENv2}
}
```

Machine-readable citation metadata is in `CITATION.cff`.
