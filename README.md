# HARDEN-V2

HARDEN-V2 is a research codebase for smart contract vulnerability detection from EVM runtime bytecode using enriched opcode features and graph-based learning over control-flow and data-flow structure.

**Website:** https://hozefa-l.github.io/HARDENv2/

## Overview

This public repository keeps the active curation, preprocessing, feature extraction, model, training, and evaluation path used by HARDEN-V2. It also includes the retained benchmark metadata layer, the active experiment configurations, and the aggregate analysis entrypoint.

## Project Structure

| Path | Purpose |
| --- | --- |
| `src/curation` | Benchmark curation and label harmonization |
| `src/preprocessing` | Runtime bytecode, CFG, DFG, and heterogeneous graph construction |
| `src/features` | Opcode, TF-IDF, pattern, and graph feature extraction |
| `src/models` | OpcodeGT model components |
| `src/baselines` | Retained classical and neural baselines |
| `src/training` | Experiment orchestration and training loops |
| `src/evaluation` | Metrics and aggregate evaluation logic |
| `configs/` | Active experiment configurations |
| `benchmark/` | Lightweight benchmark metadata |
| `scripts/level4_analysis.py` | Retained analysis entrypoint |

## Getting Started

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install EtherSolve separately for preprocessing workflows that require CFG recovery:

```bash
wget https://github.com/SeUniVr/EtherSolve/releases/download/v1.0/EtherSolve.jar
mkdir -p tools/EtherSolve
mv EtherSolve.jar tools/EtherSolve/
```

## Running the Pipeline

Run the active experiment configuration:

```bash
python -m src.training.run_experiments --config configs/phase7_balanced.yaml
```

Run the retained analysis script after collecting run outputs:

```bash
python scripts/level4_analysis.py --manifest reports/phase7_balanced/phase7_run_manifest.json
```

## Public Website

For a visual walkthrough of the public repository, open the live GitHub Pages site:

https://hozefa-l.github.io/HARDENv2/

## Public Scope

The public repository includes the active code path in `src/`, the retained benchmark metadata in `benchmark/`, the active configuration chain in `configs/`, `requirements.txt`, `index.html`, and `scripts/level4_analysis.py`.
