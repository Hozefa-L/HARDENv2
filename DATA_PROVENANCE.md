# Data Provenance and Licensing

CGT-DAppSCAN is a *derived* benchmark. This document states exactly what we
redistribute, under which license, and how to obtain the parts we deliberately
do not re-host. It mirrors the "Data Availability, Provenance, and Licensing"
section of the paper.

## Two-layer release

**Layer 1: derived artifacts we author (license: CC BY 4.0, see `LICENSE-DATA`).**
These are products of our pipeline and are ours to release:

- heterogeneous CFG/DFG graph tensors,
- opcode and TF-IDF / expert-pattern feature matrices,
- ten-SWC label matrices and per-(contract, label) assessment masks,
- fixed train/validation/test split files and manifests,
- benchmark summary/metadata.

These are **committed directly in this repository** (under `data/curated/`,
`data/splits/`, and `data/features/`), so the benchmark is self-contained and
needs no external download. They can also be reproduced bit-for-bit from the
upstream sources via the reconstruction steps in the top-level `README.md`.

None of the committed files carries raw contract bytecode. The split files
hold fingerprints, source metadata, labels, and assessment masks; the opcode
corpus holds mnemonic sequences without operand values; the graph tensors hold
the lifted CFG/DFG structure. `python scripts/verify_data.py` checks this
along with the benchmark's structural properties.

**Layer 2: raw contract bytecode (NOT re-hosted).**
Rather than redistribute upstream contract files, we publish *contract
identifiers* plus a *reconstruction script* so the corpus can be rebuilt
locally and bit-for-bit reproducibly. The manifest covers **all 2,186
benchmark contracts** across both sources (1,814 CGT-only, 363 DAppSCAN-only,
9 shared), not just the CGT layer:

- `benchmark/contract_identifiers.csv` provides one row per benchmark contract, keyed
  by the canonical runtime fingerprint `fp_runtime_unified`, with `source`
  (`cgt`/`dappscan`/`both`), the CGT mirror `runtime_hash`, on-chain
  `chain`/`address`, the `dappscan_contract_id` (`<dapp>/<contract>`), and the
  assessed/positive SWC tags.
- `scripts/build_identifier_manifest.py` regenerates that manifest as a
  faithful export of the curated benchmark, from the fixed split artifacts under
  `data/splits/main_benchmark/` (joining CGT addresses from `consolidated.csv`).
- `scripts/reconstruct_bytecode.py` resolves runtime bytecode for every
  identifier into `data/reconstructed/`: CGT contracts from a local CGT mirror
  by hash (or an Ethereum JSON-RPC endpoint by address), and DAppSCAN contracts
  from the user's own DAppSCAN clone (`--dappscan-root`). Each reconstructed
  contract is verified by re-computing `fp_runtime_unified`, so a full run
  reports `2186/2186 verified`.

## Upstream sources and their terms

| Source | Upstream license | What we do |
| --- | --- | --- |
| Consolidated Ground Truth (CGT), `gsalzer/CGT` | Code: **MIT**. On-chain runtime bytecode: *no license imposed* by the CGT repo (public-blockchain data). | We re-derive artifacts from it and list its contracts by identifier. |
| DAppSCAN, `InPlusLab/DAppSCAN` | **No license** declared (GitHub reports `license: null`; no LICENSE file). | We do **not** re-host any DAppSCAN file. DAppSCAN-derived contracts are referenced by identifier only (`<dapp>/<contract>`); users reconstruct their runtime bytecode from their own DAppSCAN clone via `--dappscan-root`. |

Anyone who reconstructs the corpus remains bound by the original per-contract
source licenses, which are surfaced in the identifier manifest where known.

## SPDX summary

- Source code (`src/`, `scripts/`): `MIT` (see `LICENSE`).
- Derived data artifacts: `CC-BY-4.0` (see `LICENSE-DATA`).
- Upstream contract bytecode: original terms retained; not relicensed here.
