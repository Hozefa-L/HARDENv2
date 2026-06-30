# CGT-DAppSCAN Benchmark Metadata

This directory summarizes the benchmark scope used by HARDENv2.

## Benchmark Snapshot

- Dataset family: CGT-DAppSCAN runtime-bytecode benchmark
- Active variant: `clean_default`
- Contracts: 2,186
- Evaluated SWCs: 101, 103, 104, 107, 113, 114, 115, 120, 128, 135
- Split counts: train 1,749 / val 219 / test 218
- Source breakdown: 1,814 CGT-only / 363 DAppSCAN-only / 9 shared

## Published Dataset Artifacts

| Path | Contents |
| --- | --- |
| `data/curated/graphs/phase2_task5_20260309T161122Z/` | 2,186 serialized graph tensors for the retained benchmark |
| `data/splits/main_benchmark/train.parquet` | Training split |
| `data/splits/main_benchmark/val.parquet` | Validation split |
| `data/splits/main_benchmark/test.parquet` | Test split |
| `data/splits/main_benchmark/manifests/phase2_task5_20260316T081319Z/clean_default.json` | Split manifest for the retained benchmark scope |

## Bytecode Reconstruction (Layer 2)

`contract_identifiers.csv` lists all 2,186 contracts (1,814 CGT-only, 363
DAppSCAN-only, 9 shared) by identifier so their runtime bytecode can be
reconstructed without re-hosting upstream files. Regenerate the manifest with
`python scripts/build_identifier_manifest.py`; reconstruct bytecode for both
sources with `python scripts/reconstruct_bytecode.py` (`--cgt-root` and
`--dappscan-root`), which verifies every contract against its canonical
fingerprint (`2186/2186 verified`). See `../DATA_PROVENANCE.md` for provenance
and licensing.

## Files

| File | Purpose |
| --- | --- |
| `clean_default_summary.json` | Machine-readable benchmark summary |
| `contract_identifiers.csv` | Layer-2 per-contract identifiers for all 2,186 contracts (CGT + DAppSCAN) |
| `README.md` | Benchmark description |

## Phase Context

| Phase | Role |
| --- | --- |
| Phase 1 | Benchmark curation and label harmonization |
| Phase 2 | CFG/DFG recovery and graph construction |
| Phase 3 | Feature extraction over retained graph outputs |
| Phase 7 | Balanced training and evaluation on the retained benchmark |
