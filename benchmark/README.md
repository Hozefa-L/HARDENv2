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

## Label Conventions

Two encodings of the same labels appear in the published files, and they are
consistent with each other.

- The split parquets use one nullable integer column per SWC (`swc_101` ...
  `swc_135`), where 1 is a known positive, 0 a known negative, and NA an
  unassessed contract-label pair.
- The feature index (`data/features/main_benchmark/phase3_feature_index_10swc.parquet`)
  uses plain integers with -1 for unassessed, plus boolean `*_assessed` mask
  columns.

SWC-132 was dropped from the evaluated set because the corpus contains only 14
positives (a single one in test). Traces of it remain for transparency rather
than for use. The split parquets keep a `swc_132` column, the graph tensors
carry 11-slot `y`/`y_mask` vectors (SWC-132 at slot 9), and the packed
`label_vector`/`label_mask` strings and `label_positive_count`/
`label_assessed_count` columns in the feature index are computed over all 11
slots. The training and evaluation code reads only the ten per-SWC columns, so
none of this affects reported results, but a consumer parsing the packed
vectors should expect 11 entries. A related consequence is that 38 contracts
are assessed only for SWC-132; they remain in the benchmark counts but
contribute no assessed label among the evaluated ten.

The split files carry contract fingerprints, source metadata, labels, and
masks. They contain no raw bytecode; use the reconstruction path below when
the bytecode itself is needed.

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
