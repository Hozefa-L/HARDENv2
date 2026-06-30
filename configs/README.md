# Configuration Files

This directory contains the experiment configurations used across the
HARDENv2 pipeline.

## Phase Map

| Phase | Files | Purpose |
| --- | --- | --- |
| Phase 2 | `phase2.yaml` | Benchmark assembly, opcode preprocessing, CFG/DFG recovery, and graph construction |
| Phase 3 | `phase3.yaml` | Feature extraction over the retained Phase 2 graph outputs |
| Phase 4 | `phase4.yaml`, `phase4_full.yaml` | OpcodeGT model specification and training configuration |
| Phase 5 | `phase5.yaml`, `phase5_full.yaml` | Baseline selection and baseline training configuration |
| Phase 6 | `phase6.yaml`, `phase6_full.yaml` | Broad multi-model experiment campaigns |
| Phase 6 levels | `phase6_level0.yaml`, `phase6_level0.5.yaml`, `phase6_level2.yaml`, `phase6_level3.yaml`, `phase6_level4.yaml` | Successive evaluation tiers |
| Phase 6 support | `phase6_tune_0.yaml`, `phase6_validation.yaml` | Tuning and validation support runs |
| Phase 7 | `phase7_balanced.yaml`, `phase7_ablations.yaml`, `phase7_data_efficiency.yaml`, `phase7_graph_confirmatory.yaml` | Final experiment families and confirmatory runs |

Config names preserve the original progression from benchmark construction
through final evaluation.