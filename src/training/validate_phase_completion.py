from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASES = (3, 4, 5, 6)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ValidationIssue:
    phase: int
    severity: str
    code: str
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": int(self.phase),
            "severity": str(self.severity),
            "code": str(self.code),
            "message": str(self.message),
        }


def _resolve_path(path_value: Any) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _safe_read_mapping(path: Path, context: str) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context}: {path}")
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a mapping: {path}")
    return payload


def _check_paths_exist(*paths: Path, phase: int, issues: List[ValidationIssue]) -> None:
    for path in paths:
        if not path.exists():
            issues.append(
                ValidationIssue(
                    phase=phase,
                    severity="error",
                    code="missing_path",
                    message=f"Missing required path: {_rel(path)}",
                )
            )


def _validate_phase3(issues: List[ValidationIssue]) -> Dict[str, Any]:
    phase = 3
    manifest_path = _resolve_path("reports/phase3/phase3_run_manifest.json")
    report_path = _resolve_path("reports/phase3/feature_extraction_report.json")
    card_path = _resolve_path("reports/phase3/feature_dataset_card.md")
    feature_index_path = _resolve_path("data/features/main_benchmark/phase3_feature_index.parquet")
    opcode_corpus_path = _resolve_path("data/features/main_benchmark/opcode_text_corpus.parquet")
    codebert_path = _resolve_path("data/features/main_benchmark/codebert_features.parquet")
    graph_features_path = _resolve_path("data/features/main_benchmark/graph_level_features.parquet")

    _check_paths_exist(
        manifest_path,
        report_path,
        card_path,
        feature_index_path,
        opcode_corpus_path,
        codebert_path,
        graph_features_path,
        phase=phase,
        issues=issues,
    )
    if any(path for path in [manifest_path, report_path] if not path.exists()):
        return {"phase": phase, "status": "failed"}

    manifest = _safe_read_mapping(manifest_path, "Phase 3 run manifest")
    report = _safe_read_mapping(report_path, "Phase 3 feature report")

    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="invalid_manifest_outputs",
                message="Phase 3 run manifest has non-mapping `outputs`.",
            )
        )
        outputs = {}
    for key in [
        "feature_index_parquet",
        "opcode_text_corpus_parquet",
        "codebert_features_parquet",
        "graph_level_features_parquet",
        "report_json",
        "dataset_card_md",
    ]:
        value = outputs.get(key)
        if not value:
            issues.append(
                ValidationIssue(
                    phase=phase,
                    severity="error",
                    code="missing_manifest_output_key",
                    message=f"Phase 3 run manifest missing outputs.{key}.",
                )
            )
            continue
        out_path = _resolve_path(value)
        if not out_path.exists():
            issues.append(
                ValidationIssue(
                    phase=phase,
                    severity="error",
                    code="missing_manifest_output_path",
                    message=f"Phase 3 output from manifest not found: {_rel(out_path)}",
                )
            )

    contracts_total = int(report.get("contracts_total", 0))
    if contracts_total <= 0:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="invalid_contract_count",
                message="Phase 3 report contracts_total must be > 0.",
            )
        )

    split_counts = report.get("split_counts", {})
    if not isinstance(split_counts, dict):
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="invalid_split_counts",
                message="Phase 3 report split_counts must be a mapping.",
            )
        )
    else:
        for split in SPLITS:
            if int(split_counts.get(split, 0)) <= 0:
                issues.append(
                    ValidationIssue(
                        phase=phase,
                        severity="error",
                        code="empty_split",
                        message=f"Phase 3 split `{split}` has zero rows.",
                    )
                )

    feature_rows = report.get("feature_row_counts", {})
    if not isinstance(feature_rows, dict):
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="invalid_feature_row_counts",
                message="Phase 3 report feature_row_counts must be a mapping.",
            )
        )
    else:
        for key, value in feature_rows.items():
            if int(value) <= 0:
                issues.append(
                    ValidationIssue(
                        phase=phase,
                        severity="error",
                        code="empty_feature_artifact",
                        message=f"Phase 3 feature artifact `{key}` has zero rows.",
                    )
                )

    return {
        "phase": phase,
        "contracts_total": contracts_total,
        "split_counts": split_counts if isinstance(split_counts, dict) else {},
        "status": "passed",
    }


def _validate_phase4(issues: List[ValidationIssue]) -> Dict[str, Any]:
    phase = 4
    report_path = _resolve_path("reports/phase4/phase4_smoke_train_report.json")
    manifest_path = _resolve_path("reports/phase4/phase4_run_manifest.json")
    model_spec_path = _resolve_path("reports/phase4/model_spec.md")
    checkpoint_dir = _resolve_path("checkpoints/phase4_smoke")

    _check_paths_exist(report_path, manifest_path, model_spec_path, checkpoint_dir, phase=phase, issues=issues)
    if any(path for path in [report_path, manifest_path] if not path.exists()):
        return {"phase": phase, "status": "failed"}

    report = _safe_read_mapping(report_path, "Phase 4 report")
    if not bool(report.get("checkpoint_roundtrip_match", False)):
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="checkpoint_roundtrip_failed",
                message="Phase 4 checkpoint roundtrip did not pass.",
            )
        )

    step_metrics = report.get("smoke_step_metrics", [])
    if not isinstance(step_metrics, list) or len(step_metrics) == 0:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="missing_step_metrics",
                message="Phase 4 report has no training step metrics.",
            )
        )

    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if len(checkpoint_files) == 0:
            issues.append(
                ValidationIssue(
                    phase=phase,
                    severity="error",
                    code="missing_checkpoint_files",
                    message="Phase 4 checkpoint directory exists but contains no .pt files.",
                )
            )

    return {"phase": phase, "status": "passed", "report_task": str(report.get("task", ""))}


def _validate_phase5(issues: List[ValidationIssue]) -> Dict[str, Any]:
    phase = 5
    report_path = _resolve_path("reports/phase5/phase5_smoke_report.json")
    manifest_path = _resolve_path("reports/phase5/phase5_run_manifest.json")
    catalog_path = _resolve_path("reports/phase5/baseline_catalog.md")
    checkpoint_dir = _resolve_path("checkpoints/phase5_smoke")

    _check_paths_exist(report_path, manifest_path, catalog_path, checkpoint_dir, phase=phase, issues=issues)
    if not report_path.exists():
        return {"phase": phase, "status": "failed"}

    report = _safe_read_mapping(report_path, "Phase 5 report")
    baseline_results = report.get("baseline_results", [])
    if not isinstance(baseline_results, list) or len(baseline_results) == 0:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="missing_baseline_results",
                message="Phase 5 report has no baseline_results rows.",
            )
        )
    elif len(baseline_results) < 4:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="warning",
                code="fewer_than_expected_baselines",
                message=f"Phase 5 has {len(baseline_results)} baselines; expected at least 4 families.",
            )
        )

    if not bool(report.get("all_checkpoint_roundtrip_match", False)):
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="baseline_checkpoint_roundtrip_failed",
                message="Phase 5 reports at least one failed checkpoint roundtrip.",
            )
        )

    return {"phase": phase, "status": "passed", "baseline_count": int(len(baseline_results))}


def _validate_phase6(issues: List[ValidationIssue]) -> Dict[str, Any]:
    phase = 6
    manifest_path = _resolve_path("reports/phase6/phase6_run_manifest.json")
    results_summary_path = _resolve_path("reports/phase6/results_summary.json")
    ablation_path = _resolve_path("reports/phase6/ablation_summary.json")
    per_swc_path = _resolve_path("reports/phase6/per_swc_metrics.parquet")
    tables_path = _resolve_path("reports/phase6/final_tables.md")
    models_dir = _resolve_path("checkpoints/phase6/models")
    metrics_dir = _resolve_path("checkpoints/phase6/metrics")

    _check_paths_exist(
        manifest_path,
        results_summary_path,
        ablation_path,
        per_swc_path,
        tables_path,
        models_dir,
        metrics_dir,
        phase=phase,
        issues=issues,
    )
    if not manifest_path.exists():
        return {"phase": phase, "status": "failed"}

    manifest = _safe_read_mapping(manifest_path, "Phase 6 run manifest")
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="invalid_runs",
                message="Phase 6 run manifest `runs` is not a list.",
            )
        )
        runs = []

    expected = int(manifest.get("matrix", {}).get("expected_run_count", len(runs))) if isinstance(manifest.get("matrix"), dict) else int(len(runs))
    completed = 0
    failed = 0
    unavailable = 0
    missing_metrics_refs = 0
    missing_checkpoints = 0
    for run in runs:
        if not isinstance(run, dict):
            continue
        status = str(run.get("status", "unknown"))
        if status == "completed":
            completed += 1
            metrics_rel = run.get("metrics_json")
            if not metrics_rel:
                missing_metrics_refs += 1
            else:
                metrics_path = _resolve_path(metrics_rel)
                if not metrics_path.exists():
                    missing_metrics_refs += 1
            checkpoint_rel = run.get("checkpoint_path")
            if not checkpoint_rel:
                missing_checkpoints += 1
            else:
                checkpoint_path = _resolve_path(checkpoint_rel)
                if not checkpoint_path.exists():
                    missing_checkpoints += 1
        elif status == "failed":
            failed += 1
        elif status == "unavailable":
            unavailable += 1

    if expected <= 0:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="invalid_expected_runs",
                message="Phase 6 expected_run_count must be > 0.",
            )
        )
    if completed != expected:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="incomplete_run_matrix",
                message=f"Phase 6 run matrix incomplete: completed={completed}, expected={expected}.",
            )
        )
    if failed > 0:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="failed_runs_present",
                message=f"Phase 6 has {failed} failed runs.",
            )
        )
    if unavailable > 0:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="unavailable_runs_present",
                message=f"Phase 6 has {unavailable} unavailable runs.",
            )
        )
    if missing_metrics_refs > 0:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="missing_metrics_files",
                message=f"Phase 6 has {missing_metrics_refs} completed runs without valid metrics files.",
            )
        )
    if missing_checkpoints > 0:
        issues.append(
            ValidationIssue(
                phase=phase,
                severity="error",
                code="missing_checkpoint_files",
                message=f"Phase 6 has {missing_checkpoints} completed runs without valid checkpoints.",
            )
        )

    if per_swc_path.exists():
        per_swc = pd.read_parquet(per_swc_path)
        if len(per_swc) == 0:
            issues.append(
                ValidationIssue(
                    phase=phase,
                    severity="error",
                    code="empty_per_swc_metrics",
                    message="Phase 6 per_swc_metrics.parquet is empty.",
                )
            )

    return {
        "phase": phase,
        "status": "passed",
        "expected_runs": expected,
        "completed_runs": completed,
    }


def validate_phases(phases: Sequence[int]) -> Dict[str, Any]:
    issues: List[ValidationIssue] = []
    checks: List[Dict[str, Any]] = []
    for phase in phases:
        if int(phase) == 3:
            checks.append(_validate_phase3(issues))
        elif int(phase) == 4:
            checks.append(_validate_phase4(issues))
        elif int(phase) == 5:
            checks.append(_validate_phase5(issues))
        elif int(phase) == 6:
            checks.append(_validate_phase6(issues))
        else:
            issues.append(
                ValidationIssue(
                    phase=int(phase),
                    severity="error",
                    code="unsupported_phase",
                    message=f"Unsupported phase: {phase}. Supported values are 3, 4, 5, 6.",
                )
            )

    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phases_checked": [int(v) for v in phases],
        "status": "passed" if len(errors) == 0 else "failed",
        "error_count": int(len(errors)),
        "warning_count": int(len(warnings)),
        "checks": checks,
        "issues": [issue.as_dict() for issue in issues],
    }


def _parse_phases(values: Sequence[str]) -> List[int]:
    if not values:
        return list(DEFAULT_PHASES)
    phases: List[int] = []
    for raw in values:
        for token in str(raw).split(","):
            text = token.strip()
            if not text:
                continue
            phases.append(int(text))
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate completion contracts for Phase 3 to Phase 6 artifacts.")
    parser.add_argument(
        "--phase",
        action="append",
        dest="phases",
        help="Phase(s) to validate. Repeatable or comma-separated. Supported: 3,4,5,6",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional output path for writing validation report JSON.",
    )
    args = parser.parse_args()

    phases = _parse_phases(args.phases or [])
    report = validate_phases(phases)

    if args.report_json is not None:
        report_path = _resolve_path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report_json: {_rel(report_path)}")

    print(f"status: {report['status']}")
    print(f"phases_checked: {report['phases_checked']}")
    print(f"errors: {report['error_count']}, warnings: {report['warning_count']}")
    if report["issues"]:
        print("issues:")
        for issue in report["issues"]:
            print(f"- phase={issue['phase']} [{issue['severity']}] {issue['code']}: {issue['message']}")

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
