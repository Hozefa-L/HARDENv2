from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE6_CONFIG_PATH = PROJECT_ROOT / "configs/phase6.yaml"


@dataclass(frozen=True)
class MonitorPaths:
    config_path: Path
    manifest_path: Path
    metrics_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class ProcessInfo:
    pid: str
    elapsed: str
    command: str


@dataclass(frozen=True)
class GpuProcessInfo:
    pid: str
    process_name: str
    used_gpu_memory: str


@dataclass(frozen=True)
class MonitorSnapshot:
    config_path: Path
    manifest_path: Path
    manifest_exists: bool
    updated_at_utc: Optional[str]
    updated_age_seconds: Optional[float]
    expected_run_count: int
    status_counts: Dict[str, int]
    running_run_ids: List[str]
    metrics_count: int
    models_count: int
    processes: List[ProcessInfo]
    matched_gpu_processes: List[GpuProcessInfo]
    gpu_processes: List[GpuProcessInfo]


def _resolve_path(path_value: Any) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _safe_read_mapping(path: Path, context: str) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context}: {path}")
    with path.open("r", encoding="utf-8") as fp:
        if path.suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(fp)
        else:
            payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a mapping: {path}")
    return payload


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized)


def _count_files(path: Path, glob_pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for candidate in path.glob(glob_pattern) if candidate.is_file())


def load_monitor_paths(config_path: Path) -> MonitorPaths:
    resolved_config_path = _resolve_path(config_path)
    raw = _safe_read_mapping(resolved_config_path, "Phase 6 config")
    outputs = raw.get("outputs", {})
    if outputs is None:
        outputs = {}
    if not isinstance(outputs, dict):
        raise ValueError("`outputs` in Phase 6 config must be a mapping when provided.")

    manifest_path = _resolve_path(outputs.get("run_manifest_json") or "reports/phase6/phase6_run_manifest.json")
    metrics_dir = _resolve_path(outputs.get("metrics_dir") or "checkpoints/phase6/metrics")
    models_dir = _resolve_path(outputs.get("models_dir") or "checkpoints/phase6/models")
    return MonitorPaths(
        config_path=resolved_config_path,
        manifest_path=manifest_path,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
    )


def _matching_processes(config_path: Path) -> List[ProcessInfo]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid,etime,args"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    target_tokens = {
        str(config_path),
        str(config_path.resolve()),
        config_path.name,
    }
    matches: List[ProcessInfo] = []
    for raw_line in completed.stdout.splitlines()[1:]:
        line = raw_line.strip()
        if "src.training.run_experiments" not in line:
            continue
        if not any(token in line for token in target_tokens):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        matches.append(ProcessInfo(pid=parts[0], elapsed=parts[1], command=parts[2]))
    return matches


def _gpu_processes() -> List[GpuProcessInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []

    rows: List[GpuProcessInfo] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        rows.append(GpuProcessInfo(pid=parts[0], process_name=parts[1], used_gpu_memory=parts[2]))
    return rows


def collect_monitor_snapshot(
    config_path: Path,
    *,
    include_processes: bool = True,
    include_gpu: bool = True,
) -> MonitorSnapshot:
    monitor_paths = load_monitor_paths(config_path)
    manifest_path = monitor_paths.manifest_path
    processes = _matching_processes(monitor_paths.config_path) if include_processes else []
    gpu_processes = _gpu_processes() if include_gpu else []
    matching_pids = {proc.pid for proc in processes}
    matched_gpu_processes = [proc for proc in gpu_processes if proc.pid in matching_pids]

    if not manifest_path.exists():
        return MonitorSnapshot(
            config_path=monitor_paths.config_path,
            manifest_path=manifest_path,
            manifest_exists=False,
            updated_at_utc=None,
            updated_age_seconds=None,
            expected_run_count=0,
            status_counts={},
            running_run_ids=[],
            metrics_count=_count_files(monitor_paths.metrics_dir, "*.json"),
            models_count=_count_files(monitor_paths.models_dir, "*"),
            processes=processes,
            matched_gpu_processes=matched_gpu_processes,
            gpu_processes=gpu_processes,
        )

    manifest = _safe_read_mapping(manifest_path, "Phase 6 run manifest")
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Phase 6 run manifest must contain a list under `runs`.")

    status_counts: Dict[str, int] = {}
    running_run_ids: List[str] = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "running":
            running_run_ids.append(str(entry.get("run_id", "unknown")))

    matrix = manifest.get("matrix", {})
    if matrix is None:
        matrix = {}
    if not isinstance(matrix, dict):
        raise ValueError("Phase 6 run manifest `matrix` must be a mapping when present.")

    updated_at_utc = manifest.get("updated_at_utc")
    updated_dt = _parse_iso_datetime(updated_at_utc)
    age_seconds: Optional[float] = None
    if updated_dt is not None:
        age_seconds = max((datetime.now(timezone.utc) - updated_dt.astimezone(timezone.utc)).total_seconds(), 0.0)

    return MonitorSnapshot(
        config_path=monitor_paths.config_path,
        manifest_path=manifest_path,
        manifest_exists=True,
        updated_at_utc=str(updated_at_utc) if updated_at_utc is not None else None,
        updated_age_seconds=age_seconds,
        expected_run_count=int(matrix.get("expected_run_count", len(runs))),
        status_counts=status_counts,
        running_run_ids=running_run_ids,
        metrics_count=_count_files(monitor_paths.metrics_dir, "*.json"),
        models_count=_count_files(monitor_paths.models_dir, "*"),
        processes=processes,
        matched_gpu_processes=matched_gpu_processes,
        gpu_processes=gpu_processes,
    )


def _activity_summary(snapshot: MonitorSnapshot) -> str:
    running_count = int(snapshot.status_counts.get("running", 0))
    if snapshot.processes:
        return "matching Phase 6 process detected"
    if not snapshot.manifest_exists:
        return "manifest not created yet"
    if running_count <= 0:
        return "no run currently marked running"
    if snapshot.updated_age_seconds is not None and snapshot.updated_age_seconds > 180:
        return "no matching process found and manifest looks stale"
    return "manifest shows a running entry, but no matching process was found"


def render_snapshot(snapshot: MonitorSnapshot) -> str:
    lines: List[str] = []
    lines.append("Phase 6 monitor")
    lines.append(f"config: {snapshot.config_path}")
    lines.append(f"manifest: {snapshot.manifest_path}")
    lines.append(f"activity: {_activity_summary(snapshot)}")

    if not snapshot.manifest_exists:
        lines.append("manifest status: not created yet")
    else:
        counts = snapshot.status_counts
        completed = int(counts.get("completed", 0))
        running = int(counts.get("running", 0))
        pending = int(counts.get("pending", 0))
        failed = int(counts.get("failed", 0))
        unavailable = int(counts.get("unavailable", 0))
        lines.append(
            "progress: "
            f"{completed}/{snapshot.expected_run_count} completed | "
            f"{running} running | {pending} pending | {failed} failed | {unavailable} unavailable"
        )
        if snapshot.updated_at_utc:
            if snapshot.updated_age_seconds is None:
                lines.append(f"manifest updated: {snapshot.updated_at_utc}")
            else:
                lines.append(
                    f"manifest updated: {snapshot.updated_at_utc} "
                    f"({snapshot.updated_age_seconds:.0f}s ago)"
                )
        lines.append(f"artifacts: {snapshot.metrics_count} metrics json | {snapshot.models_count} model files")
        if snapshot.running_run_ids:
            lines.append("current run ids:")
            for run_id in snapshot.running_run_ids[:5]:
                lines.append(f"  - {run_id}")

    if snapshot.processes:
        lines.append("matching processes:")
        for proc in snapshot.processes:
            lines.append(f"  - pid={proc.pid} etime={proc.elapsed} cmd={proc.command}")
    else:
        lines.append("matching processes: none found")

    if snapshot.matched_gpu_processes:
        lines.append("gpu compute apps for this run:")
        for proc in snapshot.matched_gpu_processes:
            lines.append(
                f"  - pid={proc.pid} process={proc.process_name} used_gpu_memory={proc.used_gpu_memory}"
            )
    elif snapshot.gpu_processes:
        lines.append("gpu compute apps for this run: none matched")
    else:
        lines.append("gpu compute apps: none found or nvidia-smi unavailable")

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Monitor a Phase 6 experiment manifest, artifact growth, and GPU/process state.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE6_CONFIG_PATH, help="Path to Phase 6 YAML config.")
    parser.add_argument("--follow", action="store_true", help="Refresh the monitor until interrupted.")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds when --follow is used.")
    parser.add_argument("--no-processes", action="store_true", help="Skip process inspection via ps.")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU inspection via nvidia-smi.")
    args = parser.parse_args(argv)

    include_processes = not args.no_processes
    include_gpu = not args.no_gpu

    try:
        while True:
            snapshot = collect_monitor_snapshot(
                args.config,
                include_processes=include_processes,
                include_gpu=include_gpu,
            )
            if args.follow:
                print("\033[2J\033[H", end="")
                print("Press Ctrl-C to stop.\n")
            print(render_snapshot(snapshot), flush=True)
            if not args.follow:
                break
            time.sleep(max(float(args.interval), 1.0))
    except KeyboardInterrupt:
        if args.follow:
            print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
