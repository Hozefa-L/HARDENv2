import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_REPORT_KEYS = {"node name", "SWCs"}
EXPECTED_FINDING_KEYS = {"category", "function", "lines", "sourcePath"}
NEGATIVE_SIGNAL_KEYS = {
    "status",
    "result",
    "label",
    "is_vulnerable",
    "vulnerable",
    "negative",
    "positive",
    "passed",
    "failed",
}
NEGATIVE_SIGNAL_VALUES = {
    "false",
    "negative",
    "not vulnerable",
    "safe",
    "clean",
    "pass",
    "passed",
    "no",
}


def _extract_swc_id(category: str) -> str:
    match = re.search(r"SWC-(\d+)", category)
    return match.group(1) if match else ""


def _collect_report_files(swc_root: Path) -> Dict[str, List[Path]]:
    dapp_to_reports: Dict[str, List[Path]] = {}
    for dapp_dir in sorted([p for p in swc_root.iterdir() if p.is_dir()]):
        files = sorted([p for p in dapp_dir.rglob("*.json") if p.is_file()])
        dapp_to_reports[dapp_dir.name] = files
    return dapp_to_reports


def _detect_negative_markers_in_finding(finding: Dict[str, Any]) -> List[str]:
    markers: List[str] = []
    for key, value in finding.items():
        key_s = str(key).strip().lower()
        value_s = str(value).strip().lower()
        if key_s in NEGATIVE_SIGNAL_KEYS:
            markers.append(f"key:{key}")
        if value_s in NEGATIVE_SIGNAL_VALUES:
            markers.append(f"value:{value}")
    return markers


def _analyze_reports(paths: List[Path]) -> Dict[str, Any]:
    root_key_counter = Counter()
    finding_key_counter = Counter()
    swc_id_counter = Counter()
    parse_errors = 0
    total_findings = 0
    empty_swcs_lists = 0
    findings_without_swc_id = 0
    report_extra_keys = Counter()
    finding_extra_keys = Counter()
    negative_markers: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []

    for path in paths:
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            parse_errors += 1
            continue

        root_keys = set(obj.keys()) if isinstance(obj, dict) else set()
        root_key_counter.update(root_keys)
        for extra in sorted(root_keys - EXPECTED_REPORT_KEYS):
            report_extra_keys[extra] += 1

        swcs = obj.get("SWCs", []) if isinstance(obj, dict) else []
        if isinstance(swcs, list) and len(swcs) == 0:
            empty_swcs_lists += 1

        if isinstance(swcs, list):
            for finding in swcs:
                if not isinstance(finding, dict):
                    continue
                total_findings += 1
                finding_key_counter.update(finding.keys())
                for extra in sorted(set(finding.keys()) - EXPECTED_FINDING_KEYS):
                    finding_extra_keys[extra] += 1

                category = str(finding.get("category", "")).strip()
                swc_id = _extract_swc_id(category)
                if swc_id:
                    swc_id_counter[swc_id] += 1
                else:
                    findings_without_swc_id += 1

                markers = _detect_negative_markers_in_finding(finding)
                if markers:
                    negative_markers.append(
                        {
                            "file": str(path.relative_to(PROJECT_ROOT)),
                            "markers": markers,
                            "finding": {
                                "category": category,
                                "function": finding.get("function", ""),
                                "lines": finding.get("lines", ""),
                            },
                        }
                    )

        if len(examples) < 200 and isinstance(swcs, list) and swcs:
            first = swcs[0] if isinstance(swcs[0], dict) else {}
            examples.append(
                {
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "report_keys": sorted(root_keys),
                    "swc_count": len(swcs),
                    "category": str(first.get("category", "")),
                    "function": str(first.get("function", "")),
                    "lines": str(first.get("lines", "")),
                }
            )

    return {
        "file_count": int(len(paths)),
        "parse_errors": int(parse_errors),
        "total_findings": int(total_findings),
        "empty_swcs_lists": int(empty_swcs_lists),
        "findings_without_swc_id": int(findings_without_swc_id),
        "root_key_counter": {str(k): int(v) for k, v in root_key_counter.items()},
        "finding_key_counter": {str(k): int(v) for k, v in finding_key_counter.items()},
        "report_extra_keys": {str(k): int(v) for k, v in report_extra_keys.items()},
        "finding_extra_keys": {str(k): int(v) for k, v in finding_extra_keys.items()},
        "swc_id_counter": {
            str(k): int(v)
            for k, v in sorted(
                swc_id_counter.items(), key=lambda kv: (-kv[1], int(kv[0]))
            )
        },
        "negative_markers": negative_markers,
        "examples": examples,
    }


def _verdict_from_analysis(sample_analysis: Dict[str, Any], global_analysis: Dict[str, Any]) -> str:
    has_explicit_negative = bool(sample_analysis["negative_markers"]) or bool(
        global_analysis["negative_markers"]
    )
    if has_explicit_negative:
        return "POS+NEG_EXPLICIT"
    return "POS_ONLY"


def negatives_allowed_from_verdict(verdict: str) -> bool:
    return verdict == "POS+NEG_EXPLICIT"


def _write_semantics_markdown(
    out_path: Path,
    verdict: str,
    seed: int,
    contracts_per_dapp: int,
    eligible_dapps_count: int,
    sampled_rows: List[Tuple[str, List[str]]],
    sample_analysis: Dict[str, Any],
    global_analysis: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# DAppSCAN SWC-bytecode Label Semantics")
    lines.append("")
    lines.append(
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}"
    )
    lines.append(f"- Sampling seed: `{seed}`")
    lines.append(
        f"- Sampling design: `20 DApps x {contracts_per_dapp} contracts` (from eligible DApps with at least {contracts_per_dapp} report files)"
    )
    lines.append(f"- Eligible DApps (>= {contracts_per_dapp} reports): **{eligible_dapps_count}**")
    lines.append(f"- Sampled DApps: **{len(sampled_rows)}**")
    lines.append("")
    lines.append("## Sampled DApps and contracts")
    lines.append("")
    lines.append("| DApp | Sampled report files |")
    lines.append("| --- | --- |")
    for dapp, files in sampled_rows:
        lines.append(f"| {dapp} | {', '.join(files)} |")
    lines.append("")
    lines.append("## Schema/semantics checks")
    lines.append("")
    lines.append("### Sample (20x5)")
    lines.append(
        f"- Files analyzed: **{sample_analysis['file_count']}**, parse errors: **{sample_analysis['parse_errors']}**"
    )
    lines.append(f"- Total SWC findings: **{sample_analysis['total_findings']}**")
    lines.append(f"- Empty `SWCs` arrays: **{sample_analysis['empty_swcs_lists']}**")
    lines.append(
        f"- Findings without parseable `SWC-<id>`: **{sample_analysis['findings_without_swc_id']}**"
    )
    lines.append(
        f"- Root keys observed: `{sorted(sample_analysis['root_key_counter'].keys())}`"
    )
    lines.append(
        f"- Finding keys observed: `{sorted(sample_analysis['finding_key_counter'].keys())}`"
    )
    lines.append(
        f"- Extra root keys beyond expected: `{sorted(sample_analysis['report_extra_keys'].keys())}`"
    )
    lines.append(
        f"- Extra finding keys beyond expected: `{sorted(sample_analysis['finding_extra_keys'].keys())}`"
    )
    lines.append(
        f"- Explicit negative markers detected: **{len(sample_analysis['negative_markers'])}**"
    )
    lines.append("")
    lines.append("### Global cross-check (all SWC_bytecode reports)")
    lines.append(
        f"- Files analyzed: **{global_analysis['file_count']}**, parse errors: **{global_analysis['parse_errors']}**"
    )
    lines.append(f"- Total SWC findings: **{global_analysis['total_findings']}**")
    lines.append(f"- Empty `SWCs` arrays: **{global_analysis['empty_swcs_lists']}**")
    lines.append(
        f"- Explicit negative markers detected: **{len(global_analysis['negative_markers'])}**"
    )
    lines.append("")
    lines.append("## Representative examples (short excerpts/paraphrases)")
    lines.append("")

    picked: List[Dict[str, Any]] = []
    seen_dapps = set()
    for ex in sample_analysis["examples"]:
        ex_path = Path(ex["file"])
        dapp_name = ex_path.parts[-2] if len(ex_path.parts) >= 2 else "<UNKNOWN_DAPP>"
        if dapp_name in seen_dapps:
            continue
        picked.append(ex)
        seen_dapps.add(dapp_name)
        if len(picked) >= 6:
            break
    if len(picked) < 6:
        for ex in sample_analysis["examples"]:
            if ex in picked:
                continue
            picked.append(ex)
            if len(picked) >= 6:
                break
    for idx, ex in enumerate(picked, start=1):
        lines.append(f"### Example {idx}")
        lines.append(f"- File: `{ex['file']}`")
        lines.append(
            f"- Structure: root keys `{ex['report_keys']}`, SWC findings count `{ex['swc_count']}`"
        )
        lines.append(
            f"- First finding excerpt: category `{ex['category']}`, function `{ex['function']}`, lines `{ex['lines']}`"
        )
        lines.append(
            "- Interpretation: this entry asserts a detected SWC finding (positive evidence), not an explicit negative label."
        )
        lines.append("")

    lines.append("## Final verdict")
    lines.append("")
    lines.append(f"**Verdict: `{verdict}`**")
    lines.append("")
    lines.append("## Pipeline rule")
    lines.append("")
    lines.append(
        "- Only generate DAppSCAN negatives if verdict is `POS+NEG_EXPLICIT`."
    )
    lines.append(
        f"- Current setting from this assessment: `ALLOW_NEGATIVE_GENERATION = {str(negatives_allowed_from_verdict(verdict)).upper()}`."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_dappscan_label_semantics_report(
    dappscan_root: Path,
    out_path: Path,
    sample_dapps: int = 20,
    contracts_per_dapp: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    swc_root = dappscan_root / "DAppSCAN-bytecode" / "SWCbytecode"
    dapp_to_reports = _collect_report_files(swc_root)

    eligible = [
        (name, files)
        for name, files in sorted(dapp_to_reports.items())
        if len(files) >= contracts_per_dapp
    ]

    if not eligible:
        raise ValueError("No eligible DApps found for semantics sampling.")

    rng = random.Random(seed)
    sampled_count = min(sample_dapps, len(eligible))
    sampled_dapps = rng.sample(eligible, sampled_count)

    sampled_file_paths: List[Path] = []
    sampled_rows: List[Tuple[str, List[str]]] = []
    for dapp_name, files in sampled_dapps:
        chosen = rng.sample(files, contracts_per_dapp)
        sampled_file_paths.extend(chosen)
        sampled_rows.append((dapp_name, [p.name for p in chosen]))

    sample_analysis = _analyze_reports(sampled_file_paths)
    global_analysis = _analyze_reports([p for files in dapp_to_reports.values() for p in files])
    verdict = _verdict_from_analysis(sample_analysis, global_analysis)

    _write_semantics_markdown(
        out_path=out_path,
        verdict=verdict,
        seed=seed,
        contracts_per_dapp=contracts_per_dapp,
        eligible_dapps_count=len(eligible),
        sampled_rows=sampled_rows,
        sample_analysis=sample_analysis,
        global_analysis=global_analysis,
    )

    return {
        "verdict": verdict,
        "allow_negative_generation": negatives_allowed_from_verdict(verdict),
        "seed": seed,
        "sample_dapps": sampled_count,
        "contracts_per_dapp": contracts_per_dapp,
        "sampled_reports": len(sampled_file_paths),
        "eligible_dapps": len(eligible),
        "output_report": str(out_path.relative_to(PROJECT_ROOT)),
    }
