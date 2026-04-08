import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = PROJECT_DIR / "eval" / "support"
SRC_DIR = PROJECT_DIR / "src"
for path in (PROJECT_DIR, RUNNER_DIR, SUPPORT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from assertions import (
    check_required_coverage,
    judge_output_with_eval_model,
    judge_with_eval_model,
    score_required_coverage,
)
from src.logging_config import setup_logging
from provider_agent_service import get_service_for_vars, run_service_query
from tests_from_cases import ONLINE_ANALYSIS_TYPES, generate_tests

ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv()

REPORT_DIR = PROJECT_DIR / "eval" / "reports"
LOGGER = setup_logging(
    log_dir=str(PROJECT_DIR / "logs"),
    log_file="project_eval.log",
    logger_name="legal_app",
)

PIPELINE_STAGE_ORDER = [
    "analysis",
    "key_points",
    "recommendations",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified evaluation for single-stage or end-to-end legal analysis."
    )
    parser.add_argument(
        "--analysis-type",
        default=os.getenv("PROMPTFOO_ANALYSIS_TYPE", "Contract Review"),
        help="Task type to evaluate, e.g. Contract Review, Risk Assessment, Local Query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of cases to run. 0 means all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output report path.",
    )
    parser.add_argument(
        "--doc-index",
        type=int,
        default=0,
        help="1-based document index in the sorted eval dataset. 0 means all.",
    )
    parser.add_argument(
        "--all-tasks-for-doc",
        action="store_true",
        help="Run the selected document across all five online task types.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "e2e"),
        default="single",
        help="single evaluates only the main analysis; e2e evaluates analysis plus the two post-processing stages.",
    )
    parser.add_argument(
        "--write-stage-reports",
        action="store_true",
        help="When running in e2e mode, also emit one flattened report per pipeline stage.",
    )
    return parser.parse_args()


def _safe_slug(value: str) -> str:
    return (
        value.strip().lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _execution_mode() -> str:
    return os.getenv("PROMPTFOO_EXECUTION_MODE", "team").strip().lower() or "team"


def _build_run_dir(
    *,
    analysis_type: str,
    mode: str,
) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{_safe_slug(analysis_type)}_{mode}_{_execution_mode()}"
    run_dir = REPORT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_final_result(
    rule_result: dict[str, Any],
    judge_result: dict[str, Any],
) -> tuple[float, bool]:
    final_score = round(
        (float(rule_result["score"]) + float(judge_result["score"])) / 2, 4
    )
    final_pass = bool(rule_result["pass"]) and bool(judge_result["pass"])
    return final_score, final_pass


def _load_source_text(source_file: str) -> str:
    path = Path(source_file)
    if path.exists():
        return path.read_text(encoding="utf-8")[:12000]
    return ""


def _base_stage_config(
    vars_data: dict[str, Any],
    *,
    stage_name: str,
    user_query: str,
    rubric: str,
    topics: list[str] | None = None,
    issues: list[str] | None = None,
    actions: list[str] | None = None,
    expected_structure: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "stage_name": stage_name,
        "analysis_type": vars_data.get("analysis_type", ""),
        "user_query": user_query,
        "evaluation_rubric": rubric,
        "must_cover_topics": topics or [],
        "must_cover_issues": issues or [],
        "recommended_actions": actions or [],
        "expected_structure": expected_structure or [],
        "source_text": _load_source_text(vars_data.get("source_file", "")),
    }


def _apply_stage_overrides(
    stage_config: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if not overrides:
        return stage_config
    merged = dict(stage_config)
    for key in (
        "user_query",
        "evaluation_rubric",
        "must_cover_topics",
        "must_cover_issues",
        "recommended_actions",
        "expected_structure",
    ):
        if key in overrides:
            merged[key] = overrides[key]
    return merged


def _derive_stage_configs(
    vars_data: dict[str, Any],
    analysis_output: str,
) -> dict[str, dict[str, Any]]:
    analysis_type = vars_data.get("analysis_type", "")
    source_text = analysis_output
    pipeline_eval = vars_data.get("pipeline_eval", {}) or {}
    active_agents = vars_data.get("expected_agents", []) or []

    key_points_query = (
        f"Based on this previous analysis:\n{source_text}\n\n"
        "Please summarize the key points in bullet points."
    )
    if analysis_type != "Local Query" and active_agents:
        key_points_query += f"\nFocus on insights from: {', '.join(active_agents)}"

    recommendations_query = (
        f"Based on this previous analysis:\n{source_text}\n\n"
        "What are your key recommendations based on the analysis, the best course of action?"
    )
    if analysis_type != "Local Query" and active_agents:
        recommendations_query += (
            f"\nProvide specific recommendations from: {', '.join(active_agents)}"
        )

    topics = vars_data.get("must_cover_topics", []) or []
    issues = vars_data.get("must_cover_issues", vars_data.get("must_cover_points", [])) or []
    actions = vars_data.get("recommended_actions", []) or []

    analysis_cfg = _base_stage_config(
        vars_data,
        stage_name="analysis",
        user_query=vars_data.get("user_query", ""),
        rubric=vars_data.get("evaluation_rubric", ""),
        topics=topics,
        issues=issues,
        actions=actions,
        expected_structure=vars_data.get("expected_structure", []) or [],
    )
    key_points_cfg = _base_stage_config(
        vars_data,
        stage_name="key_points",
        user_query=key_points_query,
        rubric=(
            "Output should summarize the earlier analysis faithfully, highlight the "
            "most important points, and avoid adding new unsupported claims."
        ),
        topics=topics[: min(5, len(topics))],
        issues=issues[: min(3, len(issues))],
        actions=[],
        expected_structure=["Bullet summary of the most important findings"],
    )
    recommendations_cfg = _base_stage_config(
        vars_data,
        stage_name="recommendations",
        user_query=recommendations_query,
        rubric=(
            "Output should provide practical recommendations grounded in the prior "
            "analysis and align with the identified contract issues."
        ),
        topics=[],
        issues=issues[: min(3, len(issues))],
        actions=actions,
        expected_structure=["Clear actionable recommendations"],
    )

    return {
        "analysis": _apply_stage_overrides(
            analysis_cfg, pipeline_eval.get("analysis")
        ),
        "key_points": _apply_stage_overrides(
            key_points_cfg, pipeline_eval.get("key_points")
        ),
        "recommendations": _apply_stage_overrides(
            recommendations_cfg, pipeline_eval.get("recommendations")
        ),
    }


def _evaluate_stage_output(
    output: str,
    stage_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], float, bool]:
    rule_result = score_required_coverage(
        output,
        analysis_type=stage_config["analysis_type"],
        topics=stage_config.get("must_cover_topics", []) or [],
        issues=stage_config.get("must_cover_issues", []) or [],
        actions=stage_config.get("recommended_actions", []) or [],
    )
    judge_result = judge_output_with_eval_model(
        output,
        analysis_type=stage_config["analysis_type"],
        user_query=stage_config.get("user_query", ""),
        rubric=stage_config.get("evaluation_rubric", ""),
        topics=stage_config.get("must_cover_topics", []) or [],
        issues=stage_config.get("must_cover_issues", []) or [],
        actions=stage_config.get("recommended_actions", []) or [],
        expected_structure=stage_config.get("expected_structure", []) or [],
        source_text=stage_config.get("source_text", ""),
    )
    final_score, final_pass = _build_final_result(rule_result, judge_result)
    return rule_result, judge_result, final_score, final_pass


def _run_single_test(test: dict[str, Any]) -> dict[str, Any]:
    vars_data = test.get("vars", {}) or {}
    prompt = vars_data.get("user_query", "")
    context = {"vars": vars_data}

    provider_started = time.time()
    print(
        "  -> provider: building knowledge base and calling analysis model...",
        flush=True,
    )
    service = get_service_for_vars(vars_data)
    output = run_service_query(service, vars_data.get("analysis_type", ""), prompt)
    provider_duration = time.time() - provider_started

    print(
        f"  -> provider done in {provider_duration:.1f}s, running rule checks...",
        flush=True,
    )
    rule_result = check_required_coverage(output, context)

    print("  -> rule checks done, calling judge model...", flush=True)
    judge_started = time.time()
    judge_result = judge_with_eval_model(output, context)
    judge_duration = time.time() - judge_started
    duration = time.time() - provider_started
    print(f"  -> judge done in {judge_duration:.1f}s", flush=True)

    final_score, final_pass = _build_final_result(rule_result, judge_result)

    return {
        "description": test.get("description", ""),
        "analysis_type": vars_data.get("analysis_type", ""),
        "source_file": vars_data.get("source_file", ""),
        "user_query": vars_data.get("user_query", ""),
        "provider_error": None,
        "output": output,
        "duration_seconds": round(duration, 3),
        "rule_check": rule_result,
        "judge_check": judge_result,
        "final_score": final_score,
        "final_pass": final_pass,
    }


def _run_e2e_test(test: dict[str, Any]) -> dict[str, Any]:
    vars_data = test.get("vars", {}) or {}
    analysis_type = vars_data.get("analysis_type", "")
    service = get_service_for_vars(vars_data)

    started = time.time()
    print(
        "  -> pipeline: running analysis, key points, and recommendations...",
        flush=True,
    )

    analysis_started = time.time()
    analysis_output = run_service_query(
        service, analysis_type, vars_data.get("user_query", "")
    )
    analysis_duration = time.time() - analysis_started

    stage_configs = _derive_stage_configs(vars_data, analysis_output)

    postprocess_analysis_type = (
        "Local Query" if analysis_type == "Local Query" else "Custom Query"
    )

    key_points_started = time.time()
    key_points_output = run_service_query(
        service,
        postprocess_analysis_type,
        stage_configs["key_points"]["user_query"],
    )
    key_points_duration = time.time() - key_points_started

    recommendations_started = time.time()
    recommendations_output = run_service_query(
        service,
        postprocess_analysis_type,
        stage_configs["recommendations"]["user_query"],
    )
    recommendations_duration = time.time() - recommendations_started

    stage_outputs = {
        "analysis": (analysis_output, analysis_duration),
        "key_points": (key_points_output, key_points_duration),
        "recommendations": (recommendations_output, recommendations_duration),
    }

    stages: dict[str, Any] = {}
    for stage_name in PIPELINE_STAGE_ORDER:
        print(f"  -> evaluating stage: {stage_name}", flush=True)
        stage_output, stage_duration = stage_outputs[stage_name]
        rule_result, judge_result, final_score, final_pass = _evaluate_stage_output(
            stage_output,
            stage_configs[stage_name],
        )
        stages[stage_name] = {
            "user_query": stage_configs[stage_name]["user_query"],
            "output": stage_output,
            "duration_seconds": round(stage_duration, 3),
            "rule_check": rule_result,
            "judge_check": judge_result,
            "final_score": final_score,
            "final_pass": final_pass,
        }

    total_duration = time.time() - started
    pipeline_score = round(
        sum(stages[name]["final_score"] for name in PIPELINE_STAGE_ORDER)
        / len(PIPELINE_STAGE_ORDER),
        4,
    )
    pipeline_pass = all(stages[name]["final_pass"] for name in PIPELINE_STAGE_ORDER)

    return {
        "description": test.get("description", ""),
        "analysis_type": analysis_type,
        "source_file": vars_data.get("source_file", ""),
        "user_query": vars_data.get("user_query", ""),
        "provider_error": None,
        "duration_seconds": round(total_duration, 3),
        "pipeline_final_score": pipeline_score,
        "pipeline_final_pass": pipeline_pass,
        "stages": stages,
    }


def _build_summary(results: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    total = len(results)
    if mode == "single":
        passed = sum(1 for item in results if item["final_pass"])
        avg_score = (
            round(sum(item["final_score"] for item in results) / total, 4)
            if total
            else 0.0
        )
    else:
        passed = sum(1 for item in results if item["pipeline_final_pass"])
        avg_score = (
            round(sum(item["pipeline_final_score"] for item in results) / total, 4)
            if total
            else 0.0
        )

    avg_duration = (
        round(sum(item["duration_seconds"] for item in results) / total, 3)
        if total
        else 0.0
    )

    summary: dict[str, Any] = {
        "mode": mode,
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_score": avg_score,
        "average_duration_seconds": avg_duration,
    }

    if mode == "e2e":
        stage_summaries = {}
        for stage_name in PIPELINE_STAGE_ORDER:
            stage_results = [item["stages"][stage_name] for item in results]
            stage_total = len(stage_results)
            stage_passed = sum(1 for item in stage_results if item["final_pass"])
            stage_summaries[stage_name] = {
                "total_cases": stage_total,
                "passed_cases": stage_passed,
                "pass_rate": round(stage_passed / stage_total, 4) if stage_total else 0.0,
                "average_score": round(
                    sum(item["final_score"] for item in stage_results) / stage_total,
                    4,
                )
                if stage_total
                else 0.0,
                "average_duration_seconds": round(
                    sum(item["duration_seconds"] for item in stage_results) / stage_total,
                    3,
                )
                if stage_total
                else 0.0,
            }
        summary["stage_summaries"] = stage_summaries

    return summary


def _stage_report_payload(
    report: dict[str, Any],
    *,
    stage_name: str,
) -> dict[str, Any]:
    stage_results = []
    for item in report["results"]:
        stage = item["stages"][stage_name]
        stage_results.append(
            {
                "description": item["description"],
                "analysis_type": item["analysis_type"],
                "source_file": item["source_file"],
                "user_query": stage["user_query"],
                "output": stage["output"],
                "duration_seconds": stage["duration_seconds"],
                "rule_check": stage["rule_check"],
                "judge_check": stage["judge_check"],
                "final_score": stage["final_score"],
                "final_pass": stage["final_pass"],
            }
        )

    total = len(stage_results)
    passed = sum(1 for item in stage_results if item["final_pass"])
    summary = {
        "stage_name": stage_name,
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_score": round(
            sum(item["final_score"] for item in stage_results) / total, 4
        )
        if total
        else 0.0,
        "average_duration_seconds": round(
            sum(item["duration_seconds"] for item in stage_results) / total, 3
        )
        if total
        else 0.0,
    }
    return {
        "analysis_type": report["analysis_type"],
        "mode": report["mode"],
        "summary": summary,
        "results": stage_results,
    }


def _default_output_path(args: argparse.Namespace) -> Path:
    suffix = "e2e_report.json" if args.mode == "e2e" else "report.json"
    run_dir = _build_run_dir(analysis_type=args.analysis_type, mode=args.mode)
    return run_dir / f"{_safe_slug(args.analysis_type)}_{suffix}"


def _write_stage_reports(report: dict[str, Any], output_path: Path) -> None:
    base = output_path.stem.replace("_e2e_report", "")
    for stage_name in PIPELINE_STAGE_ORDER:
        stage_payload = _stage_report_payload(report, stage_name=stage_name)
        stage_path = output_path.with_name(f"{base}_{stage_name}_report.json")
        stage_path.write_text(
            json.dumps(stage_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Stage report written to: {stage_path}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ["PROMPTFOO_ANALYSIS_TYPE"] = args.analysis_type.strip()

    selected_analysis_types = (
        ONLINE_ANALYSIS_TYPES if args.all_tasks_for_doc else [args.analysis_type]
    )
    doc_index = args.doc_index if args.doc_index > 0 else None

    tests = generate_tests(
        analysis_type=args.analysis_type,
        doc_index=doc_index,
        analysis_types=selected_analysis_types,
    )
    if args.limit and args.limit > 0 and doc_index is None:
        tests = tests[: args.limit]

    _ensure_report_dir()

    results = []
    if args.all_tasks_for_doc and doc_index is None:
        raise ValueError("--all-tasks-for-doc requires --doc-index")

    run_label = (
        f"doc #{doc_index} across all online tasks"
        if args.all_tasks_for_doc and doc_index is not None
        else args.analysis_type
    )
    print(f"Running {args.mode} eval for: {run_label}", flush=True)
    print(f"Loaded cases: {len(tests)}", flush=True)

    runner = _run_e2e_test if args.mode == "e2e" else _run_single_test
    for idx, test in enumerate(tests, start=1):
        print(f"[{idx}/{len(tests)}] {test.get('description', '')}", flush=True)
        try:
            result = runner(test)
        except Exception as exc:
            LOGGER.exception("[PromptfooEval] test failed")
            print(f"  -> ERROR | {exc}", flush=True)
            raise
        results.append(result)
        if args.mode == "e2e":
            status = "PASS" if result["pipeline_final_pass"] else "FAIL"
            print(
                f"  -> {status} | pipeline_score={result['pipeline_final_score']:.4f}",
                flush=True,
            )
        else:
            status = "PASS" if result["final_pass"] else "FAIL"
            print(
                f"  -> {status} | score={result['final_score']:.4f} | "
                f"rule={float(result['rule_check']['score']):.4f} | "
                f"judge={float(result['judge_check']['score']):.4f}",
                flush=True,
            )

    summary = _build_summary(results, mode=args.mode)
    report = {
        "analysis_type": args.analysis_type,
        "mode": args.mode,
        "summary": summary,
        "results": results,
    }

    output_path = args.output or _default_output_path(args)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mode == "e2e" and args.write_stage_reports:
        _write_stage_reports(report, output_path)

    print("\nSummary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nReport written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
