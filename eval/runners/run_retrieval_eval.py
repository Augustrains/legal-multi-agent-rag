import argparse
import datetime as dt
import json
import math
import os
import re
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

from src.logging_config import setup_logging
from provider_agent_service import ONLINE_TASKS, get_local_service, get_online_service
from tests_from_cases import ONLINE_ANALYSIS_TYPES, generate_tests

ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv()

REPORT_DIR = PROJECT_DIR / "eval" / "retrieval_reports"
LOGGER = setup_logging(
    log_dir=str(PROJECT_DIR / "logs"),
    log_file="retrieval_eval.log",
    logger_name="legal_app",
)


class _LocalKBAgentShim:
    def __init__(self, knowledge: Any):
        self.knowledge = knowledge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval-focused evaluation for uploaded-document and local-KB retrieval."
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
        help="Optional output report path. Defaults to a timestamped directory under eval/retrieval_reports/.",
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
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to inspect for each retrieval source.",
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


def _build_run_dir(*, analysis_type: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{_safe_slug(analysis_type)}_retrieval_{_execution_mode()}"
    run_dir = REPORT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _phrase_hit(output_norm: str, phrase: str) -> bool:
    phrase_norm = _normalize(phrase)
    if not phrase_norm:
        return False
    if phrase_norm in output_norm:
        return True
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", phrase_norm) if len(tok) > 2]
    if not tokens:
        return False
    overlap_count = sum(tok in output_norm for tok in tokens)
    required_overlap = max(1, math.ceil(len(tokens) * 0.4))
    return overlap_count >= required_overlap


def _score_hits(output: str, values: list[str]) -> tuple[float, list[str], list[str]]:
    output_norm = _normalize(output)
    hits = []
    misses = []
    for value in values:
        if _phrase_hit(output_norm, value):
            hits.append(value)
        else:
            misses.append(value)
    score = 1.0 if not values else len(hits) / len(values)
    return score, hits, misses


def _serialize_doc(doc: Any) -> dict[str, Any]:
    metadata = getattr(doc, "meta_data", {}) or {}
    content = getattr(doc, "content", "") or ""
    return {
        "preview": content[:500],
        "metadata": metadata,
    }


def _serialize_local_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "preview": (item.get("content") or "")[:500],
        "metadata": item.get("metadata", {}) or {},
    }


def _run_document_retrieval(service: Any, query: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    if getattr(service, "knowledge_base", None) is None:
        return "", []
    vector_db = service.knowledge_base.vector_db
    docs = vector_db.search(query=query, limit=top_k)
    serialized = [_serialize_doc(doc) for doc in docs]
    combined = "\n\n".join(item["preview"] for item in serialized if item["preview"])
    return combined, serialized


def _run_local_kb_retrieval(service: Any, query: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    if getattr(service, "local_knowledge_base", None) is None or getattr(service, "local_retriever", None) is None:
        return "", []
    shim_agent = _LocalKBAgentShim(service.local_knowledge_base)
    results = service.local_retriever(
        shim_agent,
        query=query,
        num_documents=top_k,
    )
    serialized = [_serialize_local_result(item) for item in results]
    combined = "\n\n".join(item["preview"] for item in serialized if item["preview"])
    return combined, serialized


def _evaluate_source(
    retrieved_text: str,
    expected_targets: list[str],
    pass_threshold: float = 0.6,
) -> dict[str, Any]:
    if not expected_targets:
        return {
            "pass": True,
            "score": 1.0,
            "reason": {
                "target_hits": [],
                "target_misses": [],
                "note": "No expected targets were provided for this retrieval source.",
            },
        }

    score, hits, misses = _score_hits(retrieved_text, expected_targets)
    return {
        "pass": score >= pass_threshold,
        "score": round(score, 4),
        "reason": {
            "target_hits": hits,
            "target_misses": misses,
        },
    }


def _run_single_test(test: dict[str, Any], top_k: int) -> dict[str, Any]:
    vars_data = test.get("vars", {}) or {}
    analysis_type = (vars_data.get("analysis_type") or "").strip()
    query = (vars_data.get("user_query") or "").strip()
    source_file = (vars_data.get("source_file") or "").strip()

    if analysis_type == "Local Query":
        service = get_local_service()
    elif analysis_type in ONLINE_TASKS:
        service = get_online_service(source_file)
    else:
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")

    started = time.time()
    document_text, document_results = _run_document_retrieval(service, query, top_k=top_k)
    local_text, local_results = _run_local_kb_retrieval(service, query, top_k=top_k)
    duration = time.time() - started

    expected_document_targets = vars_data.get("expected_document_retrieval_targets", []) or []
    expected_local_targets = vars_data.get("expected_local_kb_retrieval_targets", []) or []

    document_check = _evaluate_source(document_text, expected_document_targets)
    local_check = _evaluate_source(local_text, expected_local_targets)

    checks_to_average = []
    checks_for_pass = []
    if analysis_type != "Local Query":
        checks_to_average.append(document_check["score"])
        checks_for_pass.append(document_check["pass"])
    if expected_local_targets or analysis_type == "Local Query":
        checks_to_average.append(local_check["score"])
        checks_for_pass.append(local_check["pass"])

    final_score = round(sum(checks_to_average) / len(checks_to_average), 4) if checks_to_average else 0.0
    final_pass = all(checks_for_pass) if checks_for_pass else True

    return {
        "description": test.get("description", ""),
        "analysis_type": analysis_type,
        "source_file": source_file,
        "user_query": query,
        "duration_seconds": round(duration, 3),
        "document_retrieval": {
            "expected_targets": expected_document_targets,
            "retrieved_results": document_results,
            "check": document_check,
        },
        "local_kb_retrieval": {
            "expected_targets": expected_local_targets,
            "retrieved_results": local_results,
            "check": local_check,
        },
        "final_score": final_score,
        "final_pass": final_pass,
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item["final_pass"])
    avg_score = round(sum(item["final_score"] for item in results) / total, 4) if total else 0.0
    avg_duration = (
        round(sum(item["duration_seconds"] for item in results) / total, 3) if total else 0.0
    )
    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_score": avg_score,
        "average_duration_seconds": avg_duration,
    }


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

    if args.all_tasks_for_doc and doc_index is None:
        raise ValueError("--all-tasks-for-doc requires --doc-index")

    run_label = (
        f"doc #{doc_index} across all online tasks"
        if args.all_tasks_for_doc and doc_index is not None
        else args.analysis_type
    )
    print(f"Running retrieval eval for: {run_label}", flush=True)
    print(f"Loaded cases: {len(tests)}", flush=True)

    results = []
    for idx, test in enumerate(tests, start=1):
        print(f"[{idx}/{len(tests)}] {test.get('description', '')}", flush=True)
        try:
            result = _run_single_test(test, top_k=args.top_k)
        except Exception:
            LOGGER.exception("[RetrievalEval] test failed")
            raise
        results.append(result)
        print(
            f"  -> {'PASS' if result['final_pass'] else 'FAIL'} | "
            f"score={result['final_score']:.4f} | "
            f"doc={float(result['document_retrieval']['check']['score']):.4f} | "
            f"local={float(result['local_kb_retrieval']['check']['score']):.4f}",
            flush=True,
        )

    summary = _build_summary(results)
    report = {
        "analysis_type": args.analysis_type,
        "summary": summary,
        "results": results,
    }

    output_path = args.output or (
        _build_run_dir(analysis_type=args.analysis_type)
        / f"{_safe_slug(args.analysis_type)}_retrieval_report.json"
    )
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nReport written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
