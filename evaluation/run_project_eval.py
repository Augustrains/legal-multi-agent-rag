import argparse
import json
import os
import signal
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

from agno.models.deepseek import DeepSeek

from agent_service import AgentService
from kb_service import (
    build_local_kb,
    build_metadata,
    build_text_reader,
    create_local_kb_retriever,
    ingest_file_with_dedup,
    init_qdrant_with_index,
)
from logging_config import get_logger, setup_logging

EVAL_DIR = PROJECT_DIR / "evaluation"
DEFAULT_CASES = EVAL_DIR / "cases" / "project_eval_cases.json"
DEFAULT_SAMPLE_DOC = EVAL_DIR / "fixtures" / "sample_contract.txt"
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_CASE_TIMEOUT = 240

logger = get_logger("legal_app")


@dataclass
class EvalCheck:
    name: str
    passed: bool
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run project-level real evaluation for AI Legal Agent Team."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="JSON file containing evaluation cases.",
    )
    parser.add_argument(
        "--sample-doc",
        type=Path,
        default=DEFAULT_SAMPLE_DOC,
        help="Text fixture used to build the uploaded-document knowledge base.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Base directory for timestamped evaluation outputs.",
    )
    parser.add_argument(
        "--case-timeout",
        type=int,
        default=DEFAULT_CASE_TIMEOUT,
        help="Per-case timeout in seconds.",
    )
    return parser.parse_args()


def load_env() -> None:
    env_path = PROJECT_DIR / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    load_dotenv()


def require_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    with cases_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list):
        raise ValueError("Cases file must be a JSON array.")
    return cases


def ensure_results_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    responses_dir = run_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_collection_name(prefix: str, run_id: str) -> str:
    return f"{prefix}_{run_id}".lower()


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def build_main_knowledge_base(
    qdrant_url: str,
    qdrant_api_key: str,
    sample_doc: Path,
    collection_name: str,
):
    client, _, knowledge = init_qdrant_with_index(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
    )
    reader = build_text_reader()
    metadata = build_metadata(
        source_type="eval_contract",
        file_name=sample_doc.name,
        legal_topic="commercial_contract",
        jurisdiction="california",
    )
    ingest_file_with_dedup(
        file_path=str(sample_doc),
        knowledge=knowledge,
        client=client,
        collection_name=collection_name,
        reader=reader,
        metadata=metadata,
    )
    return knowledge


def evaluate_response(case: dict[str, Any], response_text: str) -> tuple[int, int, list[EvalCheck]]:
    checks: list[EvalCheck] = []
    normalized = normalize_text(response_text)
    min_length = int(case.get("min_length", 1))
    expectations = case.get("expectations", [])

    checks.append(
        EvalCheck(
            name="non_empty",
            passed=bool(normalized),
            details=f"response_length={len(response_text.strip())}",
        )
    )
    checks.append(
        EvalCheck(
            name="min_length",
            passed=len(response_text.strip()) >= min_length,
            details=f"required>={min_length}, actual={len(response_text.strip())}",
        )
    )

    matched = []
    missed = []
    for keyword in expectations:
        if normalize_text(keyword) in normalized:
            matched.append(keyword)
        else:
            missed.append(keyword)

    checks.append(
        EvalCheck(
            name="expectation_coverage",
            passed=not missed,
            details=f"matched={matched}; missed={missed}",
        )
    )

    score = sum(1 for check in checks if check.passed)
    return score, len(checks), checks


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_report(run_dir: Path, report: dict[str, Any]) -> None:
    save_text(run_dir / "report.json", json.dumps(report, ensure_ascii=False, indent=2))


class CaseTimeoutError(RuntimeError):
    pass


def _timeout_handler(signum, frame):
    raise CaseTimeoutError("Case execution timed out.")


def main() -> None:
    args = parse_args()
    load_env()

    deepseek_api_key = require_env("DEEPSEEK_API_KEY")
    qdrant_api_key = require_env("QDRANT_API_KEY")
    qdrant_url = require_env("QDRANT_URL")
    file_path = os.getenv("FILE_PATH", str(PROJECT_DIR / "data" / "legal_kb.txt"))
    log_dir = os.getenv("LOG_DIR", str(PROJECT_DIR / "logs"))

    setup_logging(
        log_dir=log_dir,
        log_file="project_eval.log",
        logger_name="legal_app",
    )

    if not args.sample_doc.exists():
        raise FileNotFoundError(f"Sample document not found: {args.sample_doc}")

    cases = load_cases(args.cases)
    run_dir = ensure_results_dir(args.results_dir)
    run_id = run_dir.name
    responses_dir = run_dir / "responses"
    config_path = run_dir / "run_config.json"
    save_text(
        config_path,
        json.dumps(
            {
                "cases_file": str(args.cases),
                "sample_doc": str(args.sample_doc),
                "results_dir": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    logger.info("[ProjectEval] Starting real evaluation run_id=%s", run_id)
    main_collection = make_collection_name("eval_main", run_id)
    local_collection = make_collection_name("eval_local", run_id)

    try:
        llm = DeepSeek(id="deepseek-chat", api_key=deepseek_api_key)

        main_knowledge_base = build_main_knowledge_base(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            sample_doc=args.sample_doc,
            collection_name=main_collection,
        )
        local_knowledge_base = build_local_kb(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            file_path=file_path,
            collection_name=local_collection,
        )
        local_retriever = create_local_kb_retriever(deepseek_api_key=deepseek_api_key)

        service = AgentService(
            llm=llm,
            knowledge_base=main_knowledge_base,
            local_knowledge_base=local_knowledge_base,
            local_retriever=local_retriever,
        )

        results = []
        total_score = 0
        total_max_score = 0
        signal.signal(signal.SIGALRM, _timeout_handler)

        for index, case in enumerate(cases, start=1):
            logger.info(
                "[ProjectEval] Running case %s/%s name=%s analysis_type=%s",
                index,
                len(cases),
                case["name"],
                case["analysis_type"],
            )
            response_file = responses_dir / f"{index:02d}_{case['name']}.md"
            case_result: dict[str, Any]

            try:
                signal.alarm(args.case_timeout)
                response = service.run(case["analysis_type"], case.get("user_query"))
                response_text = getattr(response, "content", "") or ""
                signal.alarm(0)

                score, max_score, checks = evaluate_response(case, response_text)
                total_score += score
                total_max_score += max_score
                save_text(response_file, response_text)

                case_result = {
                    "name": case["name"],
                    "analysis_type": case["analysis_type"],
                    "score": score,
                    "max_score": max_score,
                    "passed": score == max_score,
                    "checks": [asdict(check) for check in checks],
                    "response_file": str(response_file),
                    "response_preview": response_text[:400],
                    "user_query": case.get("user_query"),
                    "status": "completed",
                }
            except CaseTimeoutError as exc:
                signal.alarm(0)
                timeout_message = f"Case timed out after {args.case_timeout} seconds."
                save_text(response_file, timeout_message)
                total_max_score += 3
                case_result = {
                    "name": case["name"],
                    "analysis_type": case["analysis_type"],
                    "score": 0,
                    "max_score": 3,
                    "passed": False,
                    "checks": [
                        asdict(EvalCheck(name="timeout", passed=False, details=str(exc)))
                    ],
                    "response_file": str(response_file),
                    "response_preview": timeout_message,
                    "user_query": case.get("user_query"),
                    "status": "timeout",
                }
            except Exception as exc:
                signal.alarm(0)
                error_message = f"Case failed: {exc}"
                save_text(response_file, error_message)
                total_max_score += 3
                case_result = {
                    "name": case["name"],
                    "analysis_type": case["analysis_type"],
                    "score": 0,
                    "max_score": 3,
                    "passed": False,
                    "checks": [
                        asdict(
                            EvalCheck(
                                name="runtime_error",
                                passed=False,
                                details=f"{exc.__class__.__name__}: {exc}",
                            )
                        )
                    ],
                    "response_file": str(response_file),
                    "response_preview": error_message,
                    "user_query": case.get("user_query"),
                    "status": "failed",
                }

            results.append(case_result)
            partial_summary = {
                "status": "running",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "cases": len(cases),
                "completed_cases": len(results),
                "passed_cases": sum(1 for item in results if item["passed"]),
                "total_score": total_score,
                "total_max_score": total_max_score,
                "score_rate": round(total_score / total_max_score, 4) if total_max_score else 0.0,
                "main_collection": main_collection,
                "local_collection": local_collection,
                "sample_doc": str(args.sample_doc),
            }
            save_report(run_dir, {"summary": partial_summary, "results": results})

        summary = {
            "status": "success",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "cases": len(results),
            "passed_cases": sum(1 for item in results if item["passed"]),
            "total_score": total_score,
            "total_max_score": total_max_score,
            "score_rate": round(total_score / total_max_score, 4) if total_max_score else 0.0,
            "main_collection": main_collection,
            "local_collection": local_collection,
            "sample_doc": str(args.sample_doc),
        }

        report = {
            "summary": summary,
            "results": results,
        }

        save_report(run_dir, report)

        print("== Project Evaluation Complete ==")
        print(
            f"Passed {summary['passed_cases']}/{summary['cases']} cases | "
            f"Score {summary['total_score']}/{summary['total_max_score']} "
            f"({summary['score_rate']:.2%})"
        )
        print(f"Results saved to: {run_dir}")
    except Exception as exc:
        error_report = {
            "status": "failed",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "main_collection": main_collection,
            "local_collection": local_collection,
            "traceback": traceback.format_exc(),
        }
        save_text(
            run_dir / "error.json",
            json.dumps(error_report, ensure_ascii=False, indent=2),
        )
        print("== Project Evaluation Failed ==")
        print(f"Error: {exc}")
        print(f"Failure report saved to: {run_dir / 'error.json'}")
        raise


if __name__ == "__main__":
    main()
