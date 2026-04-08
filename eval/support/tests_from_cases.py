import json
import os
import sys
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

EVAL_CASE_DIR = PROJECT_DIR / "eval" / "cases"

TASK_TO_DIR = {
    "Contract Review": "contract_review",
    "Risk Assessment": "risk_assessment",
    "Compliance Check": "compliance_check",
    "Custom Query": "custom_query",
    "Legal Research": "legal_research",
    "Local Query": "local_query",
}

ONLINE_ANALYSIS_TYPES = [
    "Contract Review",
    "Risk Assessment",
    "Compliance Check",
    "Custom Query",
    "Legal Research",
]


def _analysis_type() -> str:
    return os.getenv("PROMPTFOO_ANALYSIS_TYPE", "Contract Review").strip()


def _case_dir() -> Path:
    analysis_type = _analysis_type()
    if analysis_type not in TASK_TO_DIR:
        raise ValueError(f"Unsupported PROMPTFOO_ANALYSIS_TYPE: {analysis_type}")
    return EVAL_CASE_DIR / TASK_TO_DIR[analysis_type]


def _load_case(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_case_path"] = str(path)
    return data


def _build_test(case_path: Path) -> dict:
    case = _load_case(case_path)
    return {
        "description": f"{case['analysis_type']} | {Path(case.get('source_file', case_path.name)).name}",
        "vars": case,
        "assert": [
            {
                "type": "python",
                "value": "file://eval/support/assertions.py:check_required_coverage",
            },
            {
                "type": "python",
                "value": "file://eval/support/assertions.py:judge_with_eval_model",
            },
        ],
    }


def _sorted_case_paths(case_dir: Path) -> list[Path]:
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    return sorted(case_dir.glob("*.json"))


def generate_tests(
    analysis_type: str | None = None,
    doc_index: int | None = None,
    analysis_types: Iterable[str] | None = None,
):
    selected_analysis_types = list(analysis_types) if analysis_types is not None else None
    if selected_analysis_types is None:
        selected_analysis_types = [analysis_type or _analysis_type()]

    tests = []
    for task in selected_analysis_types:
        if task not in TASK_TO_DIR:
            raise ValueError(f"Unsupported analysis type: {task}")

        case_paths = _sorted_case_paths(EVAL_CASE_DIR / TASK_TO_DIR[task])
        if doc_index is None:
            tests.extend(_build_test(case_path) for case_path in case_paths)
            continue

        if doc_index < 1 or doc_index > len(case_paths):
            raise IndexError(
                f"doc_index out of range for {task}: {doc_index} (valid: 1-{len(case_paths)})"
            )
        tests.append(_build_test(case_paths[doc_index - 1]))

    return tests
