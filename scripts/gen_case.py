import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.logging_config import setup_logging, get_logger


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "eval" / "cases" / "eval_data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "eval" / "cases"
DEFAULT_LOCAL_KB_FILE = PROJECT_DIR / "data" / "legal_kb.txt"
TASK_TYPES = [
    "Contract Review",
    "Risk Assessment",
    "Compliance Check",
    "Custom Query",
    "Legal Research",
]
TASK_DIRS = {
    "Contract Review": "contract_review",
    "Risk Assessment": "risk_assessment",
    "Compliance Check": "compliance_check",
    "Custom Query": "custom_query",
    "Legal Research": "legal_research",
}

STANDARD_TASK_QUERIES = {
    "Contract Review": """
Review the uploaded contract from a document-analysis perspective.
Extract the key clauses, obligations, rights, deadlines, payment terms, termination conditions, liabilities, and dispute-related provisions.
Identify ambiguities, missing protections, inconsistent wording, and drafting weaknesses.
Focus on what the document explicitly says and what it fails to address.
""".strip(),
    "Legal Research": """
Identify the main legal issues raised by the uploaded document and provide the relevant legal principles, precedents, regulatory concerns, or compliance background.
Distinguish clearly between what is stated in the uploaded document and what comes from legal research or external authority.
Provide stable legal background, key definitions, and jurisdiction-relevant support where helpful.
Focus on legal interpretation and authority support rather than contract redrafting.
""".strip(),
    "Risk Assessment": """
First identify the document-level issues, problematic clauses, missing protections, or ambiguous drafting in the uploaded document.
Then assess the legal and practical risks associated with those issues.
Use relevant legal background where helpful to support the analysis of identified risks and mitigation considerations.
Rank the major risks by priority and explain why they matter.
Finally, provide practical risk-mitigation recommendations.
""".strip(),
    "Compliance Check": """
Review the uploaded document for compliance-related clauses, omissions, and regulatory risk points.
Identify the relevant legal or regulatory concerns raised by the document.
Clarify the applicable compliance framework, regulatory background, and jurisdiction-relevant obligations where helpful.
Then explain the practical compliance risks and recommend remediation or follow-up actions.
Separate document findings, legal/compliance analysis, and recommended next steps.
""".strip(),
}


PROMPT_TEMPLATE = """You generate evaluation cases for a legal contract analysis system.

Input: ONE contract text.

Generate grounded evaluation data for these 5 task types:
- Contract Review
- Risk Assessment
- Compliance Check
- Custom Query
- Legal Research

Rules:
- Use ONLY the contract text.
- For local-knowledge retrieval targets, use ONLY the provided local legal knowledge-base text.
- Do NOT use external knowledge, tools, or assumptions beyond the provided contract text and local legal knowledge-base text.
- Do NOT invent facts, clauses, risks, or questions.
- Every expected point must be strictly supported by the contract.
- Output valid JSON only. No markdown, no extra text.

Task-generation rules:
- For Contract Review, Risk Assessment, Compliance Check, and Legal Research:
  - Do NOT invent a new narrow user question.
  - Set `user_query` to the exact fixed task template provided below for that analysis type.
  - Generate `must_cover_topics`, `must_cover_issues`, `recommended_actions`, `expected_structure`, and `evaluation_rubric` based on that fixed task template and the contract text.
  - Also generate:
    - `expected_document_retrieval_targets`: the contract clauses, sections, obligations, or document-level evidence that should ideally be retrieved from the uploaded contract for this task.
    - `expected_local_kb_retrieval_targets`: the legal-background or compliance-support material that should ideally be retrieved from the local legal knowledge base for this task.
- For Custom Query only:
  - Generate one grounded, contract-specific user question.
  - The question must be narrow enough to be answerable from the contract, but meaningful for legal analysis.
  - Then generate the evaluation fields around that generated question, including:
    - `expected_document_retrieval_targets`
    - `expected_local_kb_retrieval_targets`
- If a retrieval source is not necessary for a task, return an empty list for that source rather than inventing targets.

Fixed task templates for standard tasks:
- Contract Review:
{contract_review_query}

- Risk Assessment:
{risk_assessment_query}

- Compliance Check:
{compliance_check_query}

- Legal Research:
{legal_research_query}

JSON schema:
{{
  "document_summary": "short summary",
  "cases": [
    {{
      "analysis_type": "Contract Review",
      "user_query": "exactly the fixed Contract Review task template above",
      "expected_document_retrieval_targets": ["..."],
      "expected_local_kb_retrieval_targets": ["..."],
      "must_cover_topics": ["..."],
      "must_cover_issues": ["..."],
      "recommended_actions": ["..."],
      "expected_structure": ["..."],
      "evaluation_rubric": "short rubric"
    }},
    {{
      "analysis_type": "Risk Assessment",
      "user_query": "exactly the fixed Risk Assessment task template above",
      "expected_document_retrieval_targets": ["..."],
      "expected_local_kb_retrieval_targets": ["..."],
      "must_cover_topics": ["..."],
      "must_cover_issues": ["..."],
      "recommended_actions": ["..."],
      "expected_structure": ["..."],
      "evaluation_rubric": "short rubric"
    }},
    {{
      "analysis_type": "Compliance Check",
      "user_query": "exactly the fixed Compliance Check task template above",
      "expected_document_retrieval_targets": ["..."],
      "expected_local_kb_retrieval_targets": ["..."],
      "must_cover_topics": ["..."],
      "must_cover_issues": ["..."],
      "recommended_actions": ["..."],
      "expected_structure": ["..."],
      "evaluation_rubric": "short rubric"
    }},
    {{
      "analysis_type": "Custom Query",
      "user_query": "grounded question",
      "expected_document_retrieval_targets": ["..."],
      "expected_local_kb_retrieval_targets": ["..."],
      "must_cover_topics": ["..."],
      "must_cover_issues": ["..."],
      "recommended_actions": ["..."],
      "expected_structure": ["..."],
      "evaluation_rubric": "short rubric"
    }},
    {{
      "analysis_type": "Legal Research",
      "user_query": "exactly the fixed Legal Research task template above",
      "expected_document_retrieval_targets": ["..."],
      "expected_local_kb_retrieval_targets": ["..."],
      "must_cover_topics": ["..."],
      "must_cover_issues": ["..."],
      "recommended_actions": ["..."],
      "expected_structure": ["..."],
      "evaluation_rubric": "short rubric"
    }}
  ]
}}

Field guidance:
- must_cover_topics: main contract topics explicitly present in the text
- must_cover_issues: concrete issues, gaps, ambiguities, or weaknesses grounded in the text
- recommended_actions: practical next steps based only on the text
- expected_structure: expected answer sections
- evaluation_rubric: one-sentence scoring guidance
- expected_document_retrieval_targets: the contract-side evidence that should be retrieved for the task
- expected_local_kb_retrieval_targets: the local-KB-side legal background that should be retrieved for the task

Contract text:
{contract_text}

Local legal knowledge-base text:
{local_kb_text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation cases from contract documents using an OpenAI model."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing contract files (.txt, .md).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated cases by task type.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_CASE_GEN_MODEL", "qwen3-32b "),
        help="OpenAI model used for case generation.",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob pattern for contract files inside input-dir.",
    )
    parser.add_argument(
        "--local-kb-file",
        type=Path,
        default=DEFAULT_LOCAL_KB_FILE,
        help="Path to the local legal knowledge-base text file used to generate expected local retrieval targets.",
    )
    return parser.parse_args()


def load_env() -> None:
    env_path = PROJECT_DIR / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def build_client() -> OpenAI:
    api_key = require_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def load_contract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_local_kb_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_text_from_response(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                return "\n".join(part for part in text_parts if part).strip()

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is not None:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif hasattr(item, "type") and getattr(item, "type", "") == "text":
                        text_parts.append(getattr(item, "text", ""))
                return "\n".join(part for part in text_parts if part).strip()
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()

    text_parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") == "output_text":
                text_parts.append(getattr(content, "text", ""))
    return "\n".join(part for part in text_parts if part).strip()


def dump_response_preview(response: Any, limit: int = 2000) -> str:
    try:
        if hasattr(response, "model_dump_json"):
            return response.model_dump_json(indent=2)[:limit]
        if hasattr(response, "model_dump"):
            return json.dumps(response.model_dump(), ensure_ascii=False, indent=2)[:limit]
        if isinstance(response, dict):
            return json.dumps(response, ensure_ascii=False, indent=2)[:limit]
        return repr(response)[:limit]
    except Exception:
        return repr(response)[:limit]


def parse_json_response(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
    return json.loads(cleaned)


def validate_payload(payload: dict[str, Any], source_name: str) -> dict[str, Any]:
    if "document_summary" not in payload or "cases" not in payload:
        raise ValueError(f"{source_name}: missing document_summary or cases")

    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != len(TASK_TYPES):
        raise ValueError(f"{source_name}: cases must contain exactly {len(TASK_TYPES)} items")

    seen_types = set()
    for case in cases:
        analysis_type = case.get("analysis_type")
        if analysis_type not in TASK_TYPES:
            raise ValueError(f"{source_name}: invalid analysis_type={analysis_type}")
        seen_types.add(analysis_type)
        for field in [
            "user_query",
            "expected_document_retrieval_targets",
            "expected_local_kb_retrieval_targets",
            "must_cover_topics",
            "must_cover_issues",
            "recommended_actions",
            "expected_structure",
            "evaluation_rubric",
        ]:
            if field not in case:
                raise ValueError(f"{source_name}: case {analysis_type} missing field {field}")
        for list_field in [
            "expected_document_retrieval_targets",
            "expected_local_kb_retrieval_targets",
            "must_cover_topics",
            "must_cover_issues",
            "recommended_actions",
            "expected_structure",
        ]:
            if not isinstance(case[list_field], list):
                raise ValueError(
                    f"{source_name}: case {analysis_type} field {list_field} must be a list"
                )
        for list_field in [
            "must_cover_topics",
            "must_cover_issues",
            "recommended_actions",
            "expected_structure",
        ]:
            if not case[list_field]:
                raise ValueError(
                    f"{source_name}: case {analysis_type} field {list_field} must be a non-empty list"
                )
        if not isinstance(case["evaluation_rubric"], str) or not case["evaluation_rubric"].strip():
            raise ValueError(f"{source_name}: case {analysis_type} evaluation_rubric must be non-empty")

        if analysis_type in STANDARD_TASK_QUERIES:
            case["user_query"] = STANDARD_TASK_QUERIES[analysis_type]
        elif not isinstance(case["user_query"], str) or not case["user_query"].strip():
            raise ValueError(f"{source_name}: case {analysis_type} user_query must be non-empty")
    if seen_types != set(TASK_TYPES):
        raise ValueError(f"{source_name}: cases must cover all task types exactly once")
    return payload


def generate_cases(client: OpenAI, model: str, contract_text: str, local_kb_text: str) -> dict[str, Any]:
    logger = get_logger("legal_app")
    prompt = PROMPT_TEMPLATE.format(
        contract_text=contract_text,
        local_kb_text=local_kb_text,
        contract_review_query=STANDARD_TASK_QUERIES["Contract Review"],
        risk_assessment_query=STANDARD_TASK_QUERIES["Risk Assessment"],
        compliance_check_query=STANDARD_TASK_QUERIES["Compliance Check"],
        legal_research_query=STANDARD_TASK_QUERIES["Legal Research"],
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        raw_text = extract_text_from_response(response)
    except Exception:
        logger.error(
            "[CaseGen] chat.completions.create failed. Raw provider/model failure likely needs config fix."
        )
        raise

    if not raw_text:
        logger.error("[CaseGen] Empty model content. Raw response preview:\n%s", dump_response_preview(response))
        raise ValueError("Model returned empty content")
    return parse_json_response(raw_text)


def ensure_output_dirs(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for task_dir in TASK_DIRS.values():
        (base_dir / task_dir).mkdir(parents=True, exist_ok=True)
    (base_dir / "_raw").mkdir(parents=True, exist_ok=True)


def save_case_files(base_dir: Path, source_path: Path, payload: dict[str, Any], raw_text: str) -> None:
    stem = source_path.stem
    raw_record = {
        "source_file": str(source_path),
        "document_summary": payload.get("document_summary", ""),
        "cases": payload.get("cases", []),
    }
    (base_dir / "_raw" / f"{stem}.json").write_text(
        json.dumps(raw_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for case in payload["cases"]:
        task_dir = base_dir / TASK_DIRS[case["analysis_type"]]
        case_payload = {
            "source_file": str(source_path),
            "document_summary": payload["document_summary"],
            **case,
        }
        file_name = f"{stem}.json"
        task_dir.joinpath(file_name).write_text(
            json.dumps(case_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    args.model = args.model.strip()
    load_env()

    log_dir = os.getenv("LOG_DIR", str(PROJECT_DIR / "logs"))
    setup_logging(
        log_dir=log_dir,
        log_file="gen_case.log",
        logger_name="legal_app",
    )
    logger = get_logger("legal_app")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.local_kb_file.exists():
        raise FileNotFoundError(f"Local KB file not found: {args.local_kb_file}")

    client = build_client()
    ensure_output_dirs(args.output_dir)
    local_kb_text = load_local_kb_text(args.local_kb_file)

    contract_files = sorted(args.input_dir.glob(args.glob))
    if not contract_files:
        raise FileNotFoundError(
            f"No files matched pattern {args.glob} in directory {args.input_dir}"
        )

    logger.info(
        "[CaseGen] Starting generation | input_dir=%s | files=%s | model=%s | output_dir=%s | local_kb_file=%s",
        args.input_dir,
        len(contract_files),
        args.model,
        args.output_dir,
        args.local_kb_file,
    )

    for path in contract_files:
        logger.info("[CaseGen] Processing file=%s", path)
        contract_text = load_contract_text(path)
        payload = generate_cases(
            client=client,
            model=args.model,
            contract_text=contract_text,
            local_kb_text=local_kb_text,
        )
        validated = validate_payload(payload, source_name=path.name)
        raw_text = json.dumps(validated, ensure_ascii=False)
        save_case_files(args.output_dir, path, validated, raw_text)
        logger.info("[CaseGen] Finished file=%s", path)

    logger.info("[CaseGen] All files processed successfully")


if __name__ == "__main__":
    main()
