import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.logging_config import get_logger, setup_logging


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "eval" / "cases"
OUTPUT_SUBDIR = "local_query"

PROMPT_TEMPLATE = """You generate evaluation cases for a local legal knowledge-base QA system.

Input: ONE local legal knowledge text.

Generate grounded evaluation data for Local Query tasks only.

Rules:
- Use ONLY the provided knowledge text.
- Do NOT use external knowledge, tools, or assumptions.
- Do NOT invent legal facts, questions, issues, or answers that are not supported by the text.
- Every generated question must be answerable strictly from the knowledge text.
- Questions should cover different local-QA styles such as definition, elements, comparison, remedies, exceptions, and practical explanation when supported by the text.
- Output valid JSON only. No markdown, no extra text.

JSON schema:
{{
  "knowledge_summary": "short summary",
  "cases": [
    {{
      "analysis_type": "Local Query",
      "user_query": "grounded question",
      "expected_local_kb_retrieval_targets": ["..."],
      "must_cover_topics": ["..."],
      "must_cover_points": ["..."],
      "expected_evidence_terms": ["..."],
      "recommended_followups": ["..."],
      "expected_structure": ["Answer", "Evidence from KB", "Caveats", "Follow-up questions"],
      "evaluation_rubric": "short rubric"
    }}
  ]
}}

Field guidance:
- expected_local_kb_retrieval_targets: the KB passages, concepts, definitions, or answer-bearing evidence that should ideally be retrieved from the local legal knowledge base
- must_cover_topics: main legal concepts explicitly present in the text
- must_cover_points: concrete answer points directly supported by the text
- expected_evidence_terms: text terms or phrases that should plausibly appear in evidence/citation sections
- recommended_followups: useful next questions grounded in the same knowledge text
- expected_structure: the ideal answer structure for the local KB agent
- evaluation_rubric: one-sentence scoring guidance

Generate between 8 and 12 Local Query cases from the text.

Knowledge text:
{knowledge_text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local KB evaluation cases using an OpenAI-compatible model."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_DIR / "legal_kb.txt",
        help="Path to the local legal knowledge text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated local-query cases.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_CASE_GEN_MODEL", "qwen3-32b"),
        help="OpenAI-compatible model used for local case generation.",
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


def load_text(path: Path) -> str:
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

    return ""


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
    return json.loads(raw_text.strip())


def validate_payload(payload: dict[str, Any], source_name: str) -> dict[str, Any]:
    if "knowledge_summary" not in payload or "cases" not in payload:
        raise ValueError(f"{source_name}: missing knowledge_summary or cases")

    cases = payload["cases"]
    if not isinstance(cases, list) or not (8 <= len(cases) <= 12):
        raise ValueError(f"{source_name}: cases must contain between 8 and 12 items")

    for idx, case in enumerate(cases, start=1):
        if case.get("analysis_type") != "Local Query":
            raise ValueError(f"{source_name}: case #{idx} must use analysis_type=Local Query")
        for field in [
            "user_query",
            "expected_local_kb_retrieval_targets",
            "must_cover_topics",
            "must_cover_points",
            "expected_evidence_terms",
            "recommended_followups",
            "expected_structure",
            "evaluation_rubric",
        ]:
            if field not in case:
                raise ValueError(f"{source_name}: case #{idx} missing field {field}")
        for list_field in [
            "expected_local_kb_retrieval_targets",
            "must_cover_topics",
            "must_cover_points",
            "expected_evidence_terms",
            "recommended_followups",
            "expected_structure",
        ]:
            if not isinstance(case[list_field], list):
                raise ValueError(f"{source_name}: case #{idx} field {list_field} must be a list")
        for list_field in [
            "expected_local_kb_retrieval_targets",
            "must_cover_topics",
            "must_cover_points",
            "expected_evidence_terms",
            "expected_structure",
        ]:
            if not case[list_field]:
                raise ValueError(f"{source_name}: case #{idx} field {list_field} must be non-empty")
        if not isinstance(case["user_query"], str) or not case["user_query"].strip():
            raise ValueError(f"{source_name}: case #{idx} user_query must be non-empty")
        if not isinstance(case["evaluation_rubric"], str) or not case["evaluation_rubric"].strip():
            raise ValueError(f"{source_name}: case #{idx} evaluation_rubric must be non-empty")
    return payload


def generate_cases(client: OpenAI, model: str, knowledge_text: str) -> dict[str, Any]:
    logger = get_logger("legal_app")
    prompt = PROMPT_TEMPLATE.format(knowledge_text=knowledge_text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        raw_text = extract_text_from_response(response)
    except Exception:
        logger.error(
            "[LocalCaseGen] chat.completions.create failed. Raw provider/model failure likely needs config fix."
        )
        raise

    if not raw_text:
        logger.error(
            "[LocalCaseGen] Empty model content. Raw response preview:\n%s",
            dump_response_preview(response),
        )
        raise ValueError("Model returned empty content")
    return parse_json_response(raw_text)


def ensure_output_dirs(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / OUTPUT_SUBDIR).mkdir(parents=True, exist_ok=True)
    (base_dir / "_raw_local").mkdir(parents=True, exist_ok=True)


def save_case_files(base_dir: Path, source_path: Path, payload: dict[str, Any]) -> None:
    stem = source_path.stem
    raw_record = {
        "source_file": str(source_path),
        "knowledge_summary": payload.get("knowledge_summary", ""),
        "cases": payload.get("cases", []),
    }
    (base_dir / "_raw_local" / f"{stem}.json").write_text(
        json.dumps(raw_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    task_dir = base_dir / OUTPUT_SUBDIR
    for idx, case in enumerate(payload["cases"], start=1):
        case_payload = {
            "source_file": str(source_path),
            "knowledge_summary": payload["knowledge_summary"],
            **case,
        }
        file_name = f"{stem}_{idx:02d}.json"
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
        log_file="gen_local_case.log",
        logger_name="legal_app",
    )
    logger = get_logger("legal_app")

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    client = build_client()
    ensure_output_dirs(args.output_dir)

    logger.info(
        "[LocalCaseGen] Starting generation | input_path=%s | model=%s | output_dir=%s",
        args.input_path,
        args.model,
        args.output_dir,
    )

    knowledge_text = load_text(args.input_path)
    payload = generate_cases(client=client, model=args.model, knowledge_text=knowledge_text)
    validated = validate_payload(payload, source_name=args.input_path.name)
    save_case_files(args.output_dir, args.input_path, validated)

    logger.info("[LocalCaseGen] Finished file=%s", args.input_path)


if __name__ == "__main__":
    main()
