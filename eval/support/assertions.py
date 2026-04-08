import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv()


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
    return sum(tok in output_norm for tok in tokens) >= max(1, min(2, len(tokens)))


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


def score_required_coverage(
    output: str,
    *,
    analysis_type: str,
    topics: list[str] | None = None,
    issues: list[str] | None = None,
    actions: list[str] | None = None,
) -> dict[str, Any]:
    topics = topics or []
    issues = issues or []
    actions = actions or []

    topic_score, topic_hits, topic_misses = _score_hits(output, topics)
    issue_score, issue_hits, issue_misses = _score_hits(
        output, issues
    )
    action_score, action_hits, action_misses = _score_hits(output, actions)

    if analysis_type == "Contract Review":
        score = 0.4 * topic_score + 0.6 * issue_score
        pass_threshold = 0.65
    elif analysis_type == "Local Query":
        score = 0.35 * topic_score + 0.45 * issue_score + 0.20 * action_score
        pass_threshold = 0.60
    else:
        score = 0.25 * topic_score + 0.45 * issue_score + 0.30 * action_score
        pass_threshold = 0.60

    reason = {
        "topic_hits": topic_hits,
        "topic_misses": topic_misses,
        "issue_hits": issue_hits,
        "issue_misses": issue_misses,
        "action_hits": action_hits,
        "action_misses": action_misses,
    }

    return {
        "pass": score >= pass_threshold,
        "score": round(score, 4),
        "reason": json.dumps(reason, ensure_ascii=False),
    }


def check_required_coverage(output: str, context: dict) -> dict[str, Any]:
    vars_data = context.get("vars", {}) or {}
    analysis_type = vars_data.get("analysis_type", "Unknown")

    topics = vars_data.get("must_cover_topics", []) or []
    issues = vars_data.get(
        "must_cover_issues", vars_data.get("must_cover_points", [])
    ) or []
    actions = vars_data.get("recommended_actions", []) or []

    return score_required_coverage(
        output,
        analysis_type=analysis_type,
        topics=topics,
        issues=issues,
        actions=actions,
    )


def _build_eval_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("EVAL_OPENAI_API_KEY", "").strip()
    base_url = os.getenv("EVAL_OPENAI_BASE_URL", "").strip() or None
    model = (
        os.getenv("EVAL_OPENAI_MODEL", "").strip()
        or os.getenv("EVAL_OPENAI_CASE_GEN_MODEL", "").strip()
    )

    if not api_key or not model:
        raise EnvironmentError(
            "Missing EVAL_OPENAI_API_KEY and/or EVAL_OPENAI_MODEL "
            "(fallback: EVAL_OPENAI_CASE_GEN_MODEL) for Promptfoo judge model"
        )
    timeout = float(os.getenv("EVAL_OPENAI_TIMEOUT", "120"))
    max_retries = int(os.getenv("EVAL_OPENAI_MAX_RETRIES", "2"))
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    ), model


def _load_source_text(vars_data: dict) -> str:
    source_file = (vars_data.get("source_file") or "").strip()
    if source_file and Path(source_file).exists():
        return Path(source_file).read_text(encoding="utf-8")[:12000]
    return ""


def judge_output_with_eval_model(
    output: str,
    *,
    analysis_type: str,
    user_query: str,
    rubric: str,
    topics: list[str] | None = None,
    issues: list[str] | None = None,
    actions: list[str] | None = None,
    expected_structure: list[str] | None = None,
    source_text: str = "",
) -> dict[str, Any]:
    topics = topics or []
    issues = issues or []
    actions = actions or []
    expected_structure = expected_structure or []

    judge_prompt = f"""You are grading the output of a legal AI system.

Task type: {analysis_type}
User query: {user_query}
Evaluation rubric: {rubric}
Expected topics: {topics}
Expected issues: {issues}
Expected actions: {actions}
Expected structure: {expected_structure}

Reference source text (truncated if long):
{source_text}

Model output:
{output}

Return valid JSON only:
{{
  "pass": true,
  "score": 0.0,
  "reason": "short reason"
}}

Scoring guidance:
- score should be between 0 and 1
- reward relevance, grounding, coverage, and task completion
- penalize generic answers, missing core issues, and unsupported claims
"""

    client, model = _build_eval_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "response_format": {"type": "json_object"},
    }
    if "qwen" in model.lower():
        kwargs["extra_body"] = {"enable_thinking": False}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )

    data = json.loads((content or "").strip())
    score = float(data.get("score", 0.0))
    return {
        "pass": bool(data.get("pass", score >= 0.65)),
        "score": score,
        "reason": data.get("reason", "judge model did not provide reason"),
    }


def judge_with_eval_model(output: str, context: dict) -> dict[str, Any]:
    vars_data = context.get("vars", {}) or {}
    analysis_type = vars_data.get("analysis_type", "")
    user_query = vars_data.get("user_query", "")
    rubric = vars_data.get("evaluation_rubric", "")
    topics = vars_data.get("must_cover_topics", []) or []
    issues = vars_data.get(
        "must_cover_issues", vars_data.get("must_cover_points", [])
    ) or []
    actions = vars_data.get("recommended_actions", []) or []
    expected_structure = vars_data.get("expected_structure", []) or []
    source_text = _load_source_text(vars_data)

    return judge_output_with_eval_model(
        output,
        analysis_type=analysis_type,
        user_query=user_query,
        rubric=rubric,
        topics=topics,
        issues=issues,
        actions=actions,
        expected_structure=expected_structure,
        source_text=source_text,
    )
