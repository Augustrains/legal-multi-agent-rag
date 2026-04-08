import argparse
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
OFFLINE_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CASES_PATH = OFFLINE_DIR / "eval_cases.json"
DEFAULT_OUTPUT_PATH = OFFLINE_DIR / "eval_report.json"


@dataclass
class MockRunOutput:
    content: str


def install_agno_stubs() -> None:
    if "agno" in sys.modules:
        return

    agno_module = types.ModuleType("agno")
    agent_module = types.ModuleType("agno.agent")
    team_module = types.ModuleType("agno.team")
    tools_module = types.ModuleType("agno.tools")
    duckduckgo_module = types.ModuleType("agno.tools.duckduckgo")
    websearch_module = types.ModuleType("agno.tools.websearch")

    class DummyAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, prompt: str):
            return MockRunOutput(content=prompt)

    class DummyTeam(DummyAgent):
        pass

    class DummyTool:
        def __init__(self, *args, **kwargs):
            pass

    agent_module.Agent = DummyAgent
    team_module.Team = DummyTeam
    duckduckgo_module.DuckDuckGoTools = DummyTool
    websearch_module.WebSearchTools = DummyTool

    sys.modules["agno"] = agno_module
    sys.modules["agno.agent"] = agent_module
    sys.modules["agno.team"] = team_module
    sys.modules["agno.tools"] = tools_module
    sys.modules["agno.tools.duckduckgo"] = duckduckgo_module
    sys.modules["agno.tools.websearch"] = websearch_module


def load_agent_service_class():
    try:
        from src.agent_service import AgentService  # type: ignore

        return AgentService
    except ModuleNotFoundError as exc:
        if exc.name != "agno":
            raise

    install_agno_stubs()
    from src.agent_service import AgentService  # type: ignore

    return AgentService


AgentService = load_agent_service_class()


class MockRunner:
    def __init__(self, route_name: str, service: "RecordingAgentService"):
        self.route_name = route_name
        self.service = service

    def run(self, prompt: str) -> MockRunOutput:
        self.service.last_route = self.route_name
        preview = " ".join(prompt.split())[:240]
        return MockRunOutput(
            content=(
                f"Route: {self.route_name}\n"
                f"Prompt preview: {preview}\n"
                "Assessment: mock response generated for evaluation.\n"
                "Evidence: synthetic stub output.\n"
            )
        )


class RecordingAgentService(AgentService):
    def __init__(self):
        super().__init__(
            llm=None,
            knowledge_base=None,
            local_knowledge_base=None,
            local_retriever=None,
        )
        self.last_route: str | None = None

    def build_local_agent(self):
        return MockRunner("local_agent", self)

    def build_legal_team(self):
        return MockRunner("team", self)


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    with cases_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Evaluation cases file must contain a JSON array.")

    return data


def normalize_text(text: str | None) -> str:
    return (text or "").strip().lower()


def keyword_hits(prompt: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    prompt_text = normalize_text(prompt)
    hits: list[str] = []
    misses: list[str] = []

    for keyword in keywords:
        if normalize_text(keyword) in prompt_text:
            hits.append(keyword)
        else:
            misses.append(keyword)

    return hits, misses


def evaluate_case(service: RecordingAgentService, case: dict[str, Any]) -> dict[str, Any]:
    analysis_type = case["analysis_type"]
    user_query = case.get("user_query")

    config = service.get_analysis_config(analysis_type)
    prompt = service.build_prompt(analysis_type, user_query)
    response = service.run(analysis_type, user_query)

    required_hits, required_misses = keyword_hits(
        prompt,
        case.get("required_prompt_keywords", []),
    )
    forbidden_hits, _ = keyword_hits(
        prompt,
        case.get("forbidden_prompt_keywords", []),
    )

    expected_agents = case.get("expected_agents", [])
    actual_agents = config.get("agents", [])

    checks = {
        "agents_match": actual_agents == expected_agents,
        "route_match": service.last_route == case.get("expected_route"),
        "required_keywords_present": not required_misses,
        "forbidden_keywords_absent": not forbidden_hits,
        "response_not_empty": bool(getattr(response, "content", "").strip()),
    }

    passed_checks = sum(1 for passed in checks.values() if passed)
    total_checks = len(checks)

    return {
        "name": case["name"],
        "analysis_type": analysis_type,
        "score": passed_checks,
        "max_score": total_checks,
        "passed": passed_checks == total_checks,
        "checks": checks,
        "details": {
            "expected_agents": expected_agents,
            "actual_agents": actual_agents,
            "expected_route": case.get("expected_route"),
            "actual_route": service.last_route,
            "required_keyword_hits": required_hits,
            "required_keyword_misses": required_misses,
            "forbidden_keyword_hits": forbidden_hits,
            "prompt_preview": " ".join(prompt.split())[:300],
            "response_preview": " ".join(getattr(response, "content", "").split())[:240],
        },
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_score = sum(item["score"] for item in results)
    total_max_score = sum(item["max_score"] for item in results)
    passed_cases = sum(1 for item in results if item["passed"])

    return {
        "passed_cases": passed_cases,
        "total_cases": len(results),
        "total_score": total_score,
        "total_max_score": total_max_score,
        "pass_rate": round(passed_cases / len(results), 4) if results else 0.0,
        "score_rate": round(total_score / total_max_score, 4) if total_max_score else 0.0,
    }


def print_console_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("== AI Legal Agent Team Evaluation ==")
    print(
        f"Cases: {summary['passed_cases']}/{summary['total_cases']} passed | "
        f"Score: {summary['total_score']}/{summary['total_max_score']} "
        f"({summary['score_rate']:.2%})"
    )

    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"- [{status}] {result['name']} "
            f"({result['score']}/{result['max_score']})"
        )

        if not result["passed"]:
            failed_checks = [
                name for name, passed in result["checks"].items() if not passed
            ]
            print(f"  Failed checks: {', '.join(failed_checks)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AgentService prompt construction and routing logic."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to the JSON evaluation cases file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the JSON evaluation report.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    service = RecordingAgentService()
    results = [evaluate_case(service, case) for case in cases]

    report = {
        "project": "Ai_legal_agent_team",
        "mode": "offline-mock",
        "cases_file": str(args.cases),
        "summary": build_summary(results),
        "results": results,
    }

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_console_report(report)
    print(f"\nJSON report written to: {args.output}")


if __name__ == "__main__":
    main()
