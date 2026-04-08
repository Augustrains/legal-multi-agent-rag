import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from agno.models.deepseek import DeepSeek

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.agent_service import AgentService
from src.kb_service import (
    build_local_kb,
    collection_exists,
    create_local_kb_retriever,
    load_knowledge_from_collection,
)

ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv()

ONLINE_TASKS = {
    "Contract Review",
    "Risk Assessment",
    "Compliance Check",
    "Custom Query",
    "Legal Research",
}


def _execution_mode() -> str:
    mode = os.getenv("PROMPTFOO_EXECUTION_MODE", "team").strip().lower()
    if mode not in {"team", "single"}:
        raise ValueError(
            "PROMPTFOO_EXECUTION_MODE must be either 'team' or 'single'"
        )
    return mode

def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def build_runtime_llm() -> DeepSeek:
    return DeepSeek(
        id=os.getenv("PROMPTFOO_TARGET_MODEL", "deepseek-chat").strip(),
        api_key=require_env("DEEPSEEK_API_KEY"),
        timeout=float(os.getenv("PROMPTFOO_TARGET_TIMEOUT", "180")),
        max_retries=int(os.getenv("PROMPTFOO_TARGET_MAX_RETRIES", "2")),
    )


def _doc_rank_from_path(contract_path_obj: Path) -> str:
    prefix = contract_path_obj.stem.split("_", 1)[0]
    if prefix.isdigit():
        return prefix.zfill(2)
    raise ValueError(f"Cannot derive document rank from path: {contract_path_obj}")


def _online_collection_name(contract_path: str) -> str:
    rank = _doc_rank_from_path(Path(contract_path).resolve())
    return f"promptfoo_contract_doc_{rank}"


def get_online_collection_name(contract_path: str) -> str:
    return _online_collection_name(contract_path)


@lru_cache(maxsize=64)
def get_online_service(contract_path: str) -> AgentService:
    contract_path_obj = Path(contract_path).resolve()
    if not contract_path_obj.exists():
        raise FileNotFoundError(f"Contract file not found: {contract_path_obj}")

    qdrant_url = require_env("QDRANT_URL")
    qdrant_api_key = require_env("QDRANT_API_KEY")
    collection_name = _online_collection_name(str(contract_path_obj))
    if collection_exists(qdrant_url, qdrant_api_key, collection_name):
        knowledge_base = load_knowledge_from_collection(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=collection_name,
        )
    else:
        knowledge_base = build_local_kb(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            file_path=str(contract_path_obj),
            collection_name=collection_name,
        )

    file_path = require_env("FILE_PATH")
    local_collection_name = os.getenv(
        "PROMPTFOO_LOCAL_COLLECTION",
        "local_legal_documents",
    ).strip()
    if collection_exists(qdrant_url, qdrant_api_key, local_collection_name):
        local_knowledge_base = load_knowledge_from_collection(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=local_collection_name,
        )
    else:
        local_knowledge_base = build_local_kb(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            file_path=file_path,
            collection_name=local_collection_name,
        )
    local_retriever = create_local_kb_retriever(
        deepseek_api_key=require_env("DEEPSEEK_API_KEY")
    )

    return AgentService(
        llm=build_runtime_llm(),
        knowledge_base=knowledge_base,
        local_knowledge_base=local_knowledge_base,
        local_retriever=local_retriever,
    )


@lru_cache(maxsize=1)
def get_local_service() -> AgentService:
    file_path = require_env("FILE_PATH")
    qdrant_url = require_env("QDRANT_URL")
    qdrant_api_key = require_env("QDRANT_API_KEY")
    collection_name = os.getenv("PROMPTFOO_LOCAL_COLLECTION", "local_legal_documents").strip()
    if collection_exists(qdrant_url, qdrant_api_key, collection_name):
        local_knowledge_base = load_knowledge_from_collection(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=collection_name,
        )
    else:
        local_knowledge_base = build_local_kb(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            file_path=file_path,
            collection_name=collection_name,
        )
    local_retriever = create_local_kb_retriever(
        deepseek_api_key=require_env("DEEPSEEK_API_KEY")
    )
    return AgentService(
        llm=build_runtime_llm(),
        knowledge_base=None,
        local_knowledge_base=local_knowledge_base,
        local_retriever=local_retriever,
    )


def _extract_output_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if content is not None:
        return str(content)
    return str(result)


def get_service_for_vars(vars_data: dict[str, Any]) -> AgentService:
    analysis_type = (vars_data.get("analysis_type") or "").strip()
    if analysis_type == "Local Query":
        return get_local_service()
    if analysis_type in ONLINE_TASKS:
        source_file = (vars_data.get("source_file") or "").strip()
        if not source_file:
            raise ValueError("Missing source_file for online task")
        return get_online_service(source_file)
    raise ValueError(f"Unsupported analysis_type: {analysis_type}")


def run_service_query(
    service: AgentService,
    analysis_type: str,
    user_query: str,
) -> str:
    execution_mode = _execution_mode()
    if execution_mode == "single":
        result = service.run_single_agent(analysis_type, user_query)
    else:
        result = service.run(analysis_type, user_query)
    return _extract_output_text(result)


def call_api(prompt: str, options: dict, context: dict) -> dict:
    vars_data = context.get("vars", {}) or {}
    analysis_type = (vars_data.get("analysis_type") or "").strip()
    user_query = (vars_data.get("user_query") or prompt or "").strip()

    if not analysis_type:
        return {"error": "Missing analysis_type in test vars", "output": ""}
    if not user_query:
        return {"error": "Missing user_query in test vars", "output": ""}

    try:
        service = get_service_for_vars(vars_data)
        output = run_service_query(service, analysis_type, user_query)
    except Exception as exc:
        return {"error": str(exc), "output": ""}
    return {"output": output}
