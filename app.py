import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from agno.models.deepseek import DeepSeek
from agno.run.agent import RunOutput

from logging_config import setup_logging, get_logger
from kb_service import (
    init_qdrant_main,
    process_uploaded_file,
    build_local_kb,
    create_local_kb_retriever,
    compute_uploaded_file_hash,
)
from agent_service import AgentService


# =========================================================
# ENV
# =========================================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
FILE_PATH = os.getenv("FILE_PATH")
LOG_DIR = os.getenv("LOG_DIR", "logs")


# =========================================================
# Logging
# =========================================================
setup_logging(
    log_dir=LOG_DIR,
    log_file="app.log",
    logger_name="legal_app",
)
logger = get_logger("legal_app")


# =========================================================
# Session State
# =========================================================
def init_session_state():
    if "deepseek_api_key" not in st.session_state:
        st.session_state.deepseek_api_key = DEEPSEEK_API_KEY

    if "qdrant_api_key" not in st.session_state:
        st.session_state.qdrant_api_key = QDRANT_API_KEY

    if "qdrant_url" not in st.session_state:
        st.session_state.qdrant_url = QDRANT_URL

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    if "local_knowledge_base" not in st.session_state:
        st.session_state.local_knowledge_base = None

    if "local_retriever" not in st.session_state:
        st.session_state.local_retriever = None

    if "agent_service" not in st.session_state:
        st.session_state.agent_service = None

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "last_uploaded_file_hash" not in st.session_state:
        st.session_state.last_uploaded_file_hash = None


# =========================================================
# Helpers
# =========================================================
def ensure_vector_db():
    if st.session_state.vector_db is not None:
        return st.session_state.vector_db

    logger.info("[App] Initializing main vector DB")
    st.session_state.vector_db = init_qdrant_main(
        qdrant_url=st.session_state.qdrant_url,
        qdrant_api_key=st.session_state.qdrant_api_key,
    )
    return st.session_state.vector_db


def ensure_local_kb():
    if st.session_state.local_knowledge_base is not None:
        return st.session_state.local_knowledge_base

    logger.info(f"[App] Building local knowledge base from file={FILE_PATH}")
    st.session_state.local_knowledge_base = build_local_kb(
        qdrant_url=st.session_state.qdrant_url,
        qdrant_api_key=st.session_state.qdrant_api_key,
        file_path=FILE_PATH,
    )
    return st.session_state.local_knowledge_base


def ensure_local_retriever():
    if st.session_state.local_retriever is not None:
        return st.session_state.local_retriever

    logger.info("[App] Creating local retriever")
    st.session_state.local_retriever = create_local_kb_retriever(
        deepseek_api_key=st.session_state.deepseek_api_key
    )
    return st.session_state.local_retriever


def build_agent_service(llm):
    logger.info(
        "[App] Building AgentService | "
        f"has_main_kb={st.session_state.knowledge_base is not None} | "
        f"has_local_kb={st.session_state.local_knowledge_base is not None} | "
        f"has_local_retriever={st.session_state.local_retriever is not None}"
    )

    return AgentService(
        llm=llm,
        knowledge_base=st.session_state.knowledge_base,
        local_knowledge_base=st.session_state.local_knowledge_base,
        local_retriever=st.session_state.local_retriever,
    )


def refresh_agent_service(llm):
    """
    每次主知识库或本地知识库变化后，重建 AgentService
    """
    st.session_state.agent_service = build_agent_service(llm)
    logger.info("[App] AgentService refreshed")


def process_uploaded_document(uploaded_file, llm):
    file_hash = compute_uploaded_file_hash(uploaded_file)

    if file_hash in st.session_state.processed_files:
        logger.info(
            f"[App] Duplicate uploaded file skipped "
            f"file_name={uploaded_file.name} file_hash={file_hash[:12]}"
        )
        st.success("✅ Document already processed and team ready!")
        return

    with st.spinner("Processing document..."):
        logger.info(
            f"[App] Processing uploaded document "
            f"file_name={uploaded_file.name} file_hash={file_hash[:12]}"
        )
        knowledge_base = process_uploaded_file(
            uploaded_file=uploaded_file,
            vector_db=st.session_state.vector_db,
        )

        st.session_state.knowledge_base = knowledge_base
        st.session_state.processed_files.add(file_hash)
        st.session_state.last_uploaded_file_hash = file_hash

        refresh_agent_service(llm)

        logger.info(
            f"[App] Document processed successfully "
            f"file_name={uploaded_file.name} file_hash={file_hash[:12]}"
        )
        st.success("✅ Document processed and team initialized!")


def render_response_tabs(
    analysis_type: str,
    response: RunOutput,
    service: AgentService,
    active_agents: list[str],
):
    tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])

    with tabs[0]:
        st.markdown("### Detailed Analysis")
        st.markdown(response.content or "")
        logger.info(
            f"[LLMResponse] tab=Analysis analysis_type={analysis_type} "
            f"output_len={len(response.content or '')} "
            f"preview={(response.content or '')[:500]}"
        )

    with tabs[1]:
        st.markdown("### Key Points")
        key_points_prompt = f"""Based on this previous analysis:
{response.content}

Please summarize the key points in bullet points.
"""

        if analysis_type != "Local Query":
            key_points_prompt += f"Focus on insights from: {', '.join(active_agents)}"

        key_points_response = service.run("Local Query", key_points_prompt) if analysis_type == "Local Query" else service.run("Custom Query", key_points_prompt)

        st.markdown(key_points_response.content or "")
        logger.info(
            f"[LLMResponse] tab=Key Points analysis_type={analysis_type} "
            f"output_len={len(key_points_response.content or '')} "
            f"preview={(key_points_response.content or '')[:300]}"
        )

    with tabs[2]:
        st.markdown("### Recommendations")
        rec_prompt = f"""Based on this previous analysis:
{response.content}

What are your key recommendations based on the analysis, the best course of action?
"""
        if analysis_type != "Local Query":
            rec_prompt += f"\nProvide specific recommendations from: {', '.join(active_agents)}"

        recommendations_response = service.run("Local Query", rec_prompt) if analysis_type == "Local Query" else service.run("Custom Query", rec_prompt)

        st.markdown(recommendations_response.content or "")
        logger.info(
            f"[LLMResponse] tab=Recommendations analysis_type={analysis_type} "
            f"output_len={len(recommendations_response.content or '')} "
            f"preview={(recommendations_response.content or '')[:300]}"
        )


# =========================================================
# Main
# =========================================================
def main():
    logger.info("[App] main() started")

    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API Configuration")

        if st.session_state.deepseek_api_key:
            st.success("DeepSeek API Key loaded from .env")
        else:
            st.error("DeepSeek API Key not found in .env")

        if st.session_state.qdrant_api_key and st.session_state.qdrant_url:
            st.success("Qdrant config loaded from .env")
        else:
            st.error("Qdrant configuration missing in .env")

        llm = None

        if (
            st.session_state.deepseek_api_key
            and st.session_state.qdrant_api_key
            and st.session_state.qdrant_url
        ):
            try:
                ensure_vector_db()
                st.success("Successfully connected to Qdrant!")
                llm = DeepSeek(
                    id="deepseek-chat",
                    api_key=st.session_state.deepseek_api_key,
                )
                logger.info("[App] LLM initialized successfully")
            except Exception as e:
                logger.exception("[App] Failed to initialize vector DB or LLM")
                st.error(f"Initialization failed: {str(e)}")

        else:
            st.error("Failed to connect to Qdrant. Check URL / API Key.")

        st.divider()

        if llm is not None:
            st.header("📄 Document Upload")

            uploaded_file = st.file_uploader("Upload Legal Document", type=["pdf"])

            # 初始化本地知识库 + retriever + agent service
            try:
                if st.session_state.local_knowledge_base is None:
                    ensure_local_kb()

                if st.session_state.local_retriever is None:
                    ensure_local_retriever()

                if st.session_state.agent_service is None:
                    refresh_agent_service(llm)

            except Exception as e:
                logger.exception("[App] Failed to initialize local KB or AgentService")
                st.error(f"Local KB init failed: {str(e)}")

            if uploaded_file is not None:
                try:
                    process_uploaded_document(uploaded_file, llm)
                except Exception as e:
                    logger.exception("[App] Uploaded document processing failed")
                    st.error(f"Document processing error: {str(e)}")

            st.divider()
            st.header("🔍 Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Contract Review",
                    "Legal Research",
                    "Risk Assessment",
                    "Compliance Check",
                    "Custom Query",
                    "Local Query",
                ],
            )
        else:
            analysis_type = None
            st.warning("Please configure all API credentials to proceed")

    # 主区
    if not all([st.session_state.deepseek_api_key, st.session_state.vector_db]):
        st.info("👈 Please configure your API credentials in the sidebar to begin")
        return

    if analysis_type is None:
        return

    service: AgentService = st.session_state.agent_service
    if service is None:
        st.info("👈 Agent service is not ready")
        return

    config = service.get_analysis_config(analysis_type)

    if analysis_type == "Local Query":
        if st.session_state.local_knowledge_base is None:
            st.info("👈 Local legal knowledge base is not ready")
            return
    else:
        if st.session_state.knowledge_base is None:
            st.info("👈 Please upload a legal document to begin analysis")
            return

    analysis_icons = {
        "Contract Review": "📑",
        "Legal Research": "🔍",
        "Risk Assessment": "⚠️",
        "Compliance Check": "✅",
        "Custom Query": "💭",
        "Local Query": "📚",
    }

    st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
    st.info(f"📋 {config['description']}")
    st.write(f"🤖 Active Legal AI Agents: {', '.join(config['agents'])}")

    if analysis_type in ["Custom Query", "Local Query"]:
        user_query = st.text_area("Enter your specific query:")
    else:
        user_query = None

    if st.button("Analyze"):
        if analysis_type in ["Custom Query", "Local Query"] and not user_query:
            st.warning("Please enter a query")
            return

        try:
            logger.info(
                f"[AnalyzeStart] analysis_type={analysis_type} "
                f"user_query_preview={(user_query or '')[:200]}"
            )

            os.environ["DEEPSEEK_API_KEY"] = st.session_state.deepseek_api_key

            with st.spinner("Analyzing document..."):
                response: RunOutput = service.run(analysis_type, user_query)

            logger.info(
                f"[AnalyzeDone] analysis_type={analysis_type} "
                f"output_len={len(response.content or '')}"
            )

            render_response_tabs(
                analysis_type=analysis_type,
                response=response,
                service=service,
                active_agents=config["agents"],
            )

        except Exception as e:
            logger.exception("[App] Error during analysis")
            st.error(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()