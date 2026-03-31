import os
import hashlib
import tempfile
from typing import List, Callable, Optional

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.models.deepseek import DeepSeek
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.reader.text_reader import TextReader
from agno.knowledge.chunking.recursive import RecursiveChunking

from langdetect import detect_langs
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
)

from logging_config import get_logger

logger = get_logger("legal_app")

DEFAULT_MAIN_COLLECTION = "legal_documents"
DEFAULT_LOCAL_COLLECTION = "local_legal_documents"


# =========================================================
# 通用哈希 / 元数据
# =========================================================
def compute_uploaded_file_hash(uploaded_file) -> str:
    """
    计算 Streamlit UploadedFile 的 SHA256，用于文件级去重
    """
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    logger.info(
        f"[Upload:Dedup] file_name={getattr(uploaded_file, 'name', 'unknown')} "
        f"file_hash={file_hash[:12]}"
    )
    return file_hash


def compute_content_hash(text: str) -> str:
    """
    计算文本块级别哈希，用于 chunk 去重
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def build_metadata(
    source_type: str,
    file_name: str,
    legal_topic: str = "general",
    jurisdiction: str = "general",
) -> dict:
    return {
        "source_type": source_type,
        "file_name": file_name,
        "legal_topic": legal_topic,
        "jurisdiction": jurisdiction,
        "language": "",
    }


def detect_language(text: str) -> str:
    try:
        langs = detect_langs(text)
        return langs[0].lang
    except Exception:
        return "unknown"


# =========================================================
# Qdrant 初始化
# =========================================================
def init_qdrant_main(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = DEFAULT_MAIN_COLLECTION,
) -> Qdrant:
    """
    初始化主文档向量库（上传 PDF 用）
    """
    if not qdrant_url or not qdrant_api_key:
        logger.error("[Qdrant] Missing qdrant_url or qdrant_api_key")
        raise ValueError("Missing qdrant_url or qdrant_api_key")

    try:
        embedder = SentenceTransformerEmbedder(
            id="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = Qdrant(
            collection=collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key,
            embedder=embedder,
        )
        logger.info(
            f"[Qdrant] Main vector DB initialized successfully. collection={collection_name}"
        )
        return vector_db
    except Exception:
        logger.exception("[Qdrant] Failed to initialize main vector DB")
        raise


def init_qdrant_with_index(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = DEFAULT_LOCAL_COLLECTION,
):
    """
    初始化本地知识库 collection，并建立 payload index
    返回:
        client, vector_db, knowledge
    """
    try:
        embedder = SentenceTransformerEmbedder(
            id="sentence-transformers/all-MiniLM-L6-v2"
        )

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=10,
        )

        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"[LocalKB] Created collection={collection_name}")
        else:
            logger.info(f"[LocalKB] Reusing collection={collection_name}")

        for field in [
            "content_hash",
            "source_type",
            "file_name",
            "legal_topic",
            "jurisdiction",
            "language",
        ]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info(f"[LocalKB] Created payload index field={field}")
            except Exception:
                logger.info(f"[LocalKB] Payload index may already exist field={field}")

        vector_db = Qdrant(
            collection=collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key,
            embedder=embedder,
        )

        knowledge = Knowledge(vector_db=vector_db)

        return client, vector_db, knowledge

    except Exception:
        logger.exception("[LocalKB] Failed to initialize local collection")
        raise


# =========================================================
# 主文档处理（上传 PDF）
# =========================================================
def process_uploaded_file(uploaded_file, vector_db: Qdrant) -> Knowledge:
    """
    将用户上传的 PDF 文件写入临时文件后，导入主知识库
    """
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        logger.info(
            f"[ProcessDocument] Start file_name={getattr(uploaded_file, 'name', 'unknown')} "
            f"temp_path={temp_file_path}"
        )

        knowledge_base = Knowledge(vector_db=vector_db)
        knowledge_base.add_content(path=temp_file_path)

        logger.info(
            f"[ProcessDocument] Completed file_name={getattr(uploaded_file, 'name', 'unknown')}"
        )
        return knowledge_base

    except Exception:
        logger.exception("[ProcessDocument] Document processing error")
        raise

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"[ProcessDocument] Temp file removed path={temp_file_path}")
            except Exception:
                logger.warning(
                    f"[ProcessDocument] Failed to remove temp file path={temp_file_path}"
                )


# =========================================================
# 本地知识库构建
# =========================================================
def build_text_reader() -> TextReader:
    return TextReader(
        chunk=True,
        chunking_strategy=RecursiveChunking(
            chunk_size=800,
            overlap=100,
        ),
    )


def content_hash_exists(
    client: QdrantClient,
    collection_name: str,
    content_hash: str,
) -> bool:
    """
    检查某个 chunk 的 content_hash 是否已经存在
    """
    result = client.count(
        collection_name=collection_name,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="content_hash",
                    match=MatchValue(value=content_hash),
                )
            ]
        ),
        exact=True,
    )
    exists = result.count > 0
    logger.info(
        f"[Ingest:Dedup] collection={collection_name} "
        f"content_hash={content_hash[:12]} exists={exists}"
    )
    return exists


def ingest_file_with_dedup(
    file_path: str,
    knowledge: Knowledge,
    client: QdrantClient,
    collection_name: str,
    reader: TextReader,
    metadata: dict,
) -> None:
    """
    读取本地文本文件，切块、计算 content_hash、查重、写入 Qdrant
    """
    if not os.path.exists(file_path):
        logger.error(f"[Ingest] File not found: {file_path}")
        raise FileNotFoundError(file_path)

    logger.info(f"[Ingest] Start file_path={file_path} collection={collection_name}")

    with open(file_path, "rb") as f:
        documents = reader.read(f)
        logger.info(f"[Ingest] Reader returned total_chunks={len(documents)}")

    new_documents = []
    skipped = 0

    for idx, doc in enumerate(documents, start=1):
        content = getattr(doc, "content", "") or ""
        if not content.strip():
            logger.warning(f"[Ingest] Empty chunk skipped. chunk_index={idx}")
            continue

        content_hash = compute_content_hash(content)

        if content_hash_exists(client, collection_name, content_hash):
            skipped += 1
            logger.info(
                f"[Ingest] Duplicate chunk skipped. "
                f"chunk_index={idx} content_hash={content_hash[:12]}"
            )
            continue

        language = detect_language(content)
        original_meta = getattr(doc, "meta_data", {}) or {}

        doc.meta_data = {
            **original_meta,
            **metadata,
            "content_hash": content_hash,
            "language": language,
        }
        new_documents.append(doc)

    if new_documents:
        for insert_idx, doc in enumerate(new_documents, start=1):
            try:
                knowledge.vector_db.insert(
                    documents=[doc],
                    filters=doc.meta_data,
                    content_hash=doc.meta_data["content_hash"],
                )
                logger.info(
                    f"[IngestInsert] inserted_index={insert_idx} "
                    f"content_hash={doc.meta_data['content_hash'][:12]} "
                    f"file_name={doc.meta_data.get('file_name')}"
                )
            except Exception:
                logger.exception(
                    f"[IngestInsert] Failed to insert "
                    f"content_hash={doc.meta_data.get('content_hash', '')[:12]}"
                )
                raise

    logger.info(
        f"[IngestSummary] total_chunks={len(documents)} "
        f"inserted={len(new_documents)} skipped={skipped}"
    )


def build_local_kb(
    qdrant_url: str,
    qdrant_api_key: str,
    file_path: str,
    collection_name: str = DEFAULT_LOCAL_COLLECTION,
) -> Knowledge:
    """
    构建本地知识库：
    文本 -> 分块 -> hash 去重 -> 写入 local_legal_documents
    """
    logger.info(
        f"[LocalKB] build_local_kb started collection={collection_name} file_path={file_path}"
    )

    client, _, knowledge = init_qdrant_with_index(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
    )

    reader = build_text_reader()

    metadata = build_metadata(
        source_type="local_txt",
        file_name=os.path.basename(file_path),
        legal_topic="general_legal_principles",
        jurisdiction="general",
    )

    ingest_file_with_dedup(
        file_path=file_path,
        knowledge=knowledge,
        client=client,
        collection_name=collection_name,
        reader=reader,
        metadata=metadata,
    )

    logger.info("[LocalKB] build_local_kb finished")
    return knowledge


# =========================================================
# Retriever
# =========================================================
def create_local_kb_retriever(
    deepseek_api_key: str,
) -> Callable:
    """
    返回一个可直接传给 Agent(..., knowledge_retriever=...) 的 retriever
    内部带 query expander 缓存，不依赖 streamlit session_state
    """
    query_expander_agent: Optional[Agent] = None

    def parse_queries(text: str) -> List[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            ln2 = ln.lstrip("-.*").strip()
            if len(ln2) >= 3 and ln2[0].isdigit() and ln2[1] in [".", ")"]:
                ln2 = ln2[2:].strip()
            if ln2:
                cleaned.append(ln2)
        return cleaned

    def get_query_expander() -> Agent:
        nonlocal query_expander_agent

        if query_expander_agent is None:
            logger.info("[Retriever] Creating Query Expander agent")
            query_expander_agent = Agent(
                name="Query Expander",
                model=DeepSeek(
                    id="deepseek-chat",
                    api_key=deepseek_api_key,
                ),
                knowledge=None,
                search_knowledge=False,
                tools=[],
                instructions=[
                    "Generate multiple semantically similar legal search queries.",
                    "Preserve key legal terms from the original question.",
                    "Return one query per line.",
                    "Do not explain.",
                ],
                debug_mode=False,
                markdown=False,
            )
        return query_expander_agent

    def generate_queries(question: str, question_num: int = 4) -> List[str]:
        prompt = f"""
你是一个法律知识库检索系统的 Query 扩展器。
请围绕用户问题生成 {question_num} 个检索 query，用于提高召回率。

要求：
1. 保留原问题中的核心法律术语
2. 优先保持与原问题同一语义，不要泛化成过宽的问题
3. 每行输出一个 query
4. 不要编号
5. 不要解释

用户问题:
{question}
"""
        q_out = get_query_expander().run(prompt)
        queries = parse_queries(q_out.content or "")
        logger.info(f"[Retriever] Generated queries={queries}")
        return queries

    def local_kb_retriever(agent, query: str, num_documents: int = 3, **kwargs):
        logger.info(
            f"[Retriever] Start query={query[:200]} num_documents={num_documents}"
        )

        MIN_LEN = 30
        queries = [query] + generate_queries(query)

        # query 去重
        dedup_queries = []
        seen_q = set()
        for q in queries:
            q = q.strip()
            if q and q not in seen_q:
                seen_q.add(q)
                dedup_queries.append(q)

        queries = dedup_queries
        logger.info(f"[Retriever] total_queries={len(queries)} queries={queries}")

        vector_db = agent.knowledge.vector_db
        doc_map = {}

        for q_idx, q in enumerate(queries):
            logger.info(f"[Retriever] query_index={q_idx} query={q}")
            docs = vector_db.search(query=q, limit=num_documents)

            for rank, d in enumerate(docs, start=1):
                meta = getattr(d, "meta_data", {}) or {}
                content = getattr(d, "content", "") or ""

                logger.info(
                    f"[RetrieverResult] rank={rank} "
                    f"chunk={meta.get('chunk')} "
                    f"content_hash={meta.get('content_hash')} "
                    f"preview={content[:200]}"
                )

                if not content.strip():
                    continue
                if len(content.strip()) < MIN_LEN:
                    continue

                key = meta.get("content_hash") or content[:200]
                rank_score = num_documents - rank + 1

                if key not in doc_map:
                    doc_map[key] = {
                        "doc": d,
                        "score": rank_score,
                        "hits": 1,
                        "best_rank": rank,
                    }
                else:
                    doc_map[key]["score"] += rank_score
                    doc_map[key]["hits"] += 1
                    doc_map[key]["best_rank"] = min(
                        doc_map[key]["best_rank"], rank
                    )

        sorted_items = sorted(
            doc_map.values(),
            key=lambda x: (-x["score"], -x["hits"], x["best_rank"]),
        )

        uniq_docs = [item["doc"] for item in sorted_items[:num_documents]]

        for idx, item in enumerate(sorted_items[:num_documents], start=1):
            d = item["doc"]
            meta = getattr(d, "meta_data", {}) or {}
            logger.info(
                f"[RetrieverFusion] final_rank={idx} "
                f"chunk={meta.get('chunk')} "
                f"content_hash={meta.get('content_hash')} "
                f"score={item['score']} "
                f"hits={item['hits']} "
                f"best_rank={item['best_rank']}"
            )

        results = []
        for idx, r in enumerate(uniq_docs, start=1):
            content = getattr(r, "content", "") or ""
            metadata = getattr(r, "meta_data", {}) or {}

            file_name = metadata.get("file_name", "unknown")
            chunk = metadata.get("chunk", idx)
            content_hash = metadata.get("content_hash", "unknown")
            legal_topic = metadata.get("legal_topic", "unknown")
            jurisdiction = metadata.get("jurisdiction", "unknown")

            formatted_content = f"""
[Source]
file_name: {file_name}
chunk: {chunk}
content_hash: {content_hash}
legal_topic: {legal_topic}
jurisdiction: {jurisdiction}

[Content]
{content}
""".strip()

            results.append(
                {
                    "content": formatted_content,
                    "metadata": metadata,
                }
            )

        logger.info(f"[Retriever] Finished returned_docs={len(results)}")
        return results

    return local_kb_retriever


def base_local_kb_retriever(agent, query: str, num_documents: int = 3, **kwargs):
    """
    基础检索链路，不做 multi-query 扩展
    """
    logger.info(
        f"[BaseRetriever] Start query={query[:200]} num_documents={num_documents}"
    )

    vector_db = agent.knowledge.vector_db
    docs = vector_db.search(query=query, limit=num_documents)

    results = []
    for d in docs:
        content = getattr(d, "content", "") or ""
        metadata = getattr(d, "meta_data", {}) or {}

        if not content.strip():
            continue

        results.append(
            {
                "content": content,
                "metadata": metadata,
            }
        )

    logger.info(f"[BaseRetriever] Finished returned_docs={len(results)}")
    return results