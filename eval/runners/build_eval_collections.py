import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = PROJECT_DIR / "eval" / "support"
SRC_DIR = PROJECT_DIR / "src"
for path in (PROJECT_DIR, RUNNER_DIR, SUPPORT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.kb_service import collection_exists
from src.logging_config import setup_logging
from provider_agent_service import get_online_collection_name, get_online_service, require_env

ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)
load_dotenv()

LOGGER = setup_logging(
    log_dir=str(PROJECT_DIR / "logs"),
    log_file="project_eval.log",
    logger_name="legal_app",
)

INDEX_PATH = PROJECT_DIR / "eval" / "cases" / "eval_data" / "index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prebuild reusable Qdrant collections for the 10 online eval documents."
    )
    parser.add_argument(
        "--doc-index",
        type=int,
        default=0,
        help="Optional 1-based document index to build. 0 means all online documents.",
    )
    return parser.parse_args()


def load_eval_index() -> list[dict]:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    entries = load_eval_index()
    if args.doc_index > 0:
        if args.doc_index > len(entries):
            raise IndexError(
                f"doc_index out of range: {args.doc_index} (valid: 1-{len(entries)})"
            )
        entries = [entries[args.doc_index - 1]]

    qdrant_url = require_env("QDRANT_URL")
    qdrant_api_key = require_env("QDRANT_API_KEY")

    print(f"Preparing online eval collections for {len(entries)} document(s)...", flush=True)
    for entry in entries:
        contract_path = entry["path"]
        rank = int(entry["rank"])
        collection_name = get_online_collection_name(contract_path)
        exists_before = collection_exists(qdrant_url, qdrant_api_key, collection_name)

        print(
            f"[{rank:02d}] {Path(contract_path).name} -> {collection_name}",
            flush=True,
        )
        if exists_before:
            print("  -> collection already exists, loading for reuse...", flush=True)
        else:
            print("  -> building collection in Qdrant...", flush=True)

        started = time.time()
        get_online_service(contract_path)
        elapsed = time.time() - started

        status = "reused" if exists_before else "built"
        LOGGER.info(
            "[PromptfooPrebuild] rank=%s collection=%s status=%s elapsed=%.3fs",
            rank,
            collection_name,
            status,
            elapsed,
        )
        print(f"  -> {status} in {elapsed:.1f}s", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
