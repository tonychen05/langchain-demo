"""Build or load a persistent index for the OSU engineering strategic plan."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAIError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

DEFAULT_PDF = Path(__file__).with_name("Engineering_Strategic-Plan_2018.pdf")
CACHE_DIR = Path(__file__).with_name(".rag-cache")
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300


def load_chunks(pdf_path: Path) -> list[Document]:
    pages = []
    for number, page in enumerate(PdfReader(pdf_path).pages, start=1):
        # Preserve the plan's strategy/metric table columns during extraction.
        text = page.extract_text(extraction_mode="layout")
        if text and text.strip():
            pages.append(Document(
                page_content=text,
                metadata={"source": pdf_path.name, "page": number},
            ))
    if not pages:
        raise ValueError("The PDF has no extractable text; a scanned PDF needs OCR.")
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(pages)


def get_index(pdf_path: Path, cache_dir: Path = CACHE_DIR) -> InMemoryVectorStore:
    """Reuse cached document vectors, rebuilding when the PDF/settings change."""
    settings = {
        "format": 1,
        "source": pdf_path.name,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "extraction_mode": "layout",
        "pypdf": version("pypdf"),
        "splitters": version("langchain-text-splitters"),
        "vectorstore": version("langchain-core"),
    }
    digest = hashlib.sha256(pdf_path.read_bytes())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    cache_path = cache_dir / f"{digest.hexdigest()}.json"
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if cache_path.is_file():
        try:
            store = InMemoryVectorStore.load(str(cache_path), embeddings)
            if not isinstance(store.store, dict) or not store.store:
                raise ValueError("Empty or invalid cached index.")
        except (ValueError, TypeError, KeyError):
            print("Cached index is unreadable; rebuilding it.")
        else:
            print(f"Loaded cached index for {pdf_path.name}.")
            return store

    chunks = load_chunks(pdf_path)
    print(f"Indexing {len(chunks)} chunks from {pdf_path.name}...")
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents interrupted writes from leaving a partial index.
    with NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        store.dump(str(temporary))
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved index to {cache_path}.")
    return store


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args(argv)
    load_dotenv(Path(__file__).with_name(".env"))
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        parser.error("Set OPENAI_API_KEY in .env or your environment.")
    try:
        get_index(args.pdf)
    except KeyboardInterrupt:
        print("\nIndexing cancelled.")
    except (OSError, PdfReadError, ValueError, OpenAIError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
