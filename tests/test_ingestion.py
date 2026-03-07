import os
import tempfile

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents


# ── chunker ────────────────────────────────────────────────────────────────────

def _make_docs(text: str) -> list[Document]:
    return [Document(page_content=text, metadata={"source": "test"})]


def test_chunk_assigns_chunk_ids():
    docs = _make_docs("Word " * 300)
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == [f"chunk_{i}" for i in range(len(chunks))]


def test_chunk_ids_are_unique():
    docs = _make_docs("Word " * 500)
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_respects_size():
    docs = _make_docs("A" * 2000)
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=0)
    for chunk in chunks:
        assert len(chunk.page_content) <= 210  # small tolerance for splitter


def test_chunk_preserves_metadata():
    docs = _make_docs("Hello world")
    docs[0].metadata["source_file"] = "test.txt"
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=0)
    for chunk in chunks:
        assert chunk.metadata.get("source_file") == "test.txt"


# ── loader ────────────────────────────────────────────────────────────────────

def test_load_txt_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello from a text file.\nSecond line.")
        path = f.name
    try:
        docs = load_documents([path])
        assert len(docs) >= 1
        assert "Hello from a text file" in docs[0].page_content
        assert docs[0].metadata["source_file"] == os.path.basename(path)
    finally:
        os.unlink(path)


def test_load_md_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Title\n\nSome markdown content.")
        path = f.name
    try:
        docs = load_documents([path])
        assert len(docs) >= 1
        assert "markdown content" in docs[0].page_content
    finally:
        os.unlink(path)


def test_load_unsupported_file_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_documents(["/tmp/fakefile.csv"])
