import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def embed_and_store(
    chunks: List[Document],
    persist_dir: str = "./data/chroma",
) -> Chroma:
    """Embed chunks and persist them in ChromaDB."""
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(persist_dir: str = "./data/chroma") -> Chroma:
    """Load an existing ChromaDB vectorstore."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
