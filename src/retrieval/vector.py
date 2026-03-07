from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


def vector_search(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """Return top-k chunks with similarity scores (lower distance = more similar)."""
    return vectorstore.similarity_search_with_score(query, k=k)
