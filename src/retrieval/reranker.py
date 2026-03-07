from typing import List, Tuple

from langchain_core.documents import Document


def rerank_results(
    vector_results: List[Tuple[Document, float]],
    graph_docs: List[Document],
    graph_entities: List[str],
) -> List[Tuple[Document, float, str]]:
    """
    Merge vector and graph retrieval results, deduplicate, and re-rank.
    Returns list of (doc, score, source) sorted by score descending.
    """
    seen: set[str] = set()
    ranked: List[Tuple[Document, float, str]] = []

    for doc, distance in vector_results:
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        # Convert L2 distance to a [0,1] similarity score
        sim = max(0.0, 1.0 - distance)
        ranked.append((doc, sim, "vector"))

    entity_set = {e.lower() for e in graph_entities}
    for doc in graph_docs:
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        text_lower = doc.page_content.lower()
        # Base score + entity mention boost
        score = 0.45 + min(0.45, sum(0.1 for e in entity_set if e in text_lower))
        ranked.append((doc, score, "graph"))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
