import pytest
from langchain_core.documents import Document

from src.retrieval.reranker import rerank_results


def _doc(text: str, source: str = "test.txt") -> Document:
    return Document(page_content=text, metadata={"source_file": source})


# ── reranker ──────────────────────────────────────────────────────────────────

def test_rerank_deduplicates():
    doc = _doc("shared content about OpenAI and transformers")
    vector_results = [(doc, 0.2)]
    graph_docs = [doc]  # same doc
    ranked = rerank_results(vector_results, graph_docs, ["OpenAI"])
    assert len(ranked) == 1


def test_rerank_vector_scores_first():
    doc_v = _doc("vector result content")
    doc_g = _doc("graph result content")
    vector_results = [(doc_v, 0.1)]  # distance 0.1 → sim 0.9
    graph_docs = [doc_g]
    ranked = rerank_results(vector_results, graph_docs, [])
    # vector doc should score higher (0.9 > 0.45)
    assert ranked[0][0].page_content == doc_v.page_content


def test_rerank_entity_boost():
    doc_g = _doc("content mentioning OpenAI and GPT")
    vector_results = [(doc_g, 0.6)]  # sim = 0.4
    graph_docs = [_doc("entity-rich OpenAI GPT document")]
    entities = ["OpenAI", "GPT"]
    ranked = rerank_results(vector_results, graph_docs, entities)
    scores = [score for _, score, _ in ranked]
    assert all(s >= 0.0 for s in scores)


def test_rerank_empty_inputs():
    ranked = rerank_results([], [], [])
    assert ranked == []


def test_rerank_source_labels():
    doc_v = _doc("vector doc")
    doc_g = _doc("graph doc")
    vector_results = [(doc_v, 0.3)]
    graph_docs = [doc_g]
    ranked = rerank_results(vector_results, graph_docs, [])
    sources = {r[2] for r in ranked}
    assert "vector" in sources
    assert "graph" in sources


def test_rerank_sorted_descending():
    docs = [(_doc(f"doc {i}"), float(i) * 0.1) for i in range(5)]
    ranked = rerank_results(docs, [], [])
    scores = [r[1] for r in ranked]
    assert scores == sorted(scores, reverse=True)
