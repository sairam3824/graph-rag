from typing import Dict, List, Optional

import networkx as nx
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from .vector import vector_search
from .graph_search import extract_query_entities, graph_traversal
from .reranker import rerank_results


def hybrid_retrieve(
    query: str,
    vectorstore: Chroma,
    G: nx.Graph,
    llm: Optional[ChatOpenAI] = None,
    top_k: int = 5,
    hops: int = 2,
) -> Dict:
    """
    Perform hybrid retrieval:
      1. Vector similarity search
      2. Entity extraction + graph traversal
      3. Graph-entity-based vector search
      4. Merge and re-rank
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Step 1: Vector search
    vector_results = vector_search(vectorstore, query, k=top_k)

    # Step 2: Query entity extraction + graph traversal
    query_entities = extract_query_entities(query, llm)
    graph_context = graph_traversal(G, query_entities, hops=hops)

    # Step 3: Retrieve chunks for graph-connected entities
    graph_docs: List[Document] = []
    if graph_context["entities"]:
        entity_query = " ".join(graph_context["entities"][:15])
        graph_results = vector_search(vectorstore, entity_query, k=top_k)
        graph_docs = [doc for doc, _ in graph_results]

    # Step 4: Merge and re-rank
    ranked = rerank_results(vector_results, graph_docs, graph_context["entities"])

    return {
        "docs": [doc for doc, _, _ in ranked[:top_k]],
        "vector_docs": [doc for doc, _ in vector_results],
        "graph_context": graph_context,
        "query_entities": query_entities,
    }
