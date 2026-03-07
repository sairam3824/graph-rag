import json
import re
from typing import Dict, List, Optional

import networkx as nx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from thefuzz import fuzz

ENTITY_EXTRACTION_PROMPT = """List the entity names explicitly mentioned in this query.
Return ONLY a JSON array of strings. Example: ["Alice", "OpenAI", "transformer"]

Query: {query}"""


def extract_query_entities(
    query: str,
    llm: Optional[ChatOpenAI] = None,
) -> List[str]:
    """Extract entity names from a user query using the LLM."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def find_best_node(G: nx.Graph, entity: str, threshold: int = 75) -> Optional[str]:
    """Fuzzy-match an entity name to the closest graph node."""
    best_node, best_score = None, 0
    for node in G.nodes():
        score = fuzz.ratio(entity.lower(), node.lower())
        if score > best_score and score >= threshold:
            best_score = score
            best_node = node
    return best_node


def graph_traversal(
    G: nx.Graph,
    query_entities: List[str],
    hops: int = 2,
) -> Dict:
    """
    BFS from each query entity up to `hops` hops.
    Returns connected entities, relationships found, and chunk IDs to fetch.
    """
    visited: set[str] = set()
    relationships: List[Dict] = []
    seen_edges: set[tuple] = set()

    for entity in query_entities:
        matched = find_best_node(G, entity)
        if not matched:
            continue

        frontier = {matched}
        visited.add(matched)

        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in G.neighbors(node):
                    edge_data = G.edges[node, neighbor]
                    edge_key = tuple(sorted([node, neighbor]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        relationships.append({
                            "source": node,
                            "target": neighbor,
                            "relation": edge_data.get("relation", "related to"),
                        })
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

    # Collect chunk IDs from all visited nodes
    chunk_ids: set[str] = set()
    for node in visited:
        chunk_ids.update(G.nodes[node].get("chunks", []))

    return {
        "entities": list(visited),
        "relationships": relationships,
        "chunk_ids": list(chunk_ids),
    }


def get_shortest_path(
    G: nx.Graph,
    entity_a: str,
    entity_b: str,
) -> List[str]:
    """Return shortest path nodes between two entities (empty list if none)."""
    node_a = find_best_node(G, entity_a)
    node_b = find_best_node(G, entity_b)
    if not node_a or not node_b:
        return []
    try:
        return nx.shortest_path(G, node_a, node_b)
    except nx.NetworkXNoPath:
        return []
