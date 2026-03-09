from typing import Dict, List

import networkx as nx


def add_extraction_to_graph(
    G: nx.Graph,
    extraction: Dict,
    chunk_id: str,
) -> List[str]:
    """
    Add entities and relationships from one chunk extraction to the graph.
    Returns the list of entity names found in this chunk.
    """
    entity_names = []

    for entity in extraction.get("entities", []):
        name = entity.get("name", "").strip()
        if not name:
            continue
        entity_names.append(name)
        if not G.has_node(name):
            G.add_node(name, type=entity.get("type", "other"), chunks=[])
        # Record that this chunk references this entity
        chunks = G.nodes[name].setdefault("chunks", [])
        if chunk_id not in chunks:
            chunks.append(chunk_id)

    for rel in extraction.get("relationships", []):
        source = rel.get("source", "").strip()
        target = rel.get("target", "").strip()
        relation = rel.get("relation", "").strip()
        if not source or not target:
            continue
        # Ensure nodes exist even if not in entities list
        for node_name in [source, target]:
            if not G.has_node(node_name):
                G.add_node(node_name, type="other", chunks=[])
            
            # Record that this chunk references this entity
            chunks = G.nodes[node_name].setdefault("chunks", [])
            if chunk_id not in chunks:
                chunks.append(chunk_id)

        # Add or update edge (keep first relation if already exists)
        if not G.has_edge(source, target):
            G.add_edge(source, target, relation=relation)

    return entity_names
