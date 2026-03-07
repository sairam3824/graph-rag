import networkx as nx
from thefuzz import fuzz


def resolve_entities(G: nx.Graph, threshold: int = 88) -> nx.Graph:
    """
    Merge near-duplicate entity nodes using fuzzy string matching.
    The node that appears first (alphabetically) becomes the canonical form.
    """
    nodes = sorted(G.nodes())
    merge_map: dict[str, str] = {}  # duplicate -> canonical

    for i, n1 in enumerate(nodes):
        canonical = merge_map.get(n1, n1)
        for n2 in nodes[i + 1:]:
            if n2 in merge_map:
                continue
            score = fuzz.ratio(n1.lower(), n2.lower())
            if score >= threshold:
                merge_map[n2] = canonical

    # Apply merges
    for duplicate, canonical in merge_map.items():
        if not G.has_node(duplicate):
            continue
        # Redirect edges
        for neighbor in list(G.neighbors(duplicate)):
            if neighbor == canonical:
                continue
            edge_data = G.edges[duplicate, neighbor]
            if not G.has_edge(canonical, neighbor):
                G.add_edge(canonical, neighbor, **edge_data)
        # Merge chunk references
        dup_chunks = G.nodes[duplicate].get("chunks", [])
        canon_chunks = G.nodes[canonical].setdefault("chunks", [])
        for cid in dup_chunks:
            if cid not in canon_chunks:
                canon_chunks.append(cid)
        G.remove_node(duplicate)

    return G
