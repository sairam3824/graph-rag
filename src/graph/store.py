import json
import os

import networkx as nx
from networkx.readwrite import json_graph


def save_graph(G: nx.Graph, path: str = "./data/graph.json") -> None:
    """Persist the NetworkX graph to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # NetworkX 3.4+ requires explicit `edges` parameter; older versions don't support it
    try:
        data = json_graph.node_link_data(G, edges="links")
    except TypeError:
        data = json_graph.node_link_data(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str = "./data/graph.json") -> nx.Graph:
    """Load a previously saved NetworkX graph from JSON."""
    if not os.path.exists(path):
        return nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return json_graph.node_link_graph(data, edges="links")
    except TypeError:
        return json_graph.node_link_graph(data)
