import os
from typing import List, Optional

import networkx as nx
from pyvis.network import Network

TYPE_COLORS = {
    "person": "#e94560",
    "organization": "#4361ee",
    "technology": "#7209b7",
    "concept": "#f77f00",
    "location": "#2dc653",
    "other": "#6c757d",
}


def create_graph_html(
    G: nx.Graph,
    output_path: str = "./data/graph.html",
    highlight_entities: Optional[List[str]] = None,
    filter_relation: Optional[str] = None,
    height: str = "600px",
) -> str:
    """
    Generate an interactive pyvis HTML visualization of the knowledge graph.
    Returns the path to the generated HTML file.
    """
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0d1117",
        font_color="#c9d1d9",
        directed=False,
    )
    net.set_options("""{
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "stabilization": {"iterations": 200}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "color": {"inherit": "from"},
        "smooth": {"type": "continuous"}
      }
    }""")

    highlight_set = set(highlight_entities or [])

    for node, data in G.nodes(data=True):
        entity_type = data.get("type", "other")
        chunks = data.get("chunks", [])
        color = "#FFD700" if node in highlight_set else TYPE_COLORS.get(entity_type, TYPE_COLORS["other"])
        size = 12 + min(len(chunks) * 3, 20)
        title = (
            f"<b>{node}</b><br>"
            f"Type: <i>{entity_type}</i><br>"
            f"Referenced in {len(chunks)} chunk(s)"
        )
        net.add_node(
            node,
            label=node,
            color=color,
            title=title,
            size=size,
            font={"size": 12},
        )

    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "")
        if filter_relation and filter_relation.lower() not in relation.lower():
            continue
        net.add_edge(
            u, v,
            title=relation,
            label=relation[:40] if relation else "",
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    net.save_graph(output_path)
    return output_path


def get_graph_stats(G: nx.Graph) -> dict:
    """Return basic graph statistics."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_connected_components(G),
        "density": round(nx.density(G), 4),
        "top_entities": sorted(
            G.degree(), key=lambda x: x[1], reverse=True
        )[:10],
    }
