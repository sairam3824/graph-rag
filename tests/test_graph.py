import json
import os
import tempfile

import networkx as nx
import pytest

from src.graph.builder import add_extraction_to_graph
from src.graph.resolver import resolve_entities
from src.graph.store import save_graph, load_graph
from src.retrieval.graph_search import find_best_node, graph_traversal, get_shortest_path


# ── builder ────────────────────────────────────────────────────────────────────

def test_add_extraction_adds_nodes():
    G = nx.Graph()
    extraction = {
        "entities": [
            {"name": "Alice", "type": "person"},
            {"name": "OpenAI", "type": "organization"},
        ],
        "relationships": [
            {"source": "Alice", "target": "OpenAI", "relation": "works at"},
        ],
    }
    names = add_extraction_to_graph(G, extraction, "chunk_0")
    assert "Alice" in G
    assert "OpenAI" in G
    assert G.has_edge("Alice", "OpenAI")
    assert G.edges["Alice", "OpenAI"]["relation"] == "works at"
    assert "chunk_0" in G.nodes["Alice"]["chunks"]
    assert set(names) == {"Alice", "OpenAI"}


def test_add_extraction_skips_empty_names():
    G = nx.Graph()
    extraction = {
        "entities": [{"name": "", "type": "person"}, {"name": "Bob", "type": "person"}],
        "relationships": [],
    }
    add_extraction_to_graph(G, extraction, "chunk_0")
    assert list(G.nodes()) == ["Bob"]


def test_add_extraction_deduplicates_chunk_refs():
    G = nx.Graph()
    extraction = {"entities": [{"name": "X", "type": "concept"}], "relationships": []}
    add_extraction_to_graph(G, extraction, "chunk_0")
    add_extraction_to_graph(G, extraction, "chunk_0")
    assert G.nodes["X"]["chunks"].count("chunk_0") == 1


def test_add_extraction_no_duplicate_edges():
    G = nx.Graph()
    e = {
        "entities": [{"name": "A", "type": "other"}, {"name": "B", "type": "other"}],
        "relationships": [{"source": "A", "target": "B", "relation": "rel1"}],
    }
    add_extraction_to_graph(G, e, "c0")
    e2 = {
        "entities": [],
        "relationships": [{"source": "A", "target": "B", "relation": "rel2"}],
    }
    add_extraction_to_graph(G, e2, "c1")
    # First relation wins
    assert G.edges["A", "B"]["relation"] == "rel1"


def test_add_extraction_auto_creates_relationship_nodes():
    G = nx.Graph()
    extraction = {
        "entities": [],
        "relationships": [{"source": "X", "target": "Y", "relation": "linked"}],
    }
    add_extraction_to_graph(G, extraction, "chunk_0")
    assert "X" in G and "Y" in G


# ── resolver ───────────────────────────────────────────────────────────────────

def test_resolve_merges_near_duplicates():
    G = nx.Graph()
    G.add_node("OpenAI", type="organization", chunks=["c0"])
    G.add_node("openai", type="organization", chunks=["c1"])
    G.add_edge("OpenAI", "GPT-4", relation="developed")
    G.add_edge("openai", "ChatGPT", relation="powers")
    G = resolve_entities(G, threshold=88)
    nodes = list(G.nodes())
    # One of the two should survive
    assert len([n for n in nodes if n.lower() == "openai"]) == 1


def test_resolve_keeps_distinct_entities():
    G = nx.Graph()
    G.add_node("Alice", type="person", chunks=[])
    G.add_node("Bob", type="person", chunks=[])
    G = resolve_entities(G)
    assert "Alice" in G
    assert "Bob" in G


def test_resolve_merges_chunk_references():
    G = nx.Graph()
    G.add_node("OpenAI", type="organization", chunks=["c0"])
    G.add_node("openai", type="organization", chunks=["c1"])
    G = resolve_entities(G, threshold=88)
    canonical = [n for n in G.nodes() if n.lower() == "openai"][0]
    assert set(G.nodes[canonical]["chunks"]) == {"c0", "c1"}


# ── store ──────────────────────────────────────────────────────────────────────

def test_save_and_load_graph_roundtrip():
    G = nx.Graph()
    G.add_node("Alice", type="person", chunks=["c0"])
    G.add_node("ACME", type="organization", chunks=[])
    G.add_edge("Alice", "ACME", relation="works at")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "graph.json")
        save_graph(G, path)
        G2 = load_graph(path)

    assert set(G2.nodes()) == {"Alice", "ACME"}
    assert G2.has_edge("Alice", "ACME")
    assert G2.edges["Alice", "ACME"]["relation"] == "works at"
    assert G2.nodes["Alice"]["chunks"] == ["c0"]


def test_load_graph_returns_empty_when_missing():
    G = load_graph("/nonexistent/path/graph.json")
    assert G.number_of_nodes() == 0


# ── graph_search ───────────────────────────────────────────────────────────────

def _sample_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_node("OpenAI", type="organization", chunks=["c0"])
    G.add_node("Sam Altman", type="person", chunks=["c0"])
    G.add_node("GPT-4", type="technology", chunks=["c1"])
    G.add_node("Microsoft", type="organization", chunks=["c2"])
    G.add_edge("Sam Altman", "OpenAI", relation="CEO of")
    G.add_edge("OpenAI", "GPT-4", relation="developed")
    G.add_edge("OpenAI", "Microsoft", relation="partnered with")
    return G


def test_find_best_node_exact_match():
    G = _sample_graph()
    assert find_best_node(G, "OpenAI") == "OpenAI"


def test_find_best_node_fuzzy_match():
    G = _sample_graph()
    result = find_best_node(G, "open ai")
    assert result == "OpenAI"


def test_find_best_node_no_match():
    G = _sample_graph()
    assert find_best_node(G, "zzznomatch") is None


def test_graph_traversal_returns_neighbors():
    G = _sample_graph()
    result = graph_traversal(G, ["OpenAI"], hops=1)
    assert "OpenAI" in result["entities"]
    assert "GPT-4" in result["entities"]
    assert "Sam Altman" in result["entities"]
    assert "Microsoft" in result["entities"]
    assert len(result["relationships"]) > 0


def test_graph_traversal_empty_entities():
    G = _sample_graph()
    result = graph_traversal(G, [], hops=2)
    assert result["entities"] == []
    assert result["relationships"] == []


def test_graph_traversal_collects_chunk_ids():
    G = _sample_graph()
    result = graph_traversal(G, ["OpenAI"], hops=1)
    assert "c0" in result["chunk_ids"]
    assert "c1" in result["chunk_ids"]


def test_get_shortest_path_direct():
    G = _sample_graph()
    path = get_shortest_path(G, "OpenAI", "GPT-4")
    assert path == ["OpenAI", "GPT-4"]


def test_get_shortest_path_multi_hop():
    G = _sample_graph()
    path = get_shortest_path(G, "Sam Altman", "GPT-4")
    assert path[0] == "Sam Altman"
    assert path[-1] == "GPT-4"
    assert len(path) == 3


def test_get_shortest_path_no_connection():
    G = nx.Graph()
    G.add_node("A", type="other", chunks=[])
    G.add_node("B", type="other", chunks=[])
    path = get_shortest_path(G, "A", "B")
    assert path == []


def test_get_shortest_path_unknown_entity():
    G = _sample_graph()
    path = get_shortest_path(G, "OpenAI", "Unknown Entity XYZ")
    assert path == []
