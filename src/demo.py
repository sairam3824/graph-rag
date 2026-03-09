"""
Graph RAG — Streamlit Demo
Run: streamlit run src/demo.py
"""

import os
import shutil
import tempfile
from pathlib import Path

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store, load_vectorstore
from graph.extractor import extract_entities_and_relations
from graph.builder import add_extraction_to_graph
from graph.resolver import resolve_entities
from graph.store import save_graph, load_graph
from retrieval.hybrid import hybrid_retrieve
from retrieval.graph_search import get_shortest_path
from generation.generator import generate_answer
from viz.graph_visualizer import create_graph_html, get_graph_stats

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
GRAPH_PATH = str(DATA_DIR / "graph.json")
CHROMA_DIR = str(DATA_DIR / "chroma")
VIZ_PATH = str(DATA_DIR / "graph.html")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graph RAG",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🕸️ Graph RAG — Knowledge Graph-Enhanced Retrieval")
st.caption("Vector search finds similar text. Graph RAG finds *connected* knowledge.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI API key (stored in session only)",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top-K results", 3, 10, 5)
    hops = st.slider("Graph traversal hops", 1, 4, 2)
    chunk_size = st.slider("Chunk size (tokens)", 200, 1000, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, chunk_size - 50, 50, step=10)
    if chunk_overlap >= chunk_size:
        st.warning("Chunk overlap must be less than chunk size.")

    st.divider()
    if st.button("🗑️ Clear all data", type="secondary"):
        for p in [CHROMA_DIR, GRAPH_PATH, VIZ_PATH]:
            if os.path.exists(p):
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Data cleared.")
        st.rerun()

# ── Session state init ─────────────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = load_graph(GRAPH_PATH)
if "vectorstore" not in st.session_state:
    if os.path.exists(CHROMA_DIR) and api_key:
        try:
            st.session_state.vectorstore = load_vectorstore(CHROMA_DIR)
        except Exception:
            st.session_state.vectorstore = None
    else:
        st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = (
        st.session_state.vectorstore is not None
        and st.session_state.graph.number_of_nodes() > 0
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_ingest, tab_graph, tab_chat = st.tabs(
    ["📄 Ingest Documents", "🕸️ Knowledge Graph", "💬 Chat"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.subheader("Upload Documents")
    st.info(
        "Upload PDF, TXT, or MD files. The pipeline will chunk them, "
        "embed them into ChromaDB, and extract a knowledge graph."
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🚀 Run Ingestion Pipeline", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            tmp_paths = []
            try:
                with st.status("Running ingestion pipeline...", expanded=True) as status:
                    # Save uploads to temp files
                    for uf in uploaded_files:
                        suffix = Path(uf.name).suffix
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tmp.write(uf.read())
                        tmp.close()
                        tmp_paths.append(tmp.name)

                    st.write(f"**Step 1/4** — Loading {len(tmp_paths)} file(s)...")
                    docs = load_documents(tmp_paths)
                    st.write(f"  Loaded {len(docs)} document(s).")

                    st.write("**Step 2/4** — Chunking...")
                    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.write(f"  Created {len(chunks)} chunk(s).")

                    st.write("**Step 3/4** — Embedding into ChromaDB...")
                    # Clear existing collection to avoid duplicate documents on re-ingestion
                    if os.path.exists(CHROMA_DIR):
                        shutil.rmtree(CHROMA_DIR)
                    vectorstore = embed_and_store(chunks, persist_dir=CHROMA_DIR)
                    st.session_state.vectorstore = vectorstore
                    st.write("  Embeddings stored.")

                    st.write("**Step 4/4** — Extracting knowledge graph...")
                    # Reset graph for fresh ingestion
                    G = nx.Graph()
                    st.session_state.graph = G
                    progress = st.progress(0.0, text="Extracting entities...")
                    for i, chunk in enumerate(chunks):
                        extraction = extract_entities_and_relations(chunk.page_content, llm)
                        add_extraction_to_graph(G, extraction, chunk.metadata["chunk_id"])
                        progress.progress(
                            (i + 1) / len(chunks),
                            text=f"Chunk {i+1}/{len(chunks)} — {G.number_of_nodes()} entities found",
                        )

                    st.write("  Resolving duplicate entities...")
                    G = resolve_entities(G)
                    save_graph(G, GRAPH_PATH)
                    st.session_state.graph = G

                    st.session_state.ingestion_done = True
                    status.update(label="Ingestion complete!", state="complete")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    if st.session_state.ingestion_done:
        st.success("Documents are ingested and ready for querying.")
        stats = get_graph_stats(st.session_state.graph)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Graph Nodes", stats["nodes"])
        col2.metric("Graph Edges", stats["edges"])
        col3.metric("Components", stats["components"])
        col4.metric("Density", stats["density"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.subheader("Interactive Knowledge Graph")

    G: nx.Graph = st.session_state.graph

    if G.number_of_nodes() == 0:
        st.info("No graph data yet. Ingest documents first.")
    else:
        col_controls, col_legend = st.columns([3, 1])

        with col_controls:
            search_entity = st.text_input(
                "Highlight entity", placeholder="e.g. OpenAI"
            )
            filter_rel = st.text_input(
                "Filter by relationship type", placeholder="e.g. founded"
            )

        with col_legend:
            st.markdown(
                "**Legend**\n"
                "- 🔴 Person\n"
                "- 🔵 Organization\n"
                "- 🟣 Technology\n"
                "- 🟠 Concept\n"
                "- 🟢 Location\n"
                "- ⚫ Other\n"
                "- 🟡 Highlighted"
            )

        highlight = [search_entity] if search_entity else []
        html_path = create_graph_html(
            G,
            output_path=VIZ_PATH,
            highlight_entities=highlight,
            filter_relation=filter_rel or None,
        )

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=620, scrolling=False)

        # Entity details
        if search_entity and search_entity in G:
            st.subheader(f"Details: {search_entity}")
            node_data = G.nodes[search_entity]
            neighbors = list(G.neighbors(search_entity))
            st.write(f"**Type:** {node_data.get('type', 'unknown')}")
            st.write(f"**Connected entities ({len(neighbors)}):** {', '.join(neighbors[:20])}")
            rels = [
                f"{search_entity} --[{G.edges[search_entity, n].get('relation', '')}]--> {n}"
                for n in neighbors
            ]
            if rels:
                st.code("\n".join(rels))

        # Shortest path finder
        with st.expander("Find shortest path between two entities"):
            col_a, col_b = st.columns(2)
            with col_a:
                entity_a = st.text_input("Entity A", placeholder="e.g. OpenAI", key="path_a")
            with col_b:
                entity_b = st.text_input("Entity B", placeholder="e.g. Anthropic", key="path_b")
            if st.button("Find path", key="btn_path"):
                if entity_a and entity_b:
                    path = get_shortest_path(G, entity_a, entity_b)
                    if path:
                        st.success(f"Path length: {len(path) - 1} hop(s)")
                        path_parts = []
                        for i, node in enumerate(path):
                            path_parts.append(f"**{node}**")
                            if i < len(path) - 1:
                                rel = G.edges[path[i], path[i + 1]].get("relation", "→")
                                path_parts.append(f"--[{rel}]-->")
                        st.write(" ".join(path_parts))
                    else:
                        st.warning(f"No path found between '{entity_a}' and '{entity_b}'.")
                else:
                    st.warning("Enter both entity names.")

        # Top entities table
        with st.expander("Top entities by connections"):
            stats = get_graph_stats(G)
            for name, degree in stats["top_entities"]:
                st.write(f"  **{name}** — {degree} connection(s)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Ask Questions")

    if not st.session_state.ingestion_done:
        st.info("Ingest documents first before chatting.")
    else:
        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            show_graph_ctx = st.toggle("Show graph context", value=True)
        with col_opts2:
            compare_mode = st.toggle("Compare vector-only vs graph-enhanced", value=False)

        # Display chat history
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                if compare_mode and "vector_answer" in entry:
                    col_v, col_g = st.columns(2)
                    with col_v:
                        st.caption("Vector-only")
                        st.write(entry["vector_answer"])
                    with col_g:
                        st.caption("Graph-enhanced")
                        st.write(entry["graph_answer"])
                else:
                    st.write(entry.get("graph_answer", entry.get("answer", "")))

                if show_graph_ctx and entry.get("graph_context"):
                    ctx = entry["graph_context"]
                    with st.expander("Graph context"):
                        st.write(f"**Query entities:** {', '.join(entry.get('query_entities', []))}")
                        st.write(f"**Connected entities:** {', '.join(ctx.get('entities', [])[:15])}")
                        rels = ctx.get("relationships", [])[:15]
                        if rels:
                            st.write("**Relationships:**")
                            for r in rels:
                                st.write(f"  • {r['source']} --[{r['relation']}]--> {r['target']}")

        # Input
        question = st.chat_input("Ask a question about your documents...")
        if question:
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif st.session_state.vectorstore is None:
                st.error("Vectorstore not available. Please re-run ingestion.")
            else:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                vectorstore = st.session_state.vectorstore
                G = st.session_state.graph

                success = False
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        result = hybrid_retrieve(
                            question,
                            vectorstore,
                            G,
                            llm=llm,
                            top_k=top_k,
                            hops=hops,
                        )

                        graph_answer = generate_answer(
                            question,
                            result["docs"],
                            graph_context=result["graph_context"],
                            llm=llm,
                            use_graph=True,
                        )

                        entry = {
                            "question": question,
                            "graph_answer": graph_answer,
                            "graph_context": result["graph_context"],
                            "query_entities": result["query_entities"],
                        }

                        if compare_mode:
                            vector_docs = result["vector_docs"]
                            vector_answer = generate_answer(
                                question,
                                vector_docs,
                                graph_context=None,
                                llm=llm,
                                use_graph=False,
                            )
                            entry["vector_answer"] = vector_answer

                        st.session_state.chat_history.append(entry)
                        success = True
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")

                if success:
                    st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()
