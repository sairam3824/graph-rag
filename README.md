<div align="center">

# 🕸️ Graph RAG

### Knowledge Graph-Enhanced Retrieval Augmented Generation

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>

<br/>

> **Vector search finds similar text. Graph RAG finds *connected* knowledge.**

Standard RAG retrieves text chunks by similarity. Graph RAG **also** extracts a knowledge graph from your documents and traverses entity relationships — so you can answer questions that vector search alone can never touch.

---

</div>

## ✨ What makes this different?

| Question | Vector RAG | Graph RAG |
|----------|-----------|-----------|
| *"How is OpenAI related to Anthropic?"* | ❌ Finds docs mentioning both, misses the connection | ✅ Traverses the shared-founder path in the graph |
| *"What technologies does Google DeepMind use?"* | ❌ Returns similar chunks | ✅ Fetches all graph neighbors of "Google DeepMind" |
| *"Trace the link between Llama and NVIDIA"* | ❌ Cannot reason over hops | ✅ Finds the shortest path through the knowledge graph |
| *"All orgs mentioned alongside Transformers?"* | ❌ Keyword overlap | ✅ 2-hop graph neighbourhood of "Transformer" |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│                                                                 │
│  PDF / TXT / MD  ──►  Chunker  ──►  ChromaDB (embeddings)      │
│                           │                                     │
│                           └──►  LLM Extractor                  │
│                                  (entities + relations)         │
│                                       │                         │
│                                  NetworkX Graph                 │
│                              (fuzzy entity resolution)          │
│                                       │                         │
│                                  graph.json                     │
└───────────────────────────────────────┼─────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────┐
│                       HYBRID RETRIEVAL                          │
│                                                                 │
│  Query ──► Vector Search (ChromaDB)  ──────────────────┐        │
│        │                                               │        │
│        └──► Entity Extraction (LLM)                   │        │
│                  │                                     │        │
│                  └──► Graph Traversal (BFS, N hops)   │        │
│                             │                          │        │
│                        Graph Docs  ───────────────────►│        │
│                                                        ▼        │
│                                               Merge + Re-rank   │
└───────────────────────────────────────────────────┬─────────────┘
                                                    │
┌───────────────────────────────────────────────────▼─────────────┐
│                     GRAPH-AWARE GENERATION                      │
│                                                                 │
│   Retrieved Chunks  +  Entity Relationships  ──►  LLM  ──►  Answer │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/graph-rag.git
cd graph-rag
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Add your key:  OPENAI_API_KEY=sk-...
```

### 3. Launch the app

```bash
streamlit run src/demo.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## 🖥️ Demo Walkthrough

### Tab 1 — Ingest Documents
Upload PDF, TXT, or MD files. The pipeline runs automatically:

```
Step 1/4  Loading files
Step 2/4  Chunking with overlap
Step 3/4  Embedding → ChromaDB
Step 4/4  LLM entity extraction → NetworkX graph → entity resolution
```

### Tab 2 — Knowledge Graph
Interactive pyvis visualization rendered in the browser:
- **Highlight** any entity by name
- **Filter** edges by relationship type
- **Inspect** an entity's type, neighbors, and connected chunks
- **Shortest path** finder between any two entities

### Tab 3 — Chat
Ask natural language questions with two modes:
- **Graph-enhanced** — answer built from retrieved chunks *and* graph relationships
- **Compare mode** — vector-only answer side-by-side with graph-enhanced answer

Toggle **"Show graph context"** to see exactly which entities and relationships contributed to the answer.

---

## 📁 Project Structure

```
graph-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # PDF / TXT / MD loading
│   │   ├── chunker.py         # RecursiveCharacterTextSplitter + chunk IDs
│   │   └── embedder.py        # OpenAI embeddings → ChromaDB
│   ├── graph/
│   │   ├── extractor.py       # LLM entity + relationship extraction
│   │   ├── builder.py         # Incremental NetworkX graph construction
│   │   ├── resolver.py        # Fuzzy duplicate entity merging
│   │   └── store.py           # JSON graph persistence
│   ├── retrieval/
│   │   ├── vector.py          # ChromaDB similarity search
│   │   ├── graph_search.py    # BFS traversal + shortest path
│   │   ├── hybrid.py          # Vector + graph merge pipeline
│   │   └── reranker.py        # Score, deduplicate, re-rank
│   ├── generation/
│   │   └── generator.py       # Graph-aware prompt builder + LLM call
│   ├── viz/
│   │   └── graph_visualizer.py  # pyvis interactive HTML graph
│   └── demo.py                # Streamlit app (3 tabs)
├── tests/
│   ├── test_graph.py          # 20 tests: builder, resolver, store, traversal
│   ├── test_ingestion.py      # 7 tests: loader, chunker
│   └── test_retrieval.py      # 6 tests: reranker
├── examples/
│   ├── sample.txt             # AI companies & LLMs sample doc
│   └── sample2.txt            # Vector databases & infrastructure sample doc
├── conftest.py
├── requirements.txt
├── .env.example
└── LICENSE
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| LLM & Embeddings | OpenAI `gpt-4o-mini` + `text-embedding-3-small` |
| Orchestration | LangChain + LangChain Community |
| Vector Store | ChromaDB |
| Knowledge Graph | NetworkX |
| Entity Resolution | thefuzz (fuzzy string matching) |
| Visualization | pyvis → interactive HTML |
| UI | Streamlit |
| Tests | pytest (33 tests, 0 API calls needed) |

</div>

---

## ⚙️ Configuration

All retrieval settings are adjustable in the sidebar at runtime:

| Setting | Default | Description |
|---------|---------|-------------|
| Top-K results | 5 | Chunks returned per query |
| Graph hops | 2 | BFS depth from query entities |
| Chunk size | 500 | Tokens per chunk |
| Chunk overlap | 50 | Overlap between adjacent chunks |

---

## 🧠 How the Knowledge Graph is Built

For each text chunk, the LLM is prompted:

```
Extract all entities (people, organizations, concepts, technologies, locations)
and relationships from this text.
Return JSON: { entities: [{name, type}], relationships: [{source, target, relation}] }
```

Extracted triples are added to a `NetworkX` graph. After all chunks are processed, fuzzy string matching (threshold: 88%) merges near-duplicate entity nodes (e.g. *"OpenAI"* and *"openai"*). The graph is persisted to `data/graph.json` between sessions.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

Built with LangChain · ChromaDB · NetworkX · OpenAI · Streamlit · pyvis

</div>
