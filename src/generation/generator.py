from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions using both document excerpts "
    "and a structured knowledge graph. Cite relevant relationships when they help explain connections."
)

GRAPH_AWARE_TEMPLATE = """\
Answer the question using the document excerpts AND the entity relationships from the knowledge graph.

DOCUMENT EXCERPTS:
{doc_context}

ENTITY RELATIONSHIPS (Knowledge Graph):
{graph_context}

QUESTION: {question}

Provide a thorough answer that leverages both the text evidence and the graph relationships shown above.\
"""

VECTOR_ONLY_TEMPLATE = """\
Answer the question using only the document excerpts below.

DOCUMENT EXCERPTS:
{doc_context}

QUESTION: {question}\
"""


def _format_relationships(relationships: List[Dict], limit: int = 25) -> str:
    if not relationships:
        return "No specific relationships found."
    seen: set[tuple] = set()
    lines = []
    for rel in relationships:
        key = (rel["source"], rel["target"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  • {rel['source']} --[{rel['relation']}]--> {rel['target']}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def generate_answer(
    question: str,
    docs: List[Document],
    graph_context: Optional[Dict] = None,
    llm: Optional[ChatOpenAI] = None,
    use_graph: bool = True,
) -> str:
    """Generate an answer with optional graph-context enrichment."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    doc_context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
        for doc in docs[:6]
    )

    if use_graph and graph_context and graph_context.get("relationships"):
        graph_text = _format_relationships(graph_context["relationships"])
        prompt = GRAPH_AWARE_TEMPLATE.format(
            doc_context=doc_context,
            graph_context=graph_text,
            question=question,
        )
    else:
        prompt = VECTOR_ONLY_TEMPLATE.format(
            doc_context=doc_context,
            question=question,
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content
