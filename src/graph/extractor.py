import json
import re
from typing import Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

EXTRACTION_PROMPT = """Extract all entities and relationships from the text below.
Return ONLY valid JSON — no explanation, no markdown fences.

Format:
{{
  "entities": [
    {{"name": "entity name", "type": "person|organization|concept|technology|location|other"}}
  ],
  "relationships": [
    {{"source": "entity1 name", "target": "entity2 name", "relation": "short relationship description"}}
  ]
}}

Text:
{text}"""


def extract_entities_and_relations(
    text: str,
    llm: Optional[ChatOpenAI] = None,
) -> Dict:
    """Call LLM to extract entities and relationships from a text chunk."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = EXTRACTION_PROMPT.format(text=text[:3000])
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Strip markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    # Extract first JSON object
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"entities": [], "relationships": []}
