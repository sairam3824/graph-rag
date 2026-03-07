from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load PDF, TXT, and MD files into LangChain Documents."""
    docs = []
    for path in file_paths:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(p))
        elif suffix in (".txt", ".md"):
            loader = TextLoader(str(p), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md")
        loaded = loader.load()
        # Tag source metadata
        for doc in loaded:
            doc.metadata["source_file"] = p.name
        docs.extend(loaded)
    return docs
