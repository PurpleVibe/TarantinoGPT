"""
Embeddings and Vector Store Utilities
Simple helpers to create embeddings, build/load a Chroma vector store,
and get a retriever.
"""

import os
from pathlib import Path
from typing import Sequence, Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()


def get_embeddings(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """Create an OpenAI embeddings model instance."""
    return OpenAIEmbeddings(model=model_name)


def create_vectorstore(
    documents: Sequence[Document],
    persist_directory: str = "vector_db",
    collection_name: str = "knowledge_base",
    embeddings: Optional[OpenAIEmbeddings] = None,
) -> Chroma:
    """
    Build a Chroma vector store from provided documents and persist it.
    Returns the Chroma instance.
    """
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    if embeddings is None:
        embeddings = get_embeddings()

    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        print("Created ChromaDB vector store!")
        return vectorstore
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")
        raise


def load_vectorstore(
    persist_directory: str = "vector_db",
    collection_name: str = "knowledge_base",
    embeddings: Optional[OpenAIEmbeddings] = None,
) -> Chroma:
    """Load an existing Chroma vector store from disk."""
    if embeddings is None:
        embeddings = get_embeddings()

    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        print("Loaded existing ChromaDB vector store")
        return vectorstore
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        raise


def get_retriever(vectorstore: Chroma, k: int = 10, search_type: str = "mmr"):
    """Create a retriever from a vector store."""
    return vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k})