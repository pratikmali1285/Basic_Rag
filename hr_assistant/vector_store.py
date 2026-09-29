"""Step 4: store chunk embeddings so we can search them later.

Uses Qdrant Cloud when QDRANT_URL/QDRANT_API_KEY are set, otherwise falls
back to a local FAISS index saved under config.FAISS_INDEX_PATH.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


# build_vector_store

def build_vector_store(chunks):
    """Embed every chunk and store it (Qdrant Cloud if configured, else local FAISS)."""
    embeddings_model = get_embeddings_model()

    if not config.USE_QDRANT:
        logger.info("Embedding %d chunk(s) into a local FAISS index...", len(chunks))
        vector_store = FAISS.from_documents(chunks, embeddings_model)
        vector_store.save_local(config.FAISS_INDEX_PATH)
        logger.info("Saved FAISS index to '%s'", config.FAISS_INDEX_PATH)
        return vector_store

    logger.info(
        "Embedding %d chunk(s) and uploading to Qdrant collection '%s'...",
        len(chunks),
        config.QDRANT_COLLECTION_NAME,
    )
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    logger.info("Uploaded to Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    return vector_store


def load_vector_store():
    """Open a vector store that was already built before."""
    embeddings_model = get_embeddings_model()

    if not config.USE_QDRANT:
        logger.info("Loading local FAISS index from '%s'", config.FAISS_INDEX_PATH)
        # The index is pickled by us on this machine, so loading it is safe.
        return FAISS.load_local(
            config.FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )

    logger.info("Connecting to existing Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )


def vector_store_exists() -> bool:
    """Check if the vector store (Qdrant collection or local FAISS index) already exists."""
    if not config.USE_QDRANT:
        return os.path.exists(os.path.join(config.FAISS_INDEX_PATH, "index.faiss"))

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return client.collection_exists(config.QDRANT_COLLECTION_NAME)


def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever
    that returns the top-k matching chunks."""
    logger.info("Creating retriever with top_k=%d", k)
    return vector_store.as_retriever(search_kwargs={"k": k})
