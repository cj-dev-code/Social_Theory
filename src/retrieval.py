# src/feminism_rag/retrieval.py
from __future__ import annotations

from typing import List as TypingList

import qdrant_client
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

from . import config
import torch

def get_qdrant_client() -> qdrant_client.QdrantClient:
    """Create a Qdrant client pointing at the local disk index."""
    return qdrant_client.QdrantClient(path=config.QDRANT_PATH)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the HF embedding model used by the vector store."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME,
        cache_folder=config.HF_CACHE,
        model_kwargs={
            "device": "cpu",
            #"torch_dtype": torch.float16,  # cuts VRAM a lot
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 8,              # keep batch small at first
        },

    )


def get_vector_store(
    client: qdrant_client.QdrantClient | None = None,
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> Qdrant:
    """
    Build a LangChain Qdrant vector store wrapper.

    Assumes the collection already exists and has a payload key 'document'
    for the main text, consistent with the original experiment.
    """
    if client is None:
        client = get_qdrant_client()
    if embedding_model is None:
        embedding_model = get_embedding_model()

    vector_store = Qdrant(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embeddings=embedding_model,
        vector_name=config.QDRANT_VECTOR_NAME,
        content_payload_key="document",
        metadata_payload_key=None,  # keep rest of payload as metadata
    )
    return vector_store


def get_retriever(
    client: qdrant_client.QdrantClient | None = None,
    embedding_model: HuggingFaceEmbeddings | None = None,
):
    """
    Return (retriever, client) pair.

    Retriever uses MMR with configurable k, mirroring the original setup. :contentReference[oaicite:1]{index=1}
    """
    if client is None:
        client = get_qdrant_client()
    if embedding_model is None:
        embedding_model = get_embedding_model()

    vector_store = get_vector_store(client=client, embedding_model=embedding_model)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.RETRIEVAL_MMR_K},
    )
    return retriever, client


def enrich_with_payload(
    client: qdrant_client.QdrantClient,
    docs: TypingList[Document],
) -> TypingList[Document]:
    """
    Given a list of docs (with _id + _collection_name in metadata),
    pull full payload from Qdrant and reconstruct Documents with rich metadata.

    This is extracted from the original script and kept as a reusable utility.
    """
    ids = [str(d.metadata.get("_id")) for d in docs if d.metadata.get("_id") is not None]
    if not ids:
        return docs

    pts = client.retrieve(
        collection_name=config.QDRANT_COLLECTION,
        ids=ids,
        with_payload=True,
        with_vectors=False,
    )
    by_id = {str(p.id): p.payload for p in pts}
    enriched: TypingList[Document] = []

    for d in docs:
        payload = by_id.get(str(d.metadata.get("_id")), {})
        text = payload.get("document", d.page_content or "")
        meta = {k: v for k, v in payload.items() if k != "document"}
        meta.update(
            {
                "_id": d.metadata.get("_id"),
                "_collection_name": d.metadata.get("_collection_name"),
            }
        )
        enriched.append(Document(page_content=text, metadata=meta))

    return enriched
