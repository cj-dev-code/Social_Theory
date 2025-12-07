# src/feminism_rag/generate.py
from __future__ import annotations

from typing_extensions import List, TypedDict

import cohere
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from . import config
from .retrieval import get_retriever, enrich_with_payload

# ---- Cohere client ----
if not config.COHERE_API_KEY:
    raise RuntimeError(
        "COHERE_API_KEY is not set. Add it to your .env before running the app."
    )

co = cohere.Client(config.COHERE_API_KEY)

print(config.OLLAMA_MODEL_NAME)
# ---- LLM (local via Ollama) ----
llm = ChatOllama(
    model=config.OLLAMA_MODEL_NAME,
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.OLLAMA_TEMPERATURE,
    num_ctx=config.OLLAMA_NUM_CTX,
)

# ---- Prompt (from LangChain Hub) ----
prompt = ChatPromptTemplate.from_template(
    """You are a helpful research assistant specializing in feminist theory,
social science, and gender studies. Use ONLY the context provided below to
answer the question. If the context is not sufficient, say you don't know.

Context:
{context}

Question:
{question}

Answer in a clear, concise way that would make sense to a non-expert reader."""
)
_example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()
assert len(_example_messages) == 1  # basic sanity check


# ---- RAG graph state ----
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Global retrieval components
retriever, qdrant_client = get_retriever()


def retrieve(state: State) -> dict:
    """
    Retrieve relevant docs using MMR, then rerank with Cohere.
    Returns 'context' as a list of top-k Documents.
    """
    question = state["question"]

    # First-stage retrieval from Qdrant
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_docs = enrich_with_payload(qdrant_client, retrieved_docs)

    # Second-stage rerank with Cohere
    response = co.rerank(
        model=config.COHERE_RERANK_MODEL,
        query=question,
        documents=[r.page_content for r in retrieved_docs],
        top_n=config.RERANK_FINAL_K,
        return_documents=True,
    )

    top_docs = [retrieved_docs[r.index] for r in response.results]
    return {"context": top_docs}


def generate(state: State) -> dict:
    """
    Generate a grounded answer from the retrieved context.
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    message = prompt.invoke(
        {
            "question": state["question"],
            "context": docs_content,
        }
    )
    response = llm.invoke(message)
    return {"answer": response.content}


# ---- LangGraph assembly ----
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
