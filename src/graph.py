from typing import List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.config import configure_langsmith, load_settings, traceable
from src.rag_chain import generate_answer
from src.reranking import rerank_documents, reranking_results
from src.retrieval import retrieve_documents


class SynataricState(TypedDict):
    question: str
    namespace: str
    top_k: int
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
    sources: List[dict]
    reranking_results: List[dict]


@traceable(name="Synataric Retrieve Node")
def retrieve_node(state: SynataricState) -> SynataricState:
    docs = retrieve_documents(state["question"], namespace=state.get("namespace"), top_k=state.get("top_k", 10))
    return {**state, "retrieved_docs": docs}


@traceable(name="Synataric Rerank Node")
def rerank_node(state: SynataricState) -> SynataricState:
    docs = rerank_documents(state["question"], state.get("retrieved_docs", []), top_n=3)
    return {**state, "reranked_docs": docs, "reranking_results": reranking_results(docs)}


@traceable(name="Synataric Generate Node")
def generate_node(state: SynataricState) -> SynataricState:
    answer, sources = generate_answer(state["question"], state.get("reranked_docs", []))
    return {**state, "answer": answer, "sources": sources}


def build_synataric_graph():
    workflow = StateGraph(SynataricState)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("rerank_node", rerank_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "rerank_node")
    workflow.add_edge("rerank_node", "generate_node")
    workflow.add_edge("generate_node", END)
    return workflow.compile()


def run_synataric_graph(question: str, namespace: str | None = None, top_k: int = 10) -> dict:
    configure_langsmith()
    settings = load_settings()
    graph = build_synataric_graph()
    initial: SynataricState = {
        "question": question,
        "namespace": namespace or settings.semantic_namespace,
        "top_k": top_k,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "sources": [],
        "reranking_results": [],
    }
    return graph.invoke(initial)
