"""
RAG Agent setup using LangGraph.
Builds a retriever-backed agent with a single tool for retrieval.
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Optional, List, Iterator, Tuple, Set
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .embeddings import get_embeddings, create_vectorstore, load_vectorstore, get_retriever
from .data_loader import DataLoader

load_dotenv()

# Default web sources 
DEFAULT_WEB_URLS: List[str] = [
    "https://www.indiewire.com/features/general/quentin-tarantino-movies-ranked-1201784567/",
    "https://www.rottentomatoes.com/m/reservoir_dogs/reviews",
    "https://en.wikipedia.org/wiki/Quentin_Tarantino",
    "https://philosophynow.org/issues/19/Symbolism_Meaning_and_Nihilism_in_Quentin_Tarantinos_Pulp_Fiction",
    "https://indiefilmhustle.com/ultimate-guide-to-quentin-tarantino-and-his-directing-techniques/",
    "https://hibarr.substack.com/p/tarantino-1994",
]

SYSTEM_PROMPT = (
    """You are an Quentin Tarantino super fan that answers questions using the provided knowledge base about Quentin Tarantino and his work.
    URLs that include his movies ranked, his interviews, his wikipedia page, his philosophy, his directing techniques, and his movies reviews 
    are the most relevant to answer questions about Quentin Tarantino and his work.
    PDF documents that include full scripts for Pulp Fiction and Reservoir Dogs. Please use them to answer questions about the scripts and retrive full scenes scripts.
    When you use retrieved content, always cite sources using the document title and the exact URL inline in your answer (e.g., "(Title — https://...)" or as a markdown link [Title](https://...)). 
    Be concise and accurate while keeping quirky, funny and sarcastic answers similar to Quentin Tarantino's style. try to quote lines from the scripts when possible."""
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


# ---------- Query Expansion ----------

def generate_expanded_queries(question: str, num: int = 3) -> List[str]:
    """Use the LLM to produce a few targeted reformulations/keywords for retrieval."""
    llm_short = ChatOpenAI(model="gpt-4o", temperature=0.2)
    prompt = (
        "Rewrite the user question into {} short, diverse search queries that are maximally helpful for retrieving relevant passages. "
        "Prefer named entities, key phrases, and synonyms. Return one per line, no numbering.\n\nUser question: {}"
    ).format(num, question)
    resp = llm_short.invoke([HumanMessage(content=prompt)])
    lines = [ln.strip("- ") for ln in resp.content.splitlines() if ln.strip()]
    # Keep only first num unique lines
    seen: Set[str] = set()
    expansions: List[str] = []
    # build a list of expandedqueries 
    for ln in lines:
        if ln.lower() not in seen:
            expansions.append(ln)
            seen.add(ln.lower())
        if len(expansions) >= num:
            break
    if not expansions:
        expansions = [question]
    return expansions


def _doc_signature(doc) -> Tuple[str, Optional[int], str]:
    src = str(doc.metadata.get("source", "unknown"))
    idx = doc.metadata.get("chunk_index")
    head = doc.page_content[:64]
    return (src, idx, head)


def retrieve_expanded(retriever, question: str, expansions: int = 3, per_query_k: int = 4) -> List:
    """Generate expansions, retrieve per expansion, merge and dedupe while preserving order."""
    queries = [question] + generate_expanded_queries(question, num=expansions)
    merged: List = []
    seen_sigs: Set[Tuple[str, Optional[int], str]] = set()
    # retrieve per expansion
    for q in queries:
        docs = retriever.invoke(q)
        # limit per-query
        docs = docs[:per_query_k]
        # dedupe
        for d in docs:
            sig = _doc_signature(d)
            if sig not in seen_sigs:
                merged.append(d)
                seen_sigs.add(sig)
    return merged


# ---------- Retriever & Agent ----------

def build_retriever(
    web_urls: Optional[List[str]] = None,
    k: int = 10,
    persist_dir: str = "vector_db",
    collection: str = "knowledge_base",
    force_rebuild: bool = False,
):
    """Prepare the retriever for the agent."""
    embeddings = get_embeddings()
    # if the vectorstore already exists, load it
    if (not force_rebuild) and Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        vectorstore = load_vectorstore(
            persist_directory=persist_dir,
            collection_name=collection,
            embeddings=embeddings,
        )
    else:
        loader = DataLoader()
        docs = loader.load_all_pdfs()
        if web_urls is None:
            web_urls = DEFAULT_WEB_URLS
        if web_urls:
            web_docs = loader.load_web_pages(web_urls)
            docs.extend(web_docs)
        if not docs:
            raise RuntimeError("No documents loaded for retrieval.")
        chunks = loader.split_documents(docs)
        vectorstore = create_vectorstore(
            documents=chunks,
            persist_directory=persist_dir,
            collection_name=collection,
            embeddings=embeddings,
        )
    return get_retriever(vectorstore, k=k, search_type="mmr")

# get metadata for the document
def _format_header(i: int, doc) -> str:
    title = doc.metadata.get("title") or doc.metadata.get("source", "unknown")
    src = str(doc.metadata.get("source", ""))
    header = f"[{i}] {title}"
    if src.startswith("http"):
        header += f" — {src}"
    return header


def create_agent(web_urls: Optional[List[str]] = None, k: int = 10, force_rebuild: bool = False):
    retriever = build_retriever(web_urls=web_urls, k=k, force_rebuild=force_rebuild)

    @tool
    def retriever_tool(query: str) -> str:
        """
        Search knowledge base about Quentin Tarantino and his work 
        and return relevant passages with sources.
        """
        docs = retrieve_expanded(retriever, query, expansions=3, per_query_k=4)
        if not docs:
            return "No relevant sources found."
        parts = []
        for i, doc in enumerate(docs, start=1):
            header = _format_header(i, doc)
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(parts)

    tools = [retriever_tool]

    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

    # LLM Agent
    def call_llm(state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        messages = list(state["messages"])  # copy
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        message = llm.invoke(messages)
        return {"messages": [message]}

    tools_dict = {t.name: t for t in tools}

    # Retriever Agent
    def take_action(state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
            if t["name"] not in tools_dict:
                result = "Unknown tool. Please Retry and Select tool from List of Available tools"
            else:
                result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        return {"messages": results}

    # Create the graph for the agent
    # Simple state graph with two nodes: llm and retriever_agent
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, 
                                {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    return graph.compile()


def run_query(question: str, web_urls: Optional[List[str]] = None):
    """Run the agent with the given question."""
    agent = create_agent(web_urls=web_urls)
    messages = [HumanMessage(content=question)]
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


def run_query_with_history(
    question: str,
    history: Optional[Sequence[BaseMessage]] = None,
    web_urls: Optional[List[str]] = None,
    max_history_messages: Optional[int] = None,
) -> str:
    """Run the agent with prior conversation history.

    If `max_history_messages` is provided, only the most recent N messages from
    `history` are included (preserving order). If set to 0 or a negative value,
    the history is ignored. If `None`, the full provided history is used.
    """
    agent = create_agent(web_urls=web_urls)
    messages: List[BaseMessage] = []
    if history:
        if max_history_messages is None:
            messages.extend(history)
        elif max_history_messages <= 0:
            pass  # explicitly ignore history
        else:
            messages.extend(history[-max_history_messages:])
    messages.append(HumanMessage(content=question))
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


def stream_answer(question: str, web_urls: Optional[List[str]] = None, k: int = 10, history: Optional[Sequence[BaseMessage]] = None) -> Iterator[str]:
    """Retrieve context via expanded queries, then stream the LLM's answer as tokens, using optional history."""
    retriever = build_retriever(web_urls=web_urls, k=k)
    docs = retrieve_expanded(retriever, question, expansions=3, per_query_k=4)

    if not docs:
        yield "No relevant sources found."
        return

    # Build compact context block with inline-friendly citation headers
    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        header = _format_header(i, doc)
        parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n".join(parts)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    sys = SystemMessage(content=SYSTEM_PROMPT + "\nWhen the user asks for a scene/script/quote, if it spans multiple chunks, stitch them and present as one continuous block. Do not summarize unless asked. Cite sources inline with Title and URL.")

    messages: List[BaseMessage] = [sys]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=f"Question:\n{question}\n\nContext:\n{context}"))

    for chunk in llm.stream(messages):
        if getattr(chunk, "content", None):
            yield chunk.content