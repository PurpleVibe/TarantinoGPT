# TarantinoGPT
Agent for Tarantino specific questions!

## Architecture

- General overview
        - Purpose: domain-focused chatbot using RAG with LangGraph.
        - High-level flow: Flow: user → Streamlit UI → FastAPI → LangGraph agent → retriever (Chroma) → LLM (OpenAI) → response; session history stored in-memory.
- System diagram -
     <img width="1026" height="657" alt="image" src="https://github.com/user-attachments/assets/b6307df4-cfb3-45f3-984b-dc1df3b9129f" />


- Frontend–backend separation: Thin UI client communicates with a stateless HTTP API. All model execution, retrieval, and orchestration live on the backend.

- Layered RAG pipeline: Clear separation between data/preprocessing, retrieval/vector index, agent orchestration, and API interface. Each layer exposes minimal, well-typed inputs/outputs.

- Agent orchestration: A small, explicit graph (LLM node + retrieval tool node) controls the loop between reasoning and evidence gathering, making the control flow auditable and testable.

- Modular design: Data loading, chunking, embeddings, retrieval, and agent logic are isolated behind functions/modules, enabling easy swaps (e.g., different vector store or embedding model).


## Data loading and Preprocessing 

- Data: 2 full movie PDF scripts (Pulp fiction and Reservoir dogs, my favorite Tarantino movies) and 6 HTML URLs about Tarantino's life (for example Wikipedia), Articals (interview and directing analysis, etc..) and user reviews.

- Loaders:
    - PDF: UnstructuredPDFLoader(strategy=\"fast\") for speed and reasonable accuracy; Preproccessing text cleaning before embedding (for example: removing white space), metadata enriched with source and title for citations.
    - Web: trafilatura for extraction; retrives importent information and main article, cleans unnecessary information by default.

- Chunking: RecursiveCharacterTextSplitter (chunk_size=2400, overlap=400)
    - Why: robust across mixed formats; overlap preserves cross‑chunk continuity (dialogue/scene boundaries).
    - Filtering: drops tiny chunks (<100 chars) to reduce noise and cost.

## Indexing

- Embeddings: OpenAI text-embedding-3-small
    - Why: strong semantic quality, multilingual, fast, and cost‑efficient for RAG recall.

- Vector store: Chroma (local, persisted)
    - Why: simple local setup, fast dev loop, on‑disk persistence (vector_db/), mature LangChain integration.

- Retriever: MMR (Maximal Marginal Relevance)
    - Why: balances relevance and diversity; reduces redundancy and improves coverage for multi‑facet questions (especially with query expansion).

- Saving\Loading: 
    - Why: saves time at loading the program after already embedding.

## LangGraph Agent

- Graph shape: two-node graph — LLM node and Retrieval Tool node — with a conditional edge.

    - LLM node composes SYSTEM_PROMPT + conversation history and is bound to the retriever tool.
    - If the LLM emits tool_calls, control moves to the tool node; otherwise the graph ends.
    - After tool execution, control returns to the LLM for the final grounded answer.

- Orchestration details:
    - should_continue checks for tool_calls on the last message to decide whether to call the tool.
    - Retrieval tool runs a robust retrieval routine:
        - Query expansion (generate_expanded_queries) to increase recall.
        - Per‑query top_k results with MMR retriever, then merged and deduped (retrieve_expanded).
        - Returns compact, citation‑ready blocks for the LLM to synthesize.
    - Deterministic defaults: main LLM at temperature 0; small temperature bump only for query expansion.

- Grounding & citations:
    - System prompt enforces: “use retrieved content” and “always cite” (Title + URL), reducing hallucinations and improving trust.

## Basic UI

- UX:
    - Simple chatbot with message history; streaming is optional; citing sources with links.

- Endpoints:
    - /api/query: synchronous; returns the full answer.
    - /api/query_stream: streaming; yields tokens for low‑latency UX.

- Session memory:
    - In‑memory sessions keyed by session_id.
    - History is trimmed to the last 5 messages to bound token usage; applied consistently to both streaming and non‑streaming paths.
    - On stream completion, the final answer is appended to the session.


## How we addressed naive RAG pitfalls

- Poor recall from single, literal queries, low context > Query expansion (multi‑formulations merged and deduped).

- Redundant or near‑duplicate chunks > MMR retriever + signature‑based dedup across expansions.

- Answers without grounding > Inline citations required by system prompt (Title + URL), reduce hallucinations.

- Context window bloat and rising costs > History cap (last N messages) and per‑query top_k limits.

- Broken continuity across chunk boundaries → Chunk overlap (2400/400) and instruction to stitch multi‑chunk scenes in streaming mode.


## Installation

- Prerequisites
    - Python: 3.12+
    - OpenAI API key: set OPENAI_API_KEY in a .env file at the repo root

- Run: install.bat (installs requirements and create python environment) and then run.bat (runs backend api and frontend UI).

  <img width="728" height="392" alt="image" src="https://github.com/user-attachments/assets/65968079-5ce2-438e-a1b5-d8e664b25d93" />


- Visit: http://localhost:8501

Demo video:

https://github.com/user-attachments/assets/bd520b0f-148f-4b92-bdb2-3575532c784d

