from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .rag_agent import run_query_with_history, stream_answer

app = FastAPI(title="QT RAG API")

# Naive in-memory session store: {session_id: [BaseMessage, ...]}
SESSIONS: Dict[str, List[BaseMessage]] = {}
DEFAULT_SESSION = "default"

class QueryIn(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryOut(BaseModel):
    answer: str

@app.post("/api/query", response_model=QueryOut)
def query(payload: QueryIn):
    session_id = payload.session_id or DEFAULT_SESSION
    history = SESSIONS.get(session_id, [])
    answer = run_query_with_history(payload.question, history=history)
    # update history
    history = history + [HumanMessage(content=payload.question), AIMessage(content=answer)]
    SESSIONS[session_id] = history
    return QueryOut(answer=answer)

@app.post("/api/query_stream")
def query_stream(payload: QueryIn):
    session_id = payload.session_id or DEFAULT_SESSION
    history = SESSIONS.get(session_id, [])

    def generator():
        buffer = []
        for token in stream_answer(payload.question, history=history):
            buffer.append(token)
            yield token
        # finalize and append to session
        final_answer = "".join(buffer)
        SESSIONS[session_id] = history + [HumanMessage(content=payload.question), AIMessage(content=final_answer)]

    return StreamingResponse(generator(), media_type="text/plain")