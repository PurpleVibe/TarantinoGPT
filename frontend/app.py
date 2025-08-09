import os
import uuid
import streamlit as st
import requests

st.set_page_config(page_title="TarantinoGPT", page_icon="🎬", layout="centered")

st.title("TarantinoGPT 🎬")
st.caption("Ask about Quentin Tarantino, his films, themes, and directing style.")

# Resolve API URLs without requiring a secrets file
DEFAULT_API = "http://127.0.0.1:8000/api/query"
DEFAULT_API_STREAM = "http://127.0.0.1:8000/api/query_stream"
try:
    API_URL = st.secrets["API_URL"]
    API_URL_STREAM = st.secrets.get("API_URL_STREAM", DEFAULT_API_STREAM)
except Exception:
    API_URL = os.environ.get("API_URL", DEFAULT_API)
    API_URL_STREAM = os.environ.get("API_URL_STREAM", DEFAULT_API_STREAM)

# Stable session id per browser session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role: "user"|"assistant", content: str}

# Input form (always at top)
with st.form("ask_form"):
    question = st.text_area(
        "Your question",
        placeholder="e.g., What are the themes in Pulp Fiction?",
        height=100,
    )
    use_stream = st.checkbox("Stream answer", value=True)
    submitted = st.form_submit_button("Ask")

# Container where messages will be rendered below the input
messages_container = st.container()

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": question})
        payload = {"question": question, "session_id": st.session_state.session_id}

        if use_stream:
            # Stream assistant reply at the top of the messages area
            with messages_container:
                stream_placeholder = st.empty()
            try:
                buf = []
                with requests.post(API_URL_STREAM, json=payload, stream=True, timeout=180) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=None):
                        if not chunk:
                            continue
                        text = chunk.decode("utf-8", errors="ignore")
                        buf.append(text)
                        # Update the top placeholder with the latest partial text
                        stream_placeholder.chat_message("assistant").markdown("".join(buf))
                # Finalize history and rerun to render newest-first
                assistant_text = "".join(buf)
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                st.rerun()
            except Exception as e:
                st.error(f"Streaming failed: {e}")
        else:
            # Non-streaming: fetch full answer, store, and rerun
            try:
                resp = requests.post(API_URL, json=payload, timeout=60)
                if resp.ok:
                    answer = resp.json().get("answer", "")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                else:
                    st.error(f"API error: {resp.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# Render existing history newest-first below the input
with messages_container:
    for m in reversed(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])