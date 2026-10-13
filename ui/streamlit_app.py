"""Minimal Streamlit chat UI that talks to the FastAPI backend."""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("📄 Chat with your documents")

with st.sidebar:
    st.header("Upload")
    files = st.file_uploader("PDF / DOCX / TXT / MD", accept_multiple_files=True,
                             type=["pdf", "docx", "txt", "md"])
    if st.button("Ingest") and files:
        r = requests.post(f"{API}/ingest", files=[("files", (f.name, f.getvalue())) for f in files])
        st.success(r.json()) if r.ok else st.error(r.text)
    try:
        st.caption(f"Indexed: {', '.join(requests.get(f'{API}/documents').json()['sources']) or 'none'}")
    except requests.RequestException:
        st.warning("Backend not reachable")
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)

st.session_state.setdefault("history", [])
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if q := st.chat_input("Ask something about your documents"):
    st.chat_message("user").markdown(q)
    r = requests.post(f"{API}/query", json={"question": q, "top_k": top_k})
    data = r.json() if r.ok else {"answer": r.text, "citations": []}
    with st.chat_message("assistant"):
        st.markdown(data["answer"])
        for c in data["citations"]:
            page = f" p.{c['page']}" if c.get("page") else ""
            with st.expander(f"[{c['ref']}] {c['source']}{page} — score {c['score']}"):
                st.write(c["text"])
    st.session_state.history += [("user", q), ("assistant", data["answer"])]
