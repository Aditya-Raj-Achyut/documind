"""Chat UI.  Run:  streamlit run ui/streamlit_app.py   (API must be running)"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="centered")
st.title("📄 DocuMind")
st.caption("Ask questions about the indexed documents. Every answer cites its sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("About")
    st.write(
        "Hybrid retrieval (vector + BM25) → cross-encoder reranking → "
        "LLM answer grounded in the retrieved excerpts."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['source']}** — page {s['page']}")
            st.caption(s["text"])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if question := st.chat_input("Ask something about the documents..."):
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "history": history},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                answer, sources = data["answer"], data["sources"]
            except requests.exceptions.RequestException as exc:
                answer, sources = f"⚠️ Could not get an answer: {exc}", []
        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
