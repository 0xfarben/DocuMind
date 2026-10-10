import os
import tempfile
from pathlib import Path

import streamlit as st

from src_for_streamlit.data_loader import load_all_documents
from src_for_streamlit.vectorstore import FaissVectorStore
from src_for_streamlit.search import RAGSearch


st.set_page_config(
    page_title="DocuMind | RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocuMind")
st.caption("Ask questions and get answers grounded in your documents.")

# Session state
defaults = {
    "rag": None,
    "messages": [],
    "upload_dir": None,
    "document_names": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "csv", "xlsx", "docx", "json"],
        accept_multiple_files=True,
    )

    build_clicked = st.button(
        "Build Knowledge Base",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if build_clicked and uploaded_files:
        progress_bar = st.progress(0, text="Starting...")
        progress_text = st.empty()
        status_box = st.status("Preparing documents...", expanded=True)

        try:
            upload_dir = tempfile.mkdtemp(prefix="documind_uploads_")
            persist_dir = tempfile.mkdtemp(prefix="documind_index_")

            with status_box:
                st.write("Saving uploaded documents...")
                for uploaded_file in uploaded_files:
                    safe_name = Path(uploaded_file.name).name
                    file_path = os.path.join(upload_dir, safe_name)
                    with open(file_path, "wb") as file:
                        file.write(uploaded_file.getbuffer())

                progress_bar.progress(0.05, text="Files saved")
                progress_text.info("Loading documents...")
                st.write("Loading documents...")
                documents = load_all_documents(upload_dir)

                if not documents:
                    raise ValueError(
                        "No readable documents were found. Check the uploaded files."
                    )

                progress_bar.progress(0.10, text="Documents loaded")
                progress_text.info(
                    f"Loaded {len(documents):,} document entries. Preparing chunks and embeddings..."
                )
                st.write(f"Loaded {len(documents):,} document entries.")

                vectorstore = FaissVectorStore(persist_dir=persist_dir)

                def update_progress(stage: str, fraction: float, message: str):
                    # Callback runs during synchronous indexing, updating the UI.
                    progress_bar.progress(
                        max(0.0, min(1.0, fraction)),
                        text=message,
                    )
                    progress_text.info(message)

                vectorstore.build_from_documents(
                    documents,
                    progress_callback=update_progress,
                )

                progress_text.info("Initializing the RAG assistant...")
                rag = RAGSearch(persist_dir=persist_dir)

                # Update active knowledge base only after a successful build.
                st.session_state.rag = rag
                st.session_state.upload_dir = upload_dir
                st.session_state.document_names = [
                    Path(file.name).name for file in uploaded_files
                ]
                st.session_state.messages = []

                progress_bar.progress(1.0, text="Knowledge base ready")
                progress_text.success("Knowledge base ready!")
                status_box.update(
                    label="Knowledge base built successfully",
                    state="complete",
                    expanded=False,
                )

        except Exception as exc:
            status_box.update(
                label="Knowledge base build failed",
                state="error",
                expanded=True,
            )
            st.error(f"Could not build knowledge base: {exc}")

        finally:
            # Remove transient progress widgets after the operation ends.
            # The completed status remains visible.
            pass

    st.divider()

    if st.session_state.document_names:
        st.subheader("Indexed documents")
        for name in st.session_state.document_names:
            st.caption(f"📄 {name}")

    if st.session_state.rag is not None:
        st.success("Knowledge base is ready.")
    else:
        st.info("Upload documents and build your knowledge base.")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("Retrieved sources"):
                for source in sources:
                    title = source["label"]
                    if source.get("page") is not None:
                        title += f" — Page {source['page']}"
                    st.markdown(f"**{title}**")
                    st.caption(source["name"])
                    st.write(source["text"])
                    st.divider()


question = st.chat_input(
    "Ask a question about your documents...",
    disabled=st.session_state.rag is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating an answer..."):
                answer, sources = st.session_state.rag.search_with_sources(
                    question,
                    top_k=5,
                )

            st.markdown(answer)

            if sources:
                with st.expander("Retrieved sources"):
                    for source in sources:
                        title = source["label"]
                        if source.get("page") is not None:
                            title += f" — Page {source['page']}"
                        st.markdown(f"**{title}**")
                        st.caption(source["name"])
                        st.write(source["text"])
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    except Exception as exc:
        st.error(f"Could not generate an answer: {exc}")
