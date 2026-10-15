"""
Financial Report Chat — RAG Interface
======================================
Conversational Q&A over ingested financial documents using ChromaDB
for retrieval and a local Ollama LLM for generation.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dashboard.components.charts import COLORS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SESSION_KEY_PIPELINE = "rag_pipeline"
_SESSION_KEY_HISTORY = "chat_history"
_SESSION_KEY_OLLAMA_OK = "ollama_status_ok"
_SESSION_KEY_MODEL = "rag_ollama_model"
_SESSION_KEY_CHUNKS_K = "rag_chunks_k"

_DEFAULT_MODEL = "llama3"
_DEFAULT_K = 4


# ---------------------------------------------------------------------------
# Helper: lazy pipeline initialisation
# ---------------------------------------------------------------------------
def _get_pipeline(model: str = _DEFAULT_MODEL) -> "FinancialRAGPipeline | None":
    """Return the cached FinancialRAGPipeline, creating it on first call.

    Returns *None* only when the pipeline itself fails to initialise
    (e.g. embedding-model download error). Ollama connectivity is
    checked separately so that retrieval-only mode still works.
    """
    current = st.session_state.get(_SESSION_KEY_PIPELINE)
    current_model = st.session_state.get(_SESSION_KEY_MODEL)

    # Re-create if the selected model changed
    if current is not None and current_model == model:
        return current

    try:
        from src.rag_pipeline import FinancialRAGPipeline

        with st.spinner("🔧 Initialising RAG pipeline (loading embeddings model)…"):
            pipeline = FinancialRAGPipeline(ollama_model=model)
        st.session_state[_SESSION_KEY_PIPELINE] = pipeline
        st.session_state[_SESSION_KEY_MODEL] = model
        return pipeline
    except Exception as exc:
        logger.exception("Failed to initialise RAG pipeline")
        st.error(
            f"⚠️ **Pipeline initialisation failed.**\n\n"
            f"`{exc}`\n\n"
            "Make sure all dependencies are installed "
            "(`langchain-community`, `chromadb`, `sentence-transformers`)."
        )
        return None


def _check_ollama(pipeline) -> bool:
    """Verify Ollama is reachable. Caches the result per run."""
    if _SESSION_KEY_OLLAMA_OK in st.session_state:
        return st.session_state[_SESSION_KEY_OLLAMA_OK]
    ok = pipeline.verify_ollama()
    st.session_state[_SESSION_KEY_OLLAMA_OK] = ok
    return ok


def _init_history() -> list:
    """Ensure chat history exists in session state."""
    if _SESSION_KEY_HISTORY not in st.session_state:
        st.session_state[_SESSION_KEY_HISTORY] = []
    return st.session_state[_SESSION_KEY_HISTORY]


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def _render_sidebar():
    """Sidebar widgets for model config & chat management."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"### 💬 Chat Settings"
        )

        model = st.text_input(
            "Ollama Model",
            value=st.session_state.get(_SESSION_KEY_MODEL, _DEFAULT_MODEL),
            help="Name of the Ollama model to use for generation (e.g. llama3, mistral, phi3).",
        )

        k = st.slider(
            "Chunks to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.get(_SESSION_KEY_CHUNKS_K, _DEFAULT_K),
            help="Number of document chunks retrieved from ChromaDB per query.",
        )

        st.session_state[_SESSION_KEY_CHUNKS_K] = k

        st.markdown("")
        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state[_SESSION_KEY_HISTORY] = []
            # Reset Ollama status so it is re-checked next run
            st.session_state.pop(_SESSION_KEY_OLLAMA_OK, None)
            st.rerun()

    return model, k


# ---------------------------------------------------------------------------
# Document upload & ingestion
# ---------------------------------------------------------------------------
def _render_upload_section(pipeline):
    """File uploader + ingestion logic."""
    st.markdown("#### 📄 Upload Financial Documents")
    st.caption(
        "Upload a **PDF** or **TXT** file to ingest it into the vector store. "
        "Documents are chunked, embedded, and persisted in ChromaDB."
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        key="rag_file_uploader",
    )

    if uploaded is not None:
        # Save to a temp location so the loader can read from disk
        suffix = Path(uploaded.name).suffix
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="fin_rag_"
            ) as tmp:
                tmp.write(uploaded.getvalue())
                temp_path = tmp.name

            with st.spinner(f"📥 Ingesting **{uploaded.name}**…"):
                chunk_count = pipeline.ingest_document(temp_path)

            st.success(
                f"✅ **{uploaded.name}** ingested successfully — "
                f"**{chunk_count}** chunks embedded and stored."
            )
        except FileNotFoundError as exc:
            st.error(f"🚫 File not found: {exc}")
        except ValueError as exc:
            st.error(f"🚫 Unsupported file type: {exc}")
        except Exception as exc:
            logger.exception("Ingestion error")
            st.error(
                f"⚠️ **Ingestion failed.**\n\n`{exc}`\n\n"
                "Check the logs for details."
            )
        finally:
            # Clean up temp file
            try:
                if "temp_path" in locals():
                    os.unlink(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
def _render_chat(pipeline, k: int, ollama_available: bool):
    """Render history + handle new user input."""
    st.markdown("#### 💬 Ask a Question")

    history = _init_history()

    # ---- Render existing messages ----------------------------------------
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources & chunks for assistant messages that carry them
            if msg["role"] == "assistant":
                sources = msg.get("sources")
                chunks = msg.get("chunks")

                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for idx, src in enumerate(sources, 1):
                            page_str = f", Page {src['page']}" if "page" in src else ""
                            score_pct = f"{src['score']:.0%}"
                            st.markdown(
                                f"**{idx}.** `{src['file']}`{page_str} — "
                                f"relevance **{score_pct}**"
                            )

                if chunks:
                    with st.expander("🔍 Retrieved Chunks", expanded=False):
                        for idx, chunk in enumerate(chunks, 1):
                            src_info = chunk["metadata"].get("source_file", "unknown")
                            page_info = (
                                f" · p.{chunk['metadata']['page'] + 1}"
                                if "page" in chunk["metadata"]
                                else ""
                            )
                            st.markdown(
                                f"**Chunk {idx}** — `{src_info}`{page_info} "
                                f"(score: `{chunk['score']:.4f}`)"
                            )
                            st.code(chunk["content"][:600], language="text")

    # ---- New user input --------------------------------------------------
    user_input = st.chat_input(
        "Ask about your financial documents…",
        key="rag_chat_input",
    )

    if user_input:
        # Immediately render user bubble
        with st.chat_message("user"):
            st.markdown(user_input)
        history.append({"role": "user", "content": user_input})

        # ---- Generate response -------------------------------------------
        with st.chat_message("assistant"):
            try:
                # First retrieve chunks (always works if ChromaDB is up)
                with st.spinner("🔎 Retrieving relevant chunks…"):
                    retrieved_chunks = pipeline.retrieve_context(user_input, k=k)

                if not retrieved_chunks:
                    answer_text = (
                        "No relevant document chunks were found in the database. "
                        "Please ingest some documents first using the uploader above."
                    )
                    sources = []
                elif not ollama_available:
                    # Fall back to retrieval-only mode
                    answer_text = (
                        "⚠️ **Ollama is not available** — showing retrieved "
                        "context only.\n\nStart Ollama (`ollama serve`) and pull "
                        f"the **{pipeline.ollama_model}** model to enable "
                        "AI-generated answers."
                    )
                    sources = []
                else:
                    # Full RAG query
                    with st.spinner("🤖 Generating answer…"):
                        result = pipeline.query(user_input, k=k)
                    answer_text = result.get("answer", "No answer returned.")
                    sources = result.get("sources", [])

                st.markdown(answer_text)

                # Sources expander
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for idx, src in enumerate(sources, 1):
                            page_str = (
                                f", Page {src['page']}" if "page" in src else ""
                            )
                            score_pct = f"{src['score']:.0%}"
                            st.markdown(
                                f"**{idx}.** `{src['file']}`{page_str} — "
                                f"relevance **{score_pct}**"
                            )

                # Chunks expander
                if retrieved_chunks:
                    with st.expander("🔍 Retrieved Chunks", expanded=False):
                        for idx, chunk in enumerate(retrieved_chunks, 1):
                            src_info = chunk["metadata"].get(
                                "source_file", "unknown"
                            )
                            page_info = (
                                f" · p.{chunk['metadata']['page'] + 1}"
                                if "page" in chunk["metadata"]
                                else ""
                            )
                            st.markdown(
                                f"**Chunk {idx}** — `{src_info}`{page_info} "
                                f"(score: `{chunk['score']:.4f}`)"
                            )
                            st.code(chunk["content"][:600], language="text")

                # Persist assistant message
                history.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "chunks": retrieved_chunks,
                    }
                )

            except Exception as exc:
                logger.exception("Chat query error")
                error_msg = f"⚠️ **Error:** `{exc}`"
                st.error(error_msg)
                history.append({"role": "assistant", "content": error_msg})


# ---------------------------------------------------------------------------
# Main render entry-point
# ---------------------------------------------------------------------------
def render():
    """Top-level render function called by the dashboard router."""

    # ---- Header ----------------------------------------------------------
    st.markdown(
        """
        ## 🗣️ Financial Report Chat
        > **RAG-powered Q&A** — Upload financial documents, then ask questions.
        > Documents are embedded via *ChromaDB* and answers are generated with
        > a local *Ollama* LLM.
        """
    )

    # ---- Sidebar ---------------------------------------------------------
    model, k = _render_sidebar()

    # ---- Pipeline --------------------------------------------------------
    pipeline = _get_pipeline(model)
    if pipeline is None:
        st.stop()

    # ---- Ollama health check ---------------------------------------------
    ollama_available = _check_ollama(pipeline)

    if not ollama_available:
        st.warning(
            "⚠️ **Ollama is not reachable** at "
            f"`{pipeline.ollama_url}`.\n\n"
            "Document upload and chunk retrieval will still work, but "
            "AI-generated answers are disabled until Ollama is running.\n\n"
            "```bash\n"
            "ollama serve          # start the server\n"
            f"ollama pull {model}   # download the model\n"
            "```"
        )

    st.markdown("---")

    # ---- Upload section --------------------------------------------------
    _render_upload_section(pipeline)

    st.markdown("---")

    # ---- Chat interface --------------------------------------------------
    _render_chat(pipeline, k, ollama_available)
