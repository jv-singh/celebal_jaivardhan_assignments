# ============================================================
# app.py
# PURPOSE: The main entry point. A Streamlit web UI that wires
# together all the RAG pipeline steps:
#
#   load_document → split_into_chunks → get_embeddings
#   → VectorStore.add()
#   → (on query) get_single_embedding → VectorStore.search()
#   → generate_answer → display
#
# Run with: streamlit run src/app.py
# ============================================================

import os
import streamlit as st

from loader      import load_document
from chunker     import split_into_chunks
from embedder    import get_embeddings, get_single_embedding
from vector_store import VectorStore
from generator   import generate_answer


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG — Document Q&A", page_icon="📄")
st.title("📄 RAG — Document Question Answering")
st.caption("Upload a document, ask questions, get answers grounded in your document.")


# ── Sidebar: API key + how-it-works ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free key at https://console.groq.com"
    )

    st.markdown("---")
    st.markdown("**RAG pipeline (step by step):**")
    st.markdown("1. 📖 Load your document")
    st.markdown("2. ✂️  Split into small chunks")
    st.markdown("3. 🔢 Embed each chunk → vector")
    st.markdown("4. 💾 Store vectors in memory")
    st.markdown("5. ❓ You ask a question")
    st.markdown("6. 🔍 Find the closest chunks")
    st.markdown("7. 🤖 LLM answers using those chunks")


# ── Step 1: Document upload ───────────────────────────────────────────────────
st.markdown("### Step 1 — Upload your document")
st.caption("Supported: PDF and TXT files. "
           "Place your own files in the `documents/` folder, "
           "or upload directly here.")

uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])


# ── Step 2: Process the document (only when a new file is uploaded) ───────────
if uploaded_file is not None:

    # Check if this is a new file (avoid reprocessing the same file on every rerun)
    is_new_file = st.session_state.get("last_file") != uploaded_file.name

    if is_new_file:
        with st.spinner("Processing document…"):

            # Save upload to a temp file so loader can read it from disk
            temp_path = f"_temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # 2a. Load raw text
            raw_text = load_document(temp_path)

            # 2b. Split into chunks
            chunks = split_into_chunks(raw_text, chunk_size=500, overlap=50)

            # 2c. Embed all chunks
            embeddings = get_embeddings(chunks)

            # 2d. Store in our simple vector store
            vs = VectorStore()
            vs.add(chunks, embeddings)

            # Persist in session state so we don't redo this on every keystroke
            st.session_state["vector_store"] = vs
            st.session_state["last_file"]    = uploaded_file.name

            # Clean up the temp file
            os.remove(temp_path)

        st.success(f"✅ Ready! Indexed **{len(chunks)} chunks** from '{uploaded_file.name}'.")

    else:
        st.info(f"'{uploaded_file.name}' is already indexed. Ask away!")


# ── Step 3: Question input + answer ──────────────────────────────────────────
st.markdown("### Step 2 — Ask a question")

question = st.text_input("Your question:", placeholder="e.g. What is the main topic of this document?")

if st.button("Get Answer", type="primary"):

    # Guard: need a document
    if "vector_store" not in st.session_state:
        st.warning("Please upload a document first.")
        st.stop()

    # Guard: need an API key
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    # Guard: need a question
    if not question.strip():
        st.warning("Please type a question.")
        st.stop()

    with st.spinner("Searching document and generating answer…"):

        # 3a. Embed the question
        query_emb = get_single_embedding(question)

        # 3b. Retrieve the top-3 most relevant chunks
        vs = st.session_state["vector_store"]
        relevant_chunks = vs.search(query_emb, top_k=3)

        # 3c. Generate the answer
        answer = generate_answer(question, relevant_chunks, api_key)

    # ── Display the answer ────────────────────────────────────────────────────
    st.markdown("### 💬 Answer")
    st.write(answer)

    # Show retrieved chunks (great for learning — see what the LLM used!)
    with st.expander("🔍 Context the LLM used to answer (click to expand)"):
        for i, chunk in enumerate(relevant_chunks, start=1):
            st.markdown(f"**Chunk {i}** — similarity score: `{chunk['score']}`")
            st.text(chunk["text"])
            st.markdown("---")