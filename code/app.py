import streamlit as st
import os
import tempfile

from src.pipeline.rag_pipeline import SanskritRAGPipeline

st.set_page_config(
    page_title="Sanskrit RAG Assistant",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Sanskrit RAG Pipeline")
st.markdown(
    "Ask questions from Sanskrit documents using AI-powered Retrieval-Augmented Generation (RAG)."
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

st.header("1️⃣ Upload Document")

uploaded_file = st.file_uploader(
    "Upload a Sanskrit PDF or TXT document",
    type=["pdf", "txt"]
)

if uploaded_file is not None:

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File uploaded successfully: {uploaded_file.name}")

    if st.button("🚀 Build RAG Pipeline"):

        with st.spinner("Building pipeline... Please wait."):

            try:
                rag_pipeline = SanskritRAGPipeline()

                rag_pipeline.build_pipeline(temp_path)

                st.session_state.pipeline = rag_pipeline
                st.session_state.pipeline_ready = True

                st.success("✅ Pipeline built successfully!")

            except Exception as e:
                st.error(f"Pipeline build failed: {e}")

st.header("2️⃣ Ask Questions")

query = st.text_input(
    "Enter your Sanskrit or English query:"
)

top_k = st.slider(
    "Select Top-K Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=3
)

if st.button("🔍 Generate Answer"):

    if not st.session_state.pipeline_ready:
        st.warning("Please build the pipeline first.")

    elif not query.strip():
        st.warning("Please enter a valid query.")

    else:

        with st.spinner("Generating response..."):

            try:
                result = st.session_state.pipeline.ask_question(
                    query=query,
                    top_k=top_k
                )

                st.subheader("🧠 Generated Response")
                st.write(result["response"])

                with st.expander("📚 Retrieved Chunks"):

                    for idx, chunk in enumerate(
                        result["retrieved_chunks"],
                        start=1
                    ):

                        st.markdown(f"### Chunk {idx}")
                        st.write(chunk)
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error generating answer: {e}")

st.markdown("---")
st.caption(
    "Built using Streamlit + FAISS + Sentence Transformers + RAG"
)