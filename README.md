# RAG_Sanskrit_SanketMaraskolhe
This is end-to-end RAG pipeline, capable of processing and answering queries based on Sanskrit documents. The system must operate fully on CPU-based inference.

# Sanskrit Document Retrieval-Augmented Generation (RAG) System

## Project Overview

This project implements a CPU-based Retrieval-Augmented Generation (RAG) system capable of processing Sanskrit documents and generating contextual responses using Large Language Models (LLMs).

The system performs:
- Sanskrit document ingestion
- Text preprocessing and transliteration
- Semantic retrieval using FAISS
- Keyword retrieval using BM25
- Response generation using TinyLlama
- Interactive querying through Streamlit UI

The project follows a modular RAG architecture and operates completely on CPU-based inference.

---

# Features

- Sanskrit PDF/TXT document support
- Semantic retrieval using FAISS
- Keyword retrieval using BM25
- Hybrid retrieval architecture
- CPU-efficient lightweight LLM inference
- Modular and scalable architecture
- Streamlit-based user interface
- Logging and evaluation support
- Performance metrics tracking
- End-to-end RAG pipeline


# System Architecture

text
PDF Document
     ↓
Document Loader
     ↓
Text Cleaning
     ↓
Transliteration
     ↓
Chunking
     ↓
Embedding Generation
     ↓
FAISS Vector Database
     ↓
Retriever
     ↓
Prompt Engineering
     ↓
TinyLlama LLM
     ↓
Generated Response

---
# Setup and Usage Instructions
---

Setup and Usage Instructions
1. Clone the Repository
git clone <repository_link>
cd RAG_Sanskrit_SanketMaraskolhe

2. Create Virtual Environment (Windows)
python -m venv venv
venv\Scripts\activate

3. Create Virtual Environment (Linux/Mac)
python3 -m venv venv
source venv/bin/activate

4. Install Required Dependencies
pip install -r code/requirements.txt

5. Add Sanskrit Documents
Place Sanskrit documents inside:
data/raw/

Supported formats:
- PDF
- TXT
- DOCX
- CSV
- PPTX

6. Build the RAG Pipeline
python -m code.src.pipeline.rag_pipeline

7. Run the Streamlit Application
streamlit run code/app.py

Open browser:
http://localhost:8501
8. Upload Documents
1. Upload Sanskrit document
2. Click 'Build RAG Pipeline'
3. Enter Sanskrit query
4. Click 'Generate Answer'
9. Sample Sanskrit Queries
योग क्या है?
धर्म क्या है?
कर्म क्या है?
मोक्ष क्या है?
योगश्चित्तवृत्तिनिरोधः का अर्थ क्या है?

10. Run Tests
python -m code.src.tests.test_retrieval
python -m code.src.tests.test_generator
python -m code.src.tests.test_pipeline

11. Generated Artifacts
artifacts/-
faiss_index/
logs/
metrics/
responses/

13. Conclusion
The Sanskrit RAG System demonstrates Sanskrit document ingestion, semantic retrieval, keyword retrieval, CPU-based LLM inference, and end-to-end Retrieval-Augmented Generation.

