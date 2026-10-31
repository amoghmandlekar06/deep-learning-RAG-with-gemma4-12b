# College Knowledge Assistant

An AI-powered college information assistant
built using Retrieval-Augmented Generation (RAG).

## Technologies

- Python
- Streamlit
- Ollama
- Gemma 4 12B
- Nomic Embed Text
- ChromaDB
- PyMuPDF

## Architecture

1. Users upload PDF/TXT documents.
2. PyMuPDF extracts document text.
3. Text is divided into smaller chunks.
4. Nomic generates embeddings.
5. ChromaDB stores the embeddings.
6. Student questions are converted into embeddings.
7. ChromaDB retrieves relevant passages.
8. Gemma generates answers using retrieved context.
9. Sources are displayed with the answers.

## Installation

Install Python 3.12, then create and activate a virtual environment:

python3.12 -m venv .venv312

source .venv312/bin/activate

Install dependencies:

pip install -r requirements.txt

Download models:

ollama pull gemma4:12b

ollama pull nomic-embed-text

## Run Application

streamlit run app.py

## Project Files

- app.py: Streamlit interface
- ingest.py: Document extraction and chunking
- vector_store.py: Embeddings and ChromaDB
- rag_pipeline.py: Retrieval and AI response generation

## Limitations

- Scanned PDFs require OCR.
- Answer quality depends on document quality.
- Conflicting information requires human verification.
- Retrieval thresholds require evaluation.