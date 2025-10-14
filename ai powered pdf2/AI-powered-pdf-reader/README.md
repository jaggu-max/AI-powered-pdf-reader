# AI-Powered Contextual PDF Query Assistant

This is a prototype Streamlit application that lets you upload multi-page PDFs and ask natural-language questions. The system returns verbatim text snippets from the PDF (no summarization or rewriting) with page references.

Features
- Extracts text per page using PyMuPDF (fitz)
- Chunks text while preserving page numbers and block bounding boxes
- Computes embeddings with SentenceTransformers and stores them in FAISS
- Streamlit frontend for uploading PDFs and asking questions
- Renders page images for visual verification

Requirements
- Windows (tested instructions provided for PowerShell)
- Python 3.8+

Install (PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run

```powershell
streamlit run app.py
```

Notes and next steps
- This prototype uses local sentence-transformers embeddings to avoid remote API calls.
- Verbose exactness: the system returns the exact extracted text chunk as-is; it does not paraphrase.
- Improvements you can add:
  - OCR for scanned PDFs using Tesseract
  - Fine-grained bounding-box highlighting on the rendered page image
  - Persistent index saving/loading for large documents
  - Use OpenAI or other LLMs for re-ranking (careful to avoid rewriting answers)

If you'd like, I can:
- Add bounding-box highlighting onto the page image
- Add OCR fallback for scanned documents
- Add persistent index saving and a small test PDF to demo the flow
