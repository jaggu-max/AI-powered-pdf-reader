import os
import re
import pickle
import tempfile
from typing import List, Dict, Any

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None


class PDFProcessor:
    """Extracts text chunks from PDFs, computes embeddings and builds a FAISS index.

    This processor preserves verbatim text and page numbers and keeps block-level
    bounding boxes when available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None
        self.chunks: List[Dict[str, Any]] = []
        self.dim = self.model.get_sentence_embedding_dimension()

    def _split_long_text(self, text: str, max_chars: int = 800) -> List[str]:
        # split on sentences while keeping close to max_chars
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current = []
        cur_len = 0
        for s in sentences:
            if cur_len + len(s) <= max_chars:
                current.append(s)
                cur_len += len(s)
            else:
                if current:
                    chunks.append(" ".join(current).strip())
                current = [s]
                cur_len = len(s)
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def extract_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts blocks of text from each page and returns chunks with metadata.

        Each chunk dict contains:
          - text: verbatim extracted text
          - page: 1-based page number
          - bbox: block bbox (x0, y0, x1, y1) when available else None
        """
        self.chunks = []
        doc = fitz.open(pdf_path)
        for pno in range(len(doc)):
            page = doc[pno]
            blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no)
            if not blocks:
                # fallback to whole page text
                text = page.get_text("text")
                if text.strip():
                    for c in self._split_long_text(text):
                        self.chunks.append({"text": c, "page": pno + 1, "bbox": None})
                continue

            for b in blocks:
                x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
                btext = btext.strip()
                if not btext:
                    continue
                # split long blocks into smaller chunks
                small_chunks = self._split_long_text(btext)
                for sc in small_chunks:
                    self.chunks.append({"text": sc, "page": pno + 1, "bbox": (x0, y0, x1, y1)})
        doc.close()
        return self.chunks

    def build_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Compute embeddings for a list of texts using sentence-transformers."""
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embs

    def build_faiss_index(self, pdf_path: str, index_path: str = None) -> Dict[str, Any]:
        """Full pipeline: extract chunks, compute embeddings and build a FAISS index.

        Returns a metadata dict that can be pickled for later loading.
        """
        self.extract_chunks(pdf_path)
        texts = [c["text"] for c in self.chunks]
        if not texts:
            raise ValueError("No text found in PDF.")
        self.embeddings = self.build_embeddings(texts)

        if faiss is None:
            raise RuntimeError("faiss is not available. Install faiss-cpu to use the index.")

        index = faiss.IndexFlatL2(self.dim)
        index.add(self.embeddings.astype('float32'))
        self.index = index

        meta = {"chunks": self.chunks, "model_name": self.model_name, "dim": self.dim}

        if index_path:
            # save index and metadata
            faiss.write_index(self.index, index_path + ".index")
            with open(index_path + ".meta.pkl", "wb") as f:
                pickle.dump(meta, f)
        return meta

    def save(self, index_path: str):
        if self.index is None or self.chunks is None:
            raise RuntimeError("No index to save. Run build_faiss_index first.")
        faiss.write_index(self.index, index_path + ".index")
        with open(index_path + ".meta.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "model_name": self.model_name, "dim": self.dim}, f)

    def load(self, index_path: str):
        if faiss is None:
            raise RuntimeError("faiss is not available. Install faiss-cpu to use the index.")
        self.index = faiss.read_index(index_path + ".index")
        with open(index_path + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta.get("chunks", [])
        self.model_name = meta.get("model_name", self.model_name)
        # ensure model matches
        if self.model_name != self.model._first_module().config.get('name_or_path', self.model_name):
            # ignore mismatch for now
            pass

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index and return the top_k verbatim matches with metadata."""
        if self.index is None:
            raise RuntimeError("Index not loaded. Call build_faiss_index or load first.")
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb.astype('float32'), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[int(idx)]
            results.append({"score": float(dist), "text": chunk["text"], "page": chunk["page"], "bbox": chunk.get("bbox")})
        return results

    def render_page_image(self, pdf_path: str, page_number: int, zoom: float = 2.0) -> str:
        """Render a page to a temporary PNG file and return its path.

        page_number is 1-based.
        """
        doc = fitz.open(pdf_path)
        pno = page_number - 1
        if pno < 0 or pno >= len(doc):
            doc.close()
            raise IndexError("page_number out of range")
        page = doc[pno]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        tmp = tempfile.gettempdir()
        out_path = os.path.join(tmp, f"pdf_page_{os.path.basename(pdf_path)}_{page_number}.png")
        pix.save(out_path)
        doc.close()
        return out_path
