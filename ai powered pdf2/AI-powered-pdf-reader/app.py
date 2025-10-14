
import os
import io
import tempfile
import streamlit as st
from typing import List

import fitz
import re
from PIL import Image

from pdf_processor import PDFProcessor

st.set_page_config(page_title="AI PDF Query Assistant", layout="wide")

# simple white background
st.markdown("<style>body {background: white;}</style>", unsafe_allow_html=True)

st.title("AI-Powered Contextual PDF Query Assistant")
st.write("Upload a PDF, ask questions in natural language, and get verbatim answers from the document with page references.")

# --- upload ---
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"]) 
processor = None
index_meta = None

if uploaded_file is not None:
    # save to temp file
    tdir = tempfile.gettempdir()
    pdf_path = os.path.join(tdir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Saved uploaded PDF to {pdf_path}")

    # process button
    if st.button("Process PDF (extract & index)"):
        with st.spinner("Extracting text and building index (this may take a moment)..."):
            try:
                processor = PDFProcessor()
                meta = processor.build_faiss_index(pdf_path)
                # show total pages automatically
                try:
                    doc = fitz.open(pdf_path)
                    page_count = len(doc)
                    doc.close()
                except Exception:
                    page_count = None

                if page_count is not None:
                    st.info(f"Document pages: {page_count}")
                st.success(f"Indexed {len(meta['chunks'])} chunks from the document.")
                # keep in session_state
                st.session_state['processor'] = processor
                st.session_state['pdf_path'] = pdf_path
                st.session_state['page_count'] = page_count
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

# restore existing in-memory processor if present
if 'processor' in st.session_state:
    processor = st.session_state['processor']
    pdf_path = st.session_state.get('pdf_path')

# --- question input ---
if processor is not None:
    st.markdown("---")
    st.subheader("Ask a question (returns verbatim matches)")
    query = st.text_input("Your question", value="What is the definition of overfitting?")
    k = st.slider("Number of matches to return", min_value=1, max_value=10, value=3)

    # Exact-match mode: allow user to specify exact text and a specific page
    exact_mode = st.checkbox("Exact text + page (find verbatim on a specific page)", value=False)
    exact_text = None
    exact_page = None
    if exact_mode:
        exact_text = st.text_input("Exact text to find (verbatim)")
        exact_page = st.number_input("Page number (1-based)", min_value=1, step=1)

    # Heading explanation mode: user provides a heading and we return the following N chunks as the explanation
    heading_mode = st.checkbox("Heading explanation mode (show content under a heading)", value=False)
    heading_text = None
    heading_context = 3
    if heading_mode:
        heading_text = st.text_input("Heading text (verbatim or partial)")
        heading_context = st.slider("Chunks to include after heading", min_value=1, max_value=10, value=3)

    if st.button("Search") and (query.strip() or (exact_mode and exact_text) or (heading_mode and heading_text)):
        with st.spinner("Searching..."):
            try:
                results = []

                # Heading explanation flow: find heading and return the next N chunks as the explanation
                if heading_mode and heading_text:
                    found = False
                    ht_lower = heading_text.strip().lower()
                    for i, c in enumerate(processor.chunks):
                        try:
                            if ht_lower in c.get('text', '').lower():
                                # gather the following heading_context chunks (same page preferred)
                                expl_chunks = []
                                collected = 0
                                j = i + 1
                                while collected < heading_context and j < len(processor.chunks):
                                    nxt = processor.chunks[j]
                                    # prefer same page but allow next-page content
                                    expl_chunks.append(nxt['text'])
                                    collected += 1
                                    j += 1
                                explanation = "\n\n".join(expl_chunks) if expl_chunks else "(no following content found)"
                                results.append({"score": 0.0, "text": explanation, "page": c.get('page', None), "bbox": None})
                                found = True
                                break
                        except Exception:
                            continue
                    if not found:
                        results = []
                elif exact_mode and exact_text:
                    # find exact verbatim matches in the loaded chunks, restricted to the given page
                    for c in processor.chunks:
                        try:
                            page_match = (int(c.get('page', -1)) == int(exact_page)) if exact_page is not None else True
                        except Exception:
                            page_match = True
                        if page_match and exact_text in c.get('text', ''):
                            results.append({"score": 0.0, "text": c['text'], "page": c['page'], "bbox": c.get('bbox')})
                else:
                    results = processor.search(query, top_k=k)

                if not results:
                    st.info("No matches found.")
                else:
                    # Optionally extract exact snippet: find sentence containing all significant query words
                    exact_snippet_mode = st.checkbox("Return exact snippet containing all query words (when available)", value=True)
                    if exact_snippet_mode and query.strip():
                        def extract_exact_snippet(text_block: str, query_str: str):
                            # tokenise query into significant tokens (length>2)
                            q_tokens = [t.lower() for t in re.findall(r"\w+", query_str) if len(t) > 2]
                            if not q_tokens:
                                return None
                            # split block into sentences
                            sents = re.split(r'(?<=[.!?])\s+', text_block)
                            for s in sents:
                                s_low = s.lower()
                                if all(tok in s_low for tok in q_tokens):
                                    return s.strip()
                            # fallback: if whole block contains all tokens, return the block
                            blk_low = text_block.lower()
                            if all(tok in blk_low for tok in q_tokens):
                                return text_block.strip()
                            return None

                        filtered = []
                        for r in results:
                            snip = extract_exact_snippet(r['text'], query)
                            if snip:
                                # replace text with the exact snippet
                                filtered.append({**r, 'text': snip})
                        if filtered:
                            results = filtered
                    # Option to show a combined text-only (verbatim) answer assembled from top matches
                    show_text_only = st.checkbox("Show text-only answers (verbatim)", value=True)
                    if show_text_only:
                        combined = "\n\n---\n\n".join([r["text"] for r in results])
                        st.subheader("Text-only answer (verbatim from document)")
                        # use a text area for easy copy/paste while keeping verbatim content
                        st.text_area("Answer (verbatim)", value=combined, height=200)

                    # Display each match with its cropped page image
                    for i, r in enumerate(results):
                        st.write(f"### Match {i+1} — page {r['page']}")
                        st.markdown("**Verbatim:**")
                        st.code(r['text'])  # show verbatim in code block
                        
                        # show cropped page image
                        try:
                            zoom = 2.0
                            img_path = processor.render_page_image(pdf_path, r['page'], zoom=zoom)
                            bbox = r.get('bbox')
                            if bbox:
                                x0, y0, x1, y1 = bbox
                                scale = zoom
                                margin = 2 * scale  # tighter crop around the answer
                                left = max(int(x0 * scale - margin), 0)
                                top = max(int(y0 * scale - margin), 0)
                                right = int(x1 * scale + margin)
                                bottom = int(y1 * scale + margin)
                                try:
                                    img = Image.open(img_path)
                                    w, h = img.size
                                    right = min(right, w)
                                    bottom = min(bottom, h)
                                    cropped = img.crop((left, top, right, bottom))
                                    cropped_path = os.path.join(tempfile.gettempdir(), f"crop_{os.path.basename(img_path)}")
                                    cropped.save(cropped_path)
                                    st.image(cropped_path, caption=f"Page {r['page']} (cropped)")
                                except Exception:
                                    st.image(img_path, caption=f"Page {r['page']}")
                            else:
                                # bbox missing: attempt to find the exact answer text on the page and crop to that rect
                                try:
                                    doc = fitz.open(pdf_path)
                                    page = doc[r['page'] - 1]
                                    # search for the exact text (case-sensitive first, then case-insensitive)
                                    rects = page.search_for(r['text'])
                                    if not rects:
                                        rects = page.search_for(r['text'], quads=False, flags=1)  # try case-insensitive
                                    if rects:
                                        rect = rects[0]
                                        # rect is in PDF coordinates; scale to image pixels using zoom
                                        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                                        scale = zoom
                                        margin = 2 * scale
                                        left = max(int(x0 * scale - margin), 0)
                                        top = max(int(y0 * scale - margin), 0)
                                        right = int(x1 * scale + margin)
                                        bottom = int(y1 * scale + margin)
                                        img = Image.open(img_path)
                                        w, h = img.size
                                        right = min(right, w)
                                        bottom = min(bottom, h)
                                        cropped = img.crop((left, top, right, bottom))
                                        cropped_path = os.path.join(tempfile.gettempdir(), f"crop_{os.path.basename(img_path)}")
                                        cropped.save(cropped_path)
                                        st.image(cropped_path, caption=f"Page {r['page']} (cropped)")
                                    else:
                                        st.image(img_path, caption=f"Page {r['page']}")
                                    doc.close()
                                except Exception:
                                    st.image(img_path, caption=f"Page {r['page']}")
                        except Exception as e:
                            st.warning(f"Could not render page image: {e}")
            except Exception as e:
                st.error(f"Search error: {e}")

else:
    st.info("Upload and process a PDF to enable searching.")

st.markdown("---")
st.write("Constraints: the assistant returns verbatim text extracted from the PDF; it does not summarize or reword answers.")
