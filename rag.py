"""RAG pipeline for ML Study Buddy.

Steps 1-4 of the pipeline live here:
  1. INGEST   - load the markdown study guides from kb/
  2. CHUNK    - split them into ~500-token passages on heading boundaries
  3. EMBED    - turn every chunk into a vector with a sentence-transformer
  4. RETRIEVE - find the chunks most similar to a question (cosine similarity)
Step 5 (GENERATE) lives in app.py, where the LLM writes the answer.
"""

import glob
import os

import numpy as np
from sentence_transformers import SentenceTransformer

KB_DIR = os.path.join(os.path.dirname(__file__), "kb")
CHUNK_WORDS = 380     # ~500 tokens (1 token ~= 0.75 words)
OVERLAP_WORDS = 40    # ~50 tokens of overlap when a long section is split
TOP_K = 4             # how many chunks to hand to the LLM
MIN_SIMILARITY = 0.35 # below this cosine score a question is "out of scope"

# The embedding model: small, free, runs fine on Hugging Face's CPU tier.
_model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 1 — INGEST: read every markdown file in kb/ as (filename, text).
def load_documents():
    docs = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


# STEP 2 — CHUNK: split on headings first, then pack sections into chunks.
def chunk_document(source, text):
    # First pass: cut the document into sections at every markdown heading,
    # keeping each heading attached to the text below it.
    sections, current = [], []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    sections.append("\n".join(current))

    # Second pass: greedily pack adjacent sections into ~CHUNK_WORDS chunks,
    # so each chunk stays coherent but isn't wastefully small.
    chunks, buffer = [], ""
    for section in sections:
        if buffer and len((buffer + section).split()) > CHUNK_WORDS:
            chunks.append({"source": source, "text": buffer.strip()})
            buffer = ""
        buffer += section + "\n"
    if buffer.strip():
        chunks.append({"source": source, "text": buffer.strip()})

    # Third pass: if any single chunk is still too long, split it by word
    # count with a small overlap so no idea is cut cleanly in half.
    final = []
    for chunk in chunks:
        words = chunk["text"].split()
        start = 0
        while start < len(words):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            final.append({"source": chunk["source"], "text": piece})
            if start + CHUNK_WORDS >= len(words):
                break
            start += CHUNK_WORDS - OVERLAP_WORDS
    return final


# STEP 3 — EMBED: one normalized vector per chunk, stored in a numpy array.
def build_index():
    chunks = [c for src, text in load_documents()
              for c in chunk_document(src, text)]
    vectors = _model.encode([c["text"] for c in chunks],
                            normalize_embeddings=True)
    return chunks, np.asarray(vectors)


# STEP 4 — RETRIEVE: embed the question, rank chunks by cosine similarity.
def retrieve(query, chunks, vectors, top_k=TOP_K, min_similarity=MIN_SIMILARITY):
    query_vec = _model.encode([query], normalize_embeddings=True)[0]
    # Vectors are normalized, so a dot product IS the cosine similarity.
    sims = vectors @ query_vec
    best = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in best
            if sims[i] >= min_similarity]
