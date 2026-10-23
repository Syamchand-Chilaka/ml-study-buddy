"""Sanity tests for the RAG pipeline. Run with:  python test_rag.py"""

import numpy as np

import rag

print("Building index...")
chunks, vectors = rag.build_index()

# INGEST: all three study guides were loaded.
sources = {c["source"] for c in chunks}
assert sources == {"ml-fundamentals-guide.md",
                   "generative-ai-and-rag-guide.md",
                   "responsible-ai-and-study-guide.md"}, sources
print(f"ok: ingest — {len(chunks)} chunks from {len(sources)} files")

# CHUNK: no chunk exceeds the size limit, and none is empty.
assert all(0 < len(c["text"].split()) <= rag.CHUNK_WORDS for c in chunks)
print("ok: chunk — all chunks within size limit")

# EMBED: one unit-length vector per chunk.
assert vectors.shape[0] == len(chunks)
assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
print("ok: embed — one normalized vector per chunk")

# RETRIEVE: on-topic questions hit the right guide with a confident score.
hits = rag.retrieve("What is overfitting?", chunks, vectors)
assert hits and hits[0][0]["source"] == "ml-fundamentals-guide.md", hits[:1]
hits = rag.retrieve("Explain the steps of a RAG pipeline", chunks, vectors)
assert hits and hits[0][0]["source"] == "generative-ai-and-rag-guide.md"
hits = rag.retrieve("What does ethical AI mean?", chunks, vectors)
assert hits and hits[0][0]["source"] == "responsible-ai-and-study-guide.md"
assert all(len(rag.retrieve(q, chunks, vectors)) <= rag.TOP_K
           for q in ["machine learning", "quiz"])
print("ok: retrieve — correct guide ranked first, top_k respected")

# THRESHOLD: questions outside the guides retrieve nothing at all.
for off_topic in ["What is quantum computing?",
                  "Best pizza topping in Naples?",
                  "Who won the 1998 World Cup?"]:
    assert rag.retrieve(off_topic, chunks, vectors) == [], off_topic
print("ok: threshold — out-of-scope questions return no chunks")

print("\nAll RAG tests passed ✅")
