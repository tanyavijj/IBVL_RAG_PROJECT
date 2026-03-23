# ============================================================
# MODULE 3 — EMBEDDINGS
# ============================================================
# Loads from chunked_docs.json (saved by Module 2)
#
# Experiments:
# Model A — all-MiniLM-L6-v2 (HuggingFace, lightweight)
# Model B — nomic-embed-text (Ollama, fully local)
#
# Compares speed, vector dimensions, retrieval quality
#
# HOW TO RUN:
#   python module3_embeddings.py


import os
import json
import time
import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

INPUT_JSON = "chunked_docs.json"

# ── STEP 1: Load from JSON ────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1 — Loading from chunked_docs.json")
print("="*60)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
print(f"Loaded {len(chunks)} chunks from JSON instantly!")

sample_chunks = chunks[:50]
sample_texts = [c.page_content for c in sample_chunks]
print(f"Using {len(sample_texts)} sample chunks for experiments")

# ── STEP 2: MODEL A — all-MiniLM-L6-v2 ───────────────────────────────────────

print("\n" + "="*60)
print("MODEL A — all-MiniLM-L6-v2 (HuggingFace)")
print("="*60)

start = time.time()
model_a = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print(f"Model loaded in: {time.time() - start:.2f}s")

start = time.time()
embeddings_a = model_a.embed_documents(sample_texts)
embed_time_a = time.time() - start
print(f"Embedded 50 chunks in: {embed_time_a:.2f}s")
print(f"Vector dimensions: {len(embeddings_a[0])}")
print(f"Sample vector (first 5 values): {embeddings_a[0][:5]}")

# ── STEP 3: MODEL B — nomic-embed-text ────────────────────────────────────────

print("\n" + "="*60)
print("MODEL B — nomic-embed-text (Ollama)")
print("="*60)

ollama_available = False
embed_time_b = 0

try:
    start = time.time()
    model_b = OllamaEmbeddings(model="nomic-embed-text")
    print(f"Model loaded in: {time.time() - start:.2f}s")

    start = time.time()
    embeddings_b = model_b.embed_documents(sample_texts)
    embed_time_b = time.time() - start
    print(f"Embedded 50 chunks in: {embed_time_b:.2f}s")
    print(f"Vector dimensions: {len(embeddings_b[0])}")
    ollama_available = True

except Exception as e:
    print(f"Ollama not available: {e}")
    print("Make sure Ollama is running: ollama serve")

# ── STEP 4: SIMILARITY / RETRIEVAL QUALITY TEST ───────────────────────────────

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_top_chunks(query, model, chunks, top_k=3):
    query_vec = model.embed_query(query)
    chunk_vecs = model.embed_documents([c.page_content for c in chunks])
    scores = [(cosine_similarity(query_vec, v), idx, chunks[idx]) for idx, v in enumerate(chunk_vecs)]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [(s, c) for s, _, c in scores[:top_k]]

print("\n" + "="*60)
print("STEP 4 — Retrieval Quality Test")
print("="*60)

test_queries = [
    "what happens when account is locked",
    "data backup retention policy",
    "employee offboarding steps",
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: '{query}'")

    print("\nModel A (all-MiniLM-L6-v2):")
    for score, chunk in find_top_chunks(query, model_a, sample_chunks):
        print(f"  Score: {score:.4f} | {os.path.basename(chunk.metadata.get('source',''))}")
        print(f"  {chunk.page_content[:150]}")

    if ollama_available:
        print("\nModel B (nomic-embed-text):")
        for score, chunk in find_top_chunks(query, model_b, sample_chunks):
            print(f"  Score: {score:.4f} | {os.path.basename(chunk.metadata.get('source',''))}")
            print(f"  {chunk.page_content[:150]}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Model A — all-MiniLM-L6-v2:
  Speed:      {embed_time_a:.2f}s for 50 chunks
  Dimensions: {len(embeddings_a[0])}
  Size:       ~90MB
  Needs:      No Ollama, downloads automatically
""")

if ollama_available:
    print(f"""Model B — nomic-embed-text:
  Speed:      {embed_time_b:.2f}s for 50 chunks
  Dimensions: {len(embeddings_b[0])}
  Size:       ~270MB
  Needs:      Ollama running locally
""")

