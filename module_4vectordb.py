# ============================================================
# MODULE 4 — VECTOR DATABASE
# ============================================================
# Loads from chunked_docs.json (saved by Module 2)
# Builds ChromaDB and FAISS, compares both
#
# HOW TO RUN:
#   python module4_vectordb.py
#   Make sure Ollama is running: ollama serve
# ============================================================

import os
import json
import time
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

INPUT_JSON = "chunked_docs.json"
CHROMA_DIR = "chroma_db"
FAISS_DIR = "faiss_db"

# ── STEP 1: Load from JSON ────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1 — Loading from chunked_docs.json")
print("="*60)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
print(f"Loaded {len(chunks)} chunks from JSON instantly!")

# ── STEP 2: Filter oversized chunks ──────────────────────────────────────────
# nomic-embed-text has a context limit of ~8192 tokens
# We use 4000 chars as a safe limit to avoid exceeding it

print("\n" + "="*60)
print("STEP 2 — Filtering oversized chunks")
print("="*60)

MAX_CHARS = 4000

normal_chunks = [c for c in chunks if len(c.page_content) <= MAX_CHARS]
oversized_chunks = [c for c in chunks if len(c.page_content) > MAX_CHARS]

print(f"Normal chunks:    {len(normal_chunks)}")
print(f"Oversized chunks: {len(oversized_chunks)} (will be split further)")

if oversized_chunks:
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=100
    )
    split_chunks = fallback_splitter.split_documents(oversized_chunks)
    print(f"After splitting oversized: {len(split_chunks)} chunks")
    chunks = normal_chunks + split_chunks
else:
    chunks = normal_chunks

print(f"Final total chunks: {len(chunks)}")

# ── STEP 3: Initialize embedding model ───────────────────────────────────────

print("\n" + "="*60)
print("STEP 3 — Initializing embedding model")
print("="*60)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("Using nomic-embed-text (chosen from Module 3 results)")

# ── STEP 4: BUILD CHROMADB ────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT A — Building ChromaDB")
print("="*60)
print("Embedding and storing all chunks... (few minutes)")

start = time.time()
chroma_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
chroma_build_time = time.time() - start
print(f"ChromaDB built in: {chroma_build_time:.2f}s")
print(f"Saved automatically to: {CHROMA_DIR}/")

# ── STEP 5: BUILD FAISS ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT B — Building FAISS")
print("="*60)

start = time.time()
faiss_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
faiss_build_time = time.time() - start
faiss_db.save_local(FAISS_DIR)
print(f"FAISS built in: {faiss_build_time:.2f}s")
print(f"Manually saved to: {FAISS_DIR}/")

# ── STEP 6: QUERY BOTH ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 6 — Querying both databases")
print("="*60)

test_queries = [
    "what happens when an account is locked",
    "data backup retention policy",
    "employee offboarding process steps",
    "network security requirements",
    "patch management procedure",
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: '{query}'")

    start = time.time()
    chroma_results = chroma_db.similarity_search_with_score(query, k=2)
    chroma_time = time.time() - start
    print(f"\nChromaDB ({chroma_time*1000:.1f}ms):")
    for doc, score in chroma_results:
        print(f"  Score:    {score:.4f}")
        print(f"  Document: {doc.metadata.get('document', '')}")
        print(f"  Section:  {doc.metadata.get('section', '')}")
        print(f"  Content:  {doc.page_content[:150]}")

    start = time.time()
    faiss_results = faiss_db.similarity_search_with_score(query, k=2)
    faiss_time = time.time() - start
    print(f"\nFAISS ({faiss_time*1000:.1f}ms):")
    for doc, score in faiss_results:
        print(f"  Score:    {score:.4f}")
        print(f"  Document: {doc.metadata.get('document', '')}")
        print(f"  Section:  {doc.metadata.get('section', '')}")
        print(f"  Content:  {doc.page_content[:150]}")

# ── STEP 7: UPDATE TEST ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 7 — Testing document updates")
print("="*60)

new_doc = Document(
    page_content="New Password Policy: All passwords must be at least 12 characters. Changed every 90 days. MFA mandatory for all admin accounts.",
    metadata={"source": "new_password_policy.pdf", "document": "New Password Policy", "section": "Password Requirements", "page": "0"}
)

start = time.time()
chroma_db.add_documents([new_doc])
print(f"ChromaDB updated in: {(time.time()-start)*1000:.1f}ms — automatic, no rebuild!")

start = time.time()
faiss_db.add_documents([new_doc])
faiss_db.save_local(FAISS_DIR)
print(f"FAISS updated in: {(time.time()-start)*1000:.1f}ms — must manually save!")

print("\nVerifying new doc found:")
result = chroma_db.similarity_search("password policy", k=1)[0]
print(f"ChromaDB: {result.metadata.get('document')} — {result.metadata.get('section')}")
result = faiss_db.similarity_search("password policy", k=1)[0]
print(f"FAISS:    {result.metadata.get('document')} — {result.metadata.get('section')}")

# ── STEP 8: LOAD FROM DISK ────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 8 — Loading from disk")
print("="*60)

start = time.time()
chroma_loaded = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
print(f"ChromaDB loaded in: {time.time()-start:.2f}s — just point to folder!")

start = time.time()
faiss_loaded = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
print(f"FAISS loaded in: {time.time()-start:.2f}s — needs allow_dangerous_deserialization=True")

# ── FINAL COMPARISON ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL COMPARISON — ChromaDB vs FAISS")
print("="*60)
print(f"""
{'Feature':<25} {'ChromaDB':<30} {'FAISS'}
{'-'*70}
{'Build time':<25} {f'{chroma_build_time:.1f}s':<30} {f'{faiss_build_time:.1f}s'}
{'Persistence':<25} {'Automatic':<30} {'Manual save required'}
{'Update docs':<25} {'Just add, no rebuild':<30} {'Add + re-save manually'}
{'Load from disk':<25} {'Point to folder':<30} {'load_local() needed'}
{'Best for':<25} {'RAG with updates':<30} {'Speed-critical static data'}

DECISION: ChromaDB for this project
Reason: Data updates continuously (website + new policies)
ChromaDB handles this automatically without rebuild.
""")