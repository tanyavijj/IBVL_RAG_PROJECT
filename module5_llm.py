# ============================================================
# MODULE 5 — LLM + FULL RAG CHAIN
# ============================================================
# Brings everything together:
# 1. Loads ChromaDB (built in Module 4)
# 2. Query rewriting — rewrites vague questions for better retrieval
# 3. Retrieves relevant chunks from ChromaDB
# 4. Sends chunks + question to LLM (Llama3 or Mistral via Ollama)
# 5. Returns answer with source citations
# 6. Multi-turn conversation memory
# 7. Chat history saved to SQLite — auto deleted after 7 days
#
# Experiments:
# A — Llama3 vs Mistral answer quality comparison
# B — With vs without query rewriting
# C — Multi-turn conversation memory test
#
# HOW TO RUN:
#   python module5_llm.py
#   Make sure Ollama is running: ollama serve
# ============================================================

import os
import sqlite3
import datetime
import numpy as np
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

CHROMA_DIR = "chroma_db"
CHAT_DB = "chat_history.db"

# ── STEP 1: Setup SQLite chat history ─────────────────────────────────────────

def setup_database():
    conn = sqlite3.connect(CHAT_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_message(session_id, role, message):
    conn = sqlite3.connect(CHAT_DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?)",
        (session_id, role, message)
    )
    conn.commit()
    conn.close()

def load_session_history(session_id):
    conn = sqlite3.connect(CHAT_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, message FROM chat_history WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    )
    history = cursor.fetchall()
    conn.close()
    return history

def delete_old_history():
    conn = sqlite3.connect(CHAT_DB)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE timestamp < datetime('now', '-7 days')")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        print(f"Cleaned up {deleted} old messages (older than 7 days)")

def get_all_sessions():
    conn = sqlite3.connect(CHAT_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id, COUNT(*) FROM chat_history GROUP BY session_id")
    sessions = cursor.fetchall()
    conn.close()
    return sessions

print("\n" + "="*60)
print("STEP 1 — Setting up database and loading ChromaDB")
print("="*60)

setup_database()
delete_old_history()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
chroma_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = chroma_db.as_retriever(search_kwargs={"k": 4})
print("ChromaDB loaded! Retriever ready.")

# ── STEP 2: Query Rewriting ───────────────────────────────────────────────────

def rewrite_query(question, llm):
    prompt = f"""Rewrite this question to be more specific for searching company policy documents.
Return only the rewritten question, nothing else.

Original: {question}
Rewritten:"""
    return llm.invoke(prompt).strip()

# ── STEP 3: Build RAG chain with memory ───────────────────────────────────────

def build_rag_chain(llm):
    prompt_template = """You are a helpful assistant for a company knowledge base.
Answer ONLY from the provided context. If the answer is not in the context,
say "I don't have information about this in the knowledge base."
Always mention which document your answer comes from.

Context: {context}
Chat History: {chat_history}
Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )
    return chain, memory

def ask_question(chain, question, session_id, llm, use_rewriting=True):
    if use_rewriting:
        rewritten = rewrite_query(question, llm)
        print(f"  Original:  {question}")
        print(f"  Rewritten: {rewritten}")
        search_question = rewritten
    else:
        search_question = question

    result = chain.invoke({"question": search_question})
    answer = result["answer"]
    source_docs = result.get("source_documents", [])

    # get unique sources
    sources = []
    seen = set()
    for doc in source_docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            sources.append(f"{source} (p.{page})")
            seen.add(key)

    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    return answer, sources

# ── EXPERIMENT A — Llama3 vs Mistral ─────────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT A — Llama3 vs Mistral")
print("="*60)

test_questions = [
    "What happens when an employee account gets locked?",
    "What is the data backup retention period?",
    "What are the steps for employee offboarding?",
]

for model_name in ["llama3", "mistral"]:
    print(f"\n--- Testing {model_name} ---")
    try:
        llm = OllamaLLM(model=model_name, temperature=0.1)
        chain, _ = build_rag_chain(llm)

        for q in test_questions:
            print(f"\nQ: {q}")
            answer, sources = ask_question(chain, q, f"{model_name}_test", llm)
            print(f"A: {answer[:300]}")
            print(f"Sources: {', '.join(sources)}")

    except Exception as e:
        print(f"{model_name} error: {e}")
        print("Make sure model is pulled: ollama pull " + model_name)

# ── EXPERIMENT B — Query Rewriting ───────────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT B — With vs Without Query Rewriting")
print("="*60)

try:
    llm_test = OllamaLLM(model="llama3", temperature=0.1)
    chain_test, _ = build_rag_chain(llm_test)

    vague_questions = [
        "tell me about locking"
    ]

    for q in vague_questions:
        print(f"\nQuestion: '{q}'")

        print("\nWITHOUT rewriting:")
        answer, sources = ask_question(chain_test, q, "no_rewrite", llm_test, use_rewriting=False)
        print(f"Answer: {answer[:200]}")

        chain_test2, _ = build_rag_chain(llm_test)
        print("\nWITH rewriting:")
        answer, sources = ask_question(chain_test2, q, "with_rewrite", llm_test, use_rewriting=True)
        print(f"Answer: {answer[:200]}")

except Exception as e:
    print(f"Error: {e}")

# ── EXPERIMENT C — Multi-turn conversation memory ─────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT C — Multi-turn conversation memory")
print("="*60)

try:
    llm_conv = OllamaLLM(model="llama3", temperature=0.1)
    chain_conv, _ = build_rag_chain(llm_conv)

    conversation = [
        "What is the account lock policy?"
    ]

    session_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Session: {session_id}\n")

    for q in conversation:
        print(f"User: {q}")
        answer, sources = ask_question(chain_conv, q, session_id, llm_conv)
        print(f"Assistant: {answer[:250]}")
        print(f"Sources: {', '.join(sources)}\n")

    print(f"\n--- Saved to SQLite ---")
    history = load_session_history(session_id)
    print(f"Total messages saved: {len(history)}")

except Exception as e:
    print(f"Error: {e}")

# ── DATABASE SUMMARY ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CHAT HISTORY DATABASE")
print("="*60)
sessions = get_all_sessions()
print(f"Total sessions stored: {len(sessions)}")
for sid, count in sessions:
    print(f"  {sid} — {count} messages")
print("Old messages (>7 days) auto-deleted!")

print("\n" + "="*60)
print("MODULE 5 COMPLETE")
print("="*60)
