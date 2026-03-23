# IBVL RAG Project — Knowledge Base Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from company documents using local LLMs via Ollama.

---

## Live UI Preview

Frontend built with React: [docu-buddy-ai-02.lovable.app](https://docu-buddy-ai-02.lovable.app)

---

## Project Structure

```
IBVL_RAG_PROJECT/
├── Module1_document_loading.py   # PDF & Excel document ingestion
├── chunking_m2.py                # Text chunking
├── module3_embeddings.py         # Embedding generation
├── module_4vectordb.py           # ChromaDB vector store
├── module5_llm.py                # LLM integration via Ollama
├── knowledge_base/               # Source PDF documents
├── docu-buddy-ai-02/             # React frontend + FastAPI backend
│   ├── fastapi_backend/
│   │   ├── main.py               # FastAPI REST API
│   │   └── requirements.txt      # Backend dependencies
│   └── src/                      # React frontend source
└── requirements.txt              # Python dependencies
```

---

## Prerequisites

Ensure the following are installed before running the project:

- Python 3.10+
- Node.js 18+
- Ollama with llama3 or mistral model (https://ollama.com)

---

## Setup and Running

### Step 1 — Clone the repository
```bash
git clone https://github.com/tanyavijj/IBVL_RAG_PROJECT.git
cd IBVL_RAG_PROJECT
```

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Start Ollama (separate terminal)
```bash
ollama serve
ollama pull llama3
```

### Step 4 — Start the FastAPI backend (separate terminal)
```bash
cd docu-buddy-ai-02/fastapi_backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Step 5 — Start the React frontend (separate terminal)
```bash
cd docu-buddy-ai-02
npm install
npm run dev
```

### Step 6 — Open in browser
```
http://localhost:8080
```

---

## API Endpoints

| Method | Endpoint     | Description                    |
|--------|--------------|--------------------------------|
| POST   | /chat        | Send a message, get a response |
| GET    | /documents   | List all indexed documents     |
| GET    | /history     | Get chat history               |
| POST   | /clear       | Clear chat history             |
| GET    | /health      | Check backend status           |

---

## How It Works

1. **Document Loading** — PDFs from knowledge_base/ are loaded and parsed
2. **Chunking** — Documents are split into smaller chunks for processing
3. **Embeddings** — Chunks are converted to vector embeddings
4. **Vector Database** — Embeddings are stored in ChromaDB for fast retrieval
5. **LLM** — Ollama runs the language model locally to generate answers
6. **Frontend** — React UI sends questions to FastAPI and displays answers

---

## Tech Stack

- **Backend:** FastAPI, LangChain, ChromaDB, Ollama
- **Frontend:** React, Vite, TailwindCSS
- **LLM:** Llama3 / Mistral (via Ollama)
- **Embeddings:** HuggingFace / Ollama

---

## Author

Tanya — IBVL RAG Project
