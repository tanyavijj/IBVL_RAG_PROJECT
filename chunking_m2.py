# ============================================================
# MODULE 2 — CHUNKING (Section-Based for Structured PDFs)
# ============================================================
# Loads from parsed_docs.json (saved by Module 1)
#
# STRATEGY: Section-based chunking designed specifically for
# structured policy documents like IBVL's policy PDFs.
#
# WHAT THIS DOES:
# 1. Splits by numbered section headings (1., 2., Step 1: etc.)
#    → Each section becomes one complete chunk
# 2. Extracts section name into metadata
#    → e.g., metadata["section"] = "Lockout Duration"
# 3. Extracts document name into metadata
#    → e.g., metadata["document"] = "Account Lock Policy"
# 4. Filters out noise:
#    → Document control table
#    → Revision history
#    → Repeated headers/footers like "IMPERATIVE BUSINESS VENTURES LIMITED"
#    → Cover page content
# 5. Fallback: if section > 700 words → split with
#    RecursiveCharacterTextSplitter with overlap
# 6. Output: LangChain Document format with rich metadata
#
# OUTPUT: saves chunks to chunked_docs.json

#   python module2_chunking.py


import os
import re
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_JSON = "parsed_docs.json"
OUTPUT_JSON = "chunked_docs.json"

# ── STEP 1: Load from JSON ────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1 — Loading from parsed_docs.json")
print("="*60)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

all_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
print(f"Loaded {len(all_docs)} documents from JSON")


# ── STEP 2: Helper functions ──────────────────────────────────────────────────

# Noise patterns to filter out
NOISE_PATTERNS = [
    r"IMPERATIVE BUSINESS\s*\n?\s*VENTURES LIMITED",
    r"Imperative Business Ventures Limited",
    r"An Incredible Inspiration for Innovation",
    r"Confidential Do No Share",
    r"Style Collection",
    r"Document Control chart\.?",
    r"Document Revision chart\.?",
    r"Document Name\s+.*",
    r"Version\s+\d+\.\d+",
    r"Created on\s+.*",
    r"Created by\s+.*",
    r"Reviewed by\s+.*",
    r"Next Revision\s+.*",
    r"Document Approved by\s+.*",
    r"Date\s+Version\s+Reviewed by\s+Changes",
    r"\d{4}\s+\d+\.\d+\s+Director.*No changes",
    r"No changes",
]

def clean_text(text):
    """Remove noise patterns from text."""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def is_noise_chunk(text):
    """Returns True if chunk is mostly noise/boilerplate."""
    text_lower = text.lower()
    noise_indicators = [
        "document control",
        "document revision",
        "style collection",
        "confidential do no share",
        "an incredible inspiration",
        "no changes",
    ]
    # if text is very short or mostly noise
    if len(text.strip()) < 50:
        return True
    noise_count = sum(1 for n in noise_indicators if n in text_lower)
    if noise_count >= 2:
        return True
    return False

def extract_document_name(source_path):
    """Extract clean document name from file path."""
    filename = os.path.basename(source_path)
    # Remove .pdf extension and clean up
    name = filename.replace(".pdf", "").replace(".PDF", "")
    name = name.replace("_", " ").replace("-", " ")
    return name.strip()

def extract_section_name(section_text):
    """Extract the section heading from the beginning of a section."""
    lines = section_text.strip().split('\n')
    for line in lines[:3]:  # check first 3 lines
        line = line.strip()
        if line and len(line) > 3:
            # Remove leading number like "1." or "Step 1:"
            clean = re.sub(r'^[\d]+\.\s*', '', line)
            clean = re.sub(r'^Step\s+\d+[:.]\s*', '', clean, flags=re.IGNORECASE)
            if clean and len(clean) > 3:
                return clean[:80]  # max 80 chars for section name
    return "General"

def split_into_sections(text):
    """
    Split text by numbered section headers.
    Detects patterns like:
    - "1. Purpose"
    - "2. Scope"
    - "Step 1: Notification"
    - "3. Roles and Responsibilities"
    """
    # Pattern: newline followed by number+dot or Step N:
    section_pattern = re.compile(
        r'(?=\n\s*(?:\d{1,2}\.\s+[A-Z]|Step\s+\d+\s*[:.]))',
        re.MULTILINE
    )

    parts = section_pattern.split(text)
    sections = []

    for part in parts:
        part = part.strip()
        if part and len(part) > 50 and not is_noise_chunk(part):
            sections.append(part)

    return sections if sections else [text]  # fallback: return whole text


# ── STEP 3: Main chunking function ───────────────────────────────────────────

fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " "],
)

def chunk_pdf_document(doc):
    """
    Full pipeline for chunking a structured policy PDF:
    1. Clean noise/boilerplate
    2. Split by section headings
    3. Extract section name + document name into metadata
    4. Apply fallback splitter if section > 700 words
    """
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "0")
    doc_name = extract_document_name(source)

    # Clean the text first
    cleaned_text = clean_text(doc.page_content)

    # Skip if mostly noise after cleaning
    if len(cleaned_text) < 50 or is_noise_chunk(cleaned_text):
        return []

    # Split into sections
    sections = split_into_sections(cleaned_text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue

        section_name = extract_section_name(section)
        word_count = len(section.split())

        # Base metadata for every chunk
        base_metadata = {
            "source": source,
            "document": doc_name,
            "section": section_name,
            "page": str(page),
            "word_count": str(word_count),
        }

        if word_count > 700:
            # Section too large — apply fallback splitter
            sub_doc = Document(page_content=section, metadata=base_metadata)
            sub_chunks = fallback_splitter.split_documents([sub_doc])
            for i, sub in enumerate(sub_chunks):
                sub.metadata["section"] = f"{section_name} (part {i+1})"
                chunks.append(sub)
        else:
            # Section fits perfectly — keep as one chunk
            chunks.append(Document(
                page_content=section,
                metadata=base_metadata
            ))

    return chunks


# ── STEP 4: Process all documents ─────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 4 — Chunking all documents")
print("="*60)

all_chunks = []
pdf_chunks = []
other_chunks = []

for doc in all_docs:
    source = doc.metadata.get("source", "")

    if source.endswith(".pdf"):
        chunks = chunk_pdf_document(doc)
        pdf_chunks.extend(chunks)
        all_chunks.extend(chunks)
    else:
        # Excel rows and web docs — keep as is, just clean
        cleaned = clean_text(doc.page_content)
        if cleaned and len(cleaned) > 30:
            doc.page_content = cleaned
            doc.metadata["document"] = extract_document_name(source) if source else "Web Content"
            doc.metadata["section"] = "Data"
            other_chunks.append(doc)
            all_chunks.append(doc)

print(f"PDF chunks:   {len(pdf_chunks)}")
print(f"Other chunks: {len(other_chunks)}")
print(f"Total chunks: {len(all_chunks)}")


# ── STEP 5: Show sample chunks ────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 5 — Sample chunks with metadata")
print("="*60)

print("\nShowing 3 sample PDF chunks:")
shown = 0
for chunk in all_chunks:
    if chunk.metadata.get("source", "").endswith(".pdf") and shown < 3:
        print(f"\n--- Chunk {shown+1} ---")
        print(f"Document: {chunk.metadata.get('document')}")
        print(f"Section:  {chunk.metadata.get('section')}")
        print(f"Page:     {chunk.metadata.get('page')}")
        print(f"Words:    {chunk.metadata.get('word_count')}")
        print(f"Content preview:\n{chunk.page_content[:300]}")
        shown += 1


# ── STEP 6: Quality check ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 6 — Quality check")
print("="*60)

# Check if noise is being filtered
noise_found = [c for c in all_chunks if "imperative business" in c.page_content.lower()
               or "style collection" in c.page_content.lower()
               or "confidential do no share" in c.page_content.lower()]

print(f"Chunks with noise remaining: {len(noise_found)} (should be 0 or very low)")

# Show chunks from Account Lock Policy as a specific test
print("\nAccount Lock Policy chunks:")
alp_chunks = [c for c in all_chunks if "Account Lock" in c.metadata.get("document", "")]
for chunk in alp_chunks:
    print(f"  Section: {chunk.metadata.get('section')} | Words: {chunk.metadata.get('word_count')}")


# ── STEP 7: Save to JSON ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 7 — Saving to chunked_docs.json")
print("="*60)

chunks_to_save = []
for chunk in all_chunks:
    chunks_to_save.append({
        "page_content": chunk.page_content,
        "metadata": {k: str(v) for k, v in chunk.metadata.items()}
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(chunks_to_save, f, indent=2, ensure_ascii=False)

print(f"Saved {len(chunks_to_save)} chunks to {OUTPUT_JSON}")
print(f"File size: {os.path.getsize(OUTPUT_JSON) / 1024:.1f} KB")

