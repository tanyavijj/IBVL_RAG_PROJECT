# ============================================================
# MODULE 1 — DOCUMENT LOADING & INGESTION
# ============================================================
# Sources:
#   1. PDF files from knowledge_base/ folder
#   2. Excel file (All Policies Revision Dates.xlsx)
#   3. Website scraping using crawl4ai (Playwright internally)
#
# OUTPUT: saves all docs to parsed_docs.json

#   python module1_document_loading.py


import os
import json
import asyncio
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from crawl4ai import AsyncWebCrawler

KNOWLEDGE_BASE_DIR = "knowledge_base"
OUTPUT_JSON = "parsed_docs.json"

# ── PART 1: LOAD ALL PDF FILES ────────────────────────────────────────────────

print("\n" + "="*60)
print("PART 1 — Loading PDF files")
print("="*60)

all_pdf_docs = []

for filename in os.listdir(KNOWLEDGE_BASE_DIR):
    if filename.endswith(".pdf"):
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        all_pdf_docs.extend(docs)
        print(f"Loaded: {filename} → {len(docs)} page(s)")

print(f"\nTotal PDF pages loaded: {len(all_pdf_docs)}")
print(f"\nSample metadata: {all_pdf_docs[0].metadata}")
print(f"Sample content preview:\n{all_pdf_docs[0].page_content[:300]}")

# ── PART 2: LOAD EXCEL FILE ───────────────────────────────────────────────────

print("\n" + "="*60)
print("PART 2 — Loading Excel file")
print("="*60)

excel_docs = []
excel_path = os.path.join(KNOWLEDGE_BASE_DIR, "All Policies Revision Dates.xlsx")

if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    print(f"\nPreview:\n{df.head()}")

    for i, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        doc = Document(
            page_content=row_text,
            metadata={"source": excel_path, "row": str(i)}
        )
        excel_docs.append(doc)

    print(f"\nConverted {len(excel_docs)} rows into Documents")
else:
    print("Excel file not found!")

# ── PART 3: SCRAPE WEBSITE USING CRAWL4AI ────────────────────────────────────
# crawl4ai uses Playwright internally as its browser engine
# Opens pages like a real browser, renders JavaScript,
# and gives clean markdown text — much better than raw HTML scrapers

print("\n" + "="*60)
print("PART 3 — Website scraping with crawl4ai")
print("="*60)

IBVL_URLS = [
    "https://theimperative.in/",
    "https://theimperative.in/products.html",
]

async def scrape_website(urls):
    web_docs = []
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            print(f"\nScraping: {url}")
            try:
                result = await crawler.arun(url=url)
                if result.success:
                    doc = Document(
                        page_content=result.markdown,
                        metadata={
                            "source": url,
                            "title": result.metadata.get("title", ""),
                        }
                    )
                    web_docs.append(doc)
                    print(f"Success! Scraped {len(result.markdown)} characters")
                else:
                    print(f"Failed: {result.error_message}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")
    return web_docs

web_docs = asyncio.run(scrape_website(IBVL_URLS))
print(f"\nTotal web pages scraped: {len(web_docs)}")

# ── PART 4: COMBINE ALL ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("PART 4 — Combining all sources")
print("="*60)

all_docs = []
all_docs.extend(all_pdf_docs)
all_docs.extend(excel_docs)
all_docs.extend(web_docs)

print(f"PDF docs:     {len(all_pdf_docs)}")
print(f"Excel docs:   {len(excel_docs)}")
print(f"Web docs:     {len(web_docs)}")
print(f"Total:        {len(all_docs)}")

# ── PART 5: SAVE TO JSON ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("PART 5 — Saving to parsed_docs.json")
print("="*60)

docs_to_save = []
for doc in all_docs:
    docs_to_save.append({
        "page_content": doc.page_content,
        "metadata": {k: str(v) for k, v in doc.metadata.items()}
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(docs_to_save, f, indent=2, ensure_ascii=False)

print(f"Saved {len(docs_to_save)} documents to {OUTPUT_JSON}")
print(f"File size: {os.path.getsize(OUTPUT_JSON) / 1024:.1f} KB")
print("\nModule 2 will load from this JSON — no re-parsing needed!")