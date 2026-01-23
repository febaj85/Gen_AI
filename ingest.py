import os
import re
from pdf_loader import extract_sections, chunk_text
from embeddings import embed_texts
from vectordb import save_faiss
from pdf_tracker import load_processed, save_processed

DATA_DIR = "data"

processed = load_processed()

all_texts = []
all_metadatas = []

for pdf_file in os.listdir(DATA_DIR):
    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(DATA_DIR, pdf_file)
    last_modified = os.path.getmtime(pdf_path)

    # ✅ Skip already processed PDFs
    if pdf_file in processed and processed[pdf_file] == last_modified:
        print(f"⏭️ Skipping already processed: {pdf_file}")
        continue

    print(f"📄 Processing NEW PDF: {pdf_file}")

    sections = extract_sections(pdf_path)

    for s in sections:
        if s["heading"].upper() == "CONTENTS":
            continue

        chunks = chunk_text(s["text"])

        for c in chunks:
            c = re.sub(r'\s+', ' ', c).strip()
            if len(c) < 50:
                continue

            all_texts.append(c)
            all_metadatas.append({
                "pdf": pdf_file,
                "page": s["page"],
                "heading": s["heading"]
            })

    # ✅ Mark PDF as processed
    processed[pdf_file] = last_modified

# ❗ If no new PDFs
if not all_texts:
    print("✅ No new PDFs found. Nothing to ingest.")
    exit()

print("🔄 Generating embeddings...")
embeddings = embed_texts(all_texts)

print("💾 Saving to FAISS...")
save_faiss(embeddings, all_texts, all_metadatas)

save_processed(processed)

print("✅ INGEST COMPLETE")
