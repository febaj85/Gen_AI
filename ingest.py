import os
import re
from pdf_loader import extract_sections, chunk_text
from embeddings import embed_texts
from vectordb import save_faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "DeepLearningIanGoodfellow.pdf")

print("PDF exists:", os.path.exists(PDF_PATH))

sections = extract_sections(PDF_PATH)
print("Sections extracted:", len(sections))

def clean_text(text: str) -> str:
    text = re.sub(r'[\uE000-\uF8FF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

texts = []
metadatas = []

for s in sections:
    if s["heading"].upper() == "CONTENTS":
        continue

    chunks = chunk_text(s["text"])
    # print(chunks,"_______________________________")

    for c in chunks:
        c = clean_text(c)
        if len(c) < 50:
            continue

        texts.append(c)
        metadatas.append({
            "page": s["page"],
            "heading": s["heading"]
        })

print("Total chunks created:", len(texts))

print("🔄 Generating embeddings...")
embeddings = embed_texts(texts)

if len(embeddings) != len(texts):
    raise ValueError("Embedding count mismatch")

print("💾 Saving to FAISS...")
save_faiss(embeddings, texts, metadatas)

print("✅ INGEST COMPLETE")
