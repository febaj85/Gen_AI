import faiss
import pickle
import numpy as np
from embeddings import embed_texts
from config import openai_client

print("🔥 MAIN STARTED 🔥")

index = faiss.read_index("vectors.faiss")

with open("metadata.pkl", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]
metadatas = data["metadatas"]

print("Vector count:", index.ntotal)
input("➡️ Press ENTER to continue...")

def search(query, k=5):
    q_emb = embed_texts([query], batch_size=1, sleep_sec=0)[0]
    q_emb = np.array([q_emb]).astype("float32")

    distances, indices = index.search(q_emb, k)
    return [texts[i] for i in indices[0]]

def answer_question(question, chunks):
    # print(chunks)
    context = "\n\n".join(chunks)
    # print("lllllllllllllllllllllllllllllllll",context,"LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")

    response = openai_client.chat.completions.create(
        model="chatmodel",  # Azure chat deployment name
        messages=[
            {"role": "system", "content": "Answer using only the given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )

    return response.choices[0].message.content

while True:
    q = input("\nAsk something (or exit): ")
    if q.lower() == "exit":
        break

    chunks = search(q)
    answer = answer_question(q, chunks)

    print("\n🤖 Answer:\n")
    print(answer)
