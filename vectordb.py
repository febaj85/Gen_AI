import faiss
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "vectors.faiss")
META_PATH = os.path.join(BASE_DIR, "metadata.pkl")

def save_faiss(embeddings, texts, metadatas):
    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(
            {
                "texts": texts,
                "metadatas": metadatas
            },
            f
        )

    print(f"✅ FAISS index saved with {index.ntotal} vectors")
