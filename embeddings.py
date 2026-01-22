from config import embedding_client
import time

def embed_texts(texts, batch_size=10, sleep_sec=1):
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_no = i // batch_size + 1
        print(f"🔄 Embedding batch {batch_no}/{total_batches}")

        batch = texts[i:i + batch_size]

        response = embedding_client.embeddings.create(
            model="embedding-model",  # Azure deployment name
            input=batch
        )

        all_embeddings.extend([d.embedding for d in response.data])
        time.sleep(sleep_sec)

    return all_embeddings
