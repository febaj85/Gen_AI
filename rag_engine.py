from vectordb import collection
from embeddings import embed_texts
from config import openai_client

def get_heading_content(heading):
    results = collection.get(
        where={"heading": {"$contains": heading.upper()}}
    )

    if not results["documents"]:
        return None

    return "\n\n".join(results["documents"])


def retrieve_chunks(question, k=10):
    q_embedding = embed_texts([question], batch_size=1, sleep_sec=0)[0]

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=k
    )

    return results["documents"][0]


# def answer_question(question):
#     chunks = retrieve_chunks(question)
#     context = "\n\n".join(chunks)
#     print(context,"__________////////////////////////////")

#     response = openai_client.chat.completions.create(
#         model="chatmodel",  # deployment name
#         messages=[
#             {"role": "system", "content": "Answer only using the provided context."},
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
#         ]
#     )

#     return response.choices[0].message.content
