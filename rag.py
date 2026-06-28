import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

def get_embedding(text):
    response = requests.post(
        url="https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/text-embedding-3-small",
            "input": text
        }
    )
    return response.json()["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    mag1 = sum(a**2 for a in vec1) ** 0.5
    mag2 = sum(b**2 for b in vec2) ** 0.5
    return dot / (mag1 * mag2)

def ask_llm(question, context):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/auto",
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant for a wellness clinic.
Answer using ONLY the context below.
If answer not in context, say 'I don't have that information.'

Context:
{context}"""
                },
                {"role": "user", "content": question}
            ]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

def search(query, db, top_k=2):
    query_emb = get_embedding(query)
    scores = []
    for item in db:
        score = cosine_similarity(query_emb, item["embedding"])
        scores.append((score, item["text"]))
    scores.sort(reverse=True)
    return [text for _, text in scores[:top_k]]