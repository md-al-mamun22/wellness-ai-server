from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_embedding, search, ask_llm
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clinic knowledge base
documents = [
    "Acupuncture is effective for back pain and muscle tension. Sessions are 45 minutes.",
    "Yoga classes help reduce stress and improve flexibility. Morning and evening classes available.",
    "Massage therapy relieves chronic pain and improves blood circulation.",
    "Our clinic is open Monday to Friday, 9am to 6pm. Saturday 10am to 4pm.",
    "We accept most major insurance plans including BlueCross and Aetna.",
    "New patient consultation is free for the first visit.",
    "We offer nutrition counseling sessions every Tuesday.",
    "Dr. Sarah specializes in sports injury rehabilitation."
]

# Startup-এ DB বানাও
print("Building knowledge base...")
db = []
for doc in documents:
    embedding = get_embedding(doc)
    db.append({"text": doc, "embedding": embedding})
    print(f"  ✓ Indexed: {doc[:40]}...")
print("Ready!\n")

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "Wellness AI Server is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    relevant_docs = search(request.question, db)
    context = "\n".join(relevant_docs)
    answer = ask_llm(request.question, context)
    return {"answer": answer}