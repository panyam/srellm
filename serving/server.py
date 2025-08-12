# serving/server.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
from .providers import LocalMockProvider, OpenRouterProvider
from rag import qdrant_retriever as qd
from rag.retriever import build_prompt

app = FastAPI()
PROVIDER = os.getenv("PROVIDER", "mock")
provider = LocalMockProvider() if PROVIDER == "mock" else OpenRouterProvider()

retriever = qd.QdrantRetriever(collection="sre_runbooks")  # <— runbooks only

class GenRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    top_k: int = 3

@app.post("/generate")
def generate(req: GenRequest):
    chunks = retriever.search(req.prompt, k=req.top_k)
    composed = build_prompt(req.prompt, [{"text": c["text"], "meta": {"source": c["source"]}} for c in chunks])
    out = provider.generate(composed, temperature=req.temperature)
    return {"output": out, "retrieved": chunks}
