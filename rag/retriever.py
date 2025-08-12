import os, numpy as np, faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, idx_path="rag/rb.index"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(idx_path)
        self.docs = np.load("rag/docs.npy", allow_pickle=True).tolist()
        self.metas = np.load("rag/metas.npy", allow_pickle=True).tolist()

    def search(self, query: str, k=3):
        q = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        results = []
        for idx in I[0]:
            results.append({"text": self.docs[idx], "meta": self.metas[idx]})
        return results

def build_prompt(query, context_chunks):
    ctx = "\n\n---\n".join([c["text"] for c in context_chunks])
    return f"""You are an SRE assistant. Use the context to propose concrete next actions.

Context:
{ctx}

User issue: {query}

Respond with 1-3 bullet points of concrete steps."""

