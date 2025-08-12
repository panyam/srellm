# tools/prompt_preview.py
from rag import qdrant_retriever as qd
from rag.retriever import build_prompt  # your existing composer
import sys

q = sys.argv[1] # "Write operations failing on primary; disk alert at 95%"
r = qd.QdrantRetriever(collection="sre_runbooks")
ctx = r.search(q, k=2)
prompt = build_prompt(q, [{"text": c["text"], "meta": {"source": c["source"]}} for c in ctx])
print(prompt)

