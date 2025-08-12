# evals/retrieval_sanity.py
import pandas as pd
from rag import qdrant_retriever as qd

GOLD = {
    "Web 5xx after deploy; pods restarting repeatedly": "k8s_pod_crashloop.md",
    "Write operations failing on primary; disk alert at 95%": "disk_space.md",
    "Feature flags not taking effect post-push; nodes show old config version": "stale_config_cache.md",
}

def precision_at_k(k=3):
    r = qd.QdrantRetriever(collection="sre_runbooks")
    ok = 0
    for q, expected in GOLD.items():
        hits = r.search(q, k=k)
        got = [h["source"] for h in hits]
        hit = 1 if expected in got else 0
        print(f"Q: {q}\n expected={expected} | got={got}\n hit@{k}={hit}\n")
        ok += hit
    return ok / len(GOLD)

if __name__ == "__main__":
    for k in [1, 2, 3]:
        score = precision_at_k(k)
        print(f"Precision@{k} = {score:.2f}")
