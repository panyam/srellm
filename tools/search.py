# tools/search_cli.py
from rag import qdrant_retriever as qd

parser = argparse.ArgumentParser()
parser.add_argument("--collection", default="sre_runbooks")
parser.add_argument("--k", type=int, default=3)
args = parser.parse_args()

r = qd.QdrantRetriever(collection=args.collection)
print(f"Collection: {args.collection}")

while True:
    try:
        q = input("query> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        break
    hits = r.search(q, k=args.k)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['score']:.3f} {h['source']} :: {h['text'][:120]}…")
    print()
