## qdrant_setup.py  (creates & loads **runbooks → sre_runbooks**)
import argparse
import glob
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Batch
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Setup Qdrant collection and upsert RUNBOOKS into sre_runbooks.")
parser.add_argument("--clear", action="store_true", help="Clear sre_runbooks before inserting")
parser.add_argument("--runbooks_dir", type=str, default="data/runbooks", help="Directory with .md runbooks")
args = parser.parse_args()

COLLECTION = "sre_runbooks"
DIM = 384  # all-MiniLM-L6-v2

client = QdrantClient(host="localhost", port=6333)

if args.clear and COLLECTION in [c.name for c in client.get_collections().collections]:
    print(f"Deleting existing collection '{COLLECTION}'…")
    client.delete_collection(COLLECTION)

if COLLECTION not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

model = SentenceTransformer("all-MiniLM-L6-v2")

ids, vecs, payloads = [], [], []
for fp in glob.glob(os.path.join(args.runbooks_dir, "*.md")):
    with open(fp, "r") as f:
        text = f.read()
    # naive paragraph chunking
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    embs = model.encode(chunks, normalize_embeddings=True)
    for chunk, vec in zip(chunks, embs):
        ids.append(str(uuid.uuid4()))
        vecs.append(vec.tolist())
        payloads.append({
            "source": os.path.basename(fp),
            "text": chunk,
            "topic": os.path.basename(fp).split("_")[0],
        })

if ids:
    client.upsert(COLLECTION, points=Batch(ids=ids, vectors=vecs, payloads=payloads))
print(f"Upserted {len(ids)} runbook chunks into '{COLLECTION}'.")
