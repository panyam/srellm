# ingest_hf.py
import argparse
import uuid
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Batch

parser = argparse.ArgumentParser(description="Ingest a Hugging Face dataset into Qdrant (separate collection hf_corpus).")
parser.add_argument("--clear", action="store_true", help="Clear collection before inserting")
parser.add_argument("--collection", type=str, default="hf_corpus", help="Target collection (default: hf_corpus)")
parser.add_argument("--dataset", type=str, default="HuggingFaceH4/stack-exchange-preferences", help="HF dataset path")
parser.add_argument("--split", type=str, default="train", help="HF split")
parser.add_argument("--limit", type=int, default=None, help="Max rows to ingest")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for upserts")
args = parser.parse_args()

COLLECTION = args.collection
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

ds = load_dataset(args.dataset, split=args.split, streaming=True)

buf_ids, buf_vecs, buf_payloads = [], [], []
count = 0

for i, row in enumerate(ds):
    if args.limit is not None and i >= args.limit:
        break
    # minimal text field extraction (customize for your dataset)
    question = row.get("question")
    options = row.get("options")
    if not question:
        continue
    text = question
    if isinstance(options, list) and options:
        text += "\n" + " ".join(options)

    vec = model.encode(text, normalize_embeddings=True)
    buf_ids.append(str(uuid.uuid4()))
    buf_vecs.append(vec.tolist())
    buf_payloads.append({"source": "hf:stack-exchange", "text": text})

    if len(buf_ids) >= args.batch_size:
        client.upsert(COLLECTION, points=Batch(ids=buf_ids, vectors=buf_vecs, payloads=buf_payloads))
        count += len(buf_ids)
        print(f"Upserted {count} points into '{COLLECTION}'…")
        buf_ids, buf_vecs, buf_payloads = [], [], []

# flush remainder
if buf_ids:
    client.upsert(COLLECTION, points=Batch(ids=buf_ids, vectors=buf_vecs, payloads=buf_payloads))
    count += len(buf_ids)

print(f"Finished upserting {count} points into collection '{COLLECTION}'.")
