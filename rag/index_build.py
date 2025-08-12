import os, glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def chunk(text, max_tokens=150):
    # very rough chunker by paragraphs
    parts, cur = [], []
    for line in text.splitlines():
        if line.strip() == "":
            if cur:
                parts.append("\n".join(cur)); cur=[]
        else:
            cur.append(line)
    if cur: parts.append("\n".join(cur))
    return parts

def build_index(runbooks_dir="data/runbooks", out_dir="rag"):
    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs, metas = [], []
    for fp in glob.glob(os.path.join(runbooks_dir, "*.md")):
        with open(fp, "r") as f:
            for c in chunk(f.read()):
                docs.append(c)
                metas.append({"source": os.path.basename(fp), "text": c})
    embs = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embs.shape[1])
    faiss.normalize_L2(embs)
    index.add(embs)
    faiss.write_index(index, os.path.join(out_dir, "rb.index"))
    np.save(os.path.join(out_dir, "docs.npy"), docs, allow_pickle=True)
    np.save(os.path.join(out_dir, "metas.npy"), metas, allow_pickle=True)

if __name__ == "__main__":
    build_index()

