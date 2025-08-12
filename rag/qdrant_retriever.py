
## qdrant_retriever.py  (defaults to **sre_runbooks**, but configurable)
from typing import Optional, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

class QdrantRetriever:
    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection: str = "sre_runbooks",
                 model_name: str = "all-MiniLM-L6-v2",
                 normalize: bool = True):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def search(self, query: str, k: int = 3, source_equals: Optional[str] = None,
               must: Optional[List[Dict]] = None, must_not: Optional[List[Dict]] = None):
        vec = self.model.encode(query, normalize_embeddings=self.normalize)
        q_filter = None
        conds_must = []
        if source_equals:
            conds_must.append(FieldCondition(key="source", match=MatchValue(value=source_equals)))
        if must:
            conds_must.extend(must)
        if conds_must or must_not:
            q_filter = Filter(must=conds_must or None, must_not=must_not or None)

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vec.tolist(),
            query_filter=q_filter,
            limit=k,
        )
        return [{
            "text": h.payload.get("text", ""),
            "source": h.payload.get("source", ""),
            "score": h.score,
        } for h in hits]
