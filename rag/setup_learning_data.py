#!/usr/bin/env python3
"""
Learning Data Setup for RAGAS Evaluation

This script sets up a comprehensive dataset for RAGAS learning by combining:
1. Sample SRE runbooks (knowledge base)
2. HuggingFace support tickets (realistic query-answer pairs)
3. Proper Qdrant collections for retrieval

Learning Goals:
- Understand how to prepare data for RAG evaluation
- See real-world data diversity vs simple examples
- Learn data pipeline best practices
"""

import os
import sys
import uuid
from typing import List, Dict, Any
import pandas as pd
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Batch
from sentence_transformers import SentenceTransformer

def setup_sre_runbooks(client: QdrantClient, model: SentenceTransformer, clear: bool = False):
    """
    Load SRE runbooks into 'sre_runbooks' collection.
    This is your knowledge base that the retriever will search.
    """
    collection_name = "sre_runbooks"
    
    print(f"🏗️  Setting up {collection_name} collection...")
    
    if clear and collection_name in [c.name for c in client.get_collections().collections]:
        print(f"   Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
    
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # all-MiniLM-L6-v2
        )
    
    # Load runbooks from data/runbooks directory
    runbooks_dir = "data/runbooks"
    if not os.path.exists(runbooks_dir):
        print(f"❌ {runbooks_dir} not found. Please create sample runbooks first (see learning guide).")
        return 0
    
    ids, vectors, payloads = [], [], []
    
    for filename in os.listdir(runbooks_dir):
        if filename.endswith('.md'):
            filepath = os.path.join(runbooks_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple paragraph chunking (you can improve this later)
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            
            # Create embeddings
            embeddings = model.encode(chunks, normalize_embeddings=True)
            
            for chunk, embedding in zip(chunks, embeddings):
                ids.append(str(uuid.uuid4()))
                vectors.append(embedding.tolist())
                payloads.append({
                    "source": filename,
                    "text": chunk,
                    "type": "sre_runbook"
                })
    
    if ids:
        client.upsert(collection_name, points=Batch(ids=ids, vectors=vectors, payloads=payloads))
        print(f"   ✅ Loaded {len(ids)} runbook chunks")
    
    return len(ids)


def setup_support_tickets_knowledge(client: QdrantClient, model: SentenceTransformer, 
                                   max_tickets: int = 100, clear: bool = False):
    """
    Load HuggingFace support tickets as additional knowledge base.
    This gives us more realistic, diverse technical content.
    
    Why this helps learning:
    - Real-world language patterns
    - Diverse technical scenarios  
    - More comprehensive retrieval testing
    """
    collection_name = "support_knowledge"
    
    print(f"🎫 Setting up {collection_name} collection...")
    
    if clear and collection_name in [c.name for c in client.get_collections().collections]:
        print(f"   Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
    
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    
    print(f"   📥 Loading support tickets from HuggingFace...")
    
    try:
        # Load the customer support dataset we researched
        dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        
        # Filter for English IT/Technical support tickets
        df = dataset.to_pandas()
        
        # Focus on IT and Technical support for SRE relevance
        relevant_tickets = df[
            (df['language'] == 'en') & 
            (df['queue'].isin(['IT Support', 'Technical Support', 'Product Support']))
        ].head(max_tickets)
        
        print(f"   📊 Processing {len(relevant_tickets)} relevant support tickets...")
        
        ids, vectors, payloads = [], [], []
        
        for _, ticket in relevant_tickets.iterrows():
            # Combine subject and body for better context
            full_content = f"Subject: {ticket['subject']}\n\nIssue: {ticket['body']}\n\nResolution: {ticket['answer']}"
            
            # Create embedding
            embedding = model.encode(full_content, normalize_embeddings=True)
            
            ids.append(str(uuid.uuid4()))
            vectors.append(embedding.tolist())
            payloads.append({
                "source": f"ticket_{ticket.name}",
                "text": full_content,
                "type": "support_ticket",
                "queue": ticket['queue'],
                "priority": ticket.get('priority', 'unknown'),
                "subject": ticket['subject']
            })
        
        if ids:
            client.upsert(collection_name, points=Batch(ids=ids, vectors=vectors, payloads=payloads))
            print(f"   ✅ Loaded {len(ids)} support ticket knowledge chunks")
        
        return len(ids)
        
    except Exception as e:
        print(f"   ⚠️  Could not load HuggingFace dataset: {e}")
        print(f"   💡 This is optional - you can still learn with just SRE runbooks")
        return 0


def create_evaluation_samples(client: QdrantClient, num_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Create evaluation samples for RAGAS learning.
    This combines your original test cases with some from the support ticket data.
    
    Learning goal: Understand different types of evaluation scenarios
    """
    print(f"📝 Creating {num_samples} evaluation samples...")
    
    # Your original SRE test cases (known good examples)
    sre_samples = [
        {
            "query": "Web 5xx after deploy; pods restarting repeatedly",
            "expected_source": "k8s_pod_crashloop.md",
            "type": "sre_incident"
        },
        {
            "query": "Write operations failing on primary; disk alert at 95%", 
            "expected_source": "disk_space.md",
            "type": "sre_incident"
        },
        {
            "query": "Feature flags not taking effect post-push; nodes show old config version",
            "expected_source": "stale_config_cache.md", 
            "type": "sre_incident"
        }
    ]
    
    # Try to get some support ticket queries if available
    try:
        dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        df = dataset.to_pandas()
        
        # Get some IT/Technical support queries
        tech_tickets = df[
            (df['language'] == 'en') & 
            (df['queue'].isin(['IT Support', 'Technical Support']))
        ].head(num_samples - len(sre_samples))
        
        for _, ticket in tech_tickets.iterrows():
            sre_samples.append({
                "query": ticket['subject'],
                "expected_source": "support_ticket",  # We'll use this for comparison
                "type": "support_query",
                "original_answer": ticket['answer']
            })
            
    except Exception as e:
        print(f"   ⚠️  Could not load additional samples: {e}")
    
    print(f"   ✅ Created {len(sre_samples)} evaluation samples")
    return sre_samples[:num_samples]


def verify_setup(client: QdrantClient):
    """
    Verify that everything is set up correctly.
    """
    print(f"\n🔍 Verifying setup...")
    
    collections = [c.name for c in client.get_collections().collections]
    
    for collection in ["sre_runbooks", "support_knowledge"]:
        if collection in collections:
            info = client.get_collection(collection)
            count = info.points_count
            print(f"   ✅ {collection}: {count} points")
            
            # Test a simple search
            if count > 0:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    test_vector = model.encode("test query", normalize_embeddings=True)
                    
                    results = client.search(
                        collection_name=collection,
                        query_vector=test_vector.tolist(),
                        limit=1
                    )
                    print(f"      🔍 Search test: Found {len(results)} results")
                except Exception as e:
                    print(f"      ❌ Search test failed: {e}")
        else:
            print(f"   ❌ {collection}: Not found")


def main():
    """
    Main setup workflow for RAGAS learning data.
    """
    print("🎓 Setting up RAGAS Learning Data")
    print("=" * 50)
    
    # Connect to Qdrant
    try:
        client = QdrantClient(host="localhost", port=6333)
        print("✅ Connected to Qdrant")
    except Exception as e:
        print(f"❌ Could not connect to Qdrant: {e}")
        print("💡 Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        return
    
    # Load embedding model
    print("🔧 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Setup data collections
    runbook_count = setup_sre_runbooks(client, model, clear=True)
    support_count = setup_support_tickets_knowledge(client, model, max_tickets=50, clear=True)
    
    # Create evaluation samples
    eval_samples = create_evaluation_samples(client, num_samples=10)
    
    # Save evaluation samples for later use
    df = pd.DataFrame(eval_samples)
    df.to_csv("data/evaluation_samples.csv", index=False)
    print(f"💾 Saved evaluation samples to data/evaluation_samples.csv")
    
    # Verify setup
    verify_setup(client)
    
    print(f"\n🎉 Setup Complete!")
    print(f"📊 Summary:")
    print(f"   • SRE Runbooks: {runbook_count} chunks")
    print(f"   • Support Knowledge: {support_count} chunks") 
    print(f"   • Evaluation Samples: {len(eval_samples)} queries")
    print(f"\n🚀 You're ready to start RAGAS learning!")


if __name__ == "__main__":
    main()