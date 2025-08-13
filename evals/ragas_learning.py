#!/usr/bin/env python3
"""
RAGAS Learning Script - Understanding RAG Evaluation Step by Step

This script is designed to teach you RAGAS concepts incrementally.
We'll start with the Faithfulness metric to understand how LLM-as-judge works.

Learning Goals:
1. Understand how RAGAS structures evaluation data
2. See LLM-as-judge in action with real examples  
3. Compare RAGAS metrics to your current precision@k approach
4. Learn cost-effective evaluation strategies
"""

from ipdb import set_trace
import os
import sys
import asyncio
import pandas as pd
from typing import List, Dict, Any

# Add project root to path so we can import our existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import qdrant_retriever as qd
from serving.server import provider  # Use your existing LLM provider


# LEARNING STEP 1: Understanding RAGAS Data Structure
def understand_ragas_data_format():
    """
    RAGAS expects data in a specific format called SingleTurnSample.
    Let's understand what each field means:
    
    - user_input: The question/query (like your SRE incident summary)
    - response: What your RAG system generated
    - retrieved_contexts: The documents/chunks your retriever found
    - reference: Ground truth answer (optional, only needed for Context Recall)
    """
    print("=== RAGAS Data Format Learning ===")
    print("RAGAS uses SingleTurnSample with these fields:")
    print("• user_input: The user's question")
    print("• response: Your RAG system's answer") 
    print("• retrieved_contexts: List of retrieved document chunks")
    print("• reference: Ground truth (only for Context Recall)")
    print()


# LEARNING STEP 2: Convert Your Existing Data to RAGAS Format
def create_ragas_sample_from_sre_data(incident_summary: str, retriever_k: int = 3, 
                                     collection: str = "sre_runbooks") -> Dict[str, Any]:
    """
    This function shows you how to convert your existing SRE workflow 
    into RAGAS evaluation format.
    
    Why this matters:
    - You learn to bridge your current system with RAGAS
    - See how retrieval and generation work together
    - Understand the data flow in RAG evaluation
    
    New: Now supports multiple collections for richer learning!
    """
    print(f"🔍 Converting SRE incident to RAGAS format:")
    print(f"   Query: {incident_summary}")
    print(f"   Collection: {collection}")
    
    # Step 1: Use your existing retriever (just like your current eval does)
    retriever = qd.QdrantRetriever(collection=collection)
    retrieved_docs = retriever.search(incident_summary, k=retriever_k)
    
    # Step 2: Format contexts for RAGAS (it expects list of strings)
    retrieved_contexts = [doc["text"] for doc in retrieved_docs]
    print(f"   📚 Retrieved {len(retrieved_contexts)} contexts")
    
    # Show what types of sources we found (educational!)
    sources = [doc.get("source", "unknown") for doc in retrieved_docs]
    print(f"   📂 Sources: {sources}")
    
    # Step 3: Generate response using your existing provider
    # This simulates your actual RAG pipeline
    from rag.retriever import build_prompt
    prompt = build_prompt(incident_summary, [{"text": doc["text"], "meta": {"source": doc["source"]}} for doc in retrieved_docs])
    response = provider.generate(prompt, temperature=0.0)
    print(f"   💭 Generated response: {response[:100]}...")
    
    # Step 4: Create RAGAS-compatible data structure
    ragas_sample = {
        "user_input": incident_summary,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
        "collection_used": collection,  # Track which collection for learning
        "sources": sources,  # Track sources for analysis
        # Note: We don't include 'reference' since we don't have ground truth yet
    }
    
    return ragas_sample


# LEARNING STEP 2.5: Compare Different Data Sources
def compare_collections_for_query(query: str, k: int = 3):
    """
    Compare how the same query performs against different collections.
    This teaches you about data diversity and retrieval quality.
    
    Learning points:
    1. How different knowledge bases affect retrieval
    2. Quality vs quantity in your data
    3. Domain-specific vs general knowledge
    """
    print(f"\n🔍 Comparing Collections for Query: '{query}'")
    print("=" * 60)
    
    collections = ["sre_runbooks"]
    
    # Check if support_knowledge exists
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        available_collections = [c.name for c in client.get_collections().collections]
        if "support_knowledge" in available_collections:
            collections.append("support_knowledge")
    except:
        pass
    
    for collection in collections:
        print(f"\n📚 Collection: {collection}")
        try:
            retriever = qd.QdrantRetriever(collection=collection)
            results = retriever.search(query, k=k)
            
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results):
                source = result.get("source", "unknown")
                text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                print(f"   {i+1}. Source: {source}")
                print(f"      Preview: {text_preview}")
                print()
                
        except Exception as e:
            print(f"   ❌ Error with {collection}: {e}")
    
    print("💡 Learning Observation:")
    print("   • Notice differences in source types and content quality")
    print("   • Consider which collection would be better for different query types")
    print("   • Think about how to combine multiple knowledge sources")


# LEARNING STEP 3: Understanding Faithfulness Evaluation
async def test_faithfulness_metric(sample: Dict[str, Any]):
    """
    This is where the magic happens! We'll test the Faithfulness metric
    and see exactly how LLM-as-judge works.
    
    Learning points:
    1. How RAGAS breaks down the response into claims
    2. How it checks each claim against the context
    3. Why this is better than simple string matching
    """
    print("\n=== Testing Faithfulness Metric ===")
    print("Faithfulness measures: 'Are all statements in the response supported by the context?'")
    print()
    
    try:
        # Import RAGAS components
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        
        # Setup LLM for evaluation (this is the "judge")
        # Using GPT-3.5 for cost-effectiveness (remember our research!)
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0  # Deterministic for evaluation
        ))
        
        # Create RAGAS sample
        ragas_sample = SingleTurnSample(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"]
        )
        
        # Initialize Faithfulness metric
        faithfulness_scorer = Faithfulness(llm=evaluator_llm)
        
        print("🤖 Running LLM-as-judge evaluation...")
        print("   The LLM will:")
        print("   1. Break your response into individual claims")
        print("   2. Check if each claim can be inferred from the context")
        print("   3. Calculate: (supported claims) / (total claims)")
        print()
        
        # Run the evaluation
        score = await faithfulness_scorer.single_turn_ascore(ragas_sample)
        
        print(f"✅ Faithfulness Score: {score:.3f}")
        print(f"   Interpretation:")
        if score >= 0.9:
            print(f"   🟢 Excellent! Nearly all statements are supported by context")
        elif score >= 0.7:
            print(f"   🟡 Good, but some statements might not be fully supported")
        else:
            print(f"   🔴 Poor - many statements cannot be verified from context")
            
        return score
        
    except Exception as e:
        print(f"❌ Error running Faithfulness evaluation: {e}")
        print("Make sure you have OPENAI_API_KEY set in your environment")
        return None


# LEARNING STEP 4: Compare with Your Current Approach
def compare_with_precision_at_k(incident_summary: str, expected_source: str):
    """
    Let's compare RAGAS evaluation with your current Precision@K approach.
    This will help you understand the differences and benefits.
    
    Learning points:
    1. Precision@K measures retrieval accuracy
    2. Faithfulness measures generation quality  
    3. Both are needed for comprehensive evaluation
    """
    print(f"\n=== Comparing Evaluation Approaches ===")
    
    # Your current approach (from retrieval_sanity.py)
    retriever = qd.QdrantRetriever(collection="sre_runbooks")
    hits = retriever.search(incident_summary, k=3)
    got_sources = [h["source"] for h in hits]
    precision_at_3 = 1 if expected_source in got_sources else 0
    
    print(f"📊 Your Current Precision@3 Approach:")
    print(f"   Question: {incident_summary}")
    print(f"   Expected source: {expected_source}")
    print(f"   Retrieved sources: {got_sources}")
    print(f"   Precision@3: {precision_at_3}")
    print()
    print(f"💡 Key Differences:")
    print(f"   • Precision@K: Measures retrieval accuracy (did we find the right docs?)")
    print(f"   • Faithfulness: Measures generation quality (is the answer factual?)")
    print(f"   • Both needed: Good retrieval ≠ good generation")


def load_evaluation_samples() -> List[Dict[str, Any]]:
    """
    Load evaluation samples from our data setup.
    This teaches you how to work with prepared evaluation datasets.
    """
    samples = []
    
    # Try to load from the generated evaluation samples
    try:
        import pandas as pd
        df = pd.read_csv("data/evaluation_samples.csv")
        samples = df.to_dict('records')
        print(f"📊 Loaded {len(samples)} evaluation samples from CSV")
    except:
        # Fallback to your original test cases
        samples = [
            {"query": "Web 5xx after deploy; pods restarting repeatedly", 
             "expected_source": "k8s_pod_crashloop.md", "type": "sre_incident"},
            {"query": "Write operations failing on primary; disk alert at 95%", 
             "expected_source": "disk_space.md", "type": "sre_incident"},
            {"query": "Feature flags not taking effect post-push; nodes show old config version", 
             "expected_source": "stale_config_cache.md", "type": "sre_incident"}
        ]
        print(f"📊 Using {len(samples)} default SRE samples")
    
    return samples


async def main():
    """
    Main learning workflow - step by step RAGAS exploration
    """
    print("🎓 Welcome to RAGAS Learning!")
    print("This script will teach you RAG evaluation concepts step by step.\n")
    
    # Step 1: Understand data format
    understand_ragas_data_format()
    
    # Step 2: Load evaluation samples
    eval_samples = load_evaluation_samples()
    
    # Step 3: Test with first sample
    print("=== Testing with Your Data ===")
    first_sample = eval_samples[0]
    set_trace(context=21)
    test_query = first_sample["query"]
    expected_source = first_sample.get("expected_source", "unknown")
    
    print(f"📝 Testing with: {test_query}")
    
    # Step 4: Compare different collections (if available)
    compare_collections_for_query(test_query, k=3)
    
    # Step 5: Convert to RAGAS format and test faithfulness
    ragas_sample = create_ragas_sample_from_sre_data(test_query, collection="sre_runbooks")
    faithfulness_score = await test_faithfulness_metric(ragas_sample)
    
    # Step 6: Compare with your existing evaluation approach
    compare_with_precision_at_k(test_query, expected_source)
    
    # Step 7: Test with different data if available
    if len(eval_samples) > 1:
        print(f"\n🔄 Testing Additional Samples...")
        for i, sample in enumerate(eval_samples[1:3], 1):  # Test 2 more
            print(f"\n--- Sample {i+1}: {sample['type']} ---")
            print(f"Query: {sample['query']}")
            
            # Quick test without full faithfulness (to save API calls)
            ragas_sample = create_ragas_sample_from_sre_data(
                sample["query"], 
                collection="sre_runbooks", 
                retriever_k=2
            )
            print(f"✅ Generated RAGAS sample successfully")
    
    print(f"\n🎯 What You've Learned:")
    print(f"   1. How to structure data for RAGAS evaluation")
    print(f"   2. How LLM-as-judge evaluates response quality")
    print(f"   3. Difference between retrieval and generation metrics")
    print(f"   4. How different knowledge bases affect retrieval")
    print(f"   5. How to work with diverse evaluation datasets")
    
    if faithfulness_score is not None:
        print(f"\n📈 Next Steps for Learning:")
        print(f"   • Test all {len(eval_samples)} evaluation samples")
        print(f"   • Experiment with GPT-4 vs GPT-3.5 for evaluation")
        print(f"   • Try different collections (support_knowledge if available)")
        print(f"   • Test other RAGAS metrics (Answer Relevancy, Context Precision)")
        print(f"   • Generate synthetic test data for more scenarios")
        print(f"\n🔧 Advanced Experiments:")
        print(f"   • Modify retriever_k and see impact on scores")
        print(f"   • Test queries from different domains")
        print(f"   • Compare evaluation consistency across runs")


if __name__ == "__main__":
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("   This is needed for LLM-as-judge evaluation")
        sys.exit(1)
    
    # Run the learning workflow
    asyncio.run(main())
