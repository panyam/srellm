# 🎓 RAGAS Learning Guide: Your Step-by-Step Journey

This guide will walk you through learning RAGAS and RAG evaluation concepts hands-on. Follow each step carefully and observe what happens.

## 📋 Prerequisites Checklist

Before you start, make sure you have:

- [ ] **Virtual Environment Active**: Your `srellm` environment is activated
- [ ] **OpenAI API Key**: Set in your environment (required for LLM-as-judge)
- [ ] **Dependencies Installed**: RAGAS and related packages
- [ ] **Qdrant Running**: Your vector database with SRE runbooks
- [ ] **Server Ready**: Your FastAPI server can generate responses

## 🚀 Step 1: Environment Setup

### Install Dependencies
```bash
# In your srellm virtual environment
pip install -e .  # This will install the new RAGAS dependencies we added
```

### Set OpenAI API Key
```bash
# Option 1: Export in terminal
export OPENAI_API_KEY="your-api-key-here"

# Option 2: Add to your shell profile (~/.bashrc, ~/.zshrc)
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Verify Setup
```bash
python -c "import ragas; print('RAGAS imported successfully!')"
python -c "import os; print('OpenAI key set:', bool(os.getenv('OPENAI_API_KEY')))"
```

## 🎯 Step 2: Start Your Learning Journey

### 2.1 First Run - Understanding the Basics
```bash
cd /Users/dzshrh/personal/ai/srellm
python evals/ragas_learning.py
```

**What to Observe:**
- How RAGAS structures evaluation data differently from your current approach
- The step-by-step process of converting SRE data to RAGAS format
- How LLM-as-judge breaks down responses into claims
- The actual Faithfulness score and its interpretation

**Learning Questions to Ask Yourself:**
1. What's the difference between the data format RAGAS uses vs your current `incidents_gold.csv`?
2. How does the Faithfulness score relate to the quality of the generated response?
3. What happened during the LLM-as-judge evaluation step?

### 2.2 Experiment with Different Queries
Modify the script to test different SRE scenarios:

```python
# In ragas_learning.py, change line ~200 from:
test_query = "Web 5xx after deploy; pods restarting repeatedly"

# To one of these:
test_query = "Write operations failing on primary; disk alert at 95%"
# or
test_query = "Feature flags not taking effect post-push; nodes show old config version"
```

**What to Observe:**
- How different queries produce different Faithfulness scores
- Which types of incidents get better/worse scores and why
- How the retrieved contexts change based on the query

**Learning Exercise:**
Create a simple log by running each query and noting:
- Query used
- Faithfulness score
- Quality of generated response (your human judgment)
- Whether the score matches your intuition

## 🔬 Step 3: Deep Dive Experiments

### 3.1 Test Cost Optimization (GPT-4 vs GPT-3.5)
Modify the LLM setup in the script:

```python
# Current (cost-effective):
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0
))

# Try this (higher quality, higher cost):
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4",
    temperature=0.0
))
```

**What to Observe:**
- Do the scores change between GPT-3.5 and GPT-4?
- Is the difference worth the 10x cost increase?
- How consistent are the evaluations?

### 3.2 Test Retrieval Impact
Experiment with different retrieval parameters:

```python
# In create_ragas_sample_from_sre_data(), change:
ragas_sample = create_ragas_sample_from_sre_data(test_query, retriever_k=3)

# To:
ragas_sample = create_ragas_sample_from_sre_data(test_query, retriever_k=1)  # Less context
# or
ragas_sample = create_ragas_sample_from_sre_data(test_query, retriever_k=5)  # More context
```

**What to Observe:**
- How does more/less context affect the Faithfulness score?
- Does more context always lead to better scores?
- What's the sweet spot for your SRE use case?

## 📊 Step 4: Compare Evaluation Approaches

### 4.1 Run Side-by-Side Comparison
The script automatically compares your current Precision@K with RAGAS Faithfulness.

**What to Observe:**
- Cases where Precision@K is high but Faithfulness is low (good retrieval, bad generation)
- Cases where Precision@K is low but Faithfulness is high (bad retrieval, but AI generates good response anyway)
- Which metric is more useful for different aspects of your system

### 4.2 Create Your Evaluation Matrix
Create a simple spreadsheet with these columns:
- Query
- Expected Source (from your gold data)
- Precision@3 Score
- Faithfulness Score
- Your Human Assessment (1-5)
- Notes/Observations

Test 5-10 different queries and fill in this matrix.

## 🧪 Step 5: Understanding What You've Learned

### Key Concepts Checklist
After running the experiments, you should understand:

- [ ] **RAGAS Data Structure**: How SingleTurnSample differs from your current format
- [ ] **LLM-as-Judge**: How AI evaluates AI responses claim by claim
- [ ] **Faithfulness Metric**: What it measures and how to interpret scores
- [ ] **Cost vs Quality**: Trade-offs between GPT-3.5 and GPT-4 for evaluation
- [ ] **Retrieval vs Generation**: Why you need different metrics for different components
- [ ] **Evaluation Integration**: How RAGAS fits with your existing workflow

### Troubleshooting Common Issues

**❌ "OpenAI API key not found"**
```bash
echo $OPENAI_API_KEY  # Should show your key
export OPENAI_API_KEY="your-key-here"
```

**❌ "Module not found: ragas"**
```bash
pip install ragas langchain langchain-openai
```

**❌ "Qdrant connection failed"**
```bash
# Make sure your Qdrant server is running
# Check if collections exist
python -c "from rag import qdrant_retriever as qd; r = qd.QdrantRetriever('sre_runbooks'); print('Connected!')"
```

**❌ "No contexts retrieved"**
- Check if your vector database has data
- Try a simpler query first
- Verify your collection name is correct

## 🎯 Next Steps After Mastering This

Once you're comfortable with this script:

1. **Expand to All Metrics**: Add Answer Relevancy, Context Precision, Context Recall
2. **Real Dataset Testing**: Download and test with HuggingFace support tickets
3. **Synthetic Data Generation**: Create test cases using RAGAS testset generation
4. **Production Pipeline**: Build automated evaluation into your development workflow
5. **Human-in-Loop**: Add human review for edge cases

## 💡 Learning Reflection Questions

After each experiment session, ask yourself:

1. **What surprised you about the results?**
2. **Which evaluation approach (Precision@K vs Faithfulness) is more valuable for your use case?**
3. **How would you explain LLM-as-judge to a teammate?**
4. **What are the cost implications of using RAGAS in production?**
5. **How could you integrate this into your development workflow?**

## 📝 Keep a Learning Log

Document your observations in this format:

```
Date: [Date]
Experiment: [What you tested]
Query: [SRE incident you used]
Precision@K: [Score]
Faithfulness: [Score]
Observations: [What you noticed]
Questions: [What confused you or what you want to explore next]
```

This will help you track your learning progress and identify patterns in the evaluation results.

---

**Remember**: The goal isn't to get perfect scores, but to understand how these metrics help you build better RAG systems. Take your time with each step and really observe what's happening!