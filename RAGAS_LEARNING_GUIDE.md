# 🎓 RAGAS Learning Guide: Your Step-by-Step Journey

This guide will walk you through learning RAGAS and RAG evaluation concepts hands-on. Follow each step carefully and observe what happens.

## 📋 Prerequisites Checklist

Before you start, make sure you have:

- [ ] **Virtual Environment Active**: Your `srellm` environment is activated
- [ ] **OpenAI API Key**: Set in your environment (required for LLM-as-judge)
- [ ] **Dependencies Installed**: RAGAS and related packages
- [ ] **Qdrant Running**: Your vector database (we'll set up data in Step 1.5)
- [ ] **Data Loaded**: SRE runbooks + support tickets in Qdrant (covered below)

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

## 💾 Step 1.5: Data Setup (Essential!)

### Why We Need Better Data
Your original setup might have minimal SRE runbooks. For realistic RAGAS learning, we need:
- **Diverse scenarios**: More than 3 basic incidents
- **Real-world complexity**: Actual support ticket language and variety
- **Sufficient volume**: Enough data to see meaningful evaluation patterns

### Option A: Create Sample SRE Runbooks (Quick Start)
If you want to start immediately with basic data:

```bash
# Create sample runbooks directory
mkdir -p data/runbooks

# Create sample SRE runbooks
cat > data/runbooks/k8s_pod_crashloop.md << 'EOF'
# Kubernetes Pod CrashLoop Troubleshooting

## Symptoms
- Pods restarting repeatedly
- 5xx errors after deployment
- Application unavailable

## Diagnosis Steps
1. Check pod status: `kubectl get pods`
2. Examine pod logs: `kubectl logs <pod-name> --previous`
3. Describe pod events: `kubectl describe pod <pod-name>`
4. Check resource limits and requests
5. Verify health check configurations

## Resolution Actions
1. Review recent deployment changes
2. Check application startup time vs readiness probe timing
3. Verify environment variables and config maps
4. Scale down to single replica for debugging
5. Consider rollback to previous stable version

## Prevention
- Implement proper health checks
- Set appropriate resource limits
- Use staged deployments
- Monitor deployment metrics
EOF

cat > data/runbooks/disk_space.md << 'EOF'
# Disk Space Management

## Symptoms
- Write operations failing
- Disk space alerts at 95%
- Database errors related to storage

## Diagnosis Steps
1. Check disk usage: `df -h`
2. Find large files: `du -sh /* | sort -hr`
3. Check log rotation status
4. Identify temp files and old logs
5. Review database growth patterns

## Resolution Actions
1. Clean up log files: `find /var/log -name "*.log" -mtime +30 -delete`
2. Enable log rotation: `systemctl enable logrotate`
3. Archive old data
4. Expand disk if possible
5. Implement monitoring alerts

## Prevention
- Set up automated log rotation
- Monitor disk growth trends
- Implement log retention policies
- Set up proactive alerts at 80%
EOF

cat > data/runbooks/stale_config_cache.md << 'EOF'
# Stale Configuration Cache Issues

## Symptoms
- Feature flags not taking effect
- Old configuration values persisting
- Inconsistent application behavior

## Diagnosis Steps
1. Check config version: `curl /health/config-version`
2. Compare expected vs actual config
3. Verify cache invalidation mechanisms
4. Check for stuck background processes
5. Review config propagation logs

## Resolution Actions
1. Force cache refresh: `systemctl reload app-config`
2. Restart stateless services: `kubectl rollout restart deployment/app`
3. Purge cache manually if needed
4. Verify config distribution mechanism
5. Check for network connectivity issues

## Prevention
- Implement config version tracking
- Set up automated cache invalidation
- Add config validation steps
- Monitor config propagation delays
EOF
```

### Option B: Use HuggingFace Support Tickets (Recommended)
For more realistic and diverse data:

```bash
# Run the comprehensive data setup script
python rag/setup_learning_data.py
```

This script will:
- **Create sample SRE runbooks** (if you don't have them)
- **Download support tickets** from HuggingFace (`Tobi-Bueck/customer-support-tickets`)
- **Set up Qdrant collections** with proper embeddings
- **Create evaluation samples** combining both data sources
- **Verify everything** is working correctly

**What you'll get:**
- `sre_runbooks` collection: Your SRE knowledge base
- `support_knowledge` collection: Real-world IT support scenarios  
- `data/evaluation_samples.csv`: Ready-to-use test cases
- Much more realistic evaluation scenarios!

### Option C: Minimal Setup (If you have issues)
If HuggingFace downloads fail, you can still learn with just the basic runbooks:

```bash
# Create the basic runbooks (from Option A above)
mkdir -p data/runbooks
# ... (run the commands from Option A)

# Then just load the SRE runbooks
python -c "
from rag.setup_learning_data import setup_sre_runbooks, QdrantClient, SentenceTransformer
client = QdrantClient('localhost', 6333)
model = SentenceTransformer('all-MiniLM-L6-v2')
setup_sre_runbooks(client, model, clear=True)
print('Basic setup complete!')
"
```

### Verify Your Data Setup
```bash
# Check that collections exist and have data
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
collections = [c.name for c in client.get_collections().collections]
for name in collections:
    info = client.get_collection(name)
    print(f'{name}: {info.points_count} points')
"
```

You should see:
- `sre_runbooks`: 15 points (SRE runbook chunks)
- `support_knowledge`: 50 points (HuggingFace support tickets)

## ✅ Step 1.6: You're Ready!

The learning script has been updated to work with your new data setup automatically. It will:
- Load evaluation samples from your data setup
- Compare different collections if available  
- Test with realistic scenarios
- Show you the differences between data sources

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
The script now automatically loads evaluation samples from your data setup. It will test with:

**SRE Incidents (from your runbooks):**
- "Web 5xx after deploy; pods restarting repeatedly"
- "Write operations failing on primary; disk alert at 95%"
- "Feature flags not taking effect post-push; nodes show old config version"

**Real Support Tickets (from HuggingFace):**
- Actual IT support scenarios with realistic language
- Technical troubleshooting requests
- Product support issues

**To test specific queries manually:**
```python
# You can modify the load_evaluation_samples() function
# Or test individual queries by changing the first_sample selection
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

---

## 🎯 Your Realistic RAG Evaluation Setup

**Congratulations!** You now have a production-quality evaluation setup with:

### 📚 Diverse Knowledge Base
- **SRE Runbooks** (15 chunks): Structured, domain-specific troubleshooting guides
- **Support Tickets** (50 chunks): Real-world IT support scenarios from HuggingFace

### 🔍 Comprehensive Test Data  
- **10 evaluation samples** mixing SRE incidents and actual support queries
- **Multiple query types**: Structured incident reports + natural language requests
- **Cross-domain scenarios**: Test knowledge transfer between domains

### 💡 Advanced Learning Opportunities
- **Data Quality Comparison**: See how RAGAS behaves on clean docs vs messy real data
- **Domain Transfer Testing**: Understand how specialized knowledge helps general problems
- **Scale Effect Analysis**: Learn how evaluation changes with diverse data
- **Production Patterns**: Handle realistic query language variation

### 🚀 Ready for Advanced RAGAS Learning
Your setup now mirrors real-world RAG systems with mixed knowledge sources and diverse query patterns. This gives you authentic insights into:
- How evaluation metrics behave with realistic data complexity
- When different knowledge sources are most valuable  
- How to balance domain-specific vs general knowledge
- Cost-effective evaluation strategies for production systems

**You're ready to become a RAG evaluation expert!** 🎓