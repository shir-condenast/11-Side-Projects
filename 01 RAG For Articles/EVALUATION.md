# RAG Evaluation Report

**Project:** Interior Design Article Assistant
**Evaluation Framework:** RAGAS
**Date:** 2026-03-13
**Model:** OpenAI gpt-4o-mini
**Data Source:** Databricks Delta Table (`gold_us_prod.content.gld_cross_brand_live`)
**Dataset:** 100 Architectural Digest articles (2025+)

---

## Latest Evaluation Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Faithfulness** | 0.6721 | Good - responses are well-grounded in context |
| **Answer Relevancy** | 0.2378 | Needs improvement - answers could be more focused |
| **Context Precision** | 0.6417 | Good - retrieved articles are mostly relevant |
| **Context Recall** | 1.0000 | Excellent - context captures all needed information |

---

## Metric Definitions

| Metric | What It Measures |
|--------|------------------|
| **Faithfulness** | Is the generated response factually consistent with the retrieved context? Higher = less hallucination |
| **Answer Relevancy** | How relevant is the answer to the original question? |
| **Context Precision** | Are all retrieved documents relevant to the question? |
| **Context Recall** | Does the retrieved context contain all information needed to answer? |

---

## Analysis

### Strengths ✅
- **Context Recall (1.00)**: Perfect score! The hybrid retrieval system successfully finds all relevant information needed to answer questions
- **Faithfulness (0.67)**: Good improvement - responses are well-grounded in the retrieved context with minimal hallucination
- **Context Precision (0.64)**: Retrieved articles are generally relevant to the queries

### Areas for Improvement 🔧
- **Answer Relevancy (0.24)**: Significant room for improvement - generated responses may include tangential information or lack focus on the specific question
  - **Root Cause**: The conversational response format may be too verbose or include unnecessary context
  - **Recommendation**: Refine prompts to be more direct and question-focused

### Key Insights 📊
- **Perfect Recall**: The hybrid search (BM25 + Vector Search) is excellent at finding all relevant articles
- **Low Relevancy**: Despite finding the right articles, the LLM-generated answers aren't focused enough on the specific questions
- **Opportunity**: The gap between Context Recall (100%) and Answer Relevancy (24%) suggests the issue is in the generation phase, not retrieval

---

## Test Dataset

**Auto-generated from 2025+ Articles using GPT-4o-mini**

12 ground truth test cases covering:
- Timeless kitchen design features
- 2026 color trends (AD100 designers)
- Frank Gehry architectural style
- Relaxing cities to visit in 2025
- Art Nouveau museum in Paris
- Quiet luxury in residential design
- International Festival of Art, Cheese, and Wine
- Art exhibitions in Italy
- Kendrick Lamar's real estate portfolio
- Gemmyo flagship store design
- Modern kitchen materials
- Cohesive architectural design elements

**Test Generation Process:**
1. Randomly sample articles from `data/articles.json`
2. Use GPT-4o-mini to generate (Question, Ground Truth, Relevant Context) triplets
3. Save to `data/evaluation_questions.json`
4. Questions are regenerated when data is refreshed to match current content

---

## Configuration

```python
# Retrieval Settings
MAX_SEMANTIC_DISTANCE = 0.75  # Reject weak vector matches
MIN_RELEVANCE_SCORE = 2.0     # Minimum hybrid score threshold
HYBRID_ALPHA = 0.5            # Balance between keyword (0.5) and semantic (0.5)

# LLM Settings
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.5 - 0.7       # Lower for factual responses
```

---

## Recommendations

### High Priority 🔴
1. **Improve Answer Relevancy (0.24 → 0.60+ target)**:
   - Refactor prompts to be more direct and question-focused
   - Reduce conversational fluff in responses
   - Add explicit instruction: "Answer the question directly and concisely"
   - Consider A/B testing different prompt templates

### Medium Priority 🟡
2. **Maintain Context Recall (1.00)**:
   - Current hybrid search configuration is optimal
   - Monitor this metric to ensure it stays high as data grows

3. **Improve Faithfulness (0.67 → 0.80+ target)**:
   - Add stricter grounding instructions in system prompts
   - Implement citation/source attribution in responses
   - Add fact-checking layer before final response

### Low Priority 🟢
4. **Fine-tune Context Precision (0.64 → 0.75+ target)**:
   - Experiment with different `alpha` values for hybrid search
   - Consider adjusting `MIN_RELEVANCE_SCORE` threshold
   - Test different reranker models

5. **Expand Test Coverage**:
   - Increase test questions from 12 to 20-30
   - Add edge cases (ambiguous queries, multi-hop questions)
   - Include negative test cases (questions with no relevant articles)

---

## How to Run

### Option 1: Generate Questions Only (Recommended First Step)

After updating your data, generate new test questions based on the current articles:

```bash
cd "11-Side-Projects/01 RAG For Articles"
python3 scripts/generate_eval_questions.py
```

This will:
- Analyze your current articles in `data/articles.json`
- Use GPT-4o-mini to generate relevant test questions
- Save questions to `data/evaluation_questions.json`
- **Not run any scoring** - just question generation

### Option 2: Run Full Evaluation (With Pause)

Run the complete evaluation with a pause before scoring:

```bash
python3 src/evaluation.py
```

This will:
1. Load or generate test questions
2. Generate RAG responses for each question
3. **PAUSE** for you to review responses
4. Wait for you to press ENTER
5. Run RAGAS scoring
6. Save results with timestamp

### Option 3: Run End-to-End (No Pause)

Skip the pause and run everything automatically:

```bash
python3 src/evaluation.py --no-pause
```

### Option 4: Force Regenerate Questions

Force regeneration of test questions even if they exist:

```bash
python3 src/evaluation.py --regenerate
```

---

## Workflow After Data Refresh

When you update `data/articles.json` with new data:

1. **Generate new questions** based on the fresh data:
   ```bash
   python3 scripts/generate_eval_questions.py
   ```

2. **Review the questions** in `data/evaluation_questions.json`

3. **Run evaluation** with pause to review responses:
   ```bash
   python3 src/evaluation.py
   ```

4. **Review intermediate results** in `data/evaluation_responses.json`

5. **Press ENTER** to continue with RAGAS scoring

6. **Check final results** in `data/evaluation_results_YYYYMMDD_HHMMSS.json`

---

## Evaluation History

| Date | Dataset | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Notes |
|------|---------|--------------|------------------|-------------------|----------------|-------|
| 2026-03-10 | Static JSON (66 articles) | 0.5714 | 0.4002 | 0.6778 | 0.8194 | Initial baseline |
| 2026-03-13 | Databricks 2025+ (100 articles) | 0.6721 | 0.2378 | 0.6417 | 1.0000 | Live data, auto-generated questions |

**Key Changes:**
- ✅ **Faithfulness improved** from 0.57 → 0.67 (+17%)
- ✅ **Context Recall improved** from 0.82 → 1.00 (+22%)
- ⚠️ **Answer Relevancy decreased** from 0.40 → 0.24 (-40%) - needs investigation
- ➡️ **Context Precision stable** at ~0.65

---

## Data Pipeline

### Source
- **Table**: `gold_us_prod.content.gld_cross_brand_live` (Databricks Delta)
- **Brand**: Architectural Digest
- **Filter**: `published_date >= '2025-01-01'`
- **Count**: 100 articles

### ETL Process
```bash
# Fetch latest data from Databricks
python3 scripts/fetch_from_databricks.py
```

This script:
1. Connects to Databricks using SQL Statement Execution API
2. Queries the Delta table with date filter
3. Transforms schema: `hed → title`, `body → content`, `full_url → url`
4. Normalizes URLs (ensures `https://` prefix)
5. Saves to `data/articles.json`

### Refresh Workflow
When new articles are published:
1. Run `python3 scripts/fetch_from_databricks.py` to update data
2. Run `python3 scripts/generate_eval_questions.py` to generate new test questions
3. Run `python3 src/evaluation.py` to evaluate with fresh data
4. Compare results with previous evaluations

---

*Generated by RAGAS evaluation framework*

