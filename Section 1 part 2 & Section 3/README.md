# Digikala Reviews: Semantic RAG, Recommendation Classifier & LLM-as-a-Judge

A three-notebook pipeline for analyzing Digikala product reviews: a semantic
retrieval (RAG) system that answers qualitative questions about top products
using an LLM grounded in real customer comments, a lightweight classifier
that predicts purchase-recommendation status from review text, and an
independent LLM judge that scores the quality of the RAG answers.

## Project Structure

```
Section 1_2_Section_3.ipynb   # Part 1: RAG retrieval  |  Part 2: recommendation classifier
run_validation.ipynb    # Generates grounded LLM answers from Part 1's evidence
llm_as_judge.ipynb      # Scores those answers with an independent judge model
```

## Pipeline Overview

```
Section 1_2_Section_3.ipynb (Part 1)
  ↓ validation_requests.json
run_validation.ipynb
  ↓ validation_results.json
llm_as_judge.ipynb
  ↓ judge_results.json / judge_manual_review.csv
```

`Section 1_2_Section_3.ipynb` Part 2 (the classifier) is independent — it only
needs the raw comments dataset and doesn't depend on the rest of the
pipeline.

---

## 1. `Section 1_2_Section_3.ipynb`

### Part 1 — Semantic Retrieval (RAG)

Downloads the Digikala comments/products dataset from Hugging Face, finds the
20 most-reviewed products, builds sentence embeddings
(`intfloat/multilingual-e5-base`) for their comments, indexes them with
FAISS, and retrieves the most relevant comments for 4 fixed questions per
product (satisfaction, common complaints, quality, value for money).

**Output:** `validation_requests.json` — structured evidence per (product, question), consumed by `run_validation.ipynb`.

### Part 2 — Recommendation Classifier

Samples a class-balanced subset of comments, cleans the text, and trains a
`TF-IDF + LinearSVC` classifier to predict `recommendation_status`
(`recommended` / `not_recommended` / `no_idea`). Includes a train/val/test
split, hyperparameter tuning, a confusion matrix, and error analysis.

**Requirements:** `pandas`, `numpy`, `huggingface_hub`, `sentence-transformers`, `faiss-gpu` (or `faiss-cpu`), `scikit-learn`, `matplotlib`, `seaborn`

**How to run:** Run cells top to bottom. Part 1 needs internet access and benefits from a GPU. Part 2 runs independently and works fine on CPU.

---

## 2. `run_validation.ipynb`

Reads `validation_requests.json`, builds a grounded prompt per question
(explicitly instructed to answer only from the given evidence and to say so
when evidence is insufficient), sends it to an OpenAI-compatible chat
endpoint, and saves the answers alongside their evidence for review.

**Output:** `validation_results.json` (raw results) and `validation_manual_review.csv` (answer + evidence side by side).

**Requirements:** `requests`, `pandas`

**How to run:**
1. Replace the hardcoded `API_KEY` with your own key — ideally loaded from an environment variable rather than written directly in the notebook.
2. Make sure `validation_requests.json` is in the working directory.
3. Run the connectivity-check cell first to confirm your key and see available models, then run the rest top to bottom.

---

## 3. `llm_as_judge.ipynb`

Reads `validation_results.json` and scores each answer with a second,
independent model (not the one that generated the answer) on three criteria:
**groundedness** (no unsupported claims), **relevance** (answers the actual
question), and **completeness** (covers the key points in the evidence).
Produces a pass/fail verdict and a short justification per answer.

**Output:** `judge_results.json` (raw scores) and `judge_manual_review.csv` (sorted so failed/low-scoring answers surface first for manual review), plus a summary chart of average scores per criterion.

**Requirements:** `requests`, `pandas`, `matplotlib`

**How to run:** Same as `run_validation.ipynb` — set your API key via an environment variable, make sure `validation_results.json` is present, and run top to bottom.

---

## Notes & Limitations

- FAISS uses an exact index (`IndexFlatIP`); for much larger datasets, switch to an approximate index (`IndexIVFFlat` / HNSW) for speed.
- The classifier's balanced sampling avoids majority-class bias — evaluate with `f1_macro`, not accuracy, given class imbalance in the raw data.
- The judge model calls the API sequentially; for large batches, consider adding concurrency/retries.
- Never commit real API keys to version control — use environment variables or a `.env` file instead.
