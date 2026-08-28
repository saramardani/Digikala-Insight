# Digikala Reviews: Semantic RAG & Recommendation Classifier

A two-notebook pipeline for analyzing Digikala product reviews: a semantic
retrieval (RAG) system that answers qualitative questions about top products
using an LLM grounded in real customer comments, plus a lightweight
classifier that predicts purchase-recommendation status from review text.

## Project Structure

```
Section1_2_Section_3.ipynb       # Part 1: RAG retrieval  |  Part 2: classifier
run_validation.ipynb        # LLM validation runner (consumes Part 1's output)
```

## Pipeline Overview

```
Section1_2_Section_3.ipynb (Part 1)
  ↓ produces validation_requests.json
run_validation.ipynb
  ↓ produces validation_results.json / validation_manual_review.csv
```

`Section1_2_Section_3.ipynb` also contains an independent Part 2 (classifier)
that only needs the raw comments dataset — it does not depend on Part 1.

---

## 1. `Section1_2_Section_3.ipynb`

### Part 1 — Semantic Retrieval (RAG)

Downloads the Digikala comments/products dataset from Hugging Face, finds the
20 most-reviewed products, builds sentence embeddings
(`intfloat/multilingual-e5-base`) for their comments, indexes them with
FAISS, and retrieves the most relevant comments for 4 fixed validation
questions per product (satisfaction, common complaints, quality, value for
money).

**Outputs:**
- `validation_requests.json` — structured evidence per (product, question), used by `run_validation.ipynb`
- `validation_evidence.csv` — flat table of the same evidence, for manual inspection

### Part 2 — Recommendation Classifier

Independently samples a class-balanced subset of comments (max 40k per
class), cleans the text, and trains a `TF-IDF + LinearSVC` classifier to
predict `recommendation_status` (`recommended` / `not_recommended` /
`no_idea`). Includes a train/val/test split, hyperparameter tuning over `C`,
a confusion matrix, and error analysis on misclassified samples.

**Requirements:** `pandas`, `numpy`, `huggingface_hub`, `sentence-transformers`, `faiss-gpu` (or `faiss-cpu`), `scikit-learn`, `matplotlib`, `seaborn`

**How to run:** Run cells top to bottom. Part 1 needs internet access (Hugging Face download + model weights) and a GPU is recommended for embedding ~hundreds of thousands of comments. Part 2 can run on CPU; it re-reads the same comments CSV independently, so it doesn't require Part 1 to have run first.

---

## 2. `run_validation.ipynb`

Reads `validation_requests.json`, builds a grounded prompt per question
(explicitly instructed to answer *only* from the given evidence and to say
so if evidence is insufficient), sends it to an OpenAI-compatible chat
endpoint (Metis AI, model `gpt-4o-mini`), and saves the answers alongside
their evidence for human review.

**Outputs:**
- `validation_results.json` — raw results (answer, latency, evidence, errors if any)
- `validation_manual_review.csv` — answer + full evidence text side by side, for manual QA

**Requirements:** `requests`, `pandas`

**How to run:**
1. Set your own API key — **do not hardcode it**. Replace the `API_KEY = "..."` lines with:
   ```python
   import os
   API_KEY = os.environ["METIS_API_KEY"]
   ```
   and set the environment variable before launching Jupyter, or use a `.env` file with `python-dotenv`.
2. Make sure `validation_requestsSection1_2_Section_3.ipynb`, Part 1) is in the working directory.
3. Run cells top to bottom. The first cell is a quick connectivity/auth check against the `/models` endpoint — run it first to confirm your key and see available models before running the full batch.

> ⚠️ **Security note:** an earlier version of this notebook had a live API key hardcoded in two cells. If that key was ever shared or committed anywhere, revoke/rotate it in the Metis AI dashboard and switch to an environment variable as shown above.

---

## Notes & Limitations

- FAISS uses an exact index (`IndexFlatIP`); for much larger datasets, switch to an approximate index (`IndexIVFFlat` / HNSW) for speed.
- The classifier's balanced sample (40k/class) is a design choice to avoid majority-class bias — evaluate with `f1_macro`, not accuracy, given class imbalance in the raw data.
- The LLM validation step calls the API sequentially; for large batches, consider adding concurrency/retries.
