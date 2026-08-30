# Digikala Category-Level Insights: Data Prep, LLM Analysis & Judge (Part 4) 

A three-notebook pipeline for category-manager-level analysis of Digikala
reviews: a Kaggle notebook that prepares quantitative stats and text batches
for every eligible product category, a local notebook that turns those into
LLM-generated complaint themes and brand comparisons, and an independent LLM
judge that scores the quality of those outputs.

## Project Structure

```
category_analysis_prep_kaggle.ipynb   # Data prep — run on Kaggle
category_analysis_local.ipynb         # Map-Reduce & brand comparison — run locally
category_judge.ipynb                  # LLM-as-a-Judge — run locally
```

## Pipeline Overview

```
category_analysis_prep_kaggle.ipynb
  ↓ category_prep_data.json
category_analysis_local.ipynb
  ↓ category_insights.json
category_judge.ipynb
  ↓ category_judge_results.json / category_judge_manual_review.csv
```

---

## 1. `category_analysis_prep_kaggle.ipynb`

Downloads the Digikala comments/products dataset from Hugging Face, finds
every category with at least `MIN_PRODUCTS_PER_CATEGORY` products, and for
each one computes recommendation-rate stats per product, flags
"high-volume / low-recommendation" products, computes per-brand stats for
the top brands, and packages negative-signal comments into fixed-size
batches for the Map step. No LLM calls happen here — just pandas
aggregation and text-batch preparation.

**Output:** `category_prep_data.json` — a `{"categories": [...]}` array, one
entry per eligible category, consumed by `category_analysis_local.ipynb`.

**Requirements:** `pandas`, `numpy`, `huggingface_hub`, `matplotlib`

**How to run:** Run cells top to bottom on Kaggle (a CPU-only session is
enough — no GPU or embedding model needed). Download
`category_prep_data.json` from Kaggle's output panel afterward.

---

## 2. `category_analysis_local.ipynb`

Reads `category_prep_data.json` and, for the selected category (or all of
them), runs a Map-Reduce pass over the negative-signal comment batches to
extract the top recurring complaint themes, then makes one additional call
to compare the top brands using their real stats and sample comments.
Includes a shared cost/token budget guard across the whole run.

**Output:** `category_insights.json` — a `{"categories": [...]}` array (one
report per processed category), consumed by `category_judge.ipynb`.

**Requirements:** `requests`, `pandas`

**How to run:**
1. Make sure `category_prep_data.json` is in the working directory.
2. Set `RUN_MODE` — `"single"` (default) prints the list of available
   categories and asks you to pick one by name or number (fast, good for
   testing); `"all"` processes every category in the file.
3. Run top to bottom.

---

## 3. `category_judge.ipynb`

Reads `category_insights.json` and scores each complaint theme and
brand-comparison summary with a second, independent model on three
criteria: **groundedness** (matches its attached evidence), **specificity**
(concrete enough to act on), and **actionability** (useful for a category
manager's decision). Produces a pass/fail verdict and a short justification
per item, tagged with the category it belongs to.

**Output:** `category_judge_results.json` (raw scores, a flat list with a
`category` field per item) and `category_judge_manual_review.csv` (sorted so
failed/low-scoring items surface first), plus a summary chart of average
scores per criterion.

**Requirements:** `requests`, `pandas`, `matplotlib`

**How to run:** Same `RUN_MODE` pattern as `category_analysis_local.ipynb`
— if `category_insights.json` already has just one category, it's picked
automatically; otherwise, `"single"` lets you choose which one to judge,
`"all"` judges everything. Make sure `category_insights.json` is present,
then run top to bottom.

---

## Notes & Limitations

- `category_analysis_local.ipynb` and `category_judge.ipynb` share a single
  cumulative budget (`MAX_BUDGET_USD`) across all categories processed in a
  run — once it's hit, remaining categories are skipped rather than
  erroring out.
- `category_analysis_prep_kaggle.ipynb` reads the full comments CSV once
  (chunked) and filters to the union of all eligible categories' products,
  rather than re-scanning per category — this keeps prep time roughly
  constant regardless of how many categories are eligible.
- `RUN_MODE = "single"` in the two local notebooks is the fast path for
  iterating on one category; switch to `"all"` only for the final full run,
  since ~180 categories at a couple of minutes each adds up quickly.
- The per-category recommendation-rate and high-volume/low-recommendation
  thresholds are computed independently within each category (not
  globally), so they stay meaningful regardless of category size.
