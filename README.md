# Digikala product comparison

This repository currently implements **Phase 12**: pinned full-data acquisition,
canonical Parquet conversion, data-quality/full-population statistics, duplicate
product-metadata analysis, conflict-aware canonical product identities, and a
deterministic lexical product resolver, product-scoped BM25 and BGE-M3 dense
review retrieval, reproducible Reciprocal Rank Fusion (RRF), and a pinned BGE
cross-encoder reranker, a deterministic, support-aware structured
product-comparison engine, cost-bounded structured Persian generation,
deterministic grounding validation, and a reproducible final evaluation
workflow. It does not contain aspect extraction or LLM-as-a-judge evaluation.

## Requirements

- Python 3.12+
- The exact dataset revision declared in `config/default.toml`

Install the package and test dependency:

```powershell
python -m pip install -e ".[dev]"
```

## Dataset placement

Download the exact CSV files from the pinned revision in `config/default.toml`:

```text
data/raw/digikala-products.csv
data/raw/digikala-comments.csv
```

```powershell
digikala-download-data --config config/default.toml
# use --force only to replace an existing raw file
digikala-download-data --config config/default.toml --force
```

The downloader streams each file to `*.part`, checks the Hugging Face resolved
revision, verifies the received size when supplied by the source, and atomically
renames it only after completion. It records the verified revision and size in a
raw-data manifest. Raw data is ignored by Git.

## Commands

Run preprocessing, write Parquet files, and write the quality report:

```powershell
digikala-preprocess --config config/default.toml
# or
python scripts/preprocess.py --config config/default.toml
```

Generate only the quality report from the raw CSV files:

```powershell
digikala-quality-report --config config/default.toml
# or
python scripts/quality_report.py --config config/default.toml
```

Build statistics from **all canonical reviews** matched to known products:

```powershell
digikala-build-statistics --config config/default.toml
# or
python scripts/build_statistics.py --config config/default.toml
```

Analyze duplicate product metadata and build one conflict-aware canonical product
record per stable `product_id`:

```powershell
digikala-build-canonical-products --config config/default.toml
# or
python scripts/build_canonical_products.py --config config/default.toml
```

The canonicalizer chooses each displayed field by deterministic mode (then lexical
tie-break), never averages mutable values, and records source-row counts, observed
values, conflicting fields, and a canonicalization status. Conflicting title,
brand, or category identity is marked rather than silently treated as certain.

Resolve an ID, title, or structured title constraint without performing a
comparison:

```powershell
digikala-resolve-product --id 10000038 --json
digikala-resolve-product "Samsung A55 5G" --brand Samsung --json
```

Resolution uses exact ID, exact normalized title, then brand-aware RapidFuzz
lexical matching. Model-significant tokens are hard constraints: model numbers,
storage, `4G`/`5G`, `Pro`/`Pro+`, and `Max` variants cannot be silently swapped.
The default score threshold is 90 and candidates within 3 points remain
`ambiguous` instead of being auto-selected.

Run the small reproducible real-data resolver benchmark:

```powershell
digikala-evaluate-resolution --config config/default.toml
# or
python scripts/evaluate_resolution.py --config config/default.toml
```

## Phase 4: review-retrieval baseline

Build the review-evidence corpus after preprocessing. It preserves `review_id`
and `product_id`, review text (raw and normalized), title, advantages,
disadvantages, buyer flag, recommendation status, rate, likes, and dislikes.
The unit is one eligible review per document; duplicate source review IDs are
reported and deterministically reduced to their first canonical-source record.

```powershell
digikala-build-retrieval-corpus --config config/default.toml
# or
python scripts/build_retrieval_corpus.py --config config/default.toml
```

Run the explicit Persian BM25 baseline for one already-resolved product. Results
are always product-scoped and include `review_id` provenance:

```powershell
digikala-retrieve-bm25 --product-id 3331597 --query "کیفیت ساخت" --top-k 10
# or
python scripts/retrieve_bm25.py --product-id 3331597 --query "کیفیت ساخت"
```

Create or inspect the frozen seed benchmark and evaluate BM25. The corpus is one
Parquet file and per-product indexes are built on demand in a bounded LRU cache;
there are no per-review index files.

```powershell
digikala-evaluate-retrieval --config config/default.toml --method bm25 --build-benchmark
# or
python scripts/evaluate_retrieval.py --config config/default.toml --method bm25
```

The benchmark writes queries as JSONL, qrels as CSV, ranked results as Parquet,
and the report as JSON. Its labels are deterministic lexical **seed candidates
pending human review**, not relevance ground truth; do not use its scores to
claim final retrieval quality. Queries with no known relevant document remain in
the report and are excluded explicitly from aggregate Recall@K, Precision@K,
MRR, and NDCG@K. Retrieval evidence is never used to calculate full-product
satisfaction statistics.

## Phase 5: BGE-M3 dense retrieval

Phase 5 uses the pinned `BAAI/bge-m3` snapshot in `config/default.toml` through
the official `FlagEmbedding` API. It encodes the identical Phase 4 review corpus
and field composition, using only dense vectors. Each vector checkpoint has
durable `review_id`, `product_id`, and evidence metadata. FAISS `IndexFlatIP` is
built only for the requested product from its contiguous persisted vector range,
so product filtering is guaranteed without creating an index file per review.

Start with a real, small CPU-safe pilot. It is resumable and writes a measured
full-index time/storage estimate to `data/reports/bge_m3_resource_estimate.json`:

```powershell
digikala-build-dense-index --config config/default.toml --pilot
# or
python scripts/build_dense_index.py --config config/default.toml --pilot
```

Full indexing is deliberately explicit. On a CPU-only machine it additionally
requires acknowledgement of the pilot estimate; it is never started by default:

```powershell
digikala-build-dense-index --config config/default.toml --full --accept-cpu-full-estimate
```

After a completed full index, retrieve evidence or evaluate it against exactly
the Phase 4 frozen queries/qrels:

```powershell
digikala-retrieve-dense --product-id 3331597 --query "کیفیت ساخت" --top-k 10 --json
digikala-evaluate-dense --config config/default.toml
```

The dense evaluator writes its own ranked-results Parquet, evaluation JSON,
BM25-vs-BGE-M3 comparison JSON, and failure-analysis JSON. If only a pilot is
available, the dense method is explicitly `unavailable` for the controlled
comparison rather than being scored against a different corpus. As with BM25,
the present qrels are seed candidates pending human review, not final relevance
ground truth.

## Phase 6: hybrid lexical + dense retrieval

Phase 6 fuses the product-scoped BM25 and BGE-M3 candidates by Reciprocal Rank
Fusion (RRF). Candidate identity is always the stable `review_id`: text is never
used as a merge key. Each hybrid candidate retains its review evidence and
metadata, plus its BM25 score/rank, dense score/rank, and fused score/rank.

All RRF parameters are explicit in `[hybrid]` in `config/default.toml`:
BM25 candidate depth, dense candidate depth, `rrf_k`, and final `top_k`. The
default is RRF with `k=60`; unlike score fusion, it does not assume that sparse
and dense score scales are comparable.

The frozen Phase 6 development/test overlay is written on first evaluation to
`data/benchmarks/retrieval_splits_v1.json`. Only the development IDs may tune
the configured RRF candidates; the selected value is persisted in
`data/experiments/hybrid_rrf_tuning.json` and evaluated once on the separate
test IDs. The overlay does not alter the Phase 4 queries or qrels.

After a completed full dense index, retrieve one product's fused evidence or
run the three-method benchmark:

```powershell
digikala-retrieve-hybrid --product-id 3331597 --query "کیفیت ساخت" --top-k 10 --json
digikala-evaluate-hybrid --config config/default.toml
# or
python scripts/retrieve_hybrid.py --product-id 3331597 --query "کیفیت ساخت" --top-k 10 --json
python scripts/evaluate_hybrid.py --config config/default.toml
```

The benchmark reports Recall@K, Precision@K, MRR, NDCG@K, cold/warm p50/p95
latency, total/shared storage, and process peak memory for BM25, BGE-M3 dense,
and Hybrid RRF. It also records query-level improvements/regressions/ties and
candidate-depth Jaccard overlap. Its production rule retains BM25 unless hybrid
improves NDCG by at least the configured threshold and its warm p95 passes the
latency guard. If the full dense index is unavailable, dense and hybrid are
reported as unavailable and BM25 is retained; they are never silently compared
using a pilot. These seed labels remain pending human review.

## Phase 7: BGE reranking

Phase 7 adds the exact pinned `BAAI/bge-reranker-v2-m3` revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` strictly after Hybrid RRF:

```text
BM25 candidates + BGE-M3 dense candidates -> RRF hybrid candidates -> BGE reranker -> final Top-K
```

The reranker pair is the original query and `indexed_text_normalized` only. That
field is the frozen normalized composition of the review title, review body,
advantages, and disadvantages; no product-level aggregate statistic is sent to
the model. Final records preserve `review_id`, `product_id`, sparse/dense/fused
scores and ranks, `reranker_score`, and `final_rank`.

Candidate depths `20`, `50`, and `100` are tuned only on the frozen development
partition. The selected depth is then evaluated once on the separate test IDs.
The four-way report includes final-evidence metrics, candidate recall at N,
end-to-end and reranker-only latency distributions, throughput, RAM/VRAM where
available, storage, bootstrap uncertainty, and deterministic failure examples.

```powershell
digikala-retrieve-reranked --product-id 3331597 --query "کیفیت ساخت" --top-k 10 --json
digikala-evaluate-reranker --config config/default.toml
# or
python scripts/retrieve_reranked.py --product-id 3331597 --query "کیفیت ساخت" --top-k 10 --json
python scripts/evaluate_reranker.py --config config/default.toml
```

The CPU preflight records model revision, dtype, batch size, truncation policy,
available RAM, and accelerator state before downloading/loading the model. It
refuses an unsafe CPU load rather than silently substituting a different model
or risking an out-of-memory failure. A full BGE-M3 dense index remains a hard
prerequisite: if it is unavailable, Dense, Hybrid, and Hybrid+Reranker are each
reported as unavailable in the four-way benchmark.

## Phase 8: frozen retrieval experiment and evidence API

Run the complete four-method benchmark and freeze its reproducibility inputs:

```powershell
digikala-retrieval-benchmark --all --config config/default.toml
# or
python scripts/retrieval_benchmark.py --all --config config/default.toml
```

This writes separate development/test query and qrel copies under
`data/benchmarks/frozen_retrieval_v1/`, a machine-readable experiment manifest,
and a concise Markdown comparison table. The manifest records corpus eligibility
and SHA-256, tokenizer/normalization versions, all four retriever configurations,
model revisions, seed, package/hardware details, and the selected production
method. The command reports unavailable methods explicitly when the full dense
index or reranker cannot run.

Use only that frozen production selection to retrieve review evidence:

```powershell
digikala-retrieve-evidence --product-id 82098 --criterion "battery" --query "باتری و شارژدهی" --top-k 10 --json
# or
python scripts/retrieve_evidence.py --product-id 82098 --criterion "battery" --query "باتری و شارژدهی" --top-k 10 --json
```

`EvidenceSet` returns review-level provenance only: product/review IDs, rank,
final score, raw review text, buyer/recommendation metadata, likes/dislikes, and
available source scores/ranks. It includes a deterministic candidate-count and
score-distribution summary but never interprets scores as probabilities. It may
return fewer than K results or no evidence; it never fabricates filler reviews.

Full recommendation percentages remain a separate typed interface backed only
by `product_statistics.parquet`. Top-K `EvidenceSet` objects cannot be passed to
the global-recommendation summary API, preventing retrieval samples from being
mistaken for full-product statistics.

## Phase 9: deterministic structured comparison

`digikala-compare-structured` compares two or more already-resolved IDs using
typed canonical metadata, full-population Phase 2 statistics, and optional
frozen-production evidence. Numeric decisions are pure Python: no LLM or
retrieval score is used to calculate product-wide satisfaction percentages.

```powershell
digikala-compare-structured --product-id 82098 --product-id 514309 `
  --criterion price --criterion recommendation --criterion rate
```

The result is JSON with separate `direct_facts`, `aggregate_statistics`,
`retrieved_evidence`, `criterion_decisions`, and `overall` layers. It preserves
every evidence `review_id`, but labels retrieval counts as *within retrieved
evidence* and never treats them as population percentages. Direct-only criteria
such as `price`, `rate`, `Rate_cnt`, `is_fake`, and category metadata do not
retrieve irrelevant reviews.

Use `--evidence-query CRITERION=QUERY` for a review-based criterion whose
retrieval query should differ from its criterion label, and only use explicit
caller priorities for an overall result:

```powershell
digikala-compare-structured --product-id 82098 --product-id 514309 `
  --criterion price --criterion recommendation `
  --evidence-query "recommendation=مشکلات و ایرادها" `
  --weight price=0.4 --weight recommendation=0.6
```

Without `--weight`, `overall.status` is always `neutral`: the engine returns
criterion-level decisions but does not invent a universal winner. Configured
thresholds live in `[comparison]` in `config/default.toml`: recommendation
percentages need a denominator of at least 20; their practical gap is 5
percentage points; product `Rate` needs `Rate_cnt >= 10` and a 2-point gap; raw
same-snapshot prices need a 5% relative gap. Prices carry no inferred currency.
Conflicted metadata, missing fields, insufficient support, zero denominators,
unvalidated review-rate semantics, close results, and sparse evidence are all
first-class `inconclusive` outcomes with reason codes.

Run tests:

```powershell
python -m pytest
# or
python scripts/run_tests.py
```

Outputs are written to the configured `data/processed/` and `data/reports/`
paths. These data-derived artifacts are ignored by Git.

## Phases 10–11: structured generation and grounding

`digikala-generate-comparison` sends only the bounded deterministic comparison
context to the configured provider and validates the returned schema and every
claim. Review text remains untrusted data; the response must keep direct facts,
full-population statistics, retrieved-review evidence, and conditional
inference in distinct layers. Set `METIS_API_KEY` only when actually calling
the provider; it is never written to an artifact.

Each successful generation writes its complete UTF-8 answer, metadata, and
grounding result to `data/generation/outputs/`. The terminal prints only the
deterministic final decision and the saved file path.

The checked-in default is the Metis OpenAI-compatible endpoint
`https://api.metisai.ir/openai/v1`. Do not put a key in `default.toml`. In the
same PowerShell session, set a **rotated** key, list the models enabled for the
account, and copy one returned exact ID into `[generation].model`:

```powershell
$env:METIS_API_KEY = "<your rotated Metis key>"
digikala-list-llm-models --config config/default.toml
```

The model ID is intentionally not guessed. Metis token pricing is therefore
recorded as unknown until the selected model's published price is configured.

```powershell
# Inspect the exact bounded context without an API request.
digikala-generate-comparison --product-id 82098 --product-id 514309 `
  --criterion recommendation --question "مقایسه کن" --dry-run

# Validate a saved context/structured answer and write an auditable result.
digikala-validate-grounding --context context.json --answer answer.json
```

Full-dataset percentages are rechecked against `product_statistics.parquet`;
Top-K review evidence can never supply their denominator. Review citations must
exist, belong to the claimed product, and have been supplied in the exact
`EvidenceSet`. The validator rejects fabricated numeric facts, wrong-product or
out-of-context review IDs, unsupported review claims, and winners that override
an inconclusive/unauthorized deterministic decision.

## Phase 12: final evaluation and demo artifacts

Run the reproducible default evaluation. It uses the frozen selected production
retriever (currently BM25), a deterministic compact template, and the real
grounding validator. It does not make an LLM API call:

```powershell
digikala-evaluate-final --config config/default.toml --no-llm
```

The command freezes/reuses a balanced real-data Persian evaluation set, records
baseline vs final pipeline results, copies the four-way retrieval benchmark,
measures component latency/memory, checks inconclusive policy and a controlled
grounding ablation, and writes all artifacts under
`data/evaluations/final_v1/`. Use `--dry-run` to inspect only the case set and
system manifest. `--rebuild-evaluation-set` is deliberately explicit because
it changes the frozen evaluation input.

An API-bearing run is opt-in and capped; it requires the configured
`METIS_API_KEY` and is never needed for offline reproduction:

```powershell
digikala-evaluate-final --config config/default.toml --with-llm --max-llm-cases 5
```

The evaluator creates `human_answer_quality_template.csv` for a manageable
independent Persian human audit. Copy it to
`human_answer_quality_annotations.csv`, complete the 1–5 rubric with annotator
IDs, then rerun the command to aggregate human relevance, clarity, source
separation, uncertainty, citation usefulness, and semantic-support results.
The repository intentionally reports this section as pending until real human
annotations are supplied; it never substitutes an LLM or deterministic score
for human judgment.
