# Beer-Comment-Analysis-LLM-Powered-Sentiment-Filter

A two-stage sentiment filtering pipeline for the beer industry: first, rule-based rough screening using brand lexicon and keywords; then, semantic judgment by LLM on candidate rows, outputting Excel files with own-brand negatives marked in blue and competitor/industry negatives marked in yellow. The system also includes human-annotated Benchmark, multi-model evaluation, Bad Case archiving, experiment report generation, and a Hybrid RAG module (Dense + Sparse retrieval, RRF fusion, Cross-Encoder reranking) that injects retrieved judgment rules and few-shot examples into the prompt.

The project is designed for real-world business scenarios: raw CSV data comes from web-scraped beer industry posts (via Quark), which are noisy, heavily colloquial, contain OCR errors, and where "keyword hit" does not necessarily equal negative sentiment. The repository only retains desensitized sample data and synthesized Benchmark; real scraped data is not included.

## Quick Start

```bash
pip install -e ".[llm,dev]"

# Drop raw Quark CSV files into data/, then run end to end (requires DeepSeek API key in .env):
beer-sentiment run --all-time

# Equivalent long form:
beer-sentiment run --input-dir data --output-dir output --all-time --model deepseek

# Run without API Key: use mock LLM (prepare your own CSVs in data/ first)
beer-sentiment run --input-dir data --output-dir output --all-time --model mock

# Disable Hybrid RAG (plain LLM judgment only):
beer-sentiment run --all-time --no-rag

# Evaluate on Benchmark and generate report
beer-sentiment eval --model mock
```

`.env` at the project root (already git-ignored):

```text
DEEPSEEK_API_KEY=sk-...
```

Notes:

- `run` defaults to `--input-dir data --output-dir output`; colored Excel files are written next to the source file name (own-brand negatives in blue, competitor/industry negatives in yellow).
- With `--all-time` every CSV row is processed; without it, rows are filtered to the morning/afternoon time window defined in `config/pipeline.yaml`.
- Low-confidence rows (below `stage2.low_confidence_threshold`) are printed for human review at the end of the run.

Human review workflow is also supported:

```bash
beer-sentiment prepare --input-dir data --session morning --date 2026-08-24
beer-sentiment build --review-csv 待筛选_上午.csv --session morning
```

## Directory Structure

```text
beer-comment-analysis/
├── benchmark/                 # Human-annotated Benchmark (desensitized subset)
├── config/                    # Brand lexicon, keywords, pipeline, model & RAG configs
├── prompts/                   # Versioned judgment prompts
├── data/                      # Raw Quark CSV data（本地保留、不入库，见 .gitignore）
├── src/beer_sentiment/
│   ├── rules/                 # Stage 1: OCR normalization, brand matching, candidate filtering
│   ├── llm/                   # Stage 2: Mock / OpenAI-compatible models, structured output
│   ├── rag/                   # Hybrid RAG: BM25 + Dense vectors, RRF, Cross-Encoder rerank
│   ├── pipeline/              # Two-stage pipeline and end-to-end execution
│   ├── eval/                  # Benchmark, metrics, experiment reports
│   └── io/                    # CSV/Excel, time window, file naming
├── tests/                     # pytest unit tests and end-to-end tests
└── artifacts/                 # Evaluation logs and reports (not committed)
```

## Judgment Rules

- Own-brand negatives (Budweiser, Harbin, Corona, Sedrin) → **blue**.
- Competitor negatives (Tsingtao, Snow, Wusu, Heineken, RIO, Lubao) and industry-wide negatives → **yellow**.
- Must read `正文 / 封面OCR / 内容OCR / 标题` columns together; never rely on a single column.
- Keywords are signals only — do not label educational content, promotional comparisons, personal experiences, third-party counterfeiting, or nostalgic memories just because keywords appear.
- Low-confidence samples are not auto-labeled and enter the human review queue.

## Evaluation Metrics

`eval` outputs accuracy, macro-average F1, negative detection precision/recall/F1, false positive rate, false negative rate, confusion matrix, average latency, and cost. Each experiment is archived under `artifacts/runs/` with model name, prompt version, config hash, metrics, and Bad Cases. Multi-model evaluation generates an additional comparison table at `artifacts/reports/model_compare.md`.

## Hybrid RAG

`src/beer_sentiment/rag/` implements the knowledge retrieval layer for Stage 2:

- **Knowledge base** (`config/knowledge_base.yaml`): judgment rules (OCR cross-column reading, keyword-hint-only, teaching/merchant/counterfeit/nostalgia exclusions, blue-vs-yellow mapping) and labeled few-shot examples, continuously maintained from Bad Cases.
- **Sparse retrieval**: BM25 over character n-grams (`rag/sparse.py`).
- **Dense retrieval**: feature-hashed character n-gram vectors with cosine similarity, dependency-free and deterministic (`rag/dense.py`).
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60) over both rankings (`rag/hybrid.py`).
- **Reranking**: a Cross-Encoder-style LLM scorer (one batched API call per query) reorders the top-N fused candidates; on any failure it falls back to the RRF order.
- **Injection**: `RagJudge` renders the top-k entries into the "参考上下文" block of the judgment prompt.

All knobs live in `config/rag.yaml` (top-k per stage, RRF k, rerank on/off, few-shot count, context length cap).

## Connecting Real Models

`config/models.yaml` ships a `deepseek` entry (`deepseek-chat` via the OpenAI-compatible endpoint). Set the key in `.env` or export it:

```bash
export DEEPSEEK_API_KEY=...
```

## Roadmap

- M1: Engineering skeleton, configs, tests, CI, demo data
- M2: LLM judgment abstraction, Benchmark, evaluation reports
- M3: Hybrid RAG (BM25 + Dense + RRF + Cross-Encoder) + Bad Case auto-feedback
- M4: Streamlit demo page and automated scheduling

## Note

The Benchmark in this repository is a desensitized synthetic subset used to reproduce the evaluation pipeline. The full private annotated dataset and real scraped data are not committed. Prices in `config/models.yaml` are indicative; adjust according to your actual account.