# Beer-Comment-Analysis-LLM-Powered-Sentiment-Filter

A two-stage sentiment filtering pipeline for the beer industry: first, rule-based rough screening using brand lexicon and keywords; then, semantic judgment by LLM on candidate rows, outputting Excel files with own-brand negatives marked in blue and competitor/industry negatives marked in yellow. The system also includes human-annotated Benchmark, multi-model evaluation, Bad Case archiving, and experiment report generation.

The project is designed for real-world business scenarios: raw CSV data comes from web-scraped beer industry posts (via Quark), which are noisy, heavily colloquial, contain OCR errors, and where "keyword hit" does not necessarily equal negative sentiment. The repository only retains desensitized sample data and synthesized Benchmark; real scraped data is not included.

## Quick Start

```bash
pip install -e ".[dev]"

# Run without API Key: use mock LLM
beer-sentiment run \
  --input-dir data/samples \
  --output-dir object \
  --session morning \
  --date 2026-08-24 \
  --model mock

# Evaluate on Benchmark and generate report
beer-sentiment eval --model mock

# Compare multiple models (API Keys required)
beer-sentiment eval --models mock,deepseek-v4,qwen-max,kimi-k3
```

Human review workflow is also supported:

```bash
beer-sentiment prepare --input-dir original --session morning --date 2026-08-24
beer-sentiment build --review-csv 待筛选_上午.csv --session morning
```

## Directory Structure

```text
beer-comment-analysis/
├── benchmark/                 # Human-annotated Benchmark (desensitized subset)
├── config/                    # Brand lexicon, keywords, pipeline & model configs
├── prompts/                   # Versioned judgment prompts
├── data/samples/              # Public demo data
├── src/beer_sentiment/
│   ├── rules/                 # Stage 1: OCR normalization, brand matching, candidate filtering
│   ├── llm/                   # Stage 2: Mock / OpenAI-compatible models, structured output
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

## Connecting Real Models

`config/models.yaml` has placeholders for three OpenAI-compatible interfaces: DeepSeek-V4, Qwen-Max, and Kimi-K3. To connect, set the corresponding `api_key_env` environment variables and verify `base_url / model / unit_price` matches your account:

```bash
export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...
export MOONSHOT_API_KEY=...
```

## Roadmap

- M1: Engineering skeleton, configs, tests, CI, demo data
- M2: LLM judgment abstraction, Benchmark, evaluation reports
- M3: Hybrid RAG (BM25 + Dense + RRF + Cross-Encoder) + Bad Case auto-feedback
- M4: Streamlit demo page and automated scheduling

## Note

The Benchmark in this repository is a desensitized synthetic subset used to reproduce the evaluation pipeline. The full private annotated dataset and real scraped data are not committed. Model names and prices are placeholder configs; please adjust according to your actual accounts.
