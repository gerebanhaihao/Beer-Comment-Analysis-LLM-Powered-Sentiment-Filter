# Beer-Comment-Analysis-LLM-Powered-Sentiment-Filter

A two-stage sentiment filtering pipeline for the beer industry: first, rule-based rough screening using brand lexicon and keywords; then, semantic judgment by LLM on candidate rows, outputting Excel files with own-brand negatives marked in blue and competitor/industry negatives marked in yellow. The system also includes human-annotated Benchmark, multi-model evaluation, Bad Case archiving, experiment report generation, and a Hybrid RAG module (Dense + Sparse retrieval, RRF fusion, Cross-Encoder reranking) that injects retrieved judgment rules and few-shot examples into the prompt.

The project is designed for real-world business scenarios: raw CSV data comes from web-scraped beer industry posts (via Quark), which are noisy, heavily colloquial, contain OCR errors, and where "keyword hit" does not necessarily equal negative sentiment. The repository only retains desensitized sample data and synthesized Benchmark; real scraped data is not included.

## Quick Start

## Windows 桌面版

界面入口是 `python -m beer_sentiment.gui`。在项目根目录打包：

```powershell
python -m pip install -e ".[llm]" pyinstaller
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

成品位于 `dist/BeerSentiment/BeerSentiment.exe`。请保留整个 `dist/BeerSentiment` 文件夹，exe 运行需要同级的 `_internal/`。桌面界面提供“导入 CSV”“自动筛选”和“模型评测”三个功能。自动筛选和评测可选择 DeepSeek、Qwen 或 Kimi，并在界面中输入对应的 API 密钥；密钥只在当次运行的内存中使用，不写入配置文件或日志。当前轻量发布版不包含 Dense RAG 模型，但会使用品牌别名、拼音、编辑距离匹配和其他本地筛选规则。

增加新功能后，修改源码并再次运行打包脚本。脚本会把旧发布目录保留为 `dist/BeerSentiment-backup-时间戳/`，再生成新版；需要继续使用旧数据时，从备份目录复制 `data/`、`incoming/`、`output/`、`.env` 和已修改的 `config/`。exe 本身不能直接修改源码功能。

```bash
pip install -e ".[llm,dev]"

# For real-model runs with Hybrid RAG, install the embedding and CrossEncoder runtime:
pip install -e ".[rag]"

# Drop raw Quark CSV files into data/, then use a real model after configuring its API key:
beer-sentiment run --input-dir data --output-dir output --all-time --model deepseek

# Run without API Key: use mock LLM (prepare your own CSVs in data/ first)
beer-sentiment run --input-dir data --output-dir output --all-time --model mock

# Disable Hybrid RAG (plain LLM judgment only):
beer-sentiment run --all-time --no-rag

# Compare the three offline model simulations on the converted Benchmark
beer-sentiment eval --models deepseek-v4,qwen-max,kimi-k3

# Convert manually colored Benchmark Excel files to a local, Git-ignored JSONL
python scripts/convert_benchmark_excel.py

# Evaluate the private real Benchmark explicitly (the default remains synthetic)
beer-sentiment eval --benchmark data/beer_sentiment_benchmark_real.jsonl --models deepseek-v4,qwen-max,kimi-k3

# Run the local real-Benchmark smoke test; it skips automatically when the file is absent
pytest tests/test_real_benchmark_local.py

# Optional: validate and import daily Quark CSV exports from a local inbox
beer-sentiment ingest --input-dir incoming --data-dir data

# Use --move to physically move successfully imported files into data/
beer-sentiment ingest --input-dir incoming --data-dir data --move
```

`.env` at the project root (already git-ignored, only needed for real API calls):

```text
DEEPSEEK_API_KEY=sk-...
```

Notes:

- `run` defaults to `--input-dir data --output-dir output`; colored Excel files are written next to the source file name (own-brand negatives in blue, competitor/industry negatives in yellow).
- With `--all-time` every CSV row is processed; without it, rows are filtered to the morning/afternoon time window defined in `config/pipeline.yaml`.
- Low-confidence rows (below `stage2.low_confidence_threshold`) are printed for human review at the end of the run.
- `mock` is not a real model. It is an offline rule-based baseline used for tests and smoke runs.
- `deepseek-v4`, `qwen-max`, and `kimi-k3` are deterministic local simulations until their real APIs are configured. Their metrics must not be presented as real model performance.

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
├── incoming/                  # Optional local drop folder for downloaded Quark CSVs
├── data/                      # Validated raw Quark CSV data（本地保留、不入库，见 .gitignore）
├── src/beer_sentiment/
│   ├── rules/                 # Stage 1: OCR normalization, fuzzy brand-alias matching, candidate filtering
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

## Local File Ingestion and Brand Matching

The project does not connect to Quark. The optional `ingest` command validates
CSV files placed in a local `incoming/` folder, skips content duplicates, and
copies them into `data/`; add `--move` when the original download should be
removed from the inbox after successful import. Files can also be placed
directly in `data/`, which remains the runtime input directory.

Brand matching supports exact aliases, English aliases, pinyin aliases, and
one-edit-distance fuzzy matching. Product names such as `勇闯天涯` are kept as
ordinary aliases of their parent brand in `config/brands.yaml`.

For Hybrid RAG, the BM25 query uses the combined text after OCR typo
normalization and adds canonical names for matched brands. Dense retrieval and
model judgment still receive the combined original text; output columns keep
the source CSV values.

## Evaluation Metrics

`eval` outputs accuracy, macro-average F1, negative detection precision/recall/F1, false positive rate, false negative rate, confusion matrix, average latency, and cost. Each experiment is archived under `artifacts/runs/` with model name, prompt version, config hash, metrics, and Bad Cases. Multi-model evaluation generates an additional comparison table at `artifacts/reports/model_compare.md`.

The private converted Benchmark uses sequential IDs (`b000001`, `b000002`, ...), category, data scope, title, body, cover OCR, and content OCR. Source filenames and Excel row numbers are not written to JSON. The source `情感` column is not used as the gold label; row fill color is the gold label. The converter writes to the Git-ignored `data/beer_sentiment_benchmark_real.jsonl` by default. `pytest` continues to exercise the synthetic Benchmark committed under `benchmark/`; the local real-Benchmark smoke test runs only when its private file exists. Set `BEER_SENTIMENT_REAL_BENCHMARK` to test a different local JSONL path.

## Hybrid RAG

`src/beer_sentiment/rag/` implements the knowledge retrieval layer for Stage 2:

- **Knowledge base** (`config/knowledge_base.yaml`): judgment rules (OCR cross-column reading, keyword-hint-only, teaching/merchant/counterfeit/nostalgia exclusions, blue-vs-yellow mapping) and labeled few-shot examples, continuously maintained from Bad Cases.
- **Sparse retrieval**: BM25 over character n-grams (`rag/sparse.py`).
- **Dense retrieval**: `SentenceTransformer` creates semantic embeddings for knowledge entries at startup and for each query; normalized vectors are compared by cosine similarity (`rag/dense.py`). The default Chinese model is `BAAI/bge-small-zh-v1.5`.
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60) over both rankings (`rag/hybrid.py`).
- **Reranking**: disabled by default for the current small knowledge base. It can optionally use an independent `CrossEncoder` model (`BAAI/bge-reranker-base`) to score query–candidate pairs; invalid scores or inference errors fall back to the RRF order.
- **Injection**: `RagJudge` renders the top-k entries into the "参考上下文" block of the judgment prompt.

Model names and retrieval knobs live in `config/rag.yaml` (top-k per stage, RRF k, rerank on/off, few-shot count, context length cap). Install the `rag` extra before running a real model with RAG. The first run downloads both model weights from Hugging Face; subsequent runs use its local cache. Dense model loading or inference errors stop the run rather than silently replacing semantic retrieval with keyword matching. The reranker can be disabled with `rerank.enabled: false`.

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

The Benchmark JSONL committed to this repository is synthetic. Real manually colored Excel inputs and the converted `data/beer_sentiment_benchmark_real.jsonl` remain local and are ignored by Git; blank rows without text are skipped during conversion. Simulated model prices and latencies are placeholders; replace them with measured values after connecting real APIs.
