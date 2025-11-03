ArxivEmbedding
===============

An automated pipeline to build and publish arXiv paper metadata and vector embeddings:

- Incrementally fetch paper metadata from arXiv OAI-PMH (title, abstract, authors, categories, dates, ...).
- Generate embeddings with Google Gemini Batch Embedding API (about 50% cheaper than realtime endpoints).
- Persist locally in efficient Parquet format and upload to Hugging Face Datasets.
- One-command CLI to update; supports category/date incremental compute and Super Squash of dataset history.

For Chinese documentation, see: README.zh-CN.md


## Features

- Incremental sync: Use arXiv OAI-PMH to fetch only new papers (filter by categories and start_date).
- Efficient storage: Parquet with ZSTD compression; byte-stream split for embedding leaf column.
- Batch embeddings: Submit a Gemini batch job and poll until completion, then fetch results at once.
- Order alignment: Keep embeddings aligned to metadata order for reproducible downstream usage.
- Automated publishing: Upload to Hugging Face dataset repos, optionally with Super Squash.
- Simple CLI: `script/update.py` provides a single entry with common flags.


## Project Layout

```
ArxivEmbedding/
├─ config.toml                # Project config (HF repos, model, dim, start_date, categories, ...)
├─ pyproject.toml             # Dependencies and project metadata
├─ README.md                  # This file (English)
├─ README.zh-CN.md            # Chinese README
├─ data/                      # Local cache and exported data
│  └─ ArxivEmbedding/
├─ script/
│  └─ update.py               # CLI: update metadata & embeddings and upload to HF
├─ src/
│  ├─ config.py               # Pydantic config models and TOML loader
│  ├─ const.py                # Constants (project root, ...)
│  ├─ embedding.py            # Google Gemini batch embedding logic
│  ├─ io.py                   # HF read/write, incremental filtering, local persistence & upload
│  ├─ name.py                 # Column name constants
│  ├─ oai.py                  # arXiv OAI-PMH client
│  ├─ order.py                # Sorting and order alignment utilities
│  └─ schema.py               # Polars/Arrow schemas and dataclasses
└─ .github/workflows/
	 ├─ keepalive.yml           # Weekly heartbeat to keep Actions alive
	 └─ huggingface_super_squash.yml # Manual HF Super Squash workflow
```


## Requirements and Setup

- Python >= 3.12 (Linux is tested; commands below are for zsh).
- Install dependencies (choose one):
	- pip (from `pyproject.toml`):
		```zsh
		python -m venv .venv
		source .venv/bin/activate
		pip install -U pip
		pip install -e .
		```
	- uv (project includes `uv.lock` if you prefer uv):
		```zsh
		curl -LsSf https://astral.sh/uv/install.sh | sh
		uv venv
		source .venv/bin/activate
		uv pip install -e .
		```

Environment variables and config:

- Create a `.env` at repo root with your Google API key:
	```
	GEMINI_API_KEY=your_api_key
	```
- Edit `config.toml` as needed (a working example is provided):
	```toml
	[metadata]
	hf_repo = "lyk/ArxivMetaData"
	row_group = 200_000

	[embedding]
	hf_repo = "lyk/ArxivEmbedding"
	row_group = 200_000
	model = "gemini-embedding-001"
	dim = 1536
	start_date = "2020-01-01"            # optional: only embed papers updated after this date
	categories = ["cs.CL", "cs.CV", "cs.AI", "cs.LG", "stat.ML", "cs.IR", "cs.CY"]
	```


## CLI: One-command Updates

The CLI entry is `script/update.py`, powered by Typer.

- Help:
	```zsh
	python script/update.py --help
	```

- Update metadata and embeddings per config and upload to HF:
	```zsh
	# Default: update both metadata and embeddings; Super Squash after upload
	python script/update.py -c config.toml --metadata --embedding --squash
	```

- Update metadata only:
	```zsh
	python script/update.py -c config.toml --metadata --no-embedding
	```

- Update embeddings only (incremental by category and start_date):
	```zsh
	python script/update.py -c config.toml --no-metadata --embedding
	```

- Clean HF cache dir then update:
	```zsh
	python script/update.py -c config.toml --clean
	```

Selected flags:
- `--metadata/--no-metadata`: whether to update metadata
- `--embedding/--no-embedding`: whether to update embeddings
- `--squash/--no-squash`: whether to Super Squash dataset repo history on HF
- `--clean/--no-clean`: whether to clean `data/hg` before running


## Outputs and Publishing

- Local exports:
	- Metadata: `data/metadata.parquet`
	- Embeddings: `data/embedding.parquet`
- Remote publishing: controlled by `metadata.hf_repo` and `embedding.hf_repo` in `config.toml`.
- History squash: optionally call `HfApi().super_squash_history(...)` on the dataset repo to reduce git history size.


## GitHub Actions

- `keepalive.yml`: weekly heartbeat commit (`keepalive_counter.txt`) to keep Actions active.
- `huggingface_super_squash.yml`: manually trigger Super Squash using `HUGGINGFACE_TOKEN` for `lyk/ArxivEmbedding`. If you need another repo, change the `repo_id`.


## Developer Notes

- Embeddings: `src/embedding.py` uses Gemini Batch Embedding: create job → poll status → fetch all vectors on success.
- Harvesting: `src/oai.py` implements the OAI-PMH client, supports categories like `cs.AI` and `stat.ML`, date ranges, and resumption tokens.
- IO & alignment: `src/io.py` loads/uploads HF datasets, incremental filtering, and local persistence; `src/order.py` sorts and aligns rows.
- Conventions: all column names in `src/name.py`; schemas in `src/schema.py` for consistency.


## FAQ

- Q: What env vars are required?
	- A: `GEMINI_API_KEY` (Google GenAI). For CI uploads to HF, set `HUGGINGFACE_TOKEN` as a GitHub secret.

- Q: Cost and rate limits?
	- A: Batch Embedding is ~50% cheaper vs realtime endpoints; it’s more cost-effective for large batches. Mind Google quotas and limits; tune `poll_interval` or reduce `batch_size` when needed.

- Q: First run with no existing HF dataset?
	- A: Handled: if remote dataset is absent/empty, we start from an empty table and upload new results only.

- Q: What if `dim` doesn’t match the model?
	- A: Ensure `dim` matches your model output (e.g., `gemini-embedding-001` commonly supports 768/1536/3072). Mismatch causes schema errors.


## License

No explicit open-source license is declared yet. If you plan to reuse, please contact the repository owner or open an issue to discuss.


## Acknowledgments

- arXiv OAI-PMH
- Google Gemini
- Hugging Face Datasets & Hub
- Polars / PyArrow / DuckDB

