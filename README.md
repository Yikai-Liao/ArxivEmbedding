# ArxivEmbedding
Create Daily Arxiv Embedding from RSS

## Fetch Arxiv Metadata via OAI-PMH (`fetch_arxiv_oai.py`)

This script fetches paper metadata (ID, title, authors, abstract, date, categories, etc.) from the arXiv OAI-PMH v2.0 interface for specified years and categories.

### Features

*   Fetches metadata based on years provided as command-line arguments.
*   Reads target arXiv categories and OAI parameters (URL, delays, retries) from `config.toml`.
*   Saves the collected metadata into yearly Parquet files (e.g., `data/2023.parquet`) using the Polars library.
*   Supports **incremental updates**: If a Parquet file for a specific year already exists, the script reads it, fetches new records for that year, combines the data, removes duplicates based on the arXiv ID (`id` column), and overwrites the file with the updated dataset.
*   Uses `loguru` for logging.

### Configuration (`config.toml`)

*   `category`: A list of arXiv categories to fetch (e.g., `["cs.AI", "stat.ML"]`).
*   `[ArxivOAI]`: Contains parameters for interacting with the OAI endpoint:
    *   `base_url`: The OAI base URL.
    *   `metadata_prefix`: The metadata format to request (e.g., `oai_dc`).
    *   `request_delay_seconds`: Delay between paginated requests.
    *   `retry_delay_seconds`: Delay before retrying after a 503 error.
    *   `max_retries`: Maximum number of retries for failed requests.
    *   `batch_size`: (Currently unused in the script logic but defined in config).

### Dependencies

Required Python libraries are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

### Usage

Run the script from the root directory of the project, providing the years you want to fetch as arguments:

```bash
python script/fetch_arxiv_oai.py <YEAR1> [YEAR2] ...
```

**Example:**

To fetch or update data for the years 2023 and 2024:

```bash
python script/fetch_arxiv_oai.py 2023 2024
```

The script will process each year sequentially, creating or updating the corresponding Parquet file in the `data/` directory.

## 本地批量生成嵌入 (`batch_embed_local.sh`)

本仓库提供 `batch_embed_local.sh` 用于在本地批量生成历史数据嵌入并上传至 Hugging Face 数据集仓库。

### 先决条件

* 已安装 [uv](https://github.com/astral-sh/uv)（脚本会在首次执行时自动同步依赖）。
* 本地可用的 NVIDIA GPU 与 CUDA 环境。
* 已在 shell 中导出 `HF_TOKEN`，具备对目标数据集仓库的读写权限，例如：

    ```bash
    export HF_TOKEN=your_hf_token
    ```

### 目录约定

脚本会把中间任务文件和嵌入结果写入 `temp/local_matrix_tasks/` 与 `temp/local_artifacts/`，合并后的年度 Parquet 会暂存于 `merged_data/` 并自动上传到 config 中指定的 `repo-id`（默认 `lyk/ArxivEmbedding`）。

### 快速上手

```bash
chmod +x batch_embed_local.sh               # 仅需执行一次
HF_TOKEN=xxx ./batch_embed_local.sh --years 2025 --batch-size 128
```

### 常用参数

* `--years`：逗号分隔的年份列表（默认 `2025`）。
* `--batch-size`：入队的文本数量。GPU 显存充裕时可调大；若发生显存不足可调小或限制任务数。
* `--max-tasks`：限制每次处理的任务数量，便于快速验证。
* `--engine` / `--device`：底层推理引擎与设备（默认 `torch` / `cuda`）。
* `--skip-sync`：若依赖已同步，可加此参数跳过 `uv sync`。

执行完成后，脚本会：

1. 调用 `script/local_split_tasks.py` 列出缺失嵌入的论文并生成任务文件。
2. 使用 `script/process_matrix_tasks.py` 在 GPU 上批量生成嵌入（默认 FP16）。
3. 通过 `script/merge.py` 合并到最新的年度 Parquet 并推送至 Hugging Face 数据集。
