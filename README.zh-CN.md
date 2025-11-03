# ArxivEmbedding

一个用于构建与发布 arXiv 论文元数据与向量嵌入（embeddings）的自动化项目：
- 从 arXiv OAI-PMH 接口增量抓取论文元数据（标题、摘要、作者、分类、日期等）。
- 使用 Google Gemini Batch Embedding API 批量生成向量（比在线实时接口便宜约 50%）。
- 以高效 Parquet 格式本地持久化，并上传至 Hugging Face Datasets 仓库。
- 通过简单的 CLI 一键更新，支持按分类与开始日期增量计算、对历史进行 Super Squash。

> 代码基于 Python 3.12，核心依赖包括: polars、datasets、huggingface-hub、google-genai、pyarrow、tqdm、typer 等。


## 功能特性

- 增量抓取: 基于 arXiv 的 OAI-PMH 接口，仅同步新论文（可按分类与开始日期过滤）。
- 高效存储: 使用 Parquet（ZSTD 压缩），并对嵌入列启用字节流分裂，适合大规模向量。
- 向量生成: 调用 Google Gemini 的批量嵌入任务，自动轮询任务直至完成后一次性取回结果。
- 顺序对齐: 以元数据顺序对齐向量数据，便于下游一致复现。
- 自动发布: 上传至 Hugging Face 数据集仓库，可选开启 Super Squash 历史压缩。
- 易用 CLI: 通过 `script/update.py` 提供一键更新入口与常用参数。


## 目录结构

```
ArxivEmbedding/
├─ config.toml                # 项目配置（HF 仓库、模型、维度、起始日期、类别等）
├─ pyproject.toml             # 依赖与项目元信息
├─ README.md                  # 英文 README（此仓库默认）
├─ README.zh-CN.md            # 本文件（中文说明）
├─ data/                      # 本地缓存与导出数据目录
│  └─ ArxivEmbedding/
├─ script/
│  └─ update.py               # CLI：更新元数据与向量并上传 HF
├─ src/
│  ├─ config.py               # Pydantic 配置模型与 TOML 解析
│  ├─ const.py                # 常量（工程根路径等）
│  ├─ embedding.py            # Google Gemini 批量嵌入逻辑
│  ├─ io.py                   # 读写 HF / 本地 Parquet、增量过滤与上传
│  ├─ name.py                 # 列名常量
│  ├─ oai.py                  # arXiv OAI-PMH 抓取客户端
│  ├─ order.py                # 排序与顺序对齐工具
│  └─ schema.py               # Polars/Arrow Schema 与数据类
└─ .github/workflows/
   ├─ keepalive.yml           # 定时心跳，保持 Actions 活跃
   └─ huggingface_super_squash.yml # 手动触发的 HF Super Squash
```


## 安装与准备

- 环境要求: Python >= 3.12（Linux / macOS 均可，示例命令以 zsh 为准）。
- 安装依赖（任选其一）:
  - 使用 pip（基于 `pyproject.toml`）:
    ```zsh
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -e .
    ```
  - 使用 uv（仓库含 `uv.lock`，若你偏好 uv）:
    ```zsh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```
- 准备密钥与配置:
  - 在仓库根目录创建 `.env`，写入你的 Google API Key：
    ```
    GEMINI_API_KEY=你的_API_Key
    ```
  - 根据需要编辑 `config.toml`（默认已提供一个示例）：
    ```toml
    [metadata]
    hf_repo = "lyk/ArxivMetaData"
    row_group = 200_000

    [embedding]
    hf_repo = "lyk/ArxivEmbedding"
    row_group = 200_000
    model = "gemini-embedding-001"
    dim = 1536
    start_date = "2020-01-01"            # 可选：仅对该日期后论文生成嵌入
    categories = ["cs.CL", "cs.CV", "cs.AI", "cs.LG", "stat.ML", "cs.IR", "cs.CY"]
    ```


## 一键更新（CLI）

CLI 入口位于 `script/update.py`，基于 Typer：

- 查看帮助：
  ```zsh
  python script/update.py --help
  ```

- 按配置更新元数据与向量，并上传至 HF：
  ```zsh
  # 默认：同时更新元数据与向量，并在上传后 Super Squash 历史
  python script/update.py -c config.toml --metadata --embedding --squash
  ```

- 仅更新元数据：
  ```zsh
  python script/update.py -c config.toml --metadata --no-embedding
  ```

- 仅更新向量（按分类与开始日期增量生成）：
  ```zsh
  python script/update.py -c config.toml --no-metadata --embedding
  ```

- 清理 Hugging Face 缓存目录后再更新：
  ```zsh
  python script/update.py -c config.toml --clean
  ```

参数说明（节选）：
- `--metadata/--no-metadata`: 是否更新元数据
- `--embedding/--no-embedding`: 是否更新向量
- `--squash/--no-squash`: 上传后是否执行 Super Squash
- `--clean/--no-clean`: 是否先清理 `data/hg` 缓存目录


## 数据产出与发布

- 本地导出：
  - 元数据: `data/metadata.parquet`
  - 向量: `data/embedding.parquet`
- 远端发布：由 `config.toml` 中的 `metadata.hf_repo` 与 `embedding.hf_repo` 控制上传的 Hugging Face 数据集仓库路径。
- 历史压缩：上传完成后可选调用 `HfApi().super_squash_history(...)` 对数据集仓库进行历史压缩（更小的 git 存储占用）。


## GitHub Actions

- `keepalive.yml`: 每周触发一次心跳提交（写入 `keepalive_counter.txt`），以保持 Actions 活跃。
- `huggingface_super_squash.yml`: 手动触发，使用 `HUGGINGFACE_TOKEN` 对 `lyk/ArxivEmbedding` 执行 Super Squash。若你需要操作其他仓库，请修改工作流脚本中的 `repo_id`。


## 开发者指南

- 嵌入实现：`src/embedding.py` 的 `google_batch_embedding` 使用 Google Gemini 批量接口提交任务并轮询状态，成功后一次性读取全部向量。
- 抓取实现：`src/oai.py` 实现 OAI-PMH 客户端，支持按类别（如 `cs.AI`、`stat.ML`）与起止日期抓取，自动处理 Resumption Token。
- 读写与对齐：`src/io.py` 负责加载/上传 HF 数据集、增量过滤与落盘；`src/order.py` 实现按分类+日期排序与向量对齐。
- 架构约定：所有列名在 `src/name.py` 统一声明；Schema 在 `src/schema.py` 定义，便于保持数据一致性。


## 常见问题（FAQ）

- Q: 需要哪些环境变量？
  - A: 至少需要 `GEMINI_API_KEY`（Google GenAI）。若在 CI 中上传到 Hugging Face，还需要 `HUGGINGFACE_TOKEN`（作为 GitHub Secret 配置）。

- Q: 费用与速率如何？
  - A: 使用 Batch Embedding 相比在线接口价格约 50%；大批量时更划算。请留意谷歌配额与速率限制，必要时增大 `poll_interval` 或控制 `batch_size`。

- Q: 初次运行向量更新时，HF 上还没有数据会报错吗？
  - A: 代码已处理：若远端不存在数据集或为空，将以空表开始并仅上传本次生成的增量。

- Q: 维度（`dim`）与模型不一致会怎样？
  - A: 请保证 `config.toml` 中的 `dim` 与所选 `model` 输出维度一致（例如 `gemini-embedding-001` 常用 768/1536/3072）。不一致会导致下游 Schema 不匹配。


## 许可

本仓库未显式声明开源许可证，如需复用请先联系仓库作者或在 Issue 中沟通。


## 致谢

- arXiv OAI-PMH
- Google Gemini
- Hugging Face Datasets & Hub
- Polars / PyArrow / DuckDB
