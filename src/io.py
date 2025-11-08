from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import polars as pl
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from .const import BASE_DIR
from .oai import fetch_arxiv_oai
from .name import *
from .schema import PAPER_METADATE_PL_SCHEMA, paper_embedding_schema_pl


def _hf_cache_dir() -> Path:
    cache_dir = BASE_DIR / "data" / "hg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_parquet(repo_id: str, filename: str) -> Path | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=_hf_cache_dir(),
        )
    except HfHubHTTPError as err:
        if err.response is not None and err.response.status_code == 404:
            return None
        raise
    return Path(path)


def _empty_metadata(*, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    df = pl.DataFrame(schema=PAPER_METADATE_PL_SCHEMA)
    return df.lazy() if lazy else df


def _empty_embedding(dim: int, *, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    df = pl.DataFrame(schema=paper_embedding_schema_pl(dim))
    return df.lazy() if lazy else df


def _to_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    return frame.lazy()


def _parse_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def load_metadata(hf_repo: str, *, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    parquet_path = _download_parquet(hf_repo, "metadata.parquet")
    if parquet_path is None:
        return _empty_metadata(lazy=lazy)
    lf = pl.scan_parquet(parquet_path, low_memory=True)
    if lazy:
        return lf
    return lf.collect(engine="streaming").match_to_schema(
        PAPER_METADATE_PL_SCHEMA,
        extra_columns="ignore",
    )


def load_embedding(
    hf_repo: str,
    dim: int,
    *,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    schema = paper_embedding_schema_pl(dim)
    parquet_path = _download_parquet(hf_repo, "embedding.parquet")
    if parquet_path is None:
        return _empty_embedding(dim, lazy=lazy)
    lf = pl.scan_parquet(parquet_path, low_memory=True)
    if lazy:
        return lf
    return lf.collect(engine="streaming").match_to_schema(
        schema,
        extra_columns="ignore",
    )


def upload_data(path, path_in_repo, hf_repo: str, squash_history: bool = True):
    """Upload a single file to HuggingFace Hub. 
    
    Note: For uploading multiple files, consider using upload_folder instead.
    """
    api = HfApi()
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=hf_repo,
        repo_type="dataset",
    )
    if squash_history:
        api.super_squash_history(
            repo_id=hf_repo,
            repo_type="dataset",
        )


def upload_folder(folder_path: Path, hf_repo: str, squash_history: bool = True, commit_message: str = None):
    """Upload all files in a folder to HuggingFace Hub in a single commit.
    
    Args:
        folder_path: Local directory containing files to upload
        hf_repo: HuggingFace repository ID
        squash_history: Whether to squash git history after upload
        commit_message: Optional commit message
    """
    from loguru import logger
    
    api = HfApi()
    
    # Get list of files to upload
    files = list(folder_path.glob("*.parquet"))
    if not files:
        logger.warning(f"No parquet files found in {folder_path}")
        return
    
    logger.info(f"Uploading {len(files)} parquet files to {hf_repo}")
    
    # Upload entire folder in one commit
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=hf_repo,
        repo_type="dataset",
        commit_message=commit_message or f"Update {len(files)} shard files",
        allow_patterns="*.parquet",  # Only upload parquet files
    )
    
    if squash_history:
        logger.info(f"Squashing history for {hf_repo}")
        api.super_squash_history(
            repo_id=hf_repo,
            repo_type="dataset",
        )
    
    logger.success(f"Successfully uploaded {len(files)} files to {hf_repo}")


def crawl_metadata(
    ds: pl.DataFrame | pl.LazyFrame,
    categories: Iterable[str] | None = None,
    start_date: date | str | None = None,
) -> pl.DataFrame:
    from loguru import logger
    
    ds_lazy = _to_lazy(ds)
    start_dt = _parse_date(start_date)
    
    if start_dt is not None:
        max_date = start_dt
        logger.info(f"Using provided start_date: {max_date}")
    else:
        # Check if UPDATE_DATE column exists and has data
        result = ds_lazy.select(pl.col(UPDATE_DATE).max().alias("max")).collect(
            engine="streaming"
        )
        logger.debug(f"Query result: {result}, height: {result.height}")
        
        max_value = result["max"][0] if result.height > 0 else None
        logger.info(f"Max UPDATE_DATE from existing data: {max_value} (type: {type(max_value)})")
        
        if max_value is None:
            # No existing data, start from a reasonable default (e.g., 30 days ago)
            from datetime import timedelta
            max_date = date.today() - timedelta(days=30)
            logger.warning(f"No existing UPDATE_DATE found, starting from 30 days ago: {max_date}")
        else:
            max_date = max_value
            logger.info(f"Starting crawl from last update date: {max_date}")
    
    today = date.today()
    logger.info(f"Crawl date range: {max_date} to {today}")
    
    if max_date >= today:
        logger.info("max_date >= today, no new data to crawl")
        return _empty_metadata()
    
    logger.info(f"Fetching new metadata from arXiv (categories={categories})")
    return fetch_arxiv_oai(
        categories=categories,
        start=max_date,
        end=today,
    )


def filter_new_metadata(
    ds_meta: pl.DataFrame | pl.LazyFrame,
    ds_embed: pl.DataFrame | pl.LazyFrame,
    categories: Iterable[str] | None = None,
    start_date: date | str | None = None,
) -> pl.LazyFrame:
    from loguru import logger
    
    meta_lazy = _to_lazy(ds_meta)
    embed_lazy = _to_lazy(ds_embed)

    # Count existing data
    meta_count = meta_lazy.select(pl.len().alias("count")).collect(engine="streaming").item()
    embed_count = embed_lazy.select(pl.len().alias("count")).collect(engine="streaming").item()
    logger.info(f"Filtering: metadata has {meta_count} rows, embeddings has {embed_count} rows")

    # Anti-join to find metadata without embeddings
    ds_new_meta = meta_lazy.join(
        embed_lazy.select(pl.col(ID)),
        on=ID,
        how="anti",
    )
    
    after_anti_join = ds_new_meta.select(pl.len().alias("count")).collect(engine="streaming").item()
    logger.info(f"After anti-join: {after_anti_join} rows without embeddings")

    # Apply start_date filter
    start_dt = _parse_date(start_date)
    if start_dt is not None:
        ds_new_meta = ds_new_meta.filter(pl.col(UPDATE_DATE) >= start_dt)
        after_date_filter = ds_new_meta.select(pl.len().alias("count")).collect(engine="streaming").item()
        logger.info(f"After start_date filter (>= {start_dt}): {after_date_filter} rows")

    # Apply category filter
    if categories is not None:
        prefix_checks = [pl.element().str.starts_with(cat) for cat in categories]
        ds_new_meta = ds_new_meta.filter(
            pl.col(CATEGORIES)
            .list.eval(pl.any_horizontal(prefix_checks))
            .list.any()
        )
        after_category_filter = ds_new_meta.select(pl.len().alias("count")).collect(engine="streaming").item()
        logger.info(f"After category filter ({categories}): {after_category_filter} rows")

    return ds_new_meta


def wrirte_embeddings(df, path, row_group=50_000):
    import pyarrow.parquet as pq

    pq.write_table(
        df.to_arrow(),
        path,
        row_group_size=row_group,
        compression="zstd",
        compression_level=5,
        data_page_version="2.0",
        use_byte_stream_split=["embedding.list.element"],  # 针对叶子 float32
        use_dictionary={"embedding.list.element": False},  # 关闭该叶子字典
    )
