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


def crawl_metadata(
    ds: pl.DataFrame | pl.LazyFrame,
    categories: Iterable[str] | None = None,
    start_date: date | str | None = None,
) -> pl.DataFrame:
    ds_lazy = _to_lazy(ds)
    start_dt = _parse_date(start_date)
    if start_dt is not None:
        max_date = start_dt
    else:
        result = ds_lazy.select(pl.col(UPDATE_DATE).max().alias("max")).collect(
            engine="streaming"
        )
        max_value = result["max"][0] if result.height else None
        if max_value is None:
            max_date = date.today()
        else:
            max_date = max_value
    today = date.today()
    if max_date >= today:
        return _empty_metadata()
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
    meta_lazy = _to_lazy(ds_meta)
    embed_lazy = _to_lazy(ds_embed)

    ds_new_meta = meta_lazy.join(
        embed_lazy.select(pl.col(ID)),
        on=ID,
        how="anti",
    )

    start_dt = _parse_date(start_date)
    if start_dt is not None:
        ds_new_meta = ds_new_meta.filter(pl.col(UPDATE_DATE) >= start_dt)

    if categories is not None:
        prefix_checks = [pl.element().str.starts_with(cat) for cat in categories]
        ds_new_meta = ds_new_meta.filter(
            pl.col(CATEGORIES)
            .list.eval(pl.any_horizontal(prefix_checks))
            .list.any()
        )

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
