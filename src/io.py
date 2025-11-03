from .const import BASE_DIR
from .schema import PAPER_METADATE_PL_SCHEMA, paper_embedding_schema_pl
from .oai import fetch_arxiv_oai
from .name import *

import polars as pl
from huggingface_hub import HfApi
from datasets import load_dataset
from datetime import date

def load_metadata(hf_repo:str) -> pl.DataFrame:
    dataset = load_dataset(
        path=hf_repo,
        cache_dir=BASE_DIR / 'data' / 'hg',
        split='train',
    )
    return dataset.to_polars().match_to_schema(PAPER_METADATE_PL_SCHEMA, extra_columns='ignore')

def load_embedding(hf_repo:str, dim) -> pl.DataFrame:
    schema = paper_embedding_schema_pl(dim)
    
    try:
        dataset = load_dataset(
            path=hf_repo,
            cache_dir=BASE_DIR / 'data' / 'hg',
            split='train',
        )
    except Exception as e:
        print(f"Failed to load dataset from {hf_repo}: {e}")
        dataset = None
    if dataset is None or dataset.num_rows == 0:
        return pl.LazyFrame(schema=schema)
    return dataset.to_polars().match_to_schema(schema, extra_columns='ignore')

def upload_data(path, path_in_repo, hf_repo:str, squash_history:bool=True):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=hf_repo,
        repo_type='dataset',
    )
    if squash_history:
        api.super_squash_history(
            repo_id=hf_repo,
            repo_type='dataset',
        )
    
def crawl_metadata(ds:pl.DataFrame, categories: list[str]|None=None, start_date: date|None = None) -> pl.DataFrame:
    # find the last updated date
    max_date = date(*map(int, start_date.split('-'))) if start_date is not None else ds.select(pl.col(UPDATE_DATE).max()).item()
    today = date.today()
    
    if max_date >= today:
        return pl.DataFrame(schema=PAPER_METADATE_PL_SCHEMA)
    
    return fetch_arxiv_oai(
        categories=categories,
        start=max_date,
        end=today,
    )
    
    
def filter_new_metadata(ds_meta: pl.DataFrame, ds_embed: pl.DataFrame, categories=None, start_date=None):
    ds_embed_ids = ds_embed.select(pl.col(ID)).to_series().to_list()
    ds_new_meta = ds_meta.lazy().filter(
        ~pl.col(ID).is_in(ds_embed_ids)
    )
    if start_date is not None:
        ds_new_meta = ds_new_meta.filter(
            pl.col(UPDATE_DATE) >= date(*map(int, start_date.split('-')))
        )
    
    if categories is not None:
        ds_new_meta = ds_new_meta.filter(
            pl.col(CATEGORIES).list.eval(
                pl.any_horizontal([pl.element().str.starts_with(cat) for cat in categories])
            ).list.any()
        )

    return ds_new_meta.collect()

def wrirte_embeddings(df, path, row_group=50_000):
    import pyarrow.parquet as pq

    pq.write_table(
        df.to_arrow(),
        path,
        row_group_size=row_group,
        compression="zstd",
        compression_level=5,
        data_page_version="2.0",
        use_byte_stream_split=["embedding.list.element"],   # 针对叶子 float32
        use_dictionary={"embedding.list.element": False},   # 关闭该叶子字典
    )