from datasets import load_dataset
import toml
import polars as pl
import sys
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
from datetime import date

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.oai import fetch_arxiv_oai
from src.const import BASE_DIR
from src.order import order_by_category_and_date, dump_metadata, align_order
from src.schema import PAPER_METADATE_PL_SCHEMA, paper_embedding_schema_pl
from src.name import *
from src.embedding import *

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = toml.load(file)
    return config


def load_metadata(hf_repo:str) -> pl.LazyFrame:
    dataset = load_dataset(
        path=hf_repo,
        cache_dir=BASE_DIR / 'data' / 'hg',
        # keep_in_memory=,
        split='train',
    )
    return dataset.to_polars().lazy().match_to_schema(PAPER_METADATE_PL_SCHEMA, extra_columns='ignore')

def load_embedding(hf_repo:str, dim) -> pl.LazyFrame:
    schema = paper_embedding_schema_pl(dim)
    
    try:
        dataset = load_dataset(
            path=hf_repo,
            cache_dir=BASE_DIR / 'data' / 'hg',
            # keep_in_memory=,
            split='train',
        )
    except Exception as e:
        print(f"Failed to load dataset from {hf_repo}: {e}")
        dataset = None
    if dataset is None or dataset.num_rows == 0:
        return pl.LazyFrame(schema=schema)
    return dataset.to_polars().lazy().match_to_schema(schema, extra_columns='ignore')


def upload_metadata(path, path_in_repo, hf_repo:str):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=hf_repo,
        repo_type='dataset',
    )
    api.super_squash_history(
        repo_id=hf_repo,
        repo_type='dataset',
    )
    
    
def crawl_metadata(ds):
    # find the last updated date
    max_date = ds.select(pl.col(UPDATE_DATE).max()).collect().item()
    today = date.today()
    return fetch_arxiv_oai(
        categories=None,
        start=max_date,
        end=today,
    )
    
def filter_new_metadata(ds_meta, ds_embed, categories=None, start_date=None):
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

def embed(df, model, dim):
    contents = collect_content(df)
    embeddings = google_batch_embedding(
        model=model,
        output_dimensionality=dim,
        inputs=contents,
        dtype=np.float32,
    )
    df_embeddings = pl.DataFrame({
        ID: df.select(pl.col(ID)).to_series(),
        'embedding': embeddings.tolist(),
    }, schema=paper_embedding_schema_pl(dim))
    return df_embeddings
    
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

def main():
    load_dotenv(os.path.join(BASE_DIR, '.env'))
    
    config = load_config(os.path.join(BASE_DIR, 'config.toml'))
    ds = load_metadata(config['metadata']['hf_repo'])
    new_metadata = crawl_metadata(ds)
    if new_metadata.height == 0:
        print("No new metadata to update.")
        return
    

    df = pl.concat([df, new_metadata.lazy()]).unique(subset=ID, keep='last')
    df = order_by_category_and_date(ds).collect()
    
    dump_metadata(
        df,
        BASE_DIR / 'data' / 'metadata.parquet',
        row_group=config['metadata'].get('row_group', 50_000)
    )
    upload_metadata(
        BASE_DIR / 'data' / 'metadata.parquet',
        'metadata.parquet',
        config['metadata']['hf_repo'],
    )
    
    df_embed = load_embedding(config['embedding']['hf_repo'], config['embedding']['dim']).collect()
    print(df_embed)
    ds_new_meta = filter_new_metadata(
        ds.collect(), df_embed,
        categories=config['embedding'].get('categories', None),
        start_date=config['embedding'].get('start_date', None),
    )
    print(ds_new_meta)
    if ds_new_meta.height == 0:
        print("No new metadata to embed.")
        return
    
    new_embeddings = embed(
        ds_new_meta,
        model=config['embedding']['model'],
        dim=config['embedding']['dim'],
    )
    
    full_embeddings = pl.concat([df_embed, new_embeddings]).unique(subset=ID, keep='last')
    full_embeddings = align_order(df, full_embeddings, on=ID)
    print(full_embeddings)
    wrirte_embeddings(
        full_embeddings,
        BASE_DIR / 'data' / 'embedding.parquet',
        row_group=config['embedding'].get('row_group', 50_000)
    )
    
    upload_metadata(
        BASE_DIR / 'data' / 'embedding.parquet',
        'embedding.parquet',
        config['embedding']['hf_repo'],
    )
    
    
    
    # print(df.head())
    # output_dir = BASE_DIR / 'data' / 'metadata.parquet'
    # dump_metadata(
    #     df,
    #     output_dir,
    #     row_group=config['metadata'].get('row_group', 50_000)
    # )
    # upload_metadata(
    #     output_dir,
    #     'metadata.parquet',
    #     config['metadata']['hf_repo'],
    # )
    
    
    
if __name__ == "__main__":
    main()