import polars as pl
import sys
import os
from dotenv import load_dotenv
from loguru import logger
import typer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.oai import fetch_arxiv_oai
from src.const import BASE_DIR
from src.order import order_by_category_and_date, dump_metadata, align_order
from src.schema import PAPER_METADATE_PL_SCHEMA, paper_embedding_schema_pl
from src.name import *
from src.embedding import *
from src.io import *
from src.config import AppConfig, EmbeddingConfig, MetaDataConfig, load_config


def embed(df, model, dim, batch_size=500):
    contents = collect_content(df)
    ids = df.select(pl.col(ID)).to_series()
    
    all_embeddings = []
    total_batches = (len(contents) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(contents)} items in {total_batches} batches (batch_size={batch_size})")
    
    for i in range(0, len(contents), batch_size):
        batch_contents = contents[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_contents)} items)")
        
        batch_embeddings = google_batch_embedding(
            model=model,
            output_dimensionality=dim,
            inputs=batch_contents,
            dtype=np.float32,
        )
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.success(f"Generated {len(embeddings)} embeddings")
    
    df_embeddings = pl.DataFrame({
        ID: ids,
        'embedding': embeddings.tolist(),
    }, schema=paper_embedding_schema_pl(dim))
    return df_embeddings

    
def update_metadata(config: MetaDataConfig, squash_history: bool = True):
    logger.info(f"Loading metadata from {config.hf_repo}")
    df = load_metadata(config.hf_repo)
    logger.info(f"Current metadata count: {df.height}")
    
    logger.info("Crawling new metadata...")
    new_metadata = crawl_metadata(df)
    if new_metadata.height == 0:
        logger.info("No new metadata to update.")
        return

    logger.info(f"Found {new_metadata.height} new metadata entries")
    df = pl.concat([df, new_metadata]).unique(subset=ID, keep='last')
    df = order_by_category_and_date(df)
    
    logger.info(f"Dumping metadata to {BASE_DIR / 'data' / 'metadata.parquet'}")
    dump_metadata(
        df,
        BASE_DIR / 'data' / 'metadata.parquet',
        row_group=config.row_group,
    )
    
    logger.info(f"Uploading metadata to {config.hf_repo}")
    upload_data(
        path = BASE_DIR / 'data' / 'metadata.parquet',
        path_in_repo = 'metadata.parquet',
        hf_repo = config.hf_repo,
        squash_history=squash_history,
    )
    logger.success("Metadata update completed!")
    return df
    
def update_embedding(config: EmbeddingConfig, metadata: pl.DataFrame = None, squash_history: bool = True):
    logger.info(f"Loading embeddings from {config.hf_repo} (dim={config.dim})")
    df = load_embedding(config.hf_repo, config.dim)
    logger.info(f"Current embedding count: {df.height}")

    logger.info(f"Filtering new metadata (categories={config.categories}, start_date={config.start_date})")
    data_to_embed = filter_new_metadata(
        ds_meta=metadata,
        ds_embed=df,
        categories=config.categories,
        start_date=config.start_date,
    )
    if data_to_embed.height == 0:
        logger.info("No new metadata to embed.")
        return
    
    logger.info(f"Generating embeddings for {data_to_embed.height} new papers (model={config.model})")
    new_embeddings = embed(
        data_to_embed,
        model=config.model,
        dim=config.dim,
    )
    
    logger.info("Merging and aligning embeddings")
    full_embeddings = pl.concat([df, new_embeddings]).unique(subset=ID, keep='last')
    full_embeddings = align_order(metadata, full_embeddings, on=ID)
    
    logger.info(f"Writing embeddings to {BASE_DIR / 'data' / 'embedding.parquet'}")
    wrirte_embeddings(
        full_embeddings,
        BASE_DIR / 'data' / 'embedding.parquet',
        row_group=config.row_group,
    )
    
    logger.info(f"Uploading embeddings to {config.hf_repo}")
    upload_data(
        path = BASE_DIR / 'data' / 'embedding.parquet',
        path_in_repo = 'embedding.parquet',
        hf_repo = config.hf_repo,
        squash_history=squash_history,
    )
    logger.success("Embedding update completed!")
    return full_embeddings
    
    
load_dotenv(BASE_DIR / ".env")

app = typer.Typer(help="CLI update tool for Arxiv metadata and embeddings.")


@app.command()
def main(
    config_path: str = typer.Option(
        str(BASE_DIR / 'config.toml'),
        "--config", "-c",
        help="Path to the configuration file"
    ),
    update_metadata_flag: bool = typer.Option(
        True,
        "--metadata/--no-metadata",
        help="Whether to update metadata"
    ),
    update_embedding_flag: bool = typer.Option(
        True,
        "--embedding/--no-embedding",
        help="Whether to update embeddings"
    ),
    squash_history: bool = typer.Option(
        True,
        "--squash/--no-squash",
        help="Whether to squash git history when uploading"
    ),
    clean_cache: bool = typer.Option(
        False,
        "--clean/--no-clean",
        help="Whether to clean cache directory"
    ),
):
    """Update Arxiv metadata and/or embeddings."""
    import shutil
    from pathlib import Path
    
    logger.info("=" * 60)
    logger.info("Starting Arxiv update process")
    logger.info("=" * 60)
    logger.info(f"Config path: {config_path}")
    logger.info(f"Update metadata: {update_metadata_flag}")
    logger.info(f"Update embedding: {update_embedding_flag}")
    logger.info(f"Squash history: {squash_history}")
    logger.info(f"Clean cache: {clean_cache}")
    
    config = load_config(AppConfig, Path(config_path))
    
    if clean_cache:
        cache_dir = BASE_DIR / 'data' / 'hg'
        if cache_dir.exists():
            logger.warning(f"Cleaning cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            logger.success("Cache directory cleaned")
    
    metadata = None
    if update_metadata_flag:
        logger.info("UPDATING METADATA")
        logger.info("=" * 60)
        metadata = update_metadata(config.metadata, squash_history=squash_history)
        
    if metadata is None:
        metadata = load_metadata(config.metadata.hf_repo)
        
    if update_embedding_flag:
        logger.info("UPDATING EMBEDDINGS")
        logger.info("=" * 60)
        update_embedding(config.embedding, metadata=metadata, squash_history=squash_history)
    
    logger.success("All updates completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
