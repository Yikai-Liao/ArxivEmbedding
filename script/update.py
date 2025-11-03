import polars as pl
import sys
import os
from dotenv import load_dotenv
from loguru import logger
import typer
import psutil
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.oai import fetch_arxiv_oai
from src.const import BASE_DIR
from src.order import order_by_category_and_date, dump_metadata, align_order
from src.schema import PAPER_METADATE_PL_SCHEMA, paper_embedding_schema_pl
from src.name import *
from src.embedding import *
from src.io import *
from src.config import AppConfig, EmbeddingConfig, MetaDataConfig, load_config


DATA_DIR = BASE_DIR / "data"


def _directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for file_name in files:
            file_path = Path(root) / file_name
            try:
                total += file_path.stat().st_size
            except FileNotFoundError:
                continue
    return total / 1024 / 1024


def log_memory_usage(stage: str):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    data_size = _directory_size_mb(DATA_DIR)
    logger.info(f"[MEMORY] {stage}: {mem:.2f} MB | data dir: {data_size:.2f} MB")


def embed(df, model, dim, batch_size=500):
    log_memory_usage("embed: start")
    contents = collect_content(df)
    ids = df.select(pl.col(ID)).to_series()
    
    all_embeddings = []
    total_batches = (len(contents) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(contents)} items in {total_batches} batches (batch_size={batch_size})")
    
    for i in range(0, len(contents), batch_size):
        batch_contents = contents[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_contents)} items)")
        log_memory_usage(f"embed: batch {batch_num}")
        
        batch_embeddings = google_batch_embedding(
            model=model,
            output_dimensionality=dim,
            inputs=batch_contents,
            dtype=np.float32,
        )
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.success(f"Generated {len(embeddings)} embeddings")
    log_memory_usage("embed: end")
    
    df_embeddings = pl.DataFrame({
        ID: ids,
        'embedding': embeddings.tolist(),
    }, schema=paper_embedding_schema_pl(dim))
    return df_embeddings

    
def update_metadata(config: MetaDataConfig, squash_history: bool = True):
    log_memory_usage("update_metadata: start")
    logger.info(f"Loading metadata from {config.hf_repo}")
    metadata_lazy = load_metadata(config.hf_repo, lazy=True)
    log_memory_usage("update_metadata: loaded metadata")
    metadata_count = (
        metadata_lazy.select(pl.len().alias("count")).collect(engine="streaming").item()
    )
    logger.info(f"Current metadata count: {metadata_count}")

    logger.info("Crawling new metadata...")
    new_metadata = crawl_metadata(metadata_lazy)
    log_memory_usage("update_metadata: crawled new metadata")
    if new_metadata.height == 0:
        logger.info("No new metadata to update.")
        return metadata_lazy

    logger.info(f"Found {new_metadata.height} new metadata entries")
    metadata_df = metadata_lazy.collect(engine="streaming")
    metadata_df = pl.concat([metadata_df, new_metadata]).unique(subset=ID, keep="last")
    metadata_df = order_by_category_and_date(metadata_df)
    log_memory_usage("update_metadata: merged and ordered")

    logger.info(f"Dumping metadata to {BASE_DIR / 'data' / 'metadata.parquet'}")
    dump_metadata(
        metadata_df,
        BASE_DIR / "data" / "metadata.parquet",
        row_group=config.row_group,
    )
    log_memory_usage("update_metadata: dumped metadata")
    logger.info(f"Uploading metadata to {config.hf_repo}")
    upload_data(
        path=BASE_DIR / "data" / "metadata.parquet",
        path_in_repo="metadata.parquet",
        hf_repo=config.hf_repo,
        squash_history=squash_history,
    )
    log_memory_usage("update_metadata: uploaded metadata")
    logger.success("Metadata update completed!")
    del metadata_df
    return load_metadata(config.hf_repo, lazy=True)
    
def update_embedding(
    config: EmbeddingConfig,
    metadata: pl.DataFrame | pl.LazyFrame,
    squash_history: bool = True,
):
    log_memory_usage("update_embedding: start")
    logger.info(f"Loading embeddings from {config.hf_repo} (dim={config.dim})")
    embeddings_lazy = load_embedding(config.hf_repo, config.dim, lazy=True)
    log_memory_usage("update_embedding: loaded embeddings")
    embedding_count = (
        embeddings_lazy.select(pl.len().alias("count"))
        .collect(engine="streaming")
        .item()
    )
    logger.info(f"Current embedding count: {embedding_count}")

    logger.info(
        f"Filtering new metadata (categories={config.categories}, start_date={config.start_date})"
    )
    metadata_lazy = metadata.lazy() if isinstance(metadata, pl.DataFrame) else metadata
    if metadata_lazy is None:
        raise ValueError("metadata is required to update embeddings")
    data_to_embed_lazy = filter_new_metadata(
        ds_meta=metadata_lazy,
        ds_embed=embeddings_lazy,
        categories=config.categories,
        start_date=config.start_date,
    )
    new_count = (
        data_to_embed_lazy.select(pl.len().alias("count"))
        .collect(engine="streaming")
        .item()
    )
    log_memory_usage("update_embedding: filtered new metadata")
    if new_count == 0:
        logger.info("No new metadata to embed.")
        return embeddings_lazy

    logger.info(f"Found {new_count} new metadata entries to embed.")
    data_to_embed = data_to_embed_lazy.collect(engine="streaming")

    metadata_df = metadata_lazy.collect(engine="streaming")
    embeddings_df = embeddings_lazy.collect(engine="streaming")

    logger.info(f"Generating embeddings for {data_to_embed.height} new papers (model={config.model})")
    new_embeddings = embed(
        data_to_embed,
        model=config.model,
        dim=config.dim,
    )
    log_memory_usage("update_embedding: generated new embeddings")
    logger.info("Merging and aligning embeddings")
    full_embeddings = pl.concat([embeddings_df, new_embeddings]).unique(
        subset=ID, keep="last"
    )
    full_embeddings = align_order(metadata_df, full_embeddings, on=ID)
    log_memory_usage("update_embedding: merged and aligned")

    logger.info(f"Writing embeddings to {BASE_DIR / 'data' / 'embedding.parquet'}")
    wrirte_embeddings(
        full_embeddings,
        BASE_DIR / "data" / "embedding.parquet",
        row_group=config.row_group,
    )
    log_memory_usage("update_embedding: wrote embeddings")
    logger.info(f"Uploading embeddings to {config.hf_repo}")
    upload_data(
        path=BASE_DIR / "data" / "embedding.parquet",
        path_in_repo="embedding.parquet",
        hf_repo=config.hf_repo,
        squash_history=squash_history,
    )
    log_memory_usage("update_embedding: uploaded embeddings")
    logger.success("Embedding update completed!")
    del embeddings_df, metadata_df, full_embeddings
    return load_embedding(config.hf_repo, config.dim, lazy=True)
    
    
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
    log_memory_usage("main: start")
    logger.info("=" * 60)
    logger.info("Starting Arxiv update process")
    logger.info("=" * 60)
    logger.info(f"Config path: {config_path}")
    logger.info(f"Update metadata: {update_metadata_flag}")
    logger.info(f"Update embedding: {update_embedding_flag}")
    logger.info(f"Squash history: {squash_history}")
    logger.info(f"Clean cache: {clean_cache}")
    config = load_config(AppConfig, Path(config_path))
    log_memory_usage("main: loaded config")
    if clean_cache:
        cache_dir = BASE_DIR / 'data' / 'hg'
        if cache_dir.exists():
            logger.warning(f"Cleaning cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            logger.success("Cache directory cleaned")
            log_memory_usage("main: cleaned cache")
    metadata_lazy: pl.LazyFrame | None = None
    if update_metadata_flag:
        logger.info("UPDATING METADATA")
        logger.info("=" * 60)
        metadata_lazy = update_metadata(config.metadata, squash_history=squash_history)
        log_memory_usage("main: updated metadata")
    if metadata_lazy is None:
        metadata_lazy = load_metadata(config.metadata.hf_repo, lazy=True)
        log_memory_usage("main: loaded metadata fallback")
    if update_embedding_flag:
        logger.info("UPDATING EMBEDDINGS")
        logger.info("=" * 60)
        update_embedding(
            config.embedding,
            metadata=metadata_lazy,
            squash_history=squash_history,
        )
        log_memory_usage("main: updated embeddings")
    logger.success("All updates completed successfully!")
    logger.info("=" * 60)
    log_memory_usage("main: end")


if __name__ == "__main__":
    app()
