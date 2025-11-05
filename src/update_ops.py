"""Core operations for metadata and embedding updates.

This module contains the main update logic for year-based sharding.
"""
import polars as pl
import numpy as np
import os
import psutil
from pathlib import Path
from loguru import logger
from typing import Iterable
from datetime import date

from src.name import ID, PUBLISH_DATE, UPDATE_DATE, CATEGORIES
from src.schema import paper_embedding_schema_pl
from src.embedding import collect_content, google_batch_embedding
from src.shard import (
    get_year_from_date,
    update_year_shard,
    upload_year_shard,
    load_all_year_shards,
)
from src.io import crawl_metadata, filter_new_metadata
from src.config import MetaDataConfig, EmbeddingConfig


def _directory_size_mb(path: Path) -> float:
    """Calculate directory size in MB."""
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


def log_memory_usage(stage: str, data_dir: Path = None):
    """Log current memory usage and optional data directory size."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    
    msg = f"[MEMORY] {stage}: {mem:.2f} MB"
    if data_dir:
        data_size = _directory_size_mb(data_dir)
        msg += f" | data dir: {data_size:.2f} MB"
    
    logger.info(msg)


def generate_embeddings(
    df: pl.DataFrame,
    model: str,
    dim: int,
    batch_size: int = 500,
) -> pl.DataFrame:
    """Generate embeddings for papers.
    
    Args:
        df: DataFrame containing papers to embed
        model: Embedding model name
        dim: Embedding dimension
        batch_size: Batch size for API calls
    
    Returns:
        DataFrame with id and embedding columns
    """
    log_memory_usage("generate_embeddings: start")
    
    contents = collect_content(df)
    ids = df.select(pl.col(ID)).to_series()
    
    all_embeddings = []
    total_batches = (len(contents) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(contents)} items in {total_batches} batches (batch_size={batch_size})")
    
    for i in range(0, len(contents), batch_size):
        batch_contents = contents[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_contents)} items)")
        log_memory_usage(f"generate_embeddings: batch {batch_num}")
        
        batch_embeddings = google_batch_embedding(
            model=model,
            output_dimensionality=dim,
            inputs=batch_contents,
            dtype=np.float32,
        )
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.success(f"Generated {len(embeddings)} embeddings")
    log_memory_usage("generate_embeddings: end")
    
    df_embeddings = pl.DataFrame({
        ID: ids,
        'embedding': embeddings.tolist(),
    }, schema=paper_embedding_schema_pl(dim))
    
    return df_embeddings


def update_metadata_shards(
    config: MetaDataConfig,
    data_dir: Path,
    squash_history: bool = True,
) -> pl.LazyFrame:
    """Update metadata using year-based sharding.
    
    Each year's data is stored in a separate file: metadata_2024.parquet, metadata_2025.parquet, etc.
    Only affected year files are updated and uploaded.
    
    Args:
        config: Metadata configuration
        data_dir: Directory for local data storage
        squash_history: Whether to squash git history on upload
    
    Returns:
        LazyFrame of all metadata after update
    """
    log_memory_usage("update_metadata: start", data_dir)
    logger.info(f"Loading metadata from {config.hf_repo}")
    
    # Load all existing year shards lazily (download from HF if needed)
    metadata_lazy = load_all_year_shards(data_dir, "metadata", lazy=True, hf_repo=config.hf_repo)
    log_memory_usage("update_metadata: loaded metadata shards", data_dir)
    
    metadata_count = (
        metadata_lazy.select(pl.len().alias("count")).collect(engine="streaming").item()
    )
    logger.info(f"Current metadata count: {metadata_count}")

    logger.info("Crawling new metadata...")
    new_metadata = crawl_metadata(metadata_lazy)
    log_memory_usage("update_metadata: crawled new metadata", data_dir)
    
    if new_metadata.height == 0:
        logger.info("No new metadata to update.")
        return metadata_lazy

    logger.info(f"Found {new_metadata.height} new metadata entries")
    
    # Add year column to new metadata
    new_metadata = new_metadata.pipe(get_year_from_date)
    
    # Get unique years in new data
    new_years = sorted(new_metadata["publish_year"].unique().to_list())
    logger.info(f"New metadata spans years: {new_years}")
    
    # Update each affected year shard
    for year in new_years:
        year_data = new_metadata.filter(pl.col("publish_year") == year).drop("publish_year")
        logger.info(f"Updating metadata for year {year} ({year_data.height} new rows)")
        log_memory_usage(f"update_metadata: processing year {year}", data_dir)
        
        # Update year shard (merge, deduplicate, sort)
        updated_shard = update_year_shard(
            base_dir=data_dir,
            year=year,
            new_data=year_data,
            file_prefix="metadata",
            row_group=config.row_group,
        )
        log_memory_usage(f"update_metadata: updated year {year}", data_dir)
        
        # Upload updated shard
        upload_year_shard(
            base_dir=data_dir,
            year=year,
            file_prefix="metadata",
            hf_repo=config.hf_repo,
            squash_history=squash_history,
        )
        log_memory_usage(f"update_metadata: uploaded year {year}", data_dir)
        del updated_shard
    
    logger.success("Metadata update completed!")
    return load_all_year_shards(data_dir, "metadata", lazy=True, hf_repo=config.hf_repo)


def update_embedding_shards(
    config: EmbeddingConfig,
    metadata: pl.DataFrame | pl.LazyFrame,
    data_dir: Path,
    squash_history: bool = True,
) -> pl.LazyFrame:
    """Update embeddings using year-based sharding.
    
    Each year's embeddings are stored in: embedding_2024.parquet, embedding_2025.parquet, etc.
    Only affected year files are updated and uploaded.
    
    Args:
        config: Embedding configuration
        metadata: Metadata (all years) for filtering
        data_dir: Directory for local data storage
        squash_history: Whether to squash git history on upload
    
    Returns:
        LazyFrame of all embeddings after update
    """
    log_memory_usage("update_embedding: start", data_dir)
    logger.info(f"Loading embeddings from {config.hf_repo} (dim={config.dim})")
    
    # Load all existing embedding shards lazily (download from HF if needed)
    embeddings_lazy = load_all_year_shards(data_dir, "embedding", lazy=True, hf_repo=config.hf_repo)
    log_memory_usage("update_embedding: loaded embeddings", data_dir)
    
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
    
    # IMPORTANT: filter_new_metadata only needs to check IDs, but will return all columns.
    # The anti-join inside filter_new_metadata already uses .select(ID) for embeddings,
    # but metadata_lazy may still load all columns. This is acceptable because:
    # 1. It's lazy - columns are only loaded when needed
    # 2. We need all metadata columns for the returned new_metadata anyway
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
    log_memory_usage("update_embedding: filtered new metadata", data_dir)
    
    if new_count == 0:
        logger.info("No new metadata to embed.")
        return embeddings_lazy

    logger.info(f"Found {new_count} new metadata entries to embed.")
    data_to_embed = data_to_embed_lazy.collect(engine="streaming")
    
    # Add year column to new data
    data_to_embed = data_to_embed.pipe(get_year_from_date)

    logger.info(f"Generating embeddings for {data_to_embed.height} new papers (model={config.model})")
    new_embeddings = generate_embeddings(
        data_to_embed,
        model=config.model,
        dim=config.dim,
    )
    log_memory_usage("update_embedding: generated new embeddings", data_dir)
    
    # Add year column to new embeddings (join from data_to_embed)
    new_embeddings = new_embeddings.join(
        data_to_embed.select([ID, "publish_year"]),
        on=ID,
        how="left"
    )
    
    # Get unique years in new embeddings
    new_years = sorted(new_embeddings["publish_year"].unique().to_list())
    logger.info(f"New embeddings span years: {new_years}")
    
    # Update each affected year shard
    from src.order import align_order
    
    for year in new_years:
        year_embeddings = new_embeddings.filter(pl.col("publish_year") == year).drop("publish_year")
        logger.info(f"Updating embeddings for year {year} ({year_embeddings.height} new rows)")
        log_memory_usage(f"update_embedding: processing year {year}", data_dir)
        
        # Load corresponding year metadata for alignment
        from src.shard import load_year_shard
        year_metadata = load_year_shard(data_dir, year, "metadata", lazy=False)
        if year_metadata is None:
            logger.warning(f"No metadata shard for year {year}, skipping alignment")
            continue
        
        # Update year shard (merge, deduplicate)
        updated_shard = update_year_shard(
            base_dir=data_dir,
            year=year,
            new_data=year_embeddings,
            file_prefix="embedding",
            row_group=config.row_group,
        )
        log_memory_usage(f"update_embedding: updated year {year}", data_dir)
        
        # Align with metadata within this year
        updated_shard = align_order(year_metadata, updated_shard, on=ID)
        log_memory_usage(f"update_embedding: aligned year {year}", data_dir)
        
        # Save aligned shard
        from src.shard import save_year_shard
        save_year_shard(updated_shard, data_dir, year, "embedding", config.row_group)
        
        # Upload updated shard
        upload_year_shard(
            base_dir=data_dir,
            year=year,
            file_prefix="embedding",
            hf_repo=config.hf_repo,
            squash_history=squash_history,
        )
        log_memory_usage(f"update_embedding: uploaded year {year}", data_dir)
        del updated_shard, year_metadata
    
    logger.success("Embedding update completed!")
    del data_to_embed, new_embeddings
    return load_all_year_shards(data_dir, "embedding", lazy=True, hf_repo=config.hf_repo)
