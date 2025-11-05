"""Migrate existing monolithic parquet files to year-based shards.

This script:
1. Downloads existing metadata.parquet and embedding.parquet from HuggingFace
2. Splits them by publish_year
3. Uploads individual year shards back to HuggingFace
4. Deletes old monolithic files from HuggingFace
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import polars as pl
from huggingface_hub import HfApi

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.const import BASE_DIR
from src.io import load_metadata, load_embedding
from src.shard import get_year_from_date, save_year_shard, upload_year_shard
from src.config import AppConfig, load_config

DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")


def delete_old_file(hf_repo: str, filename: str):
    """Delete old monolithic file from HuggingFace."""
    api = HfApi()
    try:
        logger.info(f"Deleting old file {filename} from {hf_repo}")
        api.delete_file(
            path_in_repo=filename,
            repo_id=hf_repo,
            repo_type="dataset",
        )
        logger.success(f"Successfully deleted {filename}")
    except Exception as e:
        logger.warning(f"Failed to delete {filename}: {e}")


def migrate_metadata(hf_repo: str, row_group: int = 200000):
    """Migrate metadata.parquet to year-based shards."""
    logger.info(f"Loading metadata from {hf_repo}")
    metadata = load_metadata(hf_repo, lazy=False)
    
    if metadata.height == 0:
        logger.warning("No metadata found, skipping migration")
        return
    
    logger.info(f"Loaded {metadata.height} metadata rows")
    
    # Add year column
    metadata = metadata.pipe(get_year_from_date)
    
    # Get unique years
    years = sorted(metadata["publish_year"].unique().to_list())
    logger.info(f"Found {len(years)} years: {years}")
    
    # Split and save by year
    for year in years:
        year_data = metadata.filter(pl.col("publish_year") == year).drop("publish_year")
        logger.info(f"Saving metadata for year {year} ({year_data.height} rows)")
        
        save_year_shard(year_data, DATA_DIR, year, "metadata", row_group)
        
        # Upload to HuggingFace
        upload_year_shard(DATA_DIR, year, "metadata", hf_repo, squash_history=False)
    
    logger.success(f"Metadata migration completed! Created {len(years)} year shards")
    
    # Delete old monolithic file
    delete_old_file(hf_repo, "metadata.parquet")
    
    return years


def migrate_embeddings(hf_repo: str, dim: int, row_group: int = 200000, metadata_repo: str = None, local_embedding_path: str = None):
    """Migrate embedding.parquet to year-based shards.
    
    Requires metadata to get publish_year for each ID.
    """
    if local_embedding_path:
        logger.info(f"Loading embeddings from local file: {local_embedding_path}")
        import polars as pl
        from src.schema import paper_embedding_schema_pl
        embeddings = pl.scan_parquet(local_embedding_path).collect(streaming=True).match_to_schema(
            paper_embedding_schema_pl(dim),
            extra_columns="ignore",
        )
    else:
        logger.info(f"Loading embeddings from {hf_repo}")
        embeddings = load_embedding(hf_repo, dim, lazy=False)
    
    if embeddings.height == 0:
        logger.warning("No embeddings found, skipping migration")
        return
    
    logger.info(f"Loaded {embeddings.height} embedding rows")
    
    # Load metadata from year shards to get years
    if metadata_repo is None:
        metadata_repo = hf_repo
    
    logger.info(f"Loading metadata from local year shards to get years")
    # Load from local year shards (already migrated)
    from src.shard import load_all_year_shards
    metadata = load_all_year_shards(DATA_DIR, "metadata", lazy=False, hf_repo=None)
    metadata = metadata.pipe(get_year_from_date)
    
    # Join to get year for each embedding
    embeddings = embeddings.join(
        metadata.select(["id", "publish_year"]),
        on="id",
        how="left"
    )
    
    # Filter out rows with None year (shouldn't happen but just in case)
    embeddings = embeddings.filter(pl.col("publish_year").is_not_null())
    
    # Get unique years
    years = sorted(embeddings["publish_year"].unique().to_list())
    logger.info(f"Found {len(years)} years: {years}")
    
    # Split and save by year
    for year in years:
        year_data = embeddings.filter(pl.col("publish_year") == year).drop("publish_year")
        logger.info(f"Saving embeddings for year {year} ({year_data.height} rows)")
        
        save_year_shard(year_data, DATA_DIR, year, "embedding", row_group)
        
        # Upload to HuggingFace
        upload_year_shard(DATA_DIR, year, "embedding", hf_repo, squash_history=False)
    
    logger.success(f"Embedding migration completed! Created {len(years)} year shards")
    
    # Delete old monolithic file
    delete_old_file(hf_repo, "embedding.parquet")
    
    return years


def main():
    """Run migration for both metadata and embeddings."""
    config_path = BASE_DIR / "config.toml"
    config = load_config(AppConfig, config_path)
    
    logger.info("=" * 60)
    logger.info("Starting migration to year-based shards")
    logger.info("=" * 60)
    
    # Migrate metadata first
    logger.info("MIGRATING METADATA")
    logger.info("=" * 60)
    migrate_metadata(
        hf_repo=config.metadata.hf_repo,
        row_group=config.metadata.row_group,
    )
    
    # Then migrate embeddings
    logger.info("MIGRATING EMBEDDINGS")
    logger.info("=" * 60)
    migrate_embeddings(
        hf_repo=config.embedding.hf_repo,
        dim=config.embedding.dim,
        row_group=config.embedding.row_group,
        metadata_repo=config.metadata.hf_repo,
    )
    
    logger.success("All migrations completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
