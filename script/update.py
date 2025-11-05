"""Main update script for Arxiv metadata and embeddings.

This script orchestrates the update workflow using year-based sharding.
Core logic is implemented in src/update_ops.py.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import typer
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.const import BASE_DIR
from src.config import AppConfig, load_config
from src.update_ops import (
    update_metadata_shards,
    update_embedding_shards,
    log_memory_usage,
)
from src.shard import load_all_year_shards

DATA_DIR = BASE_DIR / "data"

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
    """Update Arxiv metadata and/or embeddings using year-based sharding."""
    log_memory_usage("main: start", DATA_DIR)
    
    logger.info("=" * 60)
    logger.info("Starting Arxiv update process")
    logger.info("=" * 60)
    logger.info(f"Config path: {config_path}")
    logger.info(f"Update metadata: {update_metadata_flag}")
    logger.info(f"Update embedding: {update_embedding_flag}")
    logger.info(f"Squash history: {squash_history}")
    logger.info(f"Clean cache: {clean_cache}")
    
    # Load configuration
    config = load_config(AppConfig, Path(config_path))
    log_memory_usage("main: loaded config", DATA_DIR)
    
    # Clean cache if requested
    if clean_cache:
        cache_dir = BASE_DIR / 'data' / 'hg'
        if cache_dir.exists():
            logger.warning(f"Cleaning cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            logger.success("Cache directory cleaned")
            log_memory_usage("main: cleaned cache", DATA_DIR)
    
    # Update metadata
    metadata_lazy = None
    if update_metadata_flag:
        logger.info("UPDATING METADATA")
        logger.info("=" * 60)
        metadata_lazy = update_metadata_shards(
            config=config.metadata,
            data_dir=DATA_DIR,
            squash_history=squash_history,
        )
        log_memory_usage("main: updated metadata", DATA_DIR)
    
    # Load metadata if not updated
    if metadata_lazy is None:
        metadata_lazy = load_all_year_shards(
            DATA_DIR,
            "metadata",
            lazy=True,
            hf_repo=config.metadata.hf_repo
        )
        log_memory_usage("main: loaded metadata fallback", DATA_DIR)
    
    # Update embeddings
    if update_embedding_flag:
        logger.info("UPDATING EMBEDDINGS")
        logger.info("=" * 60)
        update_embedding_shards(
            config=config.embedding,
            metadata=metadata_lazy,
            data_dir=DATA_DIR,
            squash_history=squash_history,
        )
        log_memory_usage("main: updated embeddings", DATA_DIR)
    
    logger.success("All updates completed successfully!")
    logger.info("=" * 60)
    log_memory_usage("main: end", DATA_DIR)


if __name__ == "__main__":
    app()
