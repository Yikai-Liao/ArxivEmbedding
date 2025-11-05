"""Compare memory usage: full merge vs single year processing.

This script investigates why single year processing uses similar memory to full merge.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import polars as pl
import psutil
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.const import BASE_DIR
from src.io import load_metadata, load_embedding
from src.shard import get_year_from_date
from src.name import ID
from src.config import AppConfig, load_config

load_dotenv(BASE_DIR / ".env")


def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[MEMORY] {stage}: {mem_mb:.2f} MB")
    return mem_mb


def test_full_embedding_load(hf_repo: str, dim: int):
    """Test loading full embedding dataset."""
    logger.info("=" * 60)
    logger.info("Test 1: Load FULL embedding dataset")
    logger.info("=" * 60)
    
    start_mem = log_memory_usage("start")
    
    # Load full embeddings
    logger.info("Loading full embeddings (lazy)")
    embeddings_lazy = load_embedding(hf_repo, dim, lazy=True)
    log_memory_usage("after lazy load")
    
    # Count rows
    count = embeddings_lazy.select(pl.len()).collect(engine="streaming").item()
    logger.info(f"Total embeddings: {count}")
    log_memory_usage("after count")
    
    # Collect all
    logger.info("Collecting full dataset")
    embeddings_df = embeddings_lazy.collect(engine="streaming")
    peak_mem = log_memory_usage("after full collect")
    
    logger.info(f"Peak memory delta: {peak_mem - start_mem:.2f} MB")
    
    del embeddings_lazy, embeddings_df
    gc.collect()
    log_memory_usage("after cleanup")
    
    return peak_mem


def test_year_filter_collect(hf_repo: str, metadata_repo: str, dim: int, year: int):
    """Test filtering to a year BEFORE collecting."""
    logger.info("=" * 60)
    logger.info(f"Test 2: Filter to year {year} BEFORE collecting")
    logger.info("=" * 60)
    
    start_mem = log_memory_usage("start")
    
    # Load metadata for year IDs
    logger.info("Loading metadata for year")
    metadata_lazy = load_metadata(metadata_repo, lazy=True)
    metadata_lazy = metadata_lazy.pipe(get_year_from_date)
    metadata_lazy = metadata_lazy.filter(pl.col("publish_year") == year)
    year_metadata = metadata_lazy.collect(engine="streaming")
    mem_after_metadata = log_memory_usage("after metadata")
    
    year_ids = set(year_metadata[ID].to_list())
    logger.info(f"Year {year} has {len(year_ids)} IDs")
    
    # Filter embeddings in lazy mode BEFORE collecting
    logger.info("Filtering embeddings (lazy)")
    embeddings_lazy = load_embedding(hf_repo, dim, lazy=True)
    embeddings_lazy = embeddings_lazy.filter(pl.col(ID).is_in(year_ids))
    mem_after_filter = log_memory_usage("after lazy filter")
    
    # Count filtered rows
    count = embeddings_lazy.select(pl.len()).collect(engine="streaming").item()
    logger.info(f"Filtered to {count} embeddings")
    
    # Now collect
    logger.info("Collecting filtered dataset")
    year_embeddings = embeddings_lazy.collect(engine="streaming")
    peak_mem = log_memory_usage("after filtered collect")
    
    logger.info(f"Peak memory delta: {peak_mem - start_mem:.2f} MB")
    logger.info(f"Rows collected: {year_embeddings.height}")
    
    del metadata_lazy, year_metadata, year_ids, embeddings_lazy, year_embeddings
    gc.collect()
    log_memory_usage("after cleanup")
    
    return peak_mem


def test_year_collect_then_filter(hf_repo: str, metadata_repo: str, dim: int, year: int):
    """Test collecting full dataset THEN filtering (inefficient)."""
    logger.info("=" * 60)
    logger.info(f"Test 3: Collect FULL dataset then filter to year {year}")
    logger.info("=" * 60)
    
    start_mem = log_memory_usage("start")
    
    # Load metadata
    logger.info("Loading metadata")
    metadata = load_metadata(metadata_repo, lazy=False)
    metadata = metadata.pipe(get_year_from_date)
    year_ids = set(metadata.filter(pl.col("publish_year") == year)[ID].to_list())
    logger.info(f"Year {year} has {len(year_ids)} IDs")
    mem_after_metadata = log_memory_usage("after metadata")
    
    # Collect FULL embeddings first (this is the problem!)
    logger.info("Collecting FULL embeddings")
    embeddings = load_embedding(hf_repo, dim, lazy=False)
    mem_after_full_collect = log_memory_usage("after FULL collect")
    
    # Then filter
    logger.info("Filtering to year")
    year_embeddings = embeddings.filter(pl.col(ID).is_in(year_ids))
    peak_mem = log_memory_usage("after filter")
    
    logger.info(f"Peak memory delta: {peak_mem - start_mem:.2f} MB")
    logger.info(f"Rows after filter: {year_embeddings.height}")
    
    del metadata, year_ids, embeddings, year_embeddings
    gc.collect()
    log_memory_usage("after cleanup")
    
    return peak_mem


def main():
    """Run comparison tests."""
    config_path = BASE_DIR / "config.toml"
    config = load_config(AppConfig, config_path)
    
    test_year = 2025
    
    results = {}
    
    logger.info("\n" + "=" * 60)
    logger.info("MEMORY USAGE INVESTIGATION")
    logger.info("=" * 60)
    
    # Test 1: Full load
    logger.info("\n")
    results['full_load'] = test_full_embedding_load(
        config.embedding.hf_repo,
        config.embedding.dim,
    )
    
    logger.info("\n" + "-" * 60 + "\n")
    gc.collect()
    
    # Test 2: Filter before collect (efficient)
    results['filter_before'] = test_year_filter_collect(
        config.embedding.hf_repo,
        config.metadata.hf_repo,
        config.embedding.dim,
        test_year,
    )
    
    logger.info("\n" + "-" * 60 + "\n")
    gc.collect()
    
    # Test 3: Collect then filter (inefficient)
    results['collect_then_filter'] = test_year_collect_then_filter(
        config.embedding.hf_repo,
        config.metadata.hf_repo,
        config.embedding.dim,
        test_year,
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Full load:              {results['full_load']:.2f} MB")
    logger.info(f"Filter before collect:  {results['filter_before']:.2f} MB")
    logger.info(f"Collect then filter:    {results['collect_then_filter']:.2f} MB")
    
    logger.info("\nConclusion:")
    if results['collect_then_filter'] > results['filter_before'] * 1.5:
        logger.error("❌ Previous test was collecting FULL dataset before filtering!")
        logger.info("   This explains the high memory usage for single year")
    else:
        logger.success("✓ Memory usage is as expected")


if __name__ == "__main__":
    main()
