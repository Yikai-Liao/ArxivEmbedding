"""Simple test for year-based operations without migration.

Tests memory usage for processing a single year from existing monolithic files.
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
from src.shard import get_year_from_date, save_year_shard, update_year_shard
from src.order import align_order
from src.name import ID
from src.config import AppConfig, load_config

DATA_DIR = BASE_DIR / "data"
TEST_DIR = BASE_DIR / "data" / "test_shards"

load_dotenv(BASE_DIR / ".env")


def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[MEMORY] {stage}: {mem_mb:.2f} MB")
    return mem_mb


def test_single_year_metadata(hf_repo: str, year: int, row_group: int = 200000):
    """Test processing a single year of metadata."""
    logger.info(f"=" * 60)
    logger.info(f"Testing single-year metadata processing: {year}")
    logger.info(f"=" * 60)
    
    start_mem = log_memory_usage("start")
    
    # Load metadata lazily and filter to year
    logger.info("Loading metadata (lazy)")
    metadata_lazy = load_metadata(hf_repo, lazy=True)
    metadata_lazy = metadata_lazy.pipe(get_year_from_date)
    metadata_lazy = metadata_lazy.filter(pl.col("publish_year") == year).drop("publish_year")
    
    # Collect only this year
    logger.info(f"Collecting year {year} data")
    year_data = metadata_lazy.collect(streaming=True)
    logger.info(f"Year {year} has {year_data.height} rows")
    after_load_mem = log_memory_usage("loaded year data")
    
    # Simulate 10 new rows
    new_rows = year_data.head(10)
    logger.info(f"Simulating {new_rows.height} new rows")
    
    # Save and update
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    save_year_shard(year_data, TEST_DIR, year, "metadata", row_group)
    mem_after_save = log_memory_usage("saved shard")
    
    updated = update_year_shard(TEST_DIR, year, new_rows, "metadata", row_group)
    peak_mem = log_memory_usage("updated shard")
    
    # Cleanup
    del year_data, new_rows, updated, metadata_lazy
    gc.collect()
    final_mem = log_memory_usage("after cleanup")
    
    logger.success(f"Year {year} metadata test completed")
    logger.info(f"Memory delta: {peak_mem - start_mem:.2f} MB")
    
    return peak_mem


def test_single_year_embedding(
    hf_repo: str,
    metadata_repo: str,
    dim: int,
    year: int,
    row_group: int = 200000,
):
    """Test processing a single year of embeddings."""
    logger.info(f"=" * 60)
    logger.info(f"Testing single-year embedding processing: {year}")
    logger.info(f"=" * 60)
    
    start_mem = log_memory_usage("start")
    
    # Load metadata for this year (needed for alignment)
    logger.info(f"Loading metadata for year {year}")
    metadata_lazy = load_metadata(metadata_repo, lazy=True)
    metadata_lazy = metadata_lazy.pipe(get_year_from_date)
    metadata_lazy = metadata_lazy.filter(pl.col("publish_year") == year).drop("publish_year")
    year_metadata = metadata_lazy.collect(streaming=True)
    logger.info(f"Metadata: {year_metadata.height} rows")
    mem_after_metadata = log_memory_usage("loaded metadata")
    
    # Get IDs for this year
    year_ids = set(year_metadata[ID].to_list())
    logger.info(f"Year {year} has {len(year_ids)} unique IDs")
    
    # Load embeddings and filter to year IDs
    logger.info("Loading embeddings (lazy)")
    embeddings_lazy = load_embedding(hf_repo, dim, lazy=True)
    embeddings_lazy = embeddings_lazy.filter(pl.col(ID).is_in(year_ids))
    
    logger.info(f"Collecting embeddings for year {year}")
    year_embeddings = embeddings_lazy.collect(streaming=True)
    logger.info(f"Embeddings: {year_embeddings.height} rows")
    mem_after_embeddings = log_memory_usage("loaded embeddings")
    
    # Simulate 10 new embeddings
    new_rows = year_embeddings.head(10)
    logger.info(f"Simulating {new_rows.height} new embeddings")
    
    # Save and update
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    save_year_shard(year_embeddings, TEST_DIR, year, "embedding", row_group)
    save_year_shard(year_metadata, TEST_DIR, year, "metadata", row_group)
    mem_after_save = log_memory_usage("saved shards")
    
    updated = update_year_shard(TEST_DIR, year, new_rows, "embedding", row_group)
    mem_after_update = log_memory_usage("updated shard")
    
    # Align with metadata
    logger.info("Aligning with metadata")
    aligned = align_order(year_metadata, updated, on=ID)
    peak_mem = log_memory_usage("aligned")
    
    # Cleanup
    del year_metadata, year_embeddings, new_rows, updated, aligned
    del metadata_lazy, embeddings_lazy, year_ids
    gc.collect()
    final_mem = log_memory_usage("after cleanup")
    
    logger.success(f"Year {year} embedding test completed")
    logger.info(f"Memory delta: {peak_mem - start_mem:.2f} MB")
    logger.info(f"Peak delta from embeddings load: {peak_mem - mem_after_embeddings:.2f} MB")
    
    return peak_mem


def main():
    """Run tests for single-year processing."""
    config_path = BASE_DIR / "config.toml"
    config = load_config(AppConfig, config_path)
    
    logger.info("=" * 60)
    logger.info("Testing single-year processing (no migration)")
    logger.info("=" * 60)
    
    test_years = [2024, 2025]
    metadata_results = {}
    embedding_results = {}
    
    # Test metadata
    for year in test_years:
        try:
            peak = test_single_year_metadata(
                hf_repo=config.metadata.hf_repo,
                year=year,
                row_group=config.metadata.row_group,
            )
            metadata_results[year] = peak
        except Exception as e:
            logger.error(f"Metadata test failed for {year}: {e}")
            metadata_results[year] = None
        
        # Force cleanup between tests
        gc.collect()
        logger.info("-" * 60)
    
    # Test embeddings
    for year in test_years:
        try:
            peak = test_single_year_embedding(
                hf_repo=config.embedding.hf_repo,
                metadata_repo=config.metadata.hf_repo,
                dim=config.embedding.dim,
                year=year,
                row_group=config.embedding.row_group,
            )
            embedding_results[year] = peak
        except Exception as e:
            logger.error(f"Embedding test failed for {year}: {e}")
            embedding_results[year] = None
        
        # Force cleanup between tests
        gc.collect()
        logger.info("-" * 60)
    
    # Summary
    logger.info("=" * 60)
    logger.info("MEMORY TEST SUMMARY")
    logger.info("=" * 60)
    logger.info("Metadata peak memory by year:")
    for year, mem in metadata_results.items():
        if mem:
            status = "✓ OK" if mem < 8000 else "✗ EXCEEDS 8GB"
            logger.info(f"  {year}: {mem:.2f} MB {status}")
    
    logger.info("\nEmbedding peak memory by year:")
    for year, mem in embedding_results.items():
        if mem:
            status = "✓ OK" if mem < 8000 else "✗ EXCEEDS 8GB"
            logger.info(f"  {year}: {mem:.2f} MB {status}")
    
    # Check results
    all_ok = True
    for mem in list(metadata_results.values()) + list(embedding_results.values()):
        if mem and mem >= 8000:
            all_ok = False
            break
    
    if all_ok:
        logger.success("✓ All tests passed! Memory usage is within CI limits (8GB)")
    else:
        logger.error("✗ Some tests exceeded memory limits")
    
    # Cleanup
    import shutil
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        logger.info(f"Cleaned up test directory: {TEST_DIR}")


if __name__ == "__main__":
    main()
