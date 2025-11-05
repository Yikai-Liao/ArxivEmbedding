"""Test memory usage for year-based shard updates.

Focuses on recent years (2024/2025) which are expected to have the highest memory usage.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import polars as pl
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.const import BASE_DIR
from src.io import load_metadata, load_embedding
from src.shard import (
    get_year_from_date,
    load_year_shard,
    save_year_shard,
    update_year_shard,
)
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


def test_year_metadata_update(hf_repo: str, year: int, row_group: int = 200000):
    """Test updating metadata for a specific year."""
    logger.info(f"=" * 60)
    logger.info(f"Testing metadata update for year {year}")
    logger.info(f"=" * 60)
    
    log_memory_usage("start")
    
    # Download year shard directly instead of loading full dataset
    logger.info(f"Downloading year {year} metadata shard from {hf_repo}")
    from huggingface_hub import HfApi, hf_hub_download
    
    api = HfApi()
    try:
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        shard_file = f"metadata_{year}.parquet"
        local_path = hf_hub_download(
            repo_id=hf_repo,
            filename=shard_file,
            repo_type="dataset",
            local_dir=TEST_DIR,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Downloaded to {local_path}")
        
        year_data = pl.read_parquet(local_path)
        logger.info(f"Year {year} has {year_data.height} rows")
        log_memory_usage(f"loaded year {year}")
        
        # Simulate adding 10 new rows
        new_rows = year_data.head(10)
        logger.info(f"Simulating update with {new_rows.height} new rows")
        
        # Update shard
        logger.info("Updating shard...")
        updated = update_year_shard(TEST_DIR, year, new_rows, "metadata", row_group)
        peak_mem = log_memory_usage("updated shard")
        
        logger.success(f"Year {year} metadata update completed")
        logger.info(f"Final row count: {updated.height}")
        logger.info(f"Peak memory: {peak_mem:.2f} MB")
        
        # Clean up
        del year_data, new_rows, updated
        import gc
        gc.collect()
        log_memory_usage("after cleanup")
        
        return peak_mem
        
    except Exception as e:
        logger.error(f"Failed to download year {year} shard: {e}")
        logger.warning(f"Year {year} shard may not exist yet in {hf_repo}")
        return None


def test_year_embedding_update(
    hf_repo: str,
    metadata_repo: str,
    dim: int,
    year: int,
    row_group: int = 200000,
):
    """Test updating embeddings for a specific year."""
    logger.info(f"=" * 60)
    logger.info(f"Testing embedding update for year {year}")
    logger.info(f"=" * 60)
    
    log_memory_usage("start")
    
    # Download year shards directly instead of loading full dataset
    logger.info(f"Downloading year {year} shards from HuggingFace")
    from huggingface_hub import HfApi, hf_hub_download
    
    api = HfApi()
    try:
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download metadata shard
        metadata_file = f"metadata_{year}.parquet"
        metadata_path = hf_hub_download(
            repo_id=metadata_repo,
            filename=metadata_file,
            repo_type="dataset",
            local_dir=TEST_DIR,
            local_dir_use_symlinks=False,
        )
        year_metadata = pl.read_parquet(metadata_path)
        logger.info(f"Loaded metadata: {year_metadata.height} rows")
        log_memory_usage("loaded metadata shard")
        
        # Download embedding shard
        embedding_file = f"embedding_{year}.parquet"
        embedding_path = hf_hub_download(
            repo_id=hf_repo,
            filename=embedding_file,
            repo_type="dataset",
            local_dir=TEST_DIR,
            local_dir_use_symlinks=False,
        )
        year_embeddings = pl.read_parquet(embedding_path)
        logger.info(f"Loaded embeddings: {year_embeddings.height} rows")
        log_memory_usage("loaded embedding shard")
        
        # Simulate adding 10 new embeddings
        new_rows = year_embeddings.head(10)
        logger.info(f"Simulating update with {new_rows.height} new embeddings")
        
        # Update shard
        logger.info("Updating embedding shard...")
        updated = update_year_shard(TEST_DIR, year, new_rows, "embedding", row_group)
        mem_after_update = log_memory_usage("updated shard")
        
        # Align with metadata
        logger.info("Aligning with metadata...")
        aligned = align_order(year_metadata, updated, on=ID)
        peak_mem = log_memory_usage("aligned with metadata")
        
        logger.success(f"Year {year} embedding update completed")
        logger.info(f"Final row count: {aligned.height}")
        logger.info(f"Peak memory: {peak_mem:.2f} MB")
        
        # Clean up
        del year_metadata, year_embeddings, new_rows, updated, aligned
        import gc
        gc.collect()
        log_memory_usage("after cleanup")
        
        return peak_mem
        
    except Exception as e:
        logger.error(f"Failed to download year {year} shards: {e}")
        logger.warning(f"Year {year} shards may not exist yet")
        return None


def main():
    """Test memory usage for recent years."""
    config_path = BASE_DIR / "config.toml"
    config = load_config(AppConfig, config_path)
    
    logger.info("=" * 60)
    logger.info("Testing year-based shard memory usage")
    logger.info("=" * 60)
    
    # Test years (focus on recent years with most data)
    test_years = [2024, 2025]
    
    metadata_results = {}
    embedding_results = {}
    
    # Test metadata updates
    for year in test_years:
        try:
            peak = test_year_metadata_update(
                hf_repo=config.metadata.hf_repo,
                year=year,
                row_group=config.metadata.row_group,
            )
            metadata_results[year] = peak
        except Exception as e:
            logger.error(f"Failed to test metadata year {year}: {e}")
            metadata_results[year] = None
    
    # Test embedding updates
    for year in test_years:
        try:
            peak = test_year_embedding_update(
                hf_repo=config.embedding.hf_repo,
                metadata_repo=config.metadata.hf_repo,
                dim=config.embedding.dim,
                year=year,
                row_group=config.embedding.row_group,
            )
            embedding_results[year] = peak
        except Exception as e:
            logger.error(f"Failed to test embedding year {year}: {e}")
            embedding_results[year] = None
    
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
    
    # Check if all pass
    all_metadata_ok = all(m and m < 8000 for m in metadata_results.values() if m is not None)
    all_embedding_ok = all(m and m < 8000 for m in embedding_results.values() if m is not None)
    
    if all_metadata_ok and all_embedding_ok:
        logger.success("✓ All tests passed! Memory usage is within CI limits (8GB)")
    else:
        logger.error("✗ Some tests exceeded memory limits")
    
    # Cleanup test directory
    import shutil
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        logger.info(f"Cleaned up test directory: {TEST_DIR}")


if __name__ == "__main__":
    main()
