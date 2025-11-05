"""Isolated memory test using subprocess for each year.

This avoids memory leaks between tests by running each in a separate process.
"""
import subprocess
import sys
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import AppConfig, load_config


def run_year_test(year: int, test_type: str, config_path: Path) -> dict:
    """Run a single year test in a subprocess."""
    script = f"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import polars as pl
import psutil
import gc

sys.path.append('{BASE_DIR}')

from src.const import BASE_DIR
from src.io import load_metadata, load_embedding
from src.shard import get_year_from_date, save_year_shard, update_year_shard
from src.order import align_order
from src.name import ID
from src.config import AppConfig, load_config

DATA_DIR = BASE_DIR / "data"
TEST_DIR = BASE_DIR / "data" / "test_shards_isolated"
load_dotenv(BASE_DIR / ".env")

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

config = load_config(AppConfig, Path('{config_path}'))
year = {year}
test_type = '{test_type}'

start_mem = get_mem()
peak_mem = start_mem

if test_type == 'metadata':
    # Load metadata for year
    metadata_lazy = load_metadata(config.metadata.hf_repo, lazy=True)
    metadata_lazy = metadata_lazy.pipe(get_year_from_date)
    metadata_lazy = metadata_lazy.filter(pl.col("publish_year") == year).drop("publish_year")
    year_data = metadata_lazy.collect(engine="streaming")
    
    mem_after_load = get_mem()
    peak_mem = max(peak_mem, mem_after_load)
    
    # Simulate update
    new_rows = year_data.head(10)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    save_year_shard(year_data, TEST_DIR, year, "metadata", config.metadata.row_group)
    
    mem_after_save = get_mem()
    peak_mem = max(peak_mem, mem_after_save)
    
    updated = update_year_shard(TEST_DIR, year, new_rows, "metadata", config.metadata.row_group)
    
    mem_after_update = get_mem()
    peak_mem = max(peak_mem, mem_after_update)
    
    print(f"RESULT:{{year}}:{{year_data.height}}:{{start_mem:.2f}}:{{peak_mem:.2f}}")

elif test_type == 'embedding':
    # Load metadata for year
    metadata_lazy = load_metadata(config.metadata.hf_repo, lazy=True)
    metadata_lazy = metadata_lazy.pipe(get_year_from_date)
    metadata_lazy = metadata_lazy.filter(pl.col("publish_year") == year).drop("publish_year")
    year_metadata = metadata_lazy.collect(engine="streaming")
    
    mem_after_metadata = get_mem()
    peak_mem = max(peak_mem, mem_after_metadata)
    
    # Get year IDs and load embeddings
    year_ids = set(year_metadata[ID].to_list())
    embeddings_lazy = load_embedding(config.embedding.hf_repo, config.embedding.dim, lazy=True)
    embeddings_lazy = embeddings_lazy.filter(pl.col(ID).is_in(year_ids))
    year_embeddings = embeddings_lazy.collect(engine="streaming")
    
    mem_after_embeddings = get_mem()
    peak_mem = max(peak_mem, mem_after_embeddings)
    
    # Simulate update
    new_rows = year_embeddings.head(10)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    save_year_shard(year_embeddings, TEST_DIR, year, "embedding", config.embedding.row_group)
    save_year_shard(year_metadata, TEST_DIR, year, "metadata", config.metadata.row_group)
    
    mem_after_save = get_mem()
    peak_mem = max(peak_mem, mem_after_save)
    
    updated = update_year_shard(TEST_DIR, year, new_rows, "embedding", config.embedding.row_group)
    
    mem_after_update = get_mem()
    peak_mem = max(peak_mem, mem_after_update)
    
    # Align
    aligned = align_order(year_metadata, updated, on=ID)
    
    mem_after_align = get_mem()
    peak_mem = max(peak_mem, mem_after_align)
    
    print(f"RESULT:{{year}}:{{year_embeddings.height}}:{{start_mem:.2f}}:{{peak_mem:.2f}}")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )
        
        # Parse output
        for line in result.stdout.split('\n'):
            if line.startswith('RESULT:'):
                parts = line.split(':')
                return {
                    'year': int(parts[1]),
                    'rows': int(parts[2]),
                    'start_mem': float(parts[3]),
                    'peak_mem': float(parts[4]),
                    'success': True,
                }
        
        # If no RESULT line found
        logger.error(f"No result found. stdout:\n{result.stdout}")
        logger.error(f"stderr:\n{result.stderr}")
        return {'success': False, 'error': 'No result line'}
        
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def main():
    """Run isolated tests for each year."""
    config_path = BASE_DIR / "config.toml"
    config = load_config(AppConfig, config_path)
    
    logger.info("=" * 60)
    logger.info("Isolated memory tests (subprocess per year)")
    logger.info("=" * 60)
    
    test_years = [2024, 2025]
    
    # Test metadata
    logger.info("\nMetadata tests:")
    logger.info("-" * 60)
    metadata_results = {}
    for year in test_years:
        logger.info(f"Testing metadata for year {year}...")
        result = run_year_test(year, 'metadata', config_path)
        metadata_results[year] = result
        
        if result.get('success'):
            logger.success(f"  Year {year}: {result['rows']} rows, peak {result['peak_mem']:.2f} MB")
        else:
            logger.error(f"  Year {year}: FAILED - {result.get('error')}")
    
    # Test embeddings
    logger.info("\nEmbedding tests:")
    logger.info("-" * 60)
    embedding_results = {}
    for year in test_years:
        logger.info(f"Testing embeddings for year {year}...")
        result = run_year_test(year, 'embedding', config_path)
        embedding_results[year] = result
        
        if result.get('success'):
            logger.success(f"  Year {year}: {result['rows']} rows, peak {result['peak_mem']:.2f} MB")
        else:
            logger.error(f"  Year {year}: FAILED - {result.get('error')}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    logger.info("\nMetadata:")
    for year, result in metadata_results.items():
        if result.get('success'):
            status = "✓" if result['peak_mem'] < 8000 else "✗"
            logger.info(f"  {year}: {result['peak_mem']:.2f} MB {status}")
    
    logger.info("\nEmbedding:")
    for year, result in embedding_results.items():
        if result.get('success'):
            status = "✓" if result['peak_mem'] < 8000 else "✗"
            logger.info(f"  {year}: {result['peak_mem']:.2f} MB {status}")
    
    # Check if all passed
    all_pass = True
    for results in [metadata_results, embedding_results]:
        for result in results.values():
            if not result.get('success') or result.get('peak_mem', 9999) >= 8000:
                all_pass = False
                break
    
    if all_pass:
        logger.success("\n✓ All tests passed! Memory within 8GB limit")
    else:
        logger.error("\n✗ Some tests failed or exceeded limit")
    
    # Cleanup
    import shutil
    test_dir = BASE_DIR / "data" / "test_shards_isolated"
    if test_dir.exists():
        shutil.rmtree(test_dir)
        logger.info(f"Cleaned up: {test_dir}")


if __name__ == "__main__":
    main()
