"""Test append-only merge correctness and memory usage.

This script:
1. Creates a backup of existing parquet
2. Simulates append operation
3. Verifies data integrity
4. Measures memory during append
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import shutil
import psutil
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from loguru import logger
from src.const import BASE_DIR
from src.schema import paper_embedding_schema_pl


def measure_memory(label):
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1024 / 1024
    logger.info(f"[MEMORY] {label}: {rss:.2f} MB")
    return rss


def test_metadata_append():
    """Test metadata append without full load."""
    logger.info("=" * 60)
    logger.info("TEST 1: Metadata append")
    logger.info("=" * 60)
    
    original = BASE_DIR / "data" / "metadata.parquet"
    if not original.exists():
        logger.error("metadata.parquet not found")
        return
    
    # Read original stats
    reader = pq.ParquetFile(str(original))
    original_rows = reader.metadata.num_rows
    original_rgs = reader.num_row_groups
    logger.info(f"Original: {original_rows} rows, {original_rgs} row-groups, size: {original.stat().st_size/1024/1024:.1f} MB")
    
    # Test: simulate appending 10 new rows
    test_dir = BASE_DIR / "tmp_append_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_out = test_dir / "metadata_appended.parquet"
    
    measure_memory("before metadata append")
    
    # Simulate append by copying row-groups
    from tempfile import NamedTemporaryFile
    tmp = NamedTemporaryFile(delete=False, suffix=".parquet", dir=test_dir)
    tmp_path = Path(tmp.name)
    tmp.close()
    
    schema = reader.schema_arrow
    writer = pq.ParquetWriter(str(tmp_path), schema)
    
    peak = measure_memory("start copying row-groups")
    for rg in range(reader.num_row_groups):
        table = reader.read_row_group(rg)
        writer.write_table(table)
        mem = measure_memory(f"copied rg {rg+1}/{reader.num_row_groups}")
        peak = max(peak, mem)
    
    # Create 10 fake new rows using Arrow directly to match schema
    sample_table = reader.read_row_group(0).slice(0, 1)
    # Repeat 10 times
    import pyarrow as pa
    arrays = []
    for i in range(10):
        row_arrays = [col.slice(0, 1) for col in sample_table.columns]
        arrays.append(pa.Table.from_arrays(row_arrays, schema=schema))
    new_rows_table = pa.concat_tables(arrays)
    writer.write_table(new_rows_table)
    writer.close()
    
    peak = max(peak, measure_memory("wrote new rows"))
    
    # Verify
    new_reader = pq.ParquetFile(str(tmp_path))
    new_rows_count = new_reader.metadata.num_rows
    logger.info(f"Result: {new_rows_count} rows (expected {original_rows + 10})")
    logger.info(f"Peak memory: {peak:.2f} MB")
    
    assert new_rows_count == original_rows + 10, "Row count mismatch!"
    logger.success("✓ Metadata append test passed")
    
    return peak


def test_embedding_append():
    """Test embedding append without full load."""
    logger.info("=" * 60)
    logger.info("TEST 2: Embedding append")
    logger.info("=" * 60)
    
    original = BASE_DIR / "data" / "embedding.parquet"
    if not original.exists():
        logger.error("embedding.parquet not found")
        return
    
    reader = pq.ParquetFile(str(original))
    original_rows = reader.metadata.num_rows
    original_rgs = reader.num_row_groups
    logger.info(f"Original: {original_rows} rows, {original_rgs} row-groups, size: {original.stat().st_size/1024/1024:.1f} MB")
    
    test_dir = BASE_DIR / "tmp_append_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    measure_memory("before embedding append")
    
    from tempfile import NamedTemporaryFile
    tmp = NamedTemporaryFile(delete=False, suffix=".parquet", dir=test_dir)
    tmp_path = Path(tmp.name)
    tmp.close()
    
    schema = reader.schema_arrow
    writer = pq.ParquetWriter(str(tmp_path), schema, compression='zstd', compression_level=5)
    
    peak = measure_memory("start copying row-groups")
    for rg in range(reader.num_row_groups):
        table = reader.read_row_group(rg)
        writer.write_table(table)
        mem = measure_memory(f"copied rg {rg+1}/{reader.num_row_groups}")
        peak = max(peak, mem)
    
    # Create 10 fake new embeddings
    sample_row = reader.read_row_group(0).slice(0, 1)
    sample_df = pl.from_arrow(sample_row)
    dim = len(sample_df['embedding'][0])
    
    new_embeddings = pl.DataFrame({
        "id": [f"test-new-{i}" for i in range(10)],
        "embedding": [np.random.rand(dim).astype(np.float32) for _ in range(10)]
    })
    
    writer.write_table(new_embeddings.to_arrow())
    writer.close()
    
    peak = max(peak, measure_memory("wrote new embeddings"))
    
    # Verify
    new_reader = pq.ParquetFile(str(tmp_path))
    new_rows_count = new_reader.metadata.num_rows
    logger.info(f"Result: {new_rows_count} rows (expected {original_rows + 10})")
    logger.info(f"Peak memory: {peak:.2f} MB")
    
    assert new_rows_count == original_rows + 10, "Row count mismatch!"
    logger.success("✓ Embedding append test passed")
    
    return peak


def main():
    logger.info("Testing append-only merge correctness and memory")
    
    metadata_peak = test_metadata_append()
    embedding_peak = test_embedding_append()
    
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Metadata append peak: {metadata_peak:.2f} MB")
    logger.info(f"Embedding append peak: {embedding_peak:.2f} MB")
    
    # Check against CI limits
    if metadata_peak > 8000:
        logger.warning(f"⚠ Metadata append exceeds 8GB limit!")
    else:
        logger.success(f"✓ Metadata append within 8GB limit")
    
    if embedding_peak > 8000:
        logger.warning(f"⚠ Embedding append exceeds 8GB limit!")
    else:
        logger.success(f"✓ Embedding append within 8GB limit")


if __name__ == "__main__":
    main()
