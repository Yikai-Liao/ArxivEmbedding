"""Analyze data distribution by year to determine optimal sharding strategy."""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import polars as pl

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.const import BASE_DIR
from src.io import load_metadata, load_embedding
from src.shard import get_year_from_date
from src.name import ID
from src.config import AppConfig, load_config

load_dotenv(BASE_DIR / ".env")


def analyze_metadata_by_year(hf_repo: str):
    """Analyze metadata row counts by year."""
    logger.info("Analyzing metadata distribution by year")
    
    metadata = load_metadata(hf_repo, lazy=True)
    metadata = metadata.pipe(get_year_from_date)
    
    # Count by year
    year_counts = (
        metadata
        .group_by("publish_year")
        .agg(pl.count().alias("count"))
        .sort("publish_year")
        .collect(engine="streaming")
    )
    
    return year_counts


def analyze_embeddings_by_year(hf_repo: str, metadata_repo: str, dim: int):
    """Analyze embedding row counts by year."""
    logger.info("Analyzing embeddings distribution by year")
    
    # Load metadata to get years
    metadata = load_metadata(metadata_repo, lazy=True)
    metadata = metadata.pipe(get_year_from_date)
    metadata_year = metadata.select([ID, "publish_year"])
    
    # Load embeddings and join with metadata to get years
    embeddings = load_embedding(hf_repo, dim, lazy=True)
    embeddings_with_year = embeddings.join(metadata_year, on=ID, how="left")
    
    # Count by year
    year_counts = (
        embeddings_with_year
        .group_by("publish_year")
        .agg(pl.count().alias("count"))
        .sort("publish_year")
        .collect(engine="streaming")
    )
    
    return year_counts


def estimate_memory_usage(row_count: int, dim: int = 1536) -> float:
    """Estimate memory usage for embedding data in MB.
    
    Args:
        row_count: Number of rows
        dim: Embedding dimension
    
    Returns:
        Estimated memory in MB (conservative estimate with 20x overhead)
    """
    # Base size: dim * 4 bytes (float32) per row
    base_size_mb = row_count * dim * 4 / 1024 / 1024
    
    # Observed overhead: polars uses ~15-20x during operations
    # Using 20x to be conservative
    estimated_mb = base_size_mb * 20
    
    return estimated_mb


def recommend_sharding_strategy(year_counts: pl.DataFrame, dim: int = 1536, max_mem_mb: float = 7000):
    """Recommend sharding strategy based on data distribution.
    
    Args:
        year_counts: DataFrame with columns [publish_year, count]
        dim: Embedding dimension
        max_mem_mb: Maximum memory limit in MB (using 7GB to have buffer)
    
    Returns:
        Dict mapping year to recommended number of shards
    """
    recommendations = {}
    
    logger.info(f"\nSharding recommendations (target < {max_mem_mb:.0f} MB per shard):")
    logger.info("-" * 70)
    
    for row in year_counts.iter_rows(named=True):
        year = row["publish_year"]
        count = row["count"]
        
        # Estimate memory for this year
        estimated_mem = estimate_memory_usage(count, dim)
        
        # Calculate required shards
        if estimated_mem <= max_mem_mb:
            shards_needed = 1
            status = "✓ Single shard OK"
        else:
            shards_needed = int((estimated_mem / max_mem_mb) + 0.5) + 1  # Round up with buffer
            status = f"⚠ Split into {shards_needed} shards"
        
        recommendations[year] = shards_needed
        
        logger.info(
            f"  {year}: {count:>7} rows → {estimated_mem:>7.0f} MB → {status}"
        )
    
    return recommendations


def main():
    """Analyze data and recommend sharding strategy."""
    config_path = BASE_DIR / "config.toml"
    config = load_config(AppConfig, config_path)
    
    logger.info("=" * 70)
    logger.info("Data Distribution Analysis")
    logger.info("=" * 70)
    
    # Analyze metadata
    logger.info("\nMetadata by year:")
    logger.info("-" * 70)
    metadata_years = analyze_metadata_by_year(config.metadata.hf_repo)
    
    for row in metadata_years.iter_rows(named=True):
        year = row["publish_year"]
        count = row["count"]
        # Rough estimate: ~3.5KB per metadata row based on observed data
        est_mem = count * 3.5 / 1024  # MB
        logger.info(f"  {year}: {count:>7} rows (~{est_mem:>6.1f} MB)")
    
    logger.info(f"\nTotal metadata: {metadata_years['count'].sum()} rows")
    
    # Analyze embeddings
    logger.info("\nEmbeddings by year:")
    logger.info("-" * 70)
    embedding_years = analyze_embeddings_by_year(
        config.embedding.hf_repo,
        config.metadata.hf_repo,
        config.embedding.dim,
    )
    
    for row in embedding_years.iter_rows(named=True):
        year = row["publish_year"]
        count = row["count"]
        est_mem = estimate_memory_usage(count, config.embedding.dim)
        logger.info(f"  {year}: {count:>7} rows (~{est_mem:>7.0f} MB estimated)")
    
    logger.info(f"\nTotal embeddings: {embedding_years['count'].sum()} rows")
    
    # Recommend sharding strategy
    logger.info("\n" + "=" * 70)
    sharding_recommendations = recommend_sharding_strategy(
        embedding_years,
        config.embedding.dim,
        max_mem_mb=7000,  # Conservative 7GB limit
    )
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    single_shard_years = [y for y, s in sharding_recommendations.items() if s == 1]
    multi_shard_years = {y: s for y, s in sharding_recommendations.items() if s > 1}
    
    logger.info(f"\n✓ Years that can use single shard: {len(single_shard_years)}")
    if single_shard_years:
        logger.info(f"  {single_shard_years}")
    
    logger.info(f"\n⚠ Years that need multiple shards: {len(multi_shard_years)}")
    if multi_shard_years:
        for year, shards in multi_shard_years.items():
            logger.info(f"  {year}: split into {shards} shards")
    
    logger.info("\nRecommendation:")
    logger.info("  - For most years: use year-based sharding (embedding_YYYY.parquet)")
    logger.info("  - For recent years (2024+): use quarterly sharding (embedding_YYYY_QN.parquet)")


if __name__ == "__main__":
    main()
