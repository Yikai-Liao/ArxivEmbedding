"""Year-based sharding utilities for metadata and embeddings.

This module provides functions to:
1. Split metadata/embeddings by publish_year into separate parquet files
2. Load and merge sharded files (from local or HuggingFace)
3. Update individual year shards incrementally
"""
import polars as pl
from pathlib import Path
from loguru import logger
from src.name import ID, PUBLISH_DATE
from src.io import upload_data
from huggingface_hub import HfApi, hf_hub_download


def get_year_from_date(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Add a 'publish_year' column extracted from publish_date."""
    if isinstance(df, pl.LazyFrame):
        return df.with_columns(
            pl.col(PUBLISH_DATE).dt.year().alias("publish_year")
        )
    return df.with_columns(
        pl.col(PUBLISH_DATE).dt.year().alias("publish_year")
    )


def get_years_in_data(df: pl.DataFrame | pl.LazyFrame) -> list[int]:
    """Get unique years from dataframe."""
    if isinstance(df, pl.LazyFrame):
        df = df.select(pl.col(PUBLISH_DATE).dt.year().alias("year")).collect(engine="streaming")
    else:
        df = df.select(pl.col(PUBLISH_DATE).dt.year().alias("year"))
    return sorted(df["year"].unique().to_list())


def load_year_shard(base_dir: Path, year: int, file_prefix: str, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame | None:
    """Load a single year shard parquet file.
    
    Args:
        base_dir: Directory containing year shards
        year: Year to load
        file_prefix: Prefix for filename (e.g., 'metadata', 'embedding')
        lazy: Whether to return LazyFrame
    
    Returns:
        DataFrame or LazyFrame, or None if file doesn't exist
    """
    path = base_dir / f"{file_prefix}_{year}.parquet"
    if not path.exists():
        return None
    
    if lazy:
        return pl.scan_parquet(str(path), low_memory=True)
    return pl.read_parquet(str(path))


def save_year_shard(df: pl.DataFrame, base_dir: Path, year: int, file_prefix: str, row_group: int = 200000):
    """Save a single year shard to parquet.
    
    Args:
        df: DataFrame to save
        base_dir: Directory to save to
        year: Year label
        file_prefix: Prefix for filename
        row_group: Row group size for parquet
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{file_prefix}_{year}.parquet"
    df.write_parquet(str(path), row_group_size=row_group, compression="zstd", compression_level=5)
    logger.info(f"Saved {df.height} rows to {path}")


def load_all_year_shards(base_dir: Path, file_prefix: str, hf_repo: str, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Load and concatenate all year shards.
    
    Args:
        base_dir: Directory containing shards
        file_prefix: Prefix for filenames
        hf_repo: HuggingFace repo to download from
        lazy: Whether to return LazyFrame
    
    Returns:
        Concatenated DataFrame or LazyFrame
    """
    from huggingface_hub import snapshot_download
    
    # Download entire repo (with auto caching)
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        local_dir=base_dir,
        local_dir_use_symlinks=False,
        allow_patterns=f"{file_prefix}_*.parquet",
    )
    
    # Load local files
    shard_files = sorted(base_dir.glob(f"{file_prefix}_*.parquet"))
    
    if not shard_files:
        raise FileNotFoundError(f"No {file_prefix}_*.parquet files found in {base_dir}")
    
    # Extract year range for concise logging
    years = [int(f.stem.split('_')[-1]) for f in shard_files]
    year_range = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
    logger.info(f"Loading {len(shard_files)} year shards ({year_range})")
    
    if lazy:
        return pl.concat([pl.scan_parquet(str(f), low_memory=True) for f in shard_files], how="vertical_relaxed")
    else:
        return pl.concat([pl.read_parquet(str(f)) for f in shard_files], how="vertical_relaxed")


def update_year_shard(
    base_dir: Path,
    year: int,
    new_data: pl.DataFrame,
    file_prefix: str,
    row_group: int = 200000,
) -> pl.DataFrame:
    """Update a single year shard with new data.
    
    Loads existing shard (if exists), merges with new data, deduplicates, sorts, and saves.
    
    Args:
        base_dir: Directory for shards
        year: Year to update
        new_data: New rows to add (should already be filtered to this year)
        file_prefix: Filename prefix
        row_group: Row group size
    
    Returns:
        Updated DataFrame for this year
    """
    existing = load_year_shard(base_dir, year, file_prefix, lazy=False)
    
    if existing is None or existing.height == 0:
        # No existing data, just use new data
        result = new_data
    else:
        # Merge and deduplicate
        result = pl.concat([existing, new_data]).unique(subset=ID, keep="last")
    
    # Sort by publish_date within year for better compression and query performance
    # Only sort if publish_date column exists (metadata has it, embeddings don't)
    if PUBLISH_DATE in result.columns:
        result = result.sort(PUBLISH_DATE)
    else:
        # For embeddings, sort by ID
        result = result.sort(ID)
    
    save_year_shard(result, base_dir, year, file_prefix, row_group)
    return result


def upload_year_shard(base_dir: Path, year: int, file_prefix: str, hf_repo: str, squash_history: bool = True):
    """Upload a year shard to Hugging Face."""
    path = base_dir / f"{file_prefix}_{year}.parquet"
    if not path.exists():
        logger.warning(f"Shard file {path} does not exist, skipping upload")
        return
    
    path_in_repo = f"{file_prefix}_{year}.parquet"
    logger.info(f"Uploading {path} to {hf_repo}/{path_in_repo}")
    upload_data(
        path=path,
        path_in_repo=path_in_repo,
        hf_repo=hf_repo,
        squash_history=squash_history,
    )
