import polars as pl
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, upload_file
from pathlib import Path
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import os

def load_artifacts(artifacts_dir: str):
    """Load all Parquet files from the artifacts directory."""
    model_keys = set()
    year2df = defaultdict(list)
    for matrix_dir in Path(artifacts_dir).glob("matrix-output-*"):
        for file in matrix_dir.glob("embedded-*.parquet"):
            year =  int(re.search(r'embedded-(.*).parquet-', file.name).group(1))
            model_key = re.search(r'.parquet-*?-(.*?)\.parquet$', file.name).group(1).replace('.parquet', '')
            model_keys.add(model_key)
            df = pl.read_parquet(file)
            year2df[year].append(df)
    print(model_keys)
    model_keys = list(model_keys)
    return year2df, model_keys
            
def load_raw_data(repo_id: str = "lyk/ArxivEmbedding", years: list[int] = None):
    year2df = {}
    for year in years:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{year}.parquet",
            repo_type="dataset",
            local_dir="./tmp_dataset"
        )
        year2df[year] = pl.read_parquet(file_path)
    return year2df
        
def merge_data(raw_data: Dict[int, pl.DataFrame], artifacts_data: Dict[int, List[pl.DataFrame]]):
    merged_data = {}
    for year, raw_df in raw_data.items():
        merged_df = raw_df.clone()
        for artifact_df in artifacts_data[year]:
            merged_df = merged_df.update(artifact_df, on='id')
        merged_data[year] = merged_df
    return merged_data

def show_non_zero(df: pl.DataFrame, model_keys: List[str]):
    # --- Build Filter Condition ---
    # Create a list to hold the boolean expression for each key
    conditions = []
    for key in model_keys:
        condition = pl.col(key).arr.sum() != 0
        conditions.append(condition)

    # Combine the individual column conditions using a row-wise OR operation.
    # pl.any_horizontal(conditions) creates a single boolean Series.
    # For each row, it will be True if *any* of the conditions in the list
    # evaluates to True for that row, and False otherwise.
    filter_condition = pl.any_horizontal(conditions)

    # --- Filter and Print ---
    # Apply the combined filter condition to the DataFrame.
    filtered_df = df.filter(filter_condition)

    # Print the header and the resulting filtered DataFrame.
    print(f"Rows where at least one vector in {model_keys} has a non-zero element:")
    print(filtered_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge embedding artifact Parquet files and upload to Hugging Face Hub.")
    parser.add_argument("--repo-id", default="lyk/ArxivEmbedding", type=str, required=True, help="Hugging Face Hub repository ID (e.g., lyk/ArxivEmbedding).")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory containing the downloaded matrix output artifacts.")
    # Add other arguments if needed, e.g., --commit-message-prefix

    args = parser.parse_args()
    artifacts_data, model_keys = load_artifacts(artifacts_dir=args.artifacts_dir)
    raw_data = load_raw_data(years=list(artifacts_data.keys()))
    merged_data = merge_data(raw_data, artifacts_data)
    # Huggingface 上传
    token = os.getenv("HF_TOKEN")
    os.makedirs("./merged_data", exist_ok=True)
    for year, df in merged_data.items():
        file_path = f"./merged_data/{year}.parquet"
        df.write_parquet(file_path)
        upload_file(
            repo_id="lyk/ArxivEmbedding",
            path_in_repo=f"{year}.parquet",
            path_or_fileobj=file_path,
            repo_type="dataset",
            commit_message=f"Merge embedding data for year {year}",
            token=token
        )
