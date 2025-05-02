#!/usr/bin/env python
import os
import sys
import glob
import argparse
import polars as pl
from huggingface_hub import HfApi, hf_hub_download, upload_file
from tqdm import tqdm
from loguru import logger
import re
import numpy as np # Need numpy for zero vector creation

def merge_and_upload(args):
    """Merges embedding Parquet files from artifacts and uploads them to the Hub."""
    repo_id = args.repo_id
    artifacts_dir = args.artifacts_dir
    hf_token = os.environ.get('HF_TOKEN') # Read token from environment

    logger.info(f"--- Merge and Upload Script ---")
    logger.info(f"Repo ID: {repo_id}")
    logger.info(f"Artifacts Directory: {artifacts_dir}")
    logger.info(f"-----------------------------")

    if not hf_token:
         logger.warning("HF_TOKEN environment variable not set. Upload might fail for private repos.")
         # Allow script to continue for public repos or if logged in via cli

    api = HfApi(token=hf_token) # Initialize API with token if available

    # Find all embedded files within the artifacts directory structure
    # Example structure: artifacts/matrix-output-0/embedded-2020.parquet-model_key.parquet
    search_pattern = os.path.join(artifacts_dir, "matrix-output-*", "embedded-*.parquet")
    embedded_files = glob.glob(search_pattern)

    if not embedded_files:
        logger.warning(f"No 'embedded-*.parquet' files found in subdirectories under '{artifacts_dir}'. Nothing to merge or upload.")
        return # Exit gracefully

    logger.info(f"Found {len(embedded_files)} embedding artifact files to process.")

    # Identify unique year files from the embedded file patterns
    year_files_to_process = set()
    # Extract year from filenames like embedded-2020.parquet-model_key.parquet
    for file_path in embedded_files:
        filename = os.path.basename(file_path)
        # Extract the original year file name (e.g., "2020.parquet")
        match = re.search(r'embedded-([0-9]+\.parquet)', filename)
        if match:
            year_file = match.group(1)
            year_files_to_process.add(year_file)
        else:
             logger.warning(f"Could not extract base year filename from artifact: {filename}")

    if not year_files_to_process:
        logger.error("Could not identify any target year Parquet files from the found artifacts. Stopping.")
        return

    logger.info(f"Identified unique year files to process: {sorted(list(year_files_to_process))}")

    for year_file in tqdm(sorted(list(year_files_to_process)), desc="Merging and Uploading Years"):
        logger.info(f"--- Processing {year_file} ---")
        local_final_path = f"final_{year_file}" # Temp local path for merged file
        
        # Find all embedding files for this specific year
        # Match filenames like embedded-2020.parquet-model_key.parquet
        embedding_files_for_year = [f for f in embedded_files if f"embedded-{year_file}-" in os.path.basename(f)]
        
        if not embedding_files_for_year:
            logger.warning(f"No embedding artifacts found specifically for {year_file} (this shouldn't happen if identified earlier). Skipping.")
            continue

        logger.info(f"Found {len(embedding_files_for_year)} artifact files for {year_file}: { [os.path.basename(f) for f in embedding_files_for_year] }")

        # Download the original file from the Hub to merge into
        original_file_path = None # Define outside try block for cleanup
        try:
            logger.info(f"Downloading original {year_file} from Hub repo {repo_id}...")
            original_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=year_file,
                repo_type='dataset',
                local_dir='.', # Download to current dir
                token=hf_token
            )
            logger.info(f"Loading original file: {original_file_path}")
            df_merged = pl.read_parquet(original_file_path)
            logger.info(f"Original {year_file} shape: {df_merged.shape}")
            
            original_schema = df_merged.schema
            logger.debug(f"Original schema: {original_schema}")

            if 'id' not in df_merged.columns:
                logger.error(f"Original file {original_file_path} is missing the required 'id' column. Skipping merge for {year_file}.")
                # Clean up downloaded original file before skipping
                if original_file_path and os.path.exists(original_file_path):
                     try: os.remove(original_file_path)
                     except OSError: logger.warning(f"Failed to cleanup downloaded file: {original_file_path}")
                continue 

        except Exception as e:
            logger.error(f"Failed to download or load original {year_file} from Hub: {e}. Skipping merge for this year.")
            # Clean up potentially downloaded file if path exists and an error occurred
            if original_file_path and os.path.exists(original_file_path):
                try: os.remove(original_file_path)
                except OSError: logger.warning(f"Failed to cleanup downloaded file: {original_file_path}")
            continue
        finally:
             # Clean up downloaded original file after loading (if it exists)
             if original_file_path and os.path.exists(original_file_path):
                 try:
                     os.remove(original_file_path)
                     logger.debug(f"Cleaned up downloaded original file: {original_file_path}")
                 except OSError as clean_err:
                      logger.warning(f"Failed to cleanup downloaded file {original_file_path}: {clean_err}")


        # Merge embedding columns from artifacts
        merged_cols_count = 0
        all_merged_cols = set()

        for embed_file in embedding_files_for_year:
            try:
                logger.debug(f"Loading embedding artifact: {embed_file}")
                df_embed = pl.read_parquet(embed_file)
                logger.debug(f"Artifact {embed_file} shape: {df_embed.shape}, columns: {df_embed.columns}")

                if 'id' not in df_embed.columns:
                    logger.warning(f"Embedding file {embed_file} is missing 'id' column. Skipping.")
                    continue

                # --- Determine embedding column name ---
                embed_col_name = None
                potential_embed_cols = [col for col in df_embed.columns if col != 'id']

                # Try extracting model key from filename (e.g., embedded-2020.parquet-BAAI_bge-large-en-v1.5.parquet)
                model_match = re.search(r'embedded-.*?-(.*?)\.parquet$', os.path.basename(embed_file))
                if model_match:
                    extracted_model_key = model_match.group(1).replace('.parquet', '') # Clean up if needed
                    if extracted_model_key in df_embed.columns:
                        embed_col_name = extracted_model_key
                        logger.debug(f"Identified embedding column '{embed_col_name}' from filename pattern in {embed_file}")
                    else:
                        logger.warning(f"Extracted model key '{extracted_model_key}' not found as column in {embed_file}. Columns available: {df_embed.columns}. Falling back to heuristic.")
                
                # Fallback heuristic: if only one non-id column exists, use it
                if not embed_col_name:
                    if len(potential_embed_cols) == 1:
                        embed_col_name = potential_embed_cols[0]
                        logger.debug(f"Using the only non-id column '{embed_col_name}' as embedding column from {embed_file}")
                    else:
                        logger.warning(f"Could not reliably determine the embedding column in {embed_file} (found non-id columns: {potential_embed_cols}). Skipping file.")
                        continue
                # --- End Determine embedding column name ---


                # --- Perform Merge/Update ---
                logger.debug(f"Preparing to update column '{embed_col_name}' from {embed_file} into {year_file} data.")

                # Select only id and the embedding column for update
                df_embed_subset = df_embed.select(['id', embed_col_name])

                # Cast ID type if necessary (optional, Polars update usually handles this)
                try:
                    target_id_type = df_merged['id'].dtype
                    if df_embed_subset['id'].dtype != target_id_type:
                         logger.debug(f"Casting ID column in subset to {target_id_type}")
                         df_embed_subset = df_embed_subset.with_columns(pl.col('id').cast(target_id_type))
                except Exception as cast_err:
                    logger.warning(f"Could not cast 'id' column in {embed_file} subset to match target type {target_id_type}. Proceeding with original type. Error: {cast_err}")

                # Check if the embedding column needs to be added first (shouldn't happen if logic is sound)
                # However, let's add robustly before update
                if embed_col_name not in df_merged.columns:
                    # Infer dtype and shape from the subset
                    first_valid_embedding = df_embed_subset.drop_nulls(subset=[embed_col_name])[embed_col_name].head(1)
                    if not first_valid_embedding.is_empty():
                        sample_value = first_valid_embedding[0]
                        if isinstance(sample_value, list): # Check if it's a list/array
                             embed_dtype = pl.List(pl.Float32) # Assume Float32, adjust if needed
                             # Use lit(None) with correct dtype to add null column
                             df_merged = df_merged.with_columns(pl.lit(None, dtype=embed_dtype).alias(embed_col_name))
                             logger.info(f"Added missing column '{embed_col_name}' with dtype {embed_dtype} to main dataframe for {year_file}.")
                        else:
                             logger.error(f"Embedding column '{embed_col_name}' in {embed_file} is not a list/array type ({type(sample_value)}). Cannot determine schema to add column.")
                             continue # Skip this file
                    else:
                         logger.error(f"Could not find any non-null embedding data in '{embed_col_name}' of {embed_file} to infer schema. Cannot add column.")
                         continue # Skip this file


                # Use update: Modifies df_merged in place where IDs match
                logger.debug(f"Performing df.update() for column '{embed_col_name}' on {year_file}...")
                df_merged.update(df_embed_subset.drop_nulls(subset=[embed_col_name]), on='id')
                
                # Verify the column was effectively updated (check null count reduction or values)
                if embed_col_name in df_merged.columns:
                    logger.info(f"Successfully updated column '{embed_col_name}' in {year_file} from {embed_file}.")
                    merged_cols_count += 1
                    all_merged_cols.add(embed_col_name)
                else:
                    # This case should ideally not be reached if column adding works
                    logger.error(f"Column '{embed_col_name}' was NOT present after the update operation from {embed_file}.")

            except Exception as e:
                logger.error(f"Failed to process or merge artifact file {embed_file}: {e}", exc_info=True)

        # --- Post-merge for the year ---
        if merged_cols_count > 0 or len(all_merged_cols.intersection(original_schema.keys())) < len(all_merged_cols):
            # Proceed to save/upload if new columns were merged OR
            # if columns that were supposed to be merged were identified (even if skipped because they existed)
            # This handles cases where a re-run processes previously missed models.

            logger.info(f"Finished processing artifacts for {year_file}. Merged/Updated {merged_cols_count} new columns. Total embedding columns identified: {len(all_merged_cols)}.")
            logger.debug(f"Final schema for {year_file}: {df_merged.schema}")

            # Optional: Validation against original schema (as before)
            for col_name, col_type in original_schema.items():
                 if col_name not in df_merged.columns:
                      logger.error(f"Validation Error: Original column '{col_name}' missing in final merged data for {year_file}!")
                 elif df_merged[col_name].dtype != col_type:
                      logger.warning(f"Validation Warning: Original column '{col_name}' dtype changed for {year_file} (original: {col_type}, final: {df_merged[col_name].dtype}).")

            # Save locally before upload
            try:
                logger.info(f"Saving final merged file locally: {local_final_path}")
                df_merged.write_parquet(local_final_path, compression='zstd')

                # Upload the final merged file
                logger.info(f"Uploading final {local_final_path} as {year_file} to Hub repo {repo_id}...")
                upload_file(
                    path_or_fileobj=local_final_path,
                    path_in_repo=year_file, # Overwrite the file in the repo root
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token,
                    commit_message=f"Add/Update embeddings for {year_file} (columns: {', '.join(sorted(list(all_merged_cols)))})"
                )
                logger.info(f"Successfully uploaded updated {year_file}.")

            except Exception as e:
                logger.error(f"Failed to save or upload final {year_file}: {e}")
            finally:
                # Clean up local final file
                if os.path.exists(local_final_path):
                    try:
                         os.remove(local_final_path)
                         logger.debug(f"Cleaned up local final file: {local_final_path}")
                    except OSError:
                         logger.warning(f"Failed to cleanup local final file: {local_final_path}")
        else:
            logger.warning(f"No new embedding columns were merged or updated for {year_file} based on available artifacts. Nothing to upload.")

    logger.info("--- Merge and Upload Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge embedding artifact Parquet files and upload to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Hub repository ID (e.g., lyk/ArxivEmbedding).")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory containing the downloaded matrix output artifacts.")
    # Add other arguments if needed, e.g., --commit-message-prefix

    args = parser.parse_args()

    # Basic logging setup
    logger.add(sys.stderr, format="{time} {level} {message}", filter="__main__", level="INFO")
    logger.add("merge_upload.log", rotation="10 MB", level="DEBUG") # Log debug info to file

    merge_and_upload(args) 