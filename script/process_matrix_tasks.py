#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
import toml
import torch
from embed import BatchedInference
from huggingface_hub import hf_hub_download
from loguru import logger
from tqdm import tqdm

# Import the embedding function from the refactored script
# We will assume generate_embeddings.py is refactored to have this function
from generate_embeddings import generate_embeddings_batch_embed, get_model_info_from_config

# Configure Loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | Matrix-{extra[matrix_id]} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


def process_model_year_batch(
    register: BatchedInference,
    model_key: str,
    model_name_or_path: str,
    year_file: str,
    tasks_for_year_model: List[Dict],
    repo_id: str,
    output_dir: Path,
    batch_size: int,
    text_column_title: str = "title",
    text_column_abstract: str = "abstract",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Downloads a year file, filters it, generates embeddings for a specific model,
    and saves the result.
    """
    processed_tasks_batch = []
    failed_tasks_batch = []
    ids_to_process = set(task["id"] for task in tasks_for_year_model)
    log = logger.bind(model_key=model_key, year_file=year_file)

    log.info(f"Processing {len(ids_to_process)} IDs for model '{model_key}' in file '{year_file}'")

    file_path = None
    df_filtered = None
    try:
        # --- Download Year File ---
        log.info(f"Downloading {year_file}...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=year_file,
            repo_type="dataset",
            local_dir=".", # Download to current dir
            local_dir_use_symlinks=False,
        )
        log.info(f"Downloaded to {file_path}")

        # --- Load and Filter Data ---
        log.info(f"Loading and filtering {year_file} for {len(ids_to_process)} IDs...")
        df_original = pl.read_parquet(file_path)
        df_filtered = df_original.filter(pl.col("id").is_in(ids_to_process))
        log.info(f"Filtered DataFrame shape: {df_filtered.shape} (Original: {df_original.shape})")

        if df_filtered.height == 0:
            log.warning(f"No matching IDs found in {year_file}. Skipping embedding generation.")
            # Mark tasks as failed because data wasn't found where expected
            for task in tasks_for_year_model:
                 failed_tasks_batch.append({**task, "error": "ID not found in downloaded year file"})
            return processed_tasks_batch, failed_tasks_batch

        # --- Prepare Text ---
        log.info("Preparing text column...")
        # Ensure required columns exist
        if text_column_title not in df_filtered.columns or text_column_abstract not in df_filtered.columns:
             missing_cols = [col for col in [text_column_title, text_column_abstract] if col not in df_filtered.columns]
             error_msg = f"Missing required text columns {missing_cols} in {year_file}"
             log.error(error_msg)
             for task in tasks_for_year_model:
                 failed_tasks_batch.append({**task, "error": error_msg})
             return processed_tasks_batch, failed_tasks_batch

        df_filtered = df_filtered.with_columns(
            pl.concat_str(
                [
                    pl.col(text_column_title).fill_null(""),
                    pl.lit("\n"), # Use newline separator
                    pl.col(text_column_abstract).fill_null(""),
                ],
                separator="",
            ).alias("_text_to_embed_")
        )
        texts_to_embed = df_filtered["_text_to_embed_"].to_list()
        log.info(f"Prepared {len(texts_to_embed)} texts for embedding.")

        # --- Generate Embeddings ---
        start_time = time.time()
        log.info(f"Generating embeddings using model '{model_key}' ({model_name_or_path})...")
        # Assuming generate_embeddings_batch_embed is adapted for direct call
        # It should handle potential errors internally and return list of embeddings (or None/NaN vectors)
        embeddings = generate_embeddings_batch_embed(
            register=register,
            model_id=model_name_or_path,
            texts=texts_to_embed,
            batch_size=batch_size, # Pass batch size for chunking inside
            model_key=model_key    # For logging inside the function
        )
        end_time = time.time()
        log.info(f"Embedding generation took {end_time - start_time:.2f} seconds.")

        if not embeddings or len(embeddings) != df_filtered.height:
            error_msg = f"Embedding generation failed or returned incorrect number of results (expected {df_filtered.height}, got {len(embeddings) if embeddings else 0})."
            log.error(error_msg)
            # Mark all tasks for this batch as failed
            for task in tasks_for_year_model:
                 failed_tasks_batch.append({**task, "error": error_msg})
            return processed_tasks_batch, failed_tasks_batch

        # --- Add Embeddings to DataFrame ---
        log.info(f"Adding embeddings column '{model_key}'...")
        embedding_series = pl.Series(name=model_key, values=embeddings, dtype=pl.List(pl.Float32))
        df_output = df_filtered.select(["id"]).with_columns(embedding_series) # Select only id and the new embedding column

        # --- Save Result ---
        output_filename = output_dir / f"embedded-{year_file}-{model_key}.parquet"
        log.info(f"Saving results to {output_filename}...")
        df_output.write_parquet(output_filename, compression='zstd')
        log.info("Results saved successfully.")

        # Mark tasks as processed
        for task in tasks_for_year_model:
             # Only include tasks whose IDs were actually in df_output (sanity check)
             if task["id"] in df_output["id"].to_list():
                 processed_tasks_batch.append({**task, "output_file": str(output_filename)})
             else:
                 log.warning(f"ID {task['id']} was expected but not found in the final output for {output_filename}. Marking as failed.")
                 failed_tasks_batch.append({**task, "error": "ID missing in final embedding output"})


    except Exception as e:
        error_msg = f"Error processing batch for model '{model_key}', year '{year_file}': {e}"
        log.error(error_msg, exc_info=True)
        # Mark all tasks intended for this batch as failed
        for task in tasks_for_year_model:
             failed_tasks_batch.append({**task, "error": error_msg})

    finally:
        # Clean up downloaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                log.debug(f"Cleaned up downloaded file: {file_path}")
            except OSError as e:
                log.warning(f"Could not remove downloaded file {file_path}: {e}")

    return processed_tasks_batch, failed_tasks_batch


def main():
    parser = argparse.ArgumentParser(description="Process embedding tasks for a specific matrix.")
    parser.add_argument("--matrix-id", type=int, required=True, help="ID of the matrix being processed.")
    parser.add_argument("--task-file", type=str, required=True, help="Path to the JSON file containing tasks for this matrix.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output Parquet files.")
    parser.add_argument("--config-file", type=str, default="config.toml", help="Path to the configuration file.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repository ID for data.")
    parser.add_argument("--batch-size", type=int, default=64, help="Processing chunk size for embedding generation.")
    parser.add_argument("--engine", type=str, default="torch", choices=["torch", "optimum"], help="Inference engine for embed.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference ('cuda', 'cpu').")
    parser.add_argument("--text-column-title", type=str, default="title", help="Name of the title column.")
    parser.add_argument("--text-column-abstract", type=str, default="abstract", help="Name of the abstract column.")

    args = parser.parse_args()

    # Bind matrix_id to logger context for all subsequent logs
    log = logger.bind(matrix_id=args.matrix_id)

    log.info(f"Starting processing for Matrix ID: {args.matrix_id}")
    log.info(f"Task File: {args.task_file}")
    log.info(f"Output Directory: {args.output_dir}")
    log.info(f"Config File: {args.config_file}")
    log.info(f"Repo ID: {args.repo_id}")
    log.info(f"Device: {args.device}, Engine: {args.engine}, Batch Size: {args.batch_size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- Load Tasks ---
    try:
        with open(args.task_file, 'r') as f:
            tasks = json.load(f)
        log.info(f"Loaded {len(tasks)} tasks from {args.task_file}")
        if not tasks:
            log.warning("Task file is empty. No processing needed.")
            # Create empty summary files
            with open(output_dir / "processed_tasks.json", "w") as f: json.dump([], f)
            with open(output_dir / "failed_tasks.json", "w") as f: json.dump([], f)
            with open(output_dir / "summary.txt", "w") as f: f.write(f"Matrix {args.matrix_id} Summary
No tasks found.
")
            return 0
    except Exception as e:
        log.error(f"Failed to load tasks from {args.task_file}: {e}")
        # Cannot proceed without tasks
        return 1

    # --- Group Tasks ---
    # Group by model first, then by year file for efficient processing
    tasks_by_model_then_year = defaultdict(lambda: defaultdict(list))
    all_model_keys = set()
    for task in tasks:
        model_key = task["model_key"]
        year_file = task["year_file"]
        tasks_by_model_then_year[model_key][year_file].append(task)
        all_model_keys.add(model_key)

    log.info(f"Tasks grouped into {len(tasks_by_model_then_year)} models.")

    # --- Load Config ---
    try:
        config = toml.load(args.config_file)
        # We need a way to get model_name for a given model_key
        model_configs = {}
        for key in all_model_keys:
             model_info = get_model_info_from_config(config, key)
             if model_info:
                 model_configs[key] = model_info
             else:
                 log.error(f"Model key '{key}' not found or missing 'model_name' in config file '{args.config_file}'.")
                 # Decide how to handle: fail matrix or skip model? Let's skip model for now.
    except Exception as e:
        log.error(f"Error loading or parsing config file {args.config_file}: {e}")
        return 1 # Config is essential

    # --- Determine Device ---
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA specified but not available. Falling back to CPU.")
        device = "cpu"
    elif device not in ["cuda", "cpu"]:
        log.warning(f"Invalid device '{device}'. Falling back to CPU.")
        device = "cpu"
    log.info(f"Final device selection: {device}")

    processed_tasks_all = []
    failed_tasks_all = []

    # --- Process by Model ---
    for model_key, tasks_by_year in tqdm(tasks_by_model_then_year.items(), desc="Processing Models"):
        log_model = log.bind(model_key=model_key) # Bind model_key for this loop

        if model_key not in model_configs:
            log_model.error(f"Skipping model '{model_key}' due to missing configuration.")
            # Mark all tasks for this model as failed
            for year_file, year_tasks in tasks_by_year.items():
                for task in year_tasks:
                    failed_tasks_all.append({**task, "error": f"Configuration missing for model {model_key}"})
            continue

        model_name_or_path = model_configs[model_key]["model_name"]
        log_model.info(f"Initializing model '{model_key}' ({model_name_or_path}) on {device} using {args.engine} engine...")

        register = None
        try:
            # Initialize BatchedInference for the current model
            register = BatchedInference(
                model_id=[model_name_or_path],
                engine=args.engine,
                device=device,
                # Consider passing batch_size from args if needed by BatchedInference init
            )
            log_model.info("BatchedInference initialized successfully.")

            # Process year files for this model
            for year_file, tasks_for_year_model in tqdm(tasks_by_year.items(), desc=f"Years for {model_key}", leave=False):
                processed_batch, failed_batch = process_model_year_batch(
                    register=register,
                    model_key=model_key,
                    model_name_or_path=model_name_or_path,
                    year_file=year_file,
                    tasks_for_year_model=tasks_for_year_model,
                    repo_id=args.repo_id,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    text_column_title=args.text_column_title,
                    text_column_abstract=args.text_column_abstract,
                )
                processed_tasks_all.extend(processed_batch)
                failed_tasks_all.extend(failed_batch)

        except Exception as e:
            log_model.error(f"Failed to initialize or run BatchedInference for model '{model_key}': {e}", exc_info=True)
            # Mark all remaining tasks for this model as failed
            for year_file, year_tasks in tasks_by_year.items():
                # Avoid double-adding if some years were already processed and failed
                processed_ids = {p['id'] for p in processed_tasks_all if p['model_key'] == model_key and p['year_file'] == year_file}
                failed_ids = {f['id'] for f in failed_tasks_all if f['model_key'] == model_key and f['year_file'] == year_file}
                for task in year_tasks:
                    if task['id'] not in processed_ids and task['id'] not in failed_ids:
                        failed_tasks_all.append({**task, "error": f"BatchedInference init/run failed: {e}"})
        finally:
            # Stop BatchedInference instance for the model
            if register:
                log_model.info("Stopping BatchedInference...")
                try:
                    register.stop()
                    log_model.info("BatchedInference stopped.")
                except Exception as e:
                    log_model.error(f"Error stopping BatchedInference: {e}")

    # --- Final Summary ---
    total_attempted = len(tasks)
    total_processed = len(processed_tasks_all)
    total_failed = len(failed_tasks_all)
    success_percent = (total_processed / total_attempted * 100) if total_attempted > 0 else 0

    log.info(f"Matrix processing finished. Attempted: {total_attempted}, Succeeded: {total_processed}, Failed: {total_failed} ({success_percent:.1f}%)")

    # Save summary files
    log.info("Saving summary files...")
    try:
        with open(output_dir / "processed_tasks.json", "w") as f:
            json.dump(processed_tasks_all, f, indent=2)
        with open(output_dir / "failed_tasks.json", "w") as f:
            json.dump(failed_tasks_all, f, indent=2)

        with open(output_dir / "summary.txt", "w") as f:
            f.write(f"Matrix {args.matrix_id} Summary\n")
            f.write("=====================\n")
            f.write(f"Total Tasks Assigned: {total_attempted}\n")
            f.write(f"Successfully Processed: {total_processed} ({success_percent:.1f}%)\n")
            f.write(f"Failed: {total_failed}\n\n")

            if failed_tasks_all:
                failure_counts = defaultdict(int)
                for task in failed_tasks_all:
                    # Simple grouping by error message prefix
                    error_prefix = task.get("error", "Unknown error")[:100]
                    failure_counts[error_prefix] += 1
                f.write("Failure Reasons (Top 10):\n")
                for reason, count in sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
                    f.write(f"- [{count} times] {reason}...\n")

                f.write("\nFirst 10 Failed Task Details:\n")
                for i, task in enumerate(failed_tasks_all[:10]):
                    f.write(f"{i+1}. ID: {task.get('id')}, Model: {task.get('model_key')}, File: {task.get('year_file')}, Error: {task.get('error', 'N/A')}\n")
    except Exception as e:
        log.error(f"Failed to write summary files: {e}")

    log.info("--- Matrix Processing Script Finished ---")

    # Return non-zero exit code if there were failures
    return 1 if total_failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main()) 