import argparse
import os
import sys
import time
from pathlib import Path
import asyncio # Needed for async handling if using embed's async nature directly

import numpy as np
import polars as pl
import toml
import torch
from loguru import logger
# from sentence_transformers import SentenceTransformer # No longer needed
from embed import BatchedInference # Import BatchedInference
from tqdm import tqdm

# Configure Loguru only if run as main script
# The process_matrix_tasks script will configure its own logger
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
else:
    # Avoid duplicate logging if imported. Rely on the importing script's logger.
    # logger.disable("generate_embeddings") # Or just don't add handlers
    pass

# Removed get_embedding_model function as initialization is different


def get_model_info_from_config(config: dict, model_key: str) -> dict | None:
    """Extracts model configuration for a given key from the loaded TOML config."""
    # Handle potential structures: [[Embedding.model]] or [Embedding.model]
    embedding_section = config.get("Embedding", None)
    if not embedding_section:
        logger.error("Config file missing [Embedding] section.")
        return None

    # Case 1: [[Embedding.model]] -> list of tables
    if isinstance(embedding_section, list):
        for item in embedding_section:
            if isinstance(item, dict) and model_key in item:
                 model_config = item[model_key]
                 if "model_name" in model_config:
                     return model_config
                 else:
                     logger.error(f"Missing 'model_name' for key '{model_key}' in [[Embedding]] list.")
                     return None

    # Case 2: [Embedding.model] -> dictionary where keys are model names
    elif isinstance(embedding_section, dict):
        if model_key in embedding_section:
            model_config = embedding_section[model_key]
            if isinstance(model_config, dict) and "model_name" in model_config:
                 return model_config
            else:
                 logger.error(f"Missing 'model_name' for key '{model_key}' in [Embedding] dict or value is not a table.")
                 return None

    logger.error(f"Model key '{model_key}' not found in [Embedding] section of the config.")
    return None


def generate_embeddings_batch_embed(register: BatchedInference, model_id: str, texts: list[str], batch_size: int, model_key: str):
    """Generates embeddings for a list of texts in batches using embed/infinity.

    Args:
        register: An initialized BatchedInference instance.
        model_id: The model identifier (name or path) known to BatchedInference.
        texts: A list of strings to embed.
        batch_size: The size for chunking requests (internal batching handled by embed).
        model_key: The logical key/name of the model (used for logging).

    Returns:
        A list of numpy arrays (embeddings), potentially containing NaN vectors on error.
        Returns an empty list if input texts is empty.
        Returns None if a critical error occurs during queuing.
    """
    all_embeddings = []
    total_texts = len(texts)
    if total_texts == 0:
        return [] # Return empty list for empty input

    # Use the global logger instance passed from the calling script
    # It should already have the matrix_id context
    logger.info(f"Generating embeddings for {total_texts} texts using model '{model_key}' ({model_id}) via embed library...")

    # embed handles internal batching, but we chunk queuing for progress/memory management
    processing_chunk_size = batch_size # Use provided batch_size for chunking
    logger.info(f"Using processing chunk size: {processing_chunk_size} for queuing tasks.")

    processed_count = 0
    # Process chunk by chunk, waiting for each chunk to complete
    with tqdm(total=total_texts, desc=f"Processing Embeddings ({model_key})", leave=False) as pbar:
        for i in range(0, total_texts, processing_chunk_size):
            batch_texts = texts[i : i + processing_chunk_size]
            if not batch_texts:
                continue

            count = len(batch_texts)
            try:
                # 1. Queue the embedding task for the current chunk
                future = register.embed(sentences=batch_texts, model_id=model_id)

                # 2. Immediately wait for the result of this chunk
                batch_embeddings_list, token_usage = future.result()

                # Ensure embeddings are float32 numpy arrays
                # This might still produce zero-size arrays if embed returns []
                batch_embeddings_np_raw = [np.array(emb, dtype=np.float32) for emb in batch_embeddings_list]

                # Check for zero-sized arrays and inconsistent dimensions
                batch_embeddings_np = []
                embedding_dim = None
                # First pass to determine expected dimension from valid embeddings
                for emb_np in batch_embeddings_np_raw:
                    if emb_np.size > 0:
                        embedding_dim = emb_np.shape[0]
                        break
                # If no valid embeddings found, try from history or default
                if embedding_dim is None:
                    if all_embeddings and all_embeddings[-1] is not None and not np.isnan(all_embeddings[-1]).all():
                        embedding_dim = all_embeddings[-1].shape[0]
                        logger.debug(f"Determined expected dimension {embedding_dim} from previous batches.")
                    else:
                        # Attempt to get from model config or use a reasonable default
                        # This part is tricky with BatchedInference, using 1024 as a common default.
                        # Ideally, get_model_info_from_config could provide this.
                        embedding_dim = 1024 # Default to 1024, adjust if other models used
                        logger.warning(f"Could not determine embedding dimension from batch or history. Using default: {embedding_dim}")

                # Second pass to correct or replace invalid embeddings
                for i, emb_np in enumerate(batch_embeddings_np_raw):
                    current_original_index = pbar.n + i # Index in the original texts list
                    if emb_np.size == 0:
                        logger.warning(f"Detected zero-size embedding for text at original index {current_original_index}. Replacing with NaN vector (dim={embedding_dim}).")
                        nan_vector = np.full((embedding_dim,), np.nan, dtype=np.float32)
                        batch_embeddings_np.append(nan_vector)
                    elif emb_np.shape[0] != embedding_dim:
                        logger.warning(f"Inconsistent embedding dimension detected ({emb_np.shape[0]} vs {embedding_dim}) for text at original index {current_original_index}. Replacing with NaN vector.")
                        nan_vector = np.full((embedding_dim,), np.nan, dtype=np.float32)
                        batch_embeddings_np.append(nan_vector)
                    else:
                        batch_embeddings_np.append(emb_np)

                if len(batch_embeddings_np) != count:
                    logger.warning(f"Mismatch in expected ({count}) vs processed ({len(batch_embeddings_np)}) embeddings for batch starting at original index {pbar.n}. Padding with NaNs.")
                    # Pad with NaN vectors if necessary, using the determined dimension
                    nan_vector = np.full((embedding_dim,), np.nan, dtype=np.float32)
                    while len(batch_embeddings_np) < count:
                        batch_embeddings_np.append(nan_vector)

                all_embeddings.extend(batch_embeddings_np)
                processed_count += count # Count based on input texts for this batch
                pbar.update(count) # Update progress bar by expected count

            except Exception as e:
                logger.error(f"Error processing embedding batch result (starting at original index {pbar.n}): {e}. Filling {count} entries with NaNs.", exc_info=True)
                # Handle errors by adding NaN vectors using determined dimension
                embedding_dim_exc = 1024 # Default
                try:
                    # If previous batches succeeded, try to get dim from there
                    if all_embeddings and all_embeddings[-1] is not None and not np.isnan(all_embeddings[-1]).all():
                        embedding_dim_exc = all_embeddings[-1].shape[0]
                        logger.debug(f"Using embedding dimension {embedding_dim_exc} from previous result for NaN padding in exception.")
                    elif embedding_dim: # Use dim determined earlier if available
                         embedding_dim_exc = embedding_dim
                         logger.debug(f"Using embedding dimension {embedding_dim_exc} determined earlier for NaN padding in exception.")
                except Exception as dim_err:
                    logger.warning(f"Could not get embedding dimension for NaN padding in exception: {dim_err}. Using placeholder {embedding_dim_exc}.")

                nan_vector = np.full((embedding_dim_exc,), np.nan, dtype=np.float32)
                all_embeddings.extend([nan_vector] * count)
                processed_count += count # Still count these as processed (with errors)
                pbar.update(count)


    logger.info(f"Generated {len(all_embeddings)} embedding results (processed {processed_count} texts).")
    # Final check and padding - should ideally not be needed if batch padding works
    if len(all_embeddings) != total_texts:
        logger.error(f"FINAL MISMATCH: Expected {total_texts}, got {len(all_embeddings)}. Check logs for errors. Padding remaining...")
        embedding_dim_final = 1024 # Default
        if embedding_dim: # Use determined dim if available
            embedding_dim_final = embedding_dim
        elif all_embeddings and all_embeddings[-1] is not None and not np.isnan(all_embeddings[-1]).all():
            embedding_dim_final = all_embeddings[-1].shape[0]
        nan_vector = np.full((embedding_dim_final,), np.nan, dtype=np.float32)
        while len(all_embeddings) < total_texts:
            all_embeddings.append(nan_vector)

    return all_embeddings

# Keep the main function for potential standalone execution or testing
# Note: Standalone execution doesn't benefit from the model-level optimization
# of the process_matrix_tasks.py script.
def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text data in a Parquet file using embed/infinity.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the output Parquet file with embeddings.")
    parser.add_argument("--model-key", type=str, required=True, help="Key of the embedding model in the config file (e.g., 'jasper_v1').")
    parser.add_argument("--config-file", type=str, default="config.toml", help="Path to the configuration file.")
    parser.add_argument("--text-column-title", type=str, default="title", help="Name of the title column.")
    parser.add_argument("--text-column-abstract", type=str, default="abstract", help="Name of the abstract column.")
    # Batch size for embed/infinity is handled internally, but we use it for chunking/progress
    parser.add_argument("--batch-size", type=int, default=64, help="Processing chunk size (influences progress updates). Internal batching is handled by embed.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cuda', 'cpu'). Defaults to 'cpu'.")
    # Add engine argument for embed
    parser.add_argument("--engine", type=str, default="torch", choices=["torch", "optimum"], help="Inference engine for embed ('torch' or 'optimum').")


    args = parser.parse_args()

    # Use the main script's logger instance
    log = logger # Already configured if __name__ == "__main__"

    # --- Validate Inputs ---
    if not os.path.exists(args.input_file):
        log.error(f"Input file not found: {args.input_file}")
        return 1
    if not os.path.exists(args.config_file):
        log.error(f"Config file not found: {args.config_file}")
        return 1

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Load Config ---
    try:
        config = toml.load(args.config_file)
        model_info = get_model_info_from_config(config, args.model_key)
        if not model_info:
            log.error(f"Failed to get model info for key '{args.model_key}' from {args.config_file}")
            return 1
        model_name_or_path = model_info["model_name"]

    except Exception as e:
        log.error(f"Error loading or parsing config file {args.config_file}: {e}")
        return 1

    # --- Determine Device ---
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA specified but not available. Falling back to CPU.")
        device = "cpu"
    elif device not in ["cuda", "cpu"]:
        log.warning(f"Invalid device '{device}'. Falling back to CPU.")
        device = "cpu"
    log.info(f"Using device: {device}")
    log.info(f"Using engine: {args.engine}")


    # --- Initialize BatchedInference ---
    register = None # Initialize register to None
    try:
        log.info(f"Initializing BatchedInference for model: {model_name_or_path}")
        register = BatchedInference(
            model_id=[model_name_or_path], # Pass model ID as a list
            engine=args.engine,
            device=device,
            # batch_size can be set here, defaults usually fine
        )
        log.info("BatchedInference initialized successfully.")
    except Exception as e:
        log.error(f"Failed to initialize BatchedInference: {e}")
        return 1 # Exit if initialization fails

    try: # Main processing block with finally for cleanup
        # --- Load Data ---
        try:
            log.info(f"Loading data from {args.input_file}...")
            df = pl.read_parquet(args.input_file)
            log.info(f"Loaded DataFrame with shape: {df.shape}")
            if df.height == 0:
                log.warning("Input DataFrame is empty. Saving an empty output file.")
                # Add the target embedding column with the correct type if possible
                try:
                    # We don't know the embedding dimension easily beforehand with embed
                    # So, just write the empty frame. Downstream merge needs to handle schema.
                    pass
                except Exception:
                     log.warning("Could not determine schema for empty output file.")
                df.write_parquet(args.output_file, compression='zstd')
                return 0
            if args.text_column_title not in df.columns or args.text_column_abstract not in df.columns:
                 log.error(f"Required columns '{args.text_column_title}' or '{args.text_column_abstract}' not found.")
                 return 1
        except Exception as e:
            log.error(f"Failed to load Parquet file {args.input_file}: {e}")
            return 1

        # --- Prepare Text ---
        log.info("Preparing text column for embedding...")
        try:
            df = df.with_columns(
                pl.concat_str(
                    [
                        pl.col(args.text_column_title).fill_null(""),
                        pl.lit("\n"),
                        pl.col(args.text_column_abstract).fill_null("")
                    ],
                    separator=""
                ).alias("_text_to_embed_")
            )
            texts_to_embed = df["_text_to_embed_"].to_list()
            log.info(f"Prepared {len(texts_to_embed)} texts.")
        except Exception as e:
            log.error(f"Error preparing text column: {e}")
            return 1
        finally:
             # Drop intermediate column immediately after use
             if "_text_to_embed_" in df.columns:
                 df = df.drop("_text_to_embed_")


        # --- Generate Embeddings ---
        start_time = time.time()
        try:
            # Use the new function with BatchedInference
            embeddings = generate_embeddings_batch_embed(register, model_name_or_path, texts_to_embed, args.batch_size, args.model_key)
            embedding_series = pl.Series(name=args.model_key, values=embeddings, dtype=pl.List(pl.Float32))

        except Exception as e:
            log.error(f"Failed during embedding generation: {e}")
            return 1
        # No finally needed here as register cleanup is outside

        end_time = time.time()
        log.info(f"Embedding generation took {end_time - start_time:.2f} seconds.")


        # --- Add Embeddings to DataFrame ---
        try:
            if args.model_key in df.columns:
                log.warning(f"Column '{args.model_key}' already exists. It will be replaced.")
                df = df.drop(args.model_key)
            df = df.with_columns(embedding_series)
            log.info(f"Added '{args.model_key}' column. DataFrame shape: {df.shape}")
        except Exception as e:
            log.error(f"Failed to add embedding column to DataFrame: {e}")
            return 1

        # --- Save Result ---
        try:
            log.info(f"Saving DataFrame with embeddings to {args.output_file}...")
            df.write_parquet(args.output_file, compression='zstd')
            log.info("Output file saved successfully.")
        except Exception as e:
            log.error(f"Failed to save output Parquet file {args.output_file}: {e}")
            return 1

        log.info("--- Embedding Generation Script Finished ---")
        return 0

    finally:
        # --- Stop BatchedInference ---
        if register:
            log.info("Stopping BatchedInference...")
            try:
                register.stop()
                log.info("BatchedInference stopped.")
            except Exception as e:
                log.error(f"Error stopping BatchedInference: {e}")


if __name__ == "__main__":
    sys.exit(main())