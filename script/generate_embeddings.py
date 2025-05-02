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

# Configure Loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Removed get_embedding_model function as initialization is different

def generate_embeddings_batch_embed(register: BatchedInference, model_id: str, texts: list[str], batch_size: int, model_key: str):
    """Generates embeddings for a list of texts in batches using embed/infinity."""
    all_embeddings = []
    total_texts = len(texts)
    logger.info(f"Generating embeddings for {total_texts} texts using model '{model_key}' ({model_id}) via embed library...")

    # embed handles batching internally, but we might still process in chunks
    # to manage memory or track progress more granularly if needed.
    # However, the simplest way is to send all texts at once.
    # Let's process in larger chunks to show progress with tqdm.
    # Infinity/embed's internal batching will optimize GPU usage.
    # We define a processing chunk size, not the inference batch size.
    processing_chunk_size = batch_size * 10 # Process 10x the inference batch size for tqdm updates

    futures = []
    for i in tqdm(range(0, total_texts, processing_chunk_size), desc=f"Queueing Embeddings ({model_key})"):
        batch_texts = texts[i : i + processing_chunk_size]
        if not batch_texts:
            continue
        # Queue the embedding task, returns a Future
        future = register.embed(sentences=batch_texts, model_id=model_id)
        futures.append((future, len(batch_texts))) # Store future and expected count

    logger.info(f"Queued {len(futures)} embedding tasks. Waiting for results...")

    # Wait for results and collect them
    # Use asyncio.gather if running in an async context, otherwise iterate and call .result()
    processed_count = 0
    with tqdm(total=total_texts, desc=f"Processing Embeddings ({model_key})") as pbar:
        for future, count in futures:
            try:
                # future.result() blocks until the result is ready
                batch_embeddings_list, token_usage = future.result()
                # Ensure embeddings are float32 numpy arrays
                batch_embeddings_np = [np.array(emb, dtype=np.float32) for emb in batch_embeddings_list]

                if len(batch_embeddings_np) != count:
                     logger.warning(f"Mismatch in expected ({count}) vs received ({len(batch_embeddings_np)}) embeddings for a batch. Padding with NaNs.")
                     # Pad with NaN vectors if necessary
                     if batch_embeddings_np:
                         embedding_dim = batch_embeddings_np[0].shape[0]
                     else:
                         # Need to get dim from model if first batch failed completely
                         # This part is tricky without access to the model object directly
                         # Let's assume a default or skip padding if dim unknown
                         embedding_dim = 768 # Placeholder, adjust if possible
                         logger.warning(f"Could not determine embedding dimension for padding. Using placeholder {embedding_dim}.")

                     nan_vector = np.full((embedding_dim,), np.nan, dtype=np.float32)
                     while len(batch_embeddings_np) < count:
                         batch_embeddings_np.append(nan_vector)

                all_embeddings.extend(batch_embeddings_np)
                processed_count += len(batch_embeddings_np) # Use actual received count
                pbar.update(count) # Update progress bar by expected count

            except Exception as e:
                logger.error(f"Error processing embedding batch: {e}. Filling {count} entries with NaNs.")
                # Handle errors by adding NaN vectors
                # Need embedding dimension here too
                embedding_dim = 768 # Placeholder
                try:
                    # Attempt to get dim if model info is accessible, otherwise use placeholder
                    pass # register object doesn't expose model details easily
                except Exception:
                    pass
                nan_vector = np.full((embedding_dim,), np.nan, dtype=np.float32)
                all_embeddings.extend([nan_vector] * count)
                processed_count += count
                pbar.update(count)


    logger.info(f"Generated {len(all_embeddings)} embeddings (processed {processed_count} texts).")
    # Final check, though padding should handle mismatches
    if len(all_embeddings) != total_texts:
         logger.error(f"FINAL MISMATCH: Expected {total_texts}, got {len(all_embeddings)}. Check logs for errors.")
         # Pad remaining if necessary
         embedding_dim = 768 # Placeholder
         nan_vector = np.full((embedding_dim,), np.nan, dtype=np.float32)
         while len(all_embeddings) < total_texts:
             all_embeddings.append(nan_vector)


    return all_embeddings


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

    # --- Validate Inputs ---
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    if not os.path.exists(args.config_file):
        logger.error(f"Config file not found: {args.config_file}")
        return 1

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Load Config ---
    try:
        config = toml.load(args.config_file)
        embedding_configs = {}
        for section in config.get("Embedding", []):
            if args.model_key in section:
                embedding_configs = section[args.model_key]
                break

        if not embedding_configs:
             logger.error(f"Model key '{args.model_key}' not found under any [[...]] section in [Embedding] in {args.config_file}")
             return 1

        model_name_or_path = embedding_configs.get("model_name")
        # trust_remote_code = embedding_configs.get("trust_remote_code", False) # embed might handle this via HF download config

        if not model_name_or_path:
             logger.error(f"Missing 'model_name' for model key '{args.model_key}' in config.")
             return 1
    except Exception as e:
        logger.error(f"Error loading or parsing config file {args.config_file}: {e}")
        return 1

    # --- Determine Device ---
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA specified but not available. Falling back to CPU.")
        device = "cpu"
    elif device not in ["cuda", "cpu"]:
        logger.warning(f"Invalid device '{device}'. Falling back to CPU.")
        device = "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Using engine: {args.engine}")


    # --- Initialize BatchedInference ---
    register = None # Initialize register to None
    try:
        logger.info(f"Initializing BatchedInference for model: {model_name_or_path}")
        register = BatchedInference(
            model_id=[model_name_or_path], # Pass model ID as a list
            engine=args.engine,
            device=device,
            # batch_size can be set here, defaults usually fine
        )
        logger.info("BatchedInference initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize BatchedInference: {e}")
        return 1 # Exit if initialization fails

    try: # Main processing block with finally for cleanup
        # --- Load Data ---
        try:
            logger.info(f"Loading data from {args.input_file}...")
            df = pl.read_parquet(args.input_file)
            logger.info(f"Loaded DataFrame with shape: {df.shape}")
            if df.height == 0:
                logger.warning("Input DataFrame is empty. Saving an empty output file.")
                # Add the target embedding column with the correct type if possible
                try:
                    # We don't know the embedding dimension easily beforehand with embed
                    # So, just write the empty frame. Downstream merge needs to handle schema.
                    pass
                except Exception:
                     logger.warning("Could not determine schema for empty output file.")
                df.write_parquet(args.output_file, compression='zstd')
                return 0
            if args.text_column_title not in df.columns or args.text_column_abstract not in df.columns:
                 logger.error(f"Required columns '{args.text_column_title}' or '{args.text_column_abstract}' not found.")
                 return 1
        except Exception as e:
            logger.error(f"Failed to load Parquet file {args.input_file}: {e}")
            return 1

        # --- Prepare Text ---
        logger.info("Preparing text column for embedding...")
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
            logger.info(f"Prepared {len(texts_to_embed)} texts.")
        except Exception as e:
            logger.error(f"Error preparing text column: {e}")
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
            logger.error(f"Failed during embedding generation: {e}")
            return 1
        # No finally needed here as register cleanup is outside

        end_time = time.time()
        logger.info(f"Embedding generation took {end_time - start_time:.2f} seconds.")


        # --- Add Embeddings to DataFrame ---
        try:
            if args.model_key in df.columns:
                logger.warning(f"Column '{args.model_key}' already exists. It will be replaced.")
                df = df.drop(args.model_key)
            df = df.with_columns(embedding_series)
            logger.info(f"Added '{args.model_key}' column. DataFrame shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to add embedding column to DataFrame: {e}")
            return 1

        # --- Save Result ---
        try:
            logger.info(f"Saving DataFrame with embeddings to {args.output_file}...")
            df.write_parquet(args.output_file, compression='zstd')
            logger.info("Output file saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save output Parquet file {args.output_file}: {e}")
            return 1

        logger.info("--- Embedding Generation Script Finished ---")
        return 0

    finally:
        # --- Stop BatchedInference ---
        if register:
            logger.info("Stopping BatchedInference...")
            try:
                register.stop()
                logger.info("BatchedInference stopped.")
            except Exception as e:
                logger.error(f"Error stopping BatchedInference: {e}")


if __name__ == "__main__":
    sys.exit(main())