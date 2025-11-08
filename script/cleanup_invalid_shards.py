#!/usr/bin/env python3
"""Clean up invalid shard files from HuggingFace Hub.

This script deletes files like embedding_None.parquet that were created
due to bugs with null publish_year values.
"""
from huggingface_hub import HfApi
from loguru import logger
import sys


def cleanup_invalid_shards():
    """Delete invalid shard files from HuggingFace."""
    api = HfApi()
    
    # Files to delete
    repos_to_clean = [
        {
            "repo_id": "lyk/ArxivEmbedding",
            "files": ["embedding_None.parquet"],
        },
        {
            "repo_id": "lyk/ArxivMetaData", 
            "files": ["metadata_None.parquet"],
        },
    ]
    
    for repo_info in repos_to_clean:
        repo_id = repo_info["repo_id"]
        files = repo_info["files"]
        
        logger.info(f"Checking repository: {repo_id}")
        
        # List all files in repo
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            logger.info(f"Found {len(repo_files)} files in {repo_id}")
        except Exception as e:
            logger.error(f"Failed to list files in {repo_id}: {e}")
            continue
        
        # Delete invalid files
        for invalid_file in files:
            if invalid_file in repo_files:
                logger.warning(f"Found invalid file: {invalid_file}, deleting...")
                try:
                    api.delete_file(
                        path_in_repo=invalid_file,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Remove invalid shard file: {invalid_file}",
                    )
                    logger.success(f"Deleted {invalid_file} from {repo_id}")
                except Exception as e:
                    logger.error(f"Failed to delete {invalid_file}: {e}")
            else:
                logger.info(f"Invalid file not found: {invalid_file} (already cleaned?)")
    
    logger.success("Cleanup completed!")


if __name__ == "__main__":
    cleanup_invalid_shards()
