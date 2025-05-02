import feedparser
import polars as pl
import toml
from pathlib import Path
from loguru import logger
import sys
from datetime import datetime
import time

# Configure Loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- Constants ---
RSS_BASE_URL = "https://rss.arxiv.org/rss/"
CONFIG_FILE = "config.toml"
OUTPUT_DIR = "./data"
OUTPUT_FILE = "current.parquet"
REQUEST_DELAY_SECONDS = 3 # Be polite to the server

def parse_arxiv_entry(entry):
    """Parses a single feedparser entry and extracts metadata consistent with OAI schema."""
    try:
        # Extract arXiv ID from link (e.g., http://arxiv.org/abs/2301.00001v1 -> 2301.00001)
        link = entry.get('link')
        if not link or '/abs/' not in link:
            logger.warning(f"Could not find valid link/ID in entry: {entry.get('id', 'N/A')}")
            return None
        arxiv_id_with_version = link.split('/abs/')[-1]
        arxiv_id = arxiv_id_with_version.split('v')[0] # Remove version if present

        title = entry.get('title', '').replace('\n', ' ').strip()
        authors = [author.get('name') for author in entry.get('authors', []) if author.get('name')]
        abstract = entry.get('summary', '').replace('\n', ' ').strip()

        # Dates
        published_parsed = entry.get('published_parsed')
        updated_parsed = entry.get('updated_parsed')

        # Use updated if available, else published for the main 'date' field
        primary_date_parsed = updated_parsed or published_parsed
        date_str = time.strftime('%Y-%m-%dT%H:%M:%SZ', primary_date_parsed) if primary_date_parsed else None

        created_str = time.strftime('%Y-%m-%dT%H:%M:%SZ', published_parsed) if published_parsed else None
        updated_str = time.strftime('%Y-%m-%dT%H:%M:%SZ', updated_parsed) if updated_parsed else None

        # Categories
        categories = [tag.get('term') for tag in entry.get('tags', []) if tag.get('term')]

        # License (often not available/parsed in RSS, default to None)
        license_url = entry.get('license')

        return {
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "date": date_str, # Primary date (updated or published)
            "categories": categories,
            "created": created_str, # Mapped from published
            "updated": updated_str,
            "license": license_url, # Add license field (often null)
            # Removed fields:
            # "link": link,
            # "published": ... (now mapped to created)
        }
    except Exception as e:
        logger.error(f"Error parsing entry {entry.get('id', 'N/A')}: {e}")
        return None

def main():
    # --- Load Config ---
    config_path = Path(CONFIG_FILE)
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    try:
        config = toml.load(config_path)
        categories = config.get("category", [])
        if not categories:
            logger.error(f"No 'category' list found or empty in {config_path}")
            return 1
        logger.info(f"Loaded categories: {categories}")
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return 1

    # --- Prepare Output Directory ---
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / OUTPUT_FILE
    logger.info(f"Output file: {output_file_path}")

    # --- Fetch and Parse RSS Feeds ---
    all_entries_data = []
    for category in categories:
        rss_url = f"{RSS_BASE_URL}{category}"
        logger.info(f"Fetching RSS feed for category '{category}' from: {rss_url}")
        try:
            feed = feedparser.parse(rss_url)

            if feed.bozo:
                bozo_exception = feed.get('bozo_exception', 'Unknown error')
                logger.warning(f"Feed for {category} may be ill-formed: {bozo_exception}")

            if feed.status != 200:
                 logger.warning(f"Received status {feed.status} for feed {category}. Skipping.")
                 continue # Skip this category if status is not OK

            parsed_count = 0
            for entry in feed.entries:
                parsed_data = parse_arxiv_entry(entry)
                if parsed_data:
                    all_entries_data.append(parsed_data)
                    parsed_count += 1
            logger.info(f"Parsed {parsed_count} entries from {category} feed.")

        except Exception as e:
            logger.error(f"Failed to fetch or parse feed for {category}: {e}")

        # Be polite
        time.sleep(REQUEST_DELAY_SECONDS)


    if not all_entries_data:
        logger.warning("No entries found across all specified category feeds. Exiting.")
        # Optionally, write an empty parquet file or delete the old one
        # For now, just exit.
        return 0

    # --- Convert to Polars DataFrame and Save ---
    logger.info(f"Converting {len(all_entries_data)} total entries to DataFrame...")
    try:
        # Define schema consistent with OAI script output
        schema = {
            "id": pl.Utf8,
            "title": pl.Utf8,
            "authors": pl.List(pl.Utf8),
            "abstract": pl.Utf8,
            "date": pl.Utf8, # Primary date
            "categories": pl.List(pl.Utf8),
            "created": pl.Utf8,
            "updated": pl.Utf8,
            "license": pl.Utf8, # Added license column
            # Removed columns:
            # "link": pl.Utf8,
            # "published": pl.Utf8,
        }
        df = pl.DataFrame(all_entries_data, schema=schema)

        # Deduplicate based on ID, keeping the first occurrence
        df = df.unique(subset=["id"], keep="first", maintain_order=True)
        logger.info(f"DataFrame shape after deduplication: {df.shape}")

        # Sort by ID before saving
        df = df.sort("id")

        logger.info(f"Saving DataFrame to {output_file_path}...")
        df.write_parquet(output_file_path, compression='zstd')
        logger.info(f"Successfully saved {output_file_path}")

    except Exception as e:
        logger.error(f"Error creating or saving DataFrame: {e}")
        return 1

    logger.info("--- RSS Fetch Script finished ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())

