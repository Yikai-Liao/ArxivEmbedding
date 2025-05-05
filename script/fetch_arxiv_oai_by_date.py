#!/usr/bin/env python
import requests
import xmltodict
import polars as pl
import toml
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from collections.abc import Iterable
import sys
from datetime import datetime, timedelta

# --- Global Variables (Defaults, will be overridden by config) ---
OAI_CONFIG = {
    "base_url": "https://oaipmh.arxiv.org/oai",
    "metadata_prefix": "oai_dc",
    "request_delay_seconds": 10,
    "retry_delay_seconds": 60,
    "max_retries": 5,
    "batch_size": 1000
}

# Configure Loguru
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def format_category_for_oai(category_string):
    """Converts category format like 'cs.AI' or 'stat.ML' to OAI setSpec 'cs:cs:AI' or 'stat:stat:ML'."""
    parts = category_string.split('.', 1)
    if len(parts) == 2:
        group = parts[0]
        archive_and_cat = parts[1]
        if '-' in archive_and_cat:
             return f"{group}:{archive_and_cat}"  # Format: group:archive
        else:
             return f"{group}:{group}:{archive_and_cat}"  # Format: group:group:CATEGORY
    else:
        return category_string  # Assume it's already a valid top-level setSpec

# --- Helper Functions ---

def parse_record(record):
    """Parses a single OAI record (dict) using the arXiv metadata format."""
    # Adjust the path to the metadata based on the arXiv format structure
    metadata = record.get("metadata", {}).get("arXiv", {})
    if not metadata:
        logger.warning(f"Could not find 'arXiv' metadata in record: {record.get('header', {}).get('identifier')}")
        return None

    arxiv_id = metadata.get("id")
    if not arxiv_id:
        logger.warning(f"Could not extract arXiv ID from metadata: {metadata}")
        return None

    title = metadata.get("title", '').replace('\n', ' ').strip()
    abstract = metadata.get("abstract", '').replace('\n', ' ').strip()
    created_date = metadata.get("created")
    updated_date = metadata.get("updated")
    license_url = metadata.get("license")

    # Parse authors
    authors_list = []
    authors_data = metadata.get("authors", {}).get("author")
    if authors_data:
        # Ensure it's a list even if there's only one author
        if not isinstance(authors_data, list):
            authors_data = [authors_data]
        for author in authors_data:
            if isinstance(author, dict):
                keyname = author.get("keyname", "")
                forenames = author.get("forenames", "")
                # Combine forenames and keyname for a full name string
                full_name = f"{forenames} {keyname}".strip()
                if full_name:
                    authors_list.append(full_name)
            elif isinstance(author, str):  # Handle potential simpler structures
                 authors_list.append(author)

    # Categories are space-separated in the arXiv format
    categories_str = metadata.get("categories", "")
    categories_list = categories_str.split() if categories_str else []

    # Use updated_date as the primary date if available, otherwise created_date
    primary_date = updated_date or created_date

    return {
        "id": arxiv_id,
        "title": title,
        "authors": authors_list,
        "abstract": abstract,
        "date": primary_date,  # Using updated/created date
        "categories": categories_list,
        # Add new fields from arXiv format
        "created": created_date,
        "updated": updated_date,
        "license": license_url,
    }


def fetch_records(from_date, until_date, category=None, resumption_token=None):
    """Fetches a batch of records from the OAI endpoint for a specific date range."""
    params = {
        "verb": "ListRecords",
    }
    if resumption_token:
        params["resumptionToken"] = resumption_token
    else:
        params["metadataPrefix"] = OAI_CONFIG["metadata_prefix"]
        params["from"] = from_date
        params["until"] = until_date
        
        if category:
            # Format the category correctly for the 'set' parameter
            set_spec = format_category_for_oai(category)
            params["set"] = set_spec
            logger.info(f"Using setSpec: {set_spec} for category: {category}")

    retries = 0
    while retries < OAI_CONFIG["max_retries"]:
        try:
            response = requests.get(OAI_CONFIG["base_url"], params=params, timeout=60)  # Increased timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            data = xmltodict.parse(response.content)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 503:
                retry_delay = OAI_CONFIG["retry_delay_seconds"]
                logger.warning(f"Received 503 error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retries += 1
            else:
                # For other request errors, maybe retry once or break
                logger.error("Non-503 request error. Aborting this batch.")
                return None  # Indicate failure
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            logger.error(f"Response content: {response.content[:500]}...")
            return None  # Indicate failure

    logger.error(f"Max retries reached for params: {params}. Skipping this batch.")
    return None


def fetch_all_for_date_range_category(from_date, until_date, category):
    """Fetches all records for a specific date range and category, handling pagination."""
    all_records_data = []
    resumption_token = None
    processed_count = 0

    pbar_desc = f"Fetching {from_date} to {until_date} [{category}]"
    pbar = tqdm(desc=pbar_desc, unit=" records")

    while True:
        data = fetch_records(from_date, until_date, category, resumption_token)

        if data is None:  # Handle fetch/parse failure
             logger.error(f"Failed to fetch data for {from_date}-{until_date}/{category} with token {resumption_token}. Skipping.")
             break  # Stop processing this date range/category combination

        oai_response = data.get("OAI-PMH", {})
        error = oai_response.get("error")
        if error:
            error_code = error.get('@code')
            error_message = error.get('#text')
            logger.warning(f"OAI Error for {from_date}-{until_date}/{category}: {error_code} - {error_message}")
            # Specific handling for 'noRecordsMatch'
            if error_code == 'noRecordsMatch':
                logger.info(f"No records found for {from_date}-{until_date}/{category}.")
            else:
                # Log other errors but continue if possible, maybe break depending on error
                logger.error(f"Unexpected OAI error encountered for {from_date}-{until_date}/{category}. Stopping fetch for this set.")
            break  # Stop processing this specific date range/category

        list_records = oai_response.get("ListRecords")
        if not list_records:
            logger.warning(f"No 'ListRecords' found in response for {from_date}-{until_date}/{category}. Response structure might be unexpected.")
            break  # Stop if the structure is wrong

        records = list_records.get("record")
        if records:
            # Ensure records is always a list, even if only one record is returned
            if not isinstance(records, list):
                records = [records]

            batch_parsed_count = 0
            for record in records:
                parsed = parse_record(record)
                if parsed:
                    all_records_data.append(parsed)
                    batch_parsed_count += 1

            processed_count += batch_parsed_count
            pbar.update(batch_parsed_count)
            pbar.set_postfix({"Total Fetched": processed_count})

        # Check for resumption token
        resumption_token_data = list_records.get("resumptionToken")
        if resumption_token_data and resumption_token_data.get('#text'):
            resumption_token = resumption_token_data.get('#text')
            total_records_estimate = resumption_token_data.get('@completeListSize')
            if total_records_estimate:
                 pbar.total = int(total_records_estimate)  # Update progress bar total if available
                 pbar.refresh()  # Refresh to show the new total
            request_delay = OAI_CONFIG["request_delay_seconds"]
            logger.debug(f"Pausing for {request_delay} seconds...")
            time.sleep(request_delay)  # Be polite
        else:
            logger.info(f"No more resumption token found for {from_date}-{until_date}/{category}. Fetch complete.")
            break  # End of records for this set

    pbar.close()
    return all_records_data

# --- Main Execution ---

def main():
    global OAI_CONFIG  # Allow modification of the global config dict
    parser = argparse.ArgumentParser(description="Fetch arXiv metadata via OAI-PMH for a specific date range and save as Parquet.")
    parser.add_argument("--from-date", type=str, 
                        help="Start date (YYYY-MM-DD format). Defaults to 2 days ago if not specified.")
    parser.add_argument("--until-date", type=str, 
                        help="End date (YYYY-MM-DD format). Defaults to today if not specified.")
    parser.add_argument("-c", "--config", type=str, default="config.toml",
                        help="Path to the configuration file (default: config.toml)")
    parser.add_argument("-o", "--output-file", type=str, default="data/incremental.parquet",
                        help="Path to save the output Parquet file (default: data/incremental.parquet)")
    parser.add_argument("--days", type=int, default=2,
                        help="Number of days to fetch if --from-date is not specified (default: 2)")

    args = parser.parse_args()

    # --- Handle date arguments ---
    today = datetime.now().date()
    
    if args.until_date:
        until_date = datetime.strptime(args.until_date, "%Y-%m-%d").date()
    else:
        until_date = today
        
    if args.from_date:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    else:
        from_date = until_date - timedelta(days=args.days)
    
    # Format dates for OAI-PMH
    from_date_str = from_date.strftime("%Y-%m-%d")
    until_date_str = until_date.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching records from {from_date_str} to {until_date_str}")

    # --- Load Config ---
    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        return 1  # Exit with error

    try:
        config = toml.load(config_path)
        categories = config.get("category", [])
        if not categories:
            logger.error(f"No 'category' list found or empty in {config_path}")
            return 1
        logger.info(f"Loaded categories: {categories}")

        # Load ArxivOAI settings, using defaults if not present
        arxiv_oai_settings = config.get("ArxivOAI", {})
        OAI_CONFIG["base_url"] = arxiv_oai_settings.get("base_url", OAI_CONFIG["base_url"])
        OAI_CONFIG["metadata_prefix"] = arxiv_oai_settings.get("metadata_prefix", OAI_CONFIG["metadata_prefix"])
        OAI_CONFIG["request_delay_seconds"] = int(arxiv_oai_settings.get("request_delay_seconds", OAI_CONFIG["request_delay_seconds"]))
        OAI_CONFIG["retry_delay_seconds"] = int(arxiv_oai_settings.get("retry_delay_seconds", OAI_CONFIG["retry_delay_seconds"]))
        OAI_CONFIG["max_retries"] = int(arxiv_oai_settings.get("max_retries", OAI_CONFIG["max_retries"]))
        OAI_CONFIG["batch_size"] = int(arxiv_oai_settings.get("batch_size", OAI_CONFIG["batch_size"]))
        logger.info(f"Loaded ArxivOAI config: {OAI_CONFIG}")

    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return 1

    # --- Prepare Output Directory ---
    output_file = Path(args.output_file)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output file will be saved to: {output_file}")

    # --- Fetch Data for Date Range ---
    logger.info(f"--- Processing date range: {from_date_str} to {until_date_str} ---")
    all_data = []
    for category in categories:
        logger.info(f"Fetching category: {category} for date range {from_date_str} to {until_date_str}")
        category_data = fetch_all_for_date_range_category(from_date_str, until_date_str, category)
        if category_data:
            all_data.extend(category_data)
        logger.info(f"Finished fetching {category}. Total records so far: {len(all_data)}")
        # Optional: Add a small delay between categories as well if needed
        time.sleep(1)

    if not all_data:
        logger.warning(f"No records found for date range {from_date_str} to {until_date_str} across all specified categories.")
        # Create empty Parquet file to indicate the process ran
        try:
            schema = {
                "id": pl.Utf8,
                "title": pl.Utf8,
                "authors": pl.List(pl.Utf8),
                "abstract": pl.Utf8,
                "date": pl.Utf8,
                "categories": pl.List(pl.Utf8),
                "created": pl.Utf8,
                "updated": pl.Utf8,
                "license": pl.Utf8,
            }
            empty_df = pl.DataFrame([], schema=schema)
            empty_df.write_parquet(output_file, compression='zstd')
            logger.info(f"Saved empty DataFrame to {output_file}")
        except Exception as e:
            logger.error(f"Error creating empty DataFrame: {e}")
        return 0

    # --- Convert to Polars DataFrame and Save ---
    logger.info(f"Converting {len(all_data)} records to DataFrame...")
    try:
        # Define schema for consistency, reflecting arXiv format
        schema = {
            "id": pl.Utf8,
            "title": pl.Utf8,
            "authors": pl.List(pl.Utf8),
            "abstract": pl.Utf8,
            "date": pl.Utf8,  # Represents updated/created date
            "categories": pl.List(pl.Utf8),
            "created": pl.Utf8,
            "updated": pl.Utf8,
            "license": pl.Utf8,
        }
        new_df = pl.DataFrame(all_data, schema=schema)

        # Optional: Deduplicate based on ID, keeping the first occurrence
        # Sort by 'updated' date descending before deduplicating to keep the most recent metadata if IDs clash
        final_df = new_df.sort("updated", descending=True, nulls_last=True).unique(subset=["id"], keep="first")
        logger.info(f"DataFrame shape after deduplication: {final_df.shape}")

        # Sort final output by ID
        final_df = final_df.sort("id")

        logger.info(f"Saving final DataFrame to {output_file}...")
        final_df.write_parquet(output_file, compression='zstd')
        logger.info(f"Successfully saved {output_file}")

    except Exception as e:
        logger.error(f"Error creating or saving DataFrame: {e}")
        return 1

    logger.info("--- Script finished ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())