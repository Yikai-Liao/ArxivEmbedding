# ArxivEmbedding
Create Daily Arxiv Embedding from RSS

## Fetch Arxiv Metadata via OAI-PMH (`fetch_arxiv_oai.py`)

This script fetches paper metadata (ID, title, authors, abstract, date, categories, etc.) from the arXiv OAI-PMH v2.0 interface for specified years and categories.

### Features

*   Fetches metadata based on years provided as command-line arguments.
*   Reads target arXiv categories and OAI parameters (URL, delays, retries) from `config.toml`.
*   Saves the collected metadata into yearly Parquet files (e.g., `data/2023.parquet`) using the Polars library.
*   Supports **incremental updates**: If a Parquet file for a specific year already exists, the script reads it, fetches new records for that year, combines the data, removes duplicates based on the arXiv ID (`id` column), and overwrites the file with the updated dataset.
*   Uses `loguru` for logging.

### Configuration (`config.toml`)

*   `category`: A list of arXiv categories to fetch (e.g., `["cs.AI", "stat.ML"]`).
*   `[ArxivOAI]`: Contains parameters for interacting with the OAI endpoint:
    *   `base_url`: The OAI base URL.
    *   `metadata_prefix`: The metadata format to request (e.g., `oai_dc`).
    *   `request_delay_seconds`: Delay between paginated requests.
    *   `retry_delay_seconds`: Delay before retrying after a 503 error.
    *   `max_retries`: Maximum number of retries for failed requests.
    *   `batch_size`: (Currently unused in the script logic but defined in config).

### Dependencies

Required Python libraries are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

### Usage

Run the script from the root directory of the project, providing the years you want to fetch as arguments:

```bash
python script/fetch_arxiv_oai.py <YEAR1> [YEAR2] ...
```

**Example:**

To fetch or update data for the years 2023 and 2024:

```bash
python script/fetch_arxiv_oai.py 2023 2024
```

The script will process each year sequentially, creating or updating the corresponding Parquet file in the `data/` directory.
