import time
import requests
import polars as pl

from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Any, Iterator
from xml.etree import ElementTree as ET
from loguru import logger
from itertools import chain
from tqdm import tqdm

from .name import *
from .schema import *

def load_arxiv_oai_snapshot(file_path: str | Path) -> pl.DataFrame:
    """Load the arXiv OAI snapshot JSONL file into a Polars DataFrame."""
    df = pl.read_ndjson(file_path, schema=ARXIV_OAI_SNAPSHOT_PL_SCHEMA)
    df = df\
        .unique(subset=ID)\
        .with_columns(
            # Split categories string into list
            pl.col(CATEGORIES).str.split(" ").alias(CATEGORIES),
            # Extract publish date from versions information
            pl.col(VERSIONS)
            .list.get(0)
            .struct.field("created")
            .str.strptime(pl.Date, "%a, %d %b %Y %H:%M:%S GMT")
            .alias(PUBLISH_DATE),
            pl.col(ABSTRACT).str.strip_chars().alias(ABSTRACT),
        )\
        .drop(VERSIONS)\
        .match_to_schema(PAPER_METADATE_PL_SCHEMA)
    # unique does not work with list data type, so we need to split categories after deduplication
    return df

def fetch_arxiv_oai(
    categories: list[str] | None,
    start: date | None,
    end: date | None,
) -> pl.DataFrame:
    client = ArxivOAIClient()
    start_str = start.strftime("%Y-%m-%d") if start else None
    end_str = end.strftime("%Y-%m-%d") if end else None
    if categories is None or len(categories) == 0:
        records = client.list_records(start_str, end_str)
    else:
        records = chain.from_iterable(
            client.list_records(start_str, end_str, category) 
            for category in categories
        )
    df = pl.DataFrame(map(asdict, records),schema=PAPER_METADATE_PL_SCHEMA).unique(subset=ID)
    return df

def format_category_for_oai(category_string: str) -> str:
    """Convert category format like 'cs.AI' or 'stat.ML' to OAI setSpec format.
    
    Examples:
        'cs.AI' -> 'cs:cs:AI'
        'stat.ML' -> 'stat:stat:ML'
        'cs' -> 'cs' (top-level, unchanged)
    """
    parts = category_string.split('.', 1)
    if len(parts) == 2:
        group = parts[0]
        category = parts[1]
        if '-' in category:
            # Format: group:archive (e.g., 'cs:cs-AI')
            return f"{group}:{category}"
        else:
            # Format: group:group:CATEGORY (e.g., 'cs:cs:AI')
            return f"{group}:{group}:{category}"
    else:
        # Assume it's already a valid top-level setSpec
        return category_string
    
    

class ArxivOAIClient:
    """Client for interacting with arXiv OAI-PMH API."""

    NAMESPACES = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "arxiv": "http://arxiv.org/OAI/arXiv/",
    }

    def __init__(
        self,
        base_url: str = "http://export.arxiv.org/oai2",
        metadata_prefix: str = "arXiv",
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.base_url = base_url
        self.metadata_prefix = metadata_prefix
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PaperDigestMono/0.1"})

    def list_records(
        self,
        from_date: str | None = None,
        until_date: str | None = None,
        set_spec: str | None = None,
    ) -> Iterator[ArxivRecord]:
        """
        Fetch records from arXiv OAI-PMH endpoint with resumption token support.

        Args:
            from_date: Start date in YYYY-MM-DD format (optional)
            until_date: End date in YYYY-MM-DD format (optional)
            set_spec: OAI set specification for filtering by category (optional).
                     Can be a formatted setSpec like 'cs:cs:AI' or a category string
                     like 'cs.AI' which will be auto-formatted.

        Yields:
            ArxivRecord objects parsed from OAI responses
        """
        params: dict[str, str] = {
            "verb": "ListRecords",
            "metadataPrefix": self.metadata_prefix,
        }
        if from_date:
            params["from"] = from_date
        if until_date:
            params["until"] = until_date
        if set_spec:
            # Auto-format category strings to OAI setSpec format
            if '.' in set_spec and ':' not in set_spec:
                set_spec = format_category_for_oai(set_spec)
            params["set"] = set_spec
            logger.debug("Using OAI set parameter: {}", set_spec)

        resumption_token: str | None = None
        total_records = 0

        while True:
            if resumption_token:
                params = {"verb": "ListRecords", "resumptionToken": resumption_token}

            response = self._make_request(params)
            if response is None:
                logger.error("Failed to fetch records after retries; stopping iteration")
                break

            try:
                root = ET.fromstring(response.text)
            except ET.ParseError as exc:
                logger.error("Failed to parse OAI-PMH XML response: {}", exc)
                break

            # Extract records
            records = root.findall(".//oai:record", self.NAMESPACES)
            for record_elem in records:
                try:
                    parsed: ArxivRecord | None = self._parse_record(record_elem)
                    if not parsed:
                        continue
                    yield parsed
                    total_records += 1
                except Exception as exc:
                    logger.warning("Failed to parse record: {}", exc)
                    continue

            # Check for resumption token
            token_elem = root.find(".//oai:resumptionToken", self.NAMESPACES)
            if token_elem is not None and token_elem.text:
                resumption_token = token_elem.text.strip()
                logger.debug(
                    "Resumption token found; fetched {} records so far",
                    total_records,
                )
            else:
                logger.info("No more resumption tokens; fetched {} records total", total_records)
                break

    def _make_request(self, params: dict[str, str]) -> requests.Response | None:
        """Make HTTP request with retries."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                logger.warning(
                    "Request failed (attempt {}/{}): {}",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries reached; request failed")
                    return None
        return None

    def _parse_record(self, record_elem: ET.Element) -> ArxivRecord | None:
        """Parse a single OAI record element into ArxivRecord."""
        # Check if record is deleted
        header = record_elem.find("oai:header", self.NAMESPACES)
        if header is not None and header.get("status") == "deleted":
            return None

        metadata = record_elem.find(".//arxiv:arXiv", self.NAMESPACES)
        if metadata is None:
            return None

        # Extract identifier
        identifier = metadata.findtext("arxiv:id", namespaces=self.NAMESPACES)
        if not identifier:
            return None

        # Extract fields
        title = metadata.findtext("arxiv:title", namespaces=self.NAMESPACES, default="").strip()
        abstract = metadata.findtext("arxiv:abstract", namespaces=self.NAMESPACES, default="").strip()

        # Parse categories
        categories_elem = metadata.find("arxiv:categories", self.NAMESPACES)
        categories = categories_elem.text.split() if categories_elem is not None and categories_elem.text else []
        # Parse authors
        authors_elems = metadata.findall("arxiv:authors/arxiv:author", self.NAMESPACES)
        authors = []
        for author_elem in authors_elems:
            keyname = author_elem.findtext("arxiv:keyname", namespaces=self.NAMESPACES, default="")
            forenames = author_elem.findtext("arxiv:forenames", namespaces=self.NAMESPACES, default="")
            full_name = f"{forenames} {keyname}".strip()
            if full_name:
                authors.append(full_name)
        authors=", ".join(authors)

        # Parse dates
        created = metadata.findtext("arxiv:created", namespaces=self.NAMESPACES, default="")
        updated = metadata.findtext("arxiv:updated", namespaces=self.NAMESPACES, default="")
        
        created = datetime.strptime(created, "%Y-%m-%d").date()
        updated = datetime.strptime(updated, "%Y-%m-%d").date()

        # Optional fields
        doi = metadata.findtext("arxiv:doi", namespaces=self.NAMESPACES)
        comments = metadata.findtext("arxiv:comments", namespaces=self.NAMESPACES)
        license_elem = metadata.find("arxiv:license", self.NAMESPACES)
        license_value: str | None = None
        if license_elem is not None:
            text_value = (license_elem.text or "").strip()
            if text_value:
                license_value = text_value
            else:
                license_value = license_elem.get("uri")

        return ArxivRecord(
            id=identifier,
            title=title,
            abstract=abstract,
            categories=categories,
            authors=authors,
            publish_date=created,
            update_date=updated,
            comments=comments,
            doi=doi,
            license=license_value,
        )