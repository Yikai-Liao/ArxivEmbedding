import pyarrow as pa
import polars as pl

from typing import Dict
from polars.datatypes import Struct, String, List, Date
from dataclasses import dataclass
from datetime import date

from .name import *

PAPER_METADATA_SCHEMA = pa.schema(
    [
        pa.field(ID, pa.string()),
        pa.field(TITLE, pa.string()),
        pa.field(ABSTRACT, pa.string()),
        pa.field(PUBLISH_DATE, pa.date32()),
        pa.field(UPDATE_DATE, pa.date32()),
        pa.field(CATEGORIES, pa.list_(pa.string()), nullable=True),
        pa.field(AUTHORS, pa.string()),
        pa.field(COMMENTS, pa.string(), nullable=True),
        pa.field(DOI, pa.string(), nullable=True),
        pa.field(LICENSE, pa.string(), nullable=True),
    ]
)

PAPER_METADATE_PL_SCHEMA: Dict[str, pl.DataType] = dict([
    (ID, String),
    (TITLE, String),
    (ABSTRACT, String),
    (PUBLISH_DATE, Date),
    (UPDATE_DATE, Date),
    (CATEGORIES, List(String)),
    (AUTHORS, String),
    (COMMENTS, String),
    (DOI, String),
    (LICENSE, String),
])


ARXIV_OAI_SNAPSHOT_PL_SCHEMA: Dict[str, pl.DataType] = dict([
    (ID, String),
    (AUTHORS, String),
    (TITLE, String),
    (COMMENTS, String),
    (DOI, String),
    (CATEGORIES, String),
    (LICENSE, String),
    (ABSTRACT, String),
    (UPDATE_DATE, Date),
    (VERSIONS, List(Struct({'version': String, 'created': String}))),
])

@dataclass(frozen=True)
class ArxivRecord:
    """Parsed arXiv paper metadata from OAI-PMH response."""
    id: str
    title: str
    abstract: str
    publish_date: date
    update_date: date
    categories: list[str]
    authors: str
    comments: str | None = None
    doi: str | None = None
    license: str | None = None
    
    def __hash__(self) -> int:
        """Hash based only on id for set deduplication."""
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        """Equality based only on id for set deduplication."""
        if not isinstance(other, ArxivRecord):
            return NotImplemented
        return self.id == other.id
    
def paper_embedding_schema(dim: int) -> pa.Schema:
    """Generate the schema for the embedding table."""
    return pa.schema(
        [
            pa.field(ID, pa.string()),
            pa.field(EMBEDDING_VECTOR, pa.list_(pa.float16(), dim)),
        ]
    )
 