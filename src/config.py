
"""Shared configuration primitives for the papersys package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar
import tomllib

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base config with strict validation rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_toml(cls: "type[C]", path: Path) -> "C":
        """Construct the config object from a TOML file."""
        data = _load_toml(path)
        return cls.model_validate(data)


C = TypeVar("C", bound="BaseConfig")


def load_config(config_cls: type[C], path: Path) -> C:
    """Parse the given TOML file into the provided config class."""
    return config_cls.from_toml(path)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Root TOML value must be a table")
    return data


__all__ = ["BaseConfig", "load_config"]



class MetaDataConfig(BaseConfig):
    hf_repo: str
    row_group: int = 200_000
    
class EmbeddingConfig(BaseConfig):
    hf_repo: str
    row_group: int = 200_000
    model: str
    dim: int
    start_date: str | None = None
    categories: list[str] | None = None


class AppConfig(BaseConfig):
    metadata: MetaDataConfig
    embedding: EmbeddingConfig