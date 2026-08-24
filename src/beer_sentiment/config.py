"""Configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AppConfig:
    """Typed view over the YAML configuration files."""

    own_brands: list[str]
    competitor_brands: list[str]
    brand_aliases: dict[str, list[str]]
    negative_keywords: list[str]
    association_keywords: list[str]
    colors: dict[str, str]
    excel: dict[str, Any]
    time: dict[str, Any]
    stage1: dict[str, Any]
    stage2: dict[str, Any]
    models: dict[str, dict[str, Any]]
    default_model: str
    config_dir: Path

    def all_brands(self) -> list[str]:
        return self.own_brands + self.competitor_brands

    def model_config(self, name: str) -> dict[str, Any]:
        if name not in self.models:
            raise KeyError(
                f"未配置模型 {name!r}，可用：{', '.join(sorted(self.models))}"
            )
        return self.models[name]

    def digest(self) -> str:
        payload = {
            "own_brands": self.own_brands,
            "competitor_brands": self.competitor_brands,
            "brand_aliases": self.brand_aliases,
            "negative_keywords": self.negative_keywords,
            "association_keywords": self.association_keywords,
            "stage1": self.stage1,
            "stage2": self.stage2,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误：{path}")
    return data


def load_config(config_dir: str | Path | None = None) -> AppConfig:
    base = Path(config_dir) if config_dir else PROJECT_ROOT / "config"
    brands = _read_yaml(base / "brands.yaml")
    keywords = _read_yaml(base / "keywords.yaml")
    pipeline = _read_yaml(base / "pipeline.yaml")
    models = _read_yaml(base / "models.yaml")
    return AppConfig(
        own_brands=brands["own_brands"],
        competitor_brands=brands["competitor_brands"],
        brand_aliases=brands.get("aliases", {}),
        negative_keywords=keywords["negative_keywords"],
        association_keywords=keywords["association_keywords"],
        colors=pipeline["colors"],
        excel=pipeline["excel"],
        time=pipeline["time"],
        stage1=pipeline.get("stage1", {}),
        stage2=pipeline.get("stage2", {}),
        models=models["models"],
        default_model=models.get("default", "mock"),
        config_dir=base,
    )
