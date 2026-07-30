"""Shared utilities: config loading, seeding, run provenance."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(
        path.read_bytes()).hexdigest()[:16]
    return cfg


def resolve(cfg: Dict[str, Any], key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def package_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "torch", "matplotlib", "yaml"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def write_provenance(out_dir: Path, tag: str, cfg: Dict[str, Any],
                     extra: Dict[str, Any] | None = None) -> Path:
    """Record exactly what produced a result file."""
    record = {
        "tag": tag,
        "seed": cfg["seed"],
        "config_path": cfg.get("_config_path"),
        "config_sha256": cfg.get("_config_sha256"),
        "versions": package_versions(),
        "argv": sys.argv,
    }
    if extra:
        record.update(extra)
    dest = out_dir / f"{tag}.provenance.json"
    dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return dest


def stable_hash(text: str, buckets: int) -> int:
    """Deterministic feature hashing (Python's hash() is salted per process)."""
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % buckets


@dataclass
class RunResult:
    name: str
    ctr: float
    ndcg: float
    crr: float
    ufi: float

    def as_row(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "CTR": round(self.ctr, 4),
            "NDCG": round(self.ndcg, 4),
            "CRR": round(self.crr, 3),
            "UFI": round(self.ufi, 3),
        }
