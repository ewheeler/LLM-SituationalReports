from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from sitrep.storage.json_io import write_json


def write_records(path: Path, records: list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_record(record) for record in records]
    if pd is None:
        return write_json(path.with_suffix(".json"), normalized)
    frame = pd.DataFrame(normalized)
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        return write_json(path.with_suffix(".json"), normalized)


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return payload
        raise ValueError(f"JSON records file must contain a list: {path}")
    if pd is None:
        json_path = path.with_suffix(".json")
        if json_path.exists():
            return load_records(json_path)
        raise RuntimeError("Pandas is required to read non-JSON record files.")
    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def _normalize_record(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported record type: {type(value)!r}")
