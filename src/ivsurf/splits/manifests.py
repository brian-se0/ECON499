"""Split manifest serialization and hashing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import orjson


@dataclass(frozen=True, slots=True)
class WalkforwardSplit:
    """Explicit walk-forward split."""

    split_id: str
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    test_dates: tuple[str, ...]


def serialize_splits(splits: list[WalkforwardSplit], output_path: Path) -> str:
    """Write split manifest and return its SHA256 hash."""

    payload = [asdict(split) for split in splits]
    raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    return sha256(raw).hexdigest()


def load_splits(path: Path) -> list[WalkforwardSplit]:
    """Load a split manifest from disk."""

    payload = orjson.loads(path.read_bytes())
    return [WalkforwardSplit(**item) for item in payload]
