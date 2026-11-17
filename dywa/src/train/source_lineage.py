"""Source fingerprints for this development repository, independent of old paper campaigns."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN_SOURCE_FILES = tuple(sorted(
    str(p.relative_to(ROOT))
    for directory, suffixes in (
        ("dywa", {".py", ".yaml", ".cu", ".cpp", ".h"}),
        ("configs", {".json"}),
        ("scripts", {".py"}),
        ("data/splits", {".json"}),
    )
    for p in (ROOT / directory).rglob("*")
    if p.is_file() and p.suffix in suffixes
))

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
