import json
import time
import sys
import threading
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional, Union

def json_dumps(obj: Any) -> str:
    """Compact JSON serialization."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def json_dumps_pretty(obj: Any) -> str:
    """Pretty JSON serialization."""
    return json.dumps(obj, ensure_ascii=False, indent=2)

def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    s = max(0, int(seconds))
    mm, ss = divmod(s, 60)
    hh, mm = divmod(mm, 60)
    if hh:
        return f"{hh:d}:{mm:02d}:{ss:02d}"
    return f"{mm:d}:{ss:02d}"

def progress_line(label: str, done: int, total: int, start_ts: float, extra: str = "") -> str:
    """Generate a progress bar string."""
    if total <= 0:
        pct = 1.0
    else:
        pct = min(1.0, max(0.0, done / total))
    bar_len = 28
    filled = int(round(pct * bar_len))
    bar = "=" * filled + "-" * (bar_len - filled)
    elapsed = format_duration(time.monotonic() - start_ts)
    suffix = f" {extra}" if extra else ""
    return f"{label} [{bar}] {done}/{total} {pct*100:5.1f}% elapsed={elapsed}{suffix}"

def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Robustly load concatenated JSON objects from a file."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    records: List[Dict[str, Any]] = []
    while True:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except Exception:
            break
        i = end
        if isinstance(obj, dict):
            records.append(obj)
    return records

def atomic_write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Atomically write records to a JSONL file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json_dumps_pretty(r) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
