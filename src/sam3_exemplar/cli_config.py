"""YAML config and shared CLI helpers for offline tools."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml


def load_config_file(path: str | pathlib.Path) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if p.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a .yaml or .yml file, got: {p.suffix}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_by_dotted_key(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def coerce_str_list(value: Any) -> list[str]:
    """Normalize a config list field that may arrive as a bare string via ``--set``.

    ``list("path.yaml")`` iterates characters; wrap strings into a one-element list.
    Also accepts a shell-stripped bracket list string such as ``[a.yaml, b.yaml,]``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        bracket = _parse_bracket_str_list(value)
        if bracket is not None:
            return bracket
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise TypeError(f"Expected str or list of str, got {type(value).__name__}")


def _parse_bracket_str_list(val: str) -> list[str] | None:
    """Parse ``[a, b,]`` / ``["a", "b"]`` when shell quoting stripped JSON quotes.

    Returns ``None`` if ``val`` is not a bracket-wrapped list.
    """
    s = val.strip()
    if len(s) < 2 or s[0] != "[" or s[-1] != "]":
        return None
    inner = s[1:-1].strip()
    if inner.endswith(","):
        inner = inner[:-1].rstrip()
    if not inner:
        return []
    items: list[str] = []
    for part in inner.split(","):
        item = part.strip().strip("\"'")
        if item:
            items.append(item)
    return items


def parse_kv_override(raw: str) -> Tuple[str, Any]:
    """Parse ``key=value`` for ``--set`` overrides (bool/None/JSON/int/float/str).

    List values may be JSON (``'["a.yaml"]'``, prefer single-quoting the whole
    ``--set`` argument) or a shell-friendly ``[a.yaml,]`` form after bash strips
    inner double quotes.
    """
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--set expects key=value")
    key, val = raw.split("=", 1)
    lowered = val.lower()
    if lowered in {"true", "false"}:
        coerced: Any = lowered == "true"
    elif lowered in {"none", "null"}:
        coerced = None
    else:
        try:
            coerced = json.loads(val)
        except json.JSONDecodeError:
            bracket = _parse_bracket_str_list(val)
            if bracket is not None:
                coerced = bracket
            else:
                try:
                    coerced = int(val)
                except ValueError:
                    try:
                        coerced = float(val)
                    except ValueError:
                        coerced = val
    return key, coerced


def write_cli_invocation(
    output_dir: pathlib.Path,
    args: argparse.Namespace,
    argv: Optional[Sequence[str]] = None,
    *,
    script: str,
    log_prefix: str = "",
) -> None:
    """Write ``run_invocation.json`` for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_path = output_dir / "run_invocation.json"
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    payload = {
        "script": script,
        "argv": argv_list,
        "args": vars(args),
    }
    with cli_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    prefix = f"{log_prefix} " if log_prefix else ""
    print(f"{prefix}Wrote CLI invocation to {cli_path}")


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return text[:max_len].strip("-") or "run"


def unique_run_dir(
    base_dir: str | Path,
    description: str = "",
    ts: dt.datetime | None = None,
) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = ts or dt.datetime.now()
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    suffix = f"_{slugify(description)}" if description else ""
    run_dir = base / f"{stamp}{suffix}"
    if run_dir.exists():
        i = 2
        while True:
            alt = base / f"{stamp}{suffix}_{i}"
            if not alt.exists():
                run_dir = alt
                break
            i += 1
    run_dir.mkdir()
    return run_dir
