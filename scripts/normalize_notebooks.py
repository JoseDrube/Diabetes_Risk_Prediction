#!/usr/bin/env python3
"""
Normalize Jupyter notebook outputs for nbformat/GitHub compatibility.

Fixes:
- Stream outputs: add "name": "stdout" when missing
- execute_result: add "metadata": {} and "execution_count" when missing
- display_data: add "metadata": {} when missing
- error: remove "metadata" (not allowed by schema)

Run manually:
  python scripts/normalize_notebooks.py
  python scripts/normalize_notebooks.py notebooks/01_eda.ipynb

Pre-commit: runs automatically on staged .ipynb files.
"""

from pathlib import Path
import json
import sys


def normalize_outputs(outputs: list) -> bool:
    """Apply nbformat-required fixes to outputs. Returns True if any change was made."""
    changed = False
    for out in outputs:
        if not isinstance(out, dict):
            continue
        ot = out.get("output_type")
        if ot == "stream":
            if "name" not in out:
                out["name"] = "stdout"
                changed = True
        elif ot == "execute_result":
            if "metadata" not in out:
                out["metadata"] = {}
                changed = True
            if "execution_count" not in out:
                out["execution_count"] = None
                changed = True
        elif ot == "display_data":
            if "metadata" not in out:
                out["metadata"] = {}
                changed = True
        elif ot == "error":
            if "metadata" in out:
                del out["metadata"]
                changed = True
    return changed


def normalize_notebook(path: Path) -> bool:
    """Normalize one notebook file. Returns True if file was modified."""
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return False
        nb = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Skip (invalid/empty): {path} — {e}", file=sys.stderr)
        return False
    changed = False
    for cell in nb.get("cells", []):
        outputs = cell.get("outputs", [])
        if normalize_outputs(outputs):
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
    return changed


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        paths = [Path(p).resolve() for p in sys.argv[1:]]
    else:
        paths = list((root / "notebooks").glob("*.ipynb"))
    if not paths:
        return 0
    modified = 0
    for path in paths:
        if not path.exists():
            print(f"Skip (not found): {path}", file=sys.stderr)
            continue
        if path.suffix != ".ipynb":
            print(f"Skip (not .ipynb): {path}", file=sys.stderr)
            continue
        if normalize_notebook(path):
            modified += 1
            print(f"Normalized: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
