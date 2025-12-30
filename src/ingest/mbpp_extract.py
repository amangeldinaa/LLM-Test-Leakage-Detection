from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from datasets import load_dataset


PARQUET_API = "https://datasets-server.huggingface.co/parquet"


def extract_function_name_from_code(code_str: str) -> str:
    if not isinstance(code_str, str) or not code_str.strip():
        return ""
    try:
        tree = ast.parse(code_str)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        pass
    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", code_str, flags=re.MULTILINE)
    return m.group(1) if m else ""


def normalize_tests(tests: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in tests or []:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def get_parquet_urls(dataset: str, config: str, split: str) -> List[str]:
    """
    Uses the dataset viewer /parquet endpoint to get Parquet URLs.
    Docs: https://huggingface.co/docs/dataset-viewer/en/parquet
    """
    r = requests.get(PARQUET_API, params={"dataset": dataset}, timeout=60)
    r.raise_for_status()
    data = r.json()

    urls = []
    for f in data.get("parquet_files", []):
        if f.get("config") == config and f.get("split") == split:
            url = f.get("url")
            if url:
                urls.append(url)
    if not urls:
        available = sorted({(f.get("config"), f.get("split")) for f in data.get("parquet_files", [])})
        raise RuntimeError(
            f"No parquet files found for dataset={dataset}, config={config}, split={split}. "
            f"Available (config, split): {available}"
        )
    return urls


def build_mbpp_reference(
    out_path: str = "data/reference/mbpp_tests.json",
    config: str = "full",    
    split: str = "test",
) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = "Muennighoff/mbpp"

    parquet_urls = get_parquet_urls(dataset_name, config=config, split=split)

    ds = load_dataset("parquet", data_files=parquet_urls, split="train")

    records: List[Dict[str, Any]] = []
    skipped = 0

    for row in ds:
        task_id = row.get("task_id")
        text = row.get("text")
        code = row.get("code", "")

        if task_id is None or text is None:
            skipped += 1
            continue

        tests = normalize_tests((row.get("test_list") or []) + (row.get("challenge_test_list") or []))
        function_name = extract_function_name_from_code(code)

        records.append(
            {
                "id": f"mbpp_{task_id}",
                "source": "mbpp",
                "task_description": text.strip() if isinstance(text, str) else str(text),
                "function_name": function_name,
                "tests": tests,
            }
        )

    out_file.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} items to {out_file}")
    if skipped:
        print(f"Skipped {skipped} rows (missing task_id/text).")


if __name__ == "__main__":
    build_mbpp_reference()
