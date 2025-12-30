# src/ingest/humaneval_extract.py
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


def humaneval_id(task_id: str) -> str:
    """
    Convert 'HumanEval/0' -> 'humaneval_0'
    """
    if not isinstance(task_id, str):
        return "humaneval_unknown"
    m = re.search(r"HumanEval/(\d+)", task_id)
    return f"humaneval_{m.group(1)}" if m else f"humaneval_{task_id.replace('/', '_')}"


def extract_docstring_from_prompt(prompt: str) -> Optional[str]:
    """
    Extract the docstring text for the first function in the HumanEval prompt,
    but truncate it at the first doctest marker (>>> or <<<), if present.

    Desired behavior:
      - Take text between triple quotes.
      - Stop at first occurrence of >>> (or <<<) inside that docstring.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    def truncate_at_doctest(doc: str) -> str:
        # Support both >>> (standard doctest) and <<< (user mentioned)
        cut_positions = []
        for marker in (">>>", "<<<"):
            idx = doc.find(marker)
            if idx != -1:
                cut_positions.append(idx)
        if cut_positions:
            doc = doc[: min(cut_positions)]
        return doc.strip()

    # 1) Preferred: AST extraction (robust to formatting), then truncate
    try:
        tree = ast.parse(prompt)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                if doc and doc.strip():
                    return truncate_at_doctest(doc)
    except SyntaxError:
        pass

    # 2) Fallback: regex between triple quotes, then truncate
    m = re.search(r'"""(.*?)"""', prompt, flags=re.DOTALL)
    if m:
        doc = m.group(1)
        doc = truncate_at_doctest(doc)
        return doc if doc else None

    m = re.search(r"'''(.*?)'''", prompt, flags=re.DOTALL)
    if m:
        doc = m.group(1)
        doc = truncate_at_doctest(doc)
        return doc if doc else None

    return None


def extract_asserts_from_test_code(test_code: str) -> List[str]:
    """
    Extract assert statements from HumanEval `test` field.

    The `test` field typically defines:
      def check(candidate):
          assert candidate(...) == ...
          ...

    We parse with AST and return strings like:
      'assert candidate([1,2], 0.3) == True'
    """
    if not isinstance(test_code, str) or not test_code.strip():
        return []

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        # If parsing fails, best-effort regex: lines starting with "assert "
        return [ln.strip() for ln in test_code.splitlines() if ln.strip().startswith("assert ")]

    tests: List[str] = []

    def unparse(node: ast.AST) -> str:
        # Python 3.9+ has ast.unparse
        try:
            return ast.unparse(node).strip()
        except Exception:
            return ""

    # Find `check` function
    check_fn: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_fn = node
            break

    if check_fn is None:
        # fallback: all Assert nodes anywhere
        for n in ast.walk(tree):
            if isinstance(n, ast.Assert):
                s = "assert " + unparse(n.test)
                if s.strip() != "assert":
                    tests.append(s)
        return _dedupe_keep_order(tests)

    # Extract asserts in check(candidate)
    for stmt in check_fn.body:
        if isinstance(stmt, ast.Assert):
            expr = unparse(stmt.test)
            s = f"assert {expr}".strip()
            if s != "assert":
                tests.append(s)

    return _dedupe_keep_order(tests)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_humaneval_reference(
    out_path: str = "data/reference/humaneval_tests.json",
    split: str = "test",
) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Official HumanEval on HF
    ds = load_dataset("openai/openai_humaneval", split=split)  # :contentReference[oaicite:1]{index=1}

    records: List[Dict[str, Any]] = []
    skipped = 0

    for row in ds:
        task_id = row.get("task_id")
        prompt = row.get("prompt", "")
        test_code = row.get("test", "")
        entry_point = row.get("entry_point", "")

        if not task_id:
            skipped += 1
            continue

        description = extract_docstring_from_prompt(prompt) or (prompt.strip() if isinstance(prompt, str) else str(prompt))
        tests = extract_asserts_from_test_code(test_code)

        records.append(
            {
                "id": humaneval_id(task_id),
                "source": "humaneval",
                "task_description": description,
                "function_name": entry_point if isinstance(entry_point, str) else str(entry_point),
                "tests": tests,
            }
        )

    out_file.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} items to {out_file}")
    if skipped:
        print(f"Skipped {skipped} rows (missing task_id).")


if __name__ == "__main__":
    build_humaneval_reference()
