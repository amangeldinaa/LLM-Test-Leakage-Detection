from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# ----------------------------
# Similarity helpers (N-gram overlap only)
# ----------------------------

def _normalize_assert(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = _normalize_assert(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}

def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def ngram_similarity(a: str, b: str, n: int = 3) -> float:
    # Jaccard over character n-gram sets
    return jaccard(char_ngrams(a, n=n), char_ngrams(b, n=n))

# ----------------------------
# Data loading
# ----------------------------

def load_reference(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Reference file must be a JSON list: {path}")
    return data

def iter_generated_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on {path}:{line_no}: {e}") from e

# ----------------------------
# Evaluation (N-gram only)
# ----------------------------

@dataclass
class MatchScore:
    ref_id: str
    max_ngram: float

def best_match_against_reference(
    gen_tests: Sequence[str],
    ref_entry: Dict[str, Any],
    ngram_n: int,
) -> MatchScore:
    ref_id = str(ref_entry.get("id", "ref_unknown"))
    ref_tests = ref_entry.get("tests", []) or []
    ref_tests = [t for t in ref_tests if isinstance(t, str)]

    best_ng = 0.0

    for ga in gen_tests:
        ga = _normalize_assert(ga)
        if not ga.startswith("assert "):
            continue
        for ra in ref_tests:
            ra = _normalize_assert(ra)
            if not ra.startswith("assert "):
                continue
            ng = ngram_similarity(ga, ra, n=ngram_n)
            if ng > best_ng:
                best_ng = ng

    return MatchScore(ref_id=ref_id, max_ngram=best_ng)

def evaluate_one_generated_record(
    gen_rec: Dict[str, Any],
    reference: List[Dict[str, Any]],
    reference_name: str,
    ngram_n: int,
    thr_ngram: float,
    top_k: int = 3,
) -> Dict[str, Any]:
    gen_tests = gen_rec.get("tests", []) or []
    gen_tests = [t for t in gen_tests if isinstance(t, str)]
    gen_tests = [_normalize_assert(t) for t in gen_tests if t.strip().startswith("assert ")]
    n_gen = len(gen_tests)

    if n_gen == 0:
        return {
            "model": gen_rec.get("model"),
            "temperature": gen_rec.get("temperature"),
            "reference_dataset": reference_name,
            "private_id": gen_rec.get("private_id"),
            "sample_id": gen_rec.get("sample_id"),
            "n_generated_asserts": 0,
            "max_ngram_score": 0.0,
            "max_ngram_ref_id": None,
            "leaked": False,
            "leak_reason": "no_asserts",
            "top_matches": [],
        }

    matches: List[MatchScore] = []
    for ref in reference:
        matches.append(best_match_against_reference(gen_tests, ref, ngram_n=ngram_n))

    matches_sorted = sorted(matches, key=lambda m: m.max_ngram, reverse=True)
    top_matches = matches_sorted[:top_k]

    max_ng = max((m.max_ngram for m in matches), default=0.0)
    max_ng_ref = top_matches[0].ref_id if top_matches else None

    leaked = max_ng >= thr_ngram
    reason = f"ngram>={thr_ngram}" if leaked else "below_threshold"

    return {
        "model": gen_rec.get("model"),
        "temperature": float(gen_rec.get("temperature")) if gen_rec.get("temperature") is not None else None,
        "reference_dataset": reference_name,
        "private_id": gen_rec.get("private_id"),
        "sample_id": gen_rec.get("sample_id"),
        "n_generated_asserts": n_gen,
        "max_ngram_score": round(max_ng, 6),
        "max_ngram_ref_id": max_ng_ref,
        "leaked": bool(leaked),
        "leak_reason": reason,
        "top_matches": [
            {"ref_id": tm.ref_id, "ngram": round(tm.max_ngram, 6)}
            for tm in top_matches
        ],
    }

# ----------------------------
# Output writers
# ----------------------------

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_flat_csv(path: Path, detailed_records: List[Dict[str, Any]], top_k: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "model",
        "temperature",
        "reference_dataset",
        "private_id",
        "sample_id",
        "n_generated_asserts",
        "max_ngram_score",
        "max_ngram_ref_id",
        "leaked",
        "leak_reason",
    ]
    top_cols = []
    for i in range(1, top_k + 1):
        top_cols += [f"top{i}_ref_id", f"top{i}_ngram"]

    cols = base_cols + top_cols

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in detailed_records:
            row = {k: r.get(k) for k in base_cols}
            tm = r.get("top_matches") or []
            for i in range(top_k):
                if i < len(tm):
                    row[f"top{i+1}_ref_id"] = tm[i].get("ref_id")
                    row[f"top{i+1}_ngram"] = tm[i].get("ngram")
                else:
                    row[f"top{i+1}_ref_id"] = None
                    row[f"top{i+1}_ngram"] = None
            w.writerow(row)

def append_summary_row(
    summary_path: Path,
    model: str,
    temperature: float,
    reference_dataset: str,
    total_samples: int,
    leaked_samples: int,
    mean_max_ngram: float,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_path.exists()

    cols = [
        "model",
        "temperature",
        "reference_dataset",
        "total_samples",
        "leaked_samples",
        "leakage_rate",
        "mean_max_ngram",
    ]

    leakage_rate = (leaked_samples / total_samples) if total_samples else 0.0

    with summary_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow(
            {
                "model": model,
                "temperature": temperature,
                "reference_dataset": reference_dataset,
                "total_samples": total_samples,
                "leaked_samples": leaked_samples,
                "leakage_rate": round(leakage_rate, 6),
                "mean_max_ngram": round(mean_max_ngram, 6),
            }
        )

# ----------------------------
# Runner
# ----------------------------

def main(
    generated_root: str = "data/generated_tests",
    reference_root: str = "data/reference",
    results_root: str = "results",
    ngram_n: int = 4,
    thr_ngram: float = 0.65,
    top_k: int = 3,
) -> None:
    generated_root_p = Path(generated_root)
    reference_root_p = Path(reference_root)
    results_root_p = Path(results_root)

    ref_files = {
        "mbpp": reference_root_p / "mbpp_tests.json",
        "humaneval": reference_root_p / "humaneval_tests.json",
    }
    references = {name: load_reference(path) for name, path in ref_files.items()}

    summary_path = results_root_p / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    for model_dir in sorted([p for p in generated_root_p.iterdir() if p.is_dir()]):
        jsonl_files = sorted(model_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue

        for gen_path in jsonl_files:
            gen_stem = gen_path.stem
            gen_records = list(iter_generated_jsonl(gen_path))
            if not gen_records:
                continue

            model_name = str(gen_records[0].get("model", model_dir.name))
            temperature = float(gen_records[0].get("temperature", 0.0))

            for ref_name, ref_entries in references.items():
                detailed: List[Dict[str, Any]] = []
                leaked_count = 0
                sum_max_ng = 0.0

                for rec in gen_records:
                    res = evaluate_one_generated_record(
                        rec,
                        ref_entries,
                        reference_name=ref_name,
                        ngram_n=ngram_n,
                        thr_ngram=thr_ngram,
                        top_k=top_k,
                    )
                    detailed.append(res)
                    leaked_count += 1 if res["leaked"] else 0
                    sum_max_ng += float(res["max_ngram_score"])

                total = len(detailed)
                mean_ng = sum_max_ng / total if total else 0.0

                out_dir = results_root_p / model_dir.name
                out_jsonl = out_dir / f"{gen_stem}_vs_{ref_name}.jsonl"
                out_csv = out_dir / f"{gen_stem}_vs_{ref_name}.csv"

                write_jsonl(out_jsonl, detailed)
                write_flat_csv(out_csv, detailed, top_k=top_k)

                append_summary_row(
                    summary_path=summary_path,
                    model=model_name,
                    temperature=temperature,
                    reference_dataset=ref_name,
                    total_samples=total,
                    leaked_samples=leaked_count,
                    mean_max_ngram=mean_ng,
                )

                print(
                    f"[{model_dir.name}] {gen_stem} vs {ref_name}: "
                    f"{leaked_count}/{total} leaked "
                    f"(thr_ng={thr_ngram}) "
                    f"-> wrote {out_jsonl.name}, {out_csv.name}"
                )

if __name__ == "__main__":
    main()
