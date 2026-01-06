from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.private_functions.private_functions import PRIVATE_FUNCTIONS


MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def build_prompt(entry: Dict[str, Any]) -> str:
    return (
        "Write unit tests for the Python function below.\n\n"
        "Specification:\n"
        f"{entry['task_description']}\n\n"
        "Rules:\n"
        "- Output ONLY lines of the form: assert <expression>\n"
        "- One assert per line\n"
        "- Do NOT include comments, blank lines, or function definitions\n"
        "- Output between 4 and 8 asserts\n\n"
        "Function:\n"
        f"{entry['code']}\n"
    )


def extract_asserts(text: str) -> List[str]:
    """
    Best-effort extraction: keep lines starting with 'assert '.
    If the model outputs a full test file, this still works.
    """
    asserts = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("assert "):
            asserts.append(s)
    seen = set()
    out = []
    for a in asserts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def generate_one(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 300,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)


def main(
    out_dir: str = "data/generated_tests/llama",
    temperatures=(0.2, 0.5, 0.8),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_per_temperature = {
        0.2: 2,
        0.5: 3,
        0.8: 3,
    }

    private_entries = PRIVATE_FUNCTIONS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        use_fast=False,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    for temp in temperatures:
        out_file = out_dir / f"LLaMA2-7B-Chat-Temperature-{temp}.jsonl"

        num_samples = samples_per_temperature.get(float(temp))
        if num_samples is None:
            raise ValueError(f"No samples_per_task defined for temperature {temp}")

        with out_file.open("w", encoding="utf-8") as f:
            for entry in private_entries:
                prompt = build_prompt(entry)

                for sample_id in range(num_samples):
                    raw = generate_one(model, tokenizer, prompt, temperature=float(temp))
                    tests = extract_asserts(raw)[:8] # in case LLM generates too many tests

                    record = {
                        "private_id": entry["id"],
                        "model": MODEL_NAME,
                        "temperature": float(temp),
                        "sample_id": sample_id,
                        "prompt": prompt,
                        "task_description": entry["task_description"],
                        "function_name": entry["function_name"],
                        "tests": tests,
                        "raw_output": raw,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

        print(f"Wrote: {out_file}")

    # Create pretty-printed snapshots for easier inspection
    for temp in temperatures:
        records = []
        in_path = out_dir / f"LLaMA2-7B-Chat-Temperature-{temp}.jsonl"
        out_path = out_dir / f"LLaMA2-7B-Chat-Temperature-{temp}-pretty.json"

        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote pretty JSON: {out_path}")


if __name__ == "__main__":
    main()
