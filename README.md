# Data Leakage Analysis for LLM-Generated Unit Tests

This project studies **data leakage in Large Language Models (LLMs)** when generating **Python unit tests**. We analyze whether generated test cases overlap with known benchmark datasets using **character n-gram overlap similarity**.

---

## Project Overview

**Goal:**  
Determine whether LLM-generated test cases show signs of leakage from benchmark datasets such as **MBPP** and **HumanEval**.

**Core idea:**  
Compare generated `assert` statements against reference test cases using **character n-gram overlap similarity** (Jaccard similarity over n-grams).

---

## Project Structure

```
├── data/
│ ├── private_functions/ # Private target functions for test generation
│ ├── generated_tests/ # Model-generated test cases (JSONL)
│ │ ├── qwen/
│ │ └── starcoder/
│ └── reference/ # Reference datasets (MBPP, HumanEval)
│
├── src/
│ ├── generate/ # Test generation scripts
│ └── evaluate/ # Leakage evaluation scripts
│
├── notebooks/
│ └── analysis.ipynb # Plotting and result analysis
│
├── results/
│ ├── summary.csv # Aggregate leakage statistics
│ ├── figures/ # Generated plots
│ └── <model_name>/ # Detailed per-model evaluation outputs
│
├── requirements.txt
└── README.md
```

---

## Datasets

### Reference Datasets
- **MBPP** — Mostly Basic Python Problems
- **HumanEval** — Python function correctness benchmark

Each reference entry contains a function specification and a small set of unit tests.

### Generated Data
LLMs generate unit tests (`assert` statements only) for a set of private functions.
Outputs are stored in JSONL format with metadata including:
- model name
- temperature
- sample ID
- generated assertions

---

## Models

Currently supported models:
- **Qwen2.5-Coder-1.5B-Instruct**
- **StarCoder2-3B**

---

## Similarity Metric

### Character n-gram Jaccard Similarity

The project uses **character-level n-gram overlap** with Jaccard similarity:

- Each `assert` statement is decomposed into overlapping character n-grams
- Similarity is computed as: |A ∩ B| / |A ∪ B|
- The maximum similarity across all assert-pairs between generated and reference test sets is used
- A configurable threshold determines whether a generated test is flagged as leaked

---

## Pipeline Overview

### 1. Installation 
Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Reference datasets are preprocessed by running the following scripts:

```bash
python src/ingest/humaneval_extract.py
python src/ingest/mbpp_extract.py
```

Private functions are written manually to ensure no overlap with reference datasets.

### 3. Test Generation
Run model-specific generation scripts:

```bash
python src/generate/qwen_generate_tests.py
python src/generate/starcoder_generate_tests.py
python src/generate/llama_generate_tests.py
```

Each script:
- samples multiple outputs per function
- varies decoding temperature
- saves results as JSONL

### 4. Leakage Evaluation

Run the evaluation script:

```bash
python src/evaluate/evaluate_leakage.py
```

This produces:
- per-sample detailed results (JSONL, CSV)
- an aggregate summary.csv

### 5. Analysis and Visualization

Open the notebook:

```bash
notebooks/analysis.ipynb
```

The notebook:
- loads summary.csv
- produces plots by model, temperature, and reference dataset
- saves figures to results/figures/

## Outputs

```bash
summary.csv
```

Contains aggregated leakage statistics:
- model
- temperature
- reference dataset
- total samples
- leaked samples
- leakage rate
- mean maximum n-gram similarity

## Detailed Results

For each configuration:
- per-sample leakage decision
- similarity scores
- top matching reference entries
