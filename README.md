# Data Leakage Analysis for LLM-Generated Unit Tests

This project studies **data leakage and memorization in Large Language Models (LLMs)** when generating **Python unit tests**.  
We analyze whether generated test cases overlap with known benchmark datasets using **character n-gram overlap similarity**, a commonly used surface-level leakage metric in prior work.

The project is designed to be **model-agnostic**, reproducible, and extensible to additional similarity metrics or datasets.

---

## Project Overview

**Goal:**  
Determine whether LLM-generated test cases show signs of memorization or leakage from benchmark datasets such as **MBPP** and **HumanEval**.

**Core idea:**  
Compare generated `assert` statements against reference test cases using **character n-gram overlap similarity** (Jaccard similarity over n-grams).

---

## Project Structure

