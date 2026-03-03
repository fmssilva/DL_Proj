# Installation Guide

This project is structured as a proper Python package so imports like `from src.data import dataset` work everywhere — locally on Windows/Mac and in Colab/Kaggle — with no `sys.path` hacks.

---

## Local setup (Windows & Mac) — run once per clone

```bash
# 1. clone the repo
git clone https://github.com/fmssilva/DL_Proj.git
cd DL_Proj/assignment_1

# 2. (optional but recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. install in editable mode — this is the key step
pip install -e .
```

`pip install -e .` registers the project root on Python's path permanently for this environment.  
After this, `from src.data import dataset` works from any directory, any script, any notebook.  
No need to ever run this again unless you recreate the environment.

---

## Google Colab

Paste this in the **first cell** of your notebook:

```python
# clone and install — run once per Colab session
!git clone https://github.com/fmssilva/DL_Proj.git
%cd DL_Proj/assignment_1
!pip install -e . -q

# (download data from Kaggle — see notebook.ipynb for full instructions)
```

`pip install -e .` in Colab installs into the session's Python environment.  
After that cell, all `from src.*` imports work in every subsequent cell.

---

## Kaggle Kernels

In a code cell at the top of your notebook:

```python
import subprocess, os

# clone into /kaggle/working
subprocess.run(["git", "clone", "https://github.com/fmssilva/DL_Proj.git"], check=True)
os.chdir("DL_Proj/assignment_1")
subprocess.run(["pip", "install", "-e", ".", "-q"], check=True)
```

Or equivalently with shell cells:

```bash
%%bash
git clone https://github.com/fmssilva/DL_Proj.git
cd DL_Proj/assignment_1
pip install -e . -q
```

---

## Why editable install (`-e`)?

| Approach               | Local   | Colab   | Kaggle  | No code changes | Cross-platform |
| ---------------------- | ------- | ------- | ------- | --------------- | -------------- |
| `pip install -e .`     | Yes     | Yes     | Yes     | Yes             | Yes            |
| `sys.path.insert(...)` | Fragile | Fragile | Fragile | No              | No             |
| `PYTHONPATH=...`       | Manual  | Manual  | Manual  | No              | No             |
| Docker                 | Yes     | No      | No      | Yes             | Complex        |

The editable install approach is the only one that is clean, standard, cross-platform, and requires no changes to any import statement in the codebase.

---

## Running the project

```bash
# Task 1 — MLP
python task1_mlp.py

# Run individual test files (from assignment_1/ root)
python src/data/dataset_test.py
python src/training/train_test.py
python src/models/models_test.py
python src/evaluation/submission.py
python src/evaluation/plots.py
```
