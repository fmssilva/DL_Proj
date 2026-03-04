# DL Assignment 1 - Pokemon Type Classification

## Running on Google Colab

Open `assignment_1/colab_test.ipynb` or `assignment_1/notebook.ipynb` from GitHub via Colab.

Run all cells top-to-bottom. The setup cell handles everything automatically:
- clones the repo (or pulls latest if already cloned)
- installs dependencies from `requirements.txt`
- downloads data from Google Drive

No manual configuration needed.

## Running locally

```bash
# from assignment_1/
pip install -r requirements.txt
python task1_mlp.py
```

Ensure `data/` contains `train_labels.csv`, `Train/` and `Test/`.

## Import approach

All imports use `from src.*` (e.g. `from src.datasets.dataset import ...`).
This works when running from `assignment_1/` as the working directory.
No `pip install -e .` or `PYTHONPATH` needed.

## Running tests

```bash
# from assignment_1/
python -m src.datasets.dataset_test
python -m src.training.train_test
python -m src.models.models_test
```
