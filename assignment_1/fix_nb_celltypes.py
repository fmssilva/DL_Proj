"""Fix wrong cell types in task1/task1.ipynb."""
import json

NB_PATH = "task1/task1.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")

def show(idx):
    c = cells[idx]
    src = c["source"]
    first = (src[0] if isinstance(src, list) else src)[:60]
    print(f"  [{idx}] {c['cell_type']:8s} | {first!r}")

# Inspect problem cells before fix
for i in [30, 32, 34, 36]:
    show(i)

# Cell 30: markdown -> code  (Experiment B Python code)
assert cells[30]["cell_type"] == "markdown", f"Expected markdown at 30, got {cells[30]['cell_type']}"
cells[30]["cell_type"] = "code"
cells[30]["outputs"] = []
cells[30]["execution_count"] = None

# Cell 32: markdown -> code  (Experiment D Python code)
assert cells[32]["cell_type"] == "markdown", f"Expected markdown at 32, got {cells[32]['cell_type']}"
cells[32]["cell_type"] = "code"
cells[32]["outputs"] = []
cells[32]["execution_count"] = None

# Cell 34: code -> markdown  (--- Part 3 separator)
assert cells[34]["cell_type"] == "code", f"Expected code at 34, got {cells[34]['cell_type']}"
cells[34]["cell_type"] = "markdown"
cells[34].pop("outputs", None)
cells[34].pop("execution_count", None)

# Cell 36: markdown -> code  (Load best experiment / classification report)
assert cells[36]["cell_type"] == "markdown", f"Expected markdown at 36, got {cells[36]['cell_type']}"
cells[36]["cell_type"] = "code"
cells[36]["outputs"] = []
cells[36]["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nAfter fix:")
for i in [30, 32, 34, 36]:
    show(i)
print("Done.")
