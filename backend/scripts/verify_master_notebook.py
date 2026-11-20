"""
Dry-run verification of the master notebook code cells.
"""
import json
from pathlib import Path

nb_path = Path("AeroSense_AI_Master_Colab_Pipeline.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded master notebook with {len(nb['cells'])} cells.")
for idx, c in enumerate(nb['cells']):
    c_type = c['cell_type']
    first_line = c['source'][0].strip() if c['source'] else ''
    print(f"Cell {idx+1}: [{c_type}] {first_line[:60]}")
