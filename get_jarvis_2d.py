"""Download JARVIS-DFT 2D (dft_2d) and save JSON + CSV.
Run:  pip install jarvis-tools pandas  &&  python get_jarvis_2d.py
"""
import json
import pandas as pd
from jarvis.db.figshare import data

records = data("dft_2d")                      # downloads from figshare, cached locally
with open("jarvis_dft_2d.json", "w") as f:    # full data incl. atomic structures
    json.dump(records, f)

df = pd.DataFrame(records)
cols = [c for c in df.columns if c != "atoms"]
df[cols].to_csv("jarvis_dft_2d.csv", index=False)   # scalar properties only

print("Rows:", len(df))
print("Columns:", cols)
print("Elastic-related columns:", [c for c in cols if "elast" in c.lower() or "modul" in c.lower()])