Raw data are not stored in this repository (C2DB alone is about 2.3 GB). `config.yaml: paths.data_root` points to `../Dataset/`.

Expected layout under `Dataset/` (relative to the repository root):

| Path | What | Source |
|---|---|---|
| `c2db/materials/<anon formula>/<uid folder>/...` | per-material `data.json`, `structure.json`, `results-asr.stiffness.json` | C2DB, <https://c2db.fysik.dtu.dk/> |
| `jarvis_dft_2d.json`, `jarvis_dft_2d.csv` | JARVIS-DFT 2D (dft_2d), structures and properties | JARVIS-tools (`jarvis.db.figshare.data("dft_2d")`, see `get_jarvis_2d.py` in the repository root) |
| `bidb_properties.csv`, `bidb_structures.json`, `bidb.db` | BiDB: 992 monolayers, 10,192 homobilayers | van der Waals Bilayer Database, <https://2dhub.org/bidb/bidb.html>; Pakdel, Rasmussen, Taghizadeh, Kruse, Olsen, Thygesen, Nat. Commun. 15, 932 (2024) |
| `bidb_to_c2db_uid_map.csv` | BiDB monolayer uid -> C2DB uid (982 of 992 mapped; verified at formula level only; provenance undocumented) | supplied with the project |

`youngs_modulus.csv` (bulk metals) is in `Dataset/` but is not used.
