"""Loaders and cleaning for C2DB (primary), JARVIS (cross-check) and BiDB (Task B).

Conventions (also in README):
  * Stiffness is in N/m (2D elastic constants), never converted to GPa.
  * Every C2DB / JARVIS row is a single vdW layer (n_layers = 1). Columns:
      Y2D            stiffness of the whole slab as computed
      Y2D_per_layer  Y2D / n_layers   (identical to Y2D for every current row)
  * BiDB binding energy is energy per unit area of the single interface of a bilayer; it is NOT
    divided by the number of layers. Its unit is unconfirmed and left as-is.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Composition, Element

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(cfg: dict, *parts: str) -> Path:
    return (ROOT / cfg["paths"]["data_root"]).joinpath(*parts).resolve()


def cache_path(cfg: dict, name: str) -> Path:
    p = ROOT / cfg["paths"]["cache_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p / name


# ----------------------------------------------------------------------------- geometry helpers
_SYM: dict[int, str] = {}


def _symbol(z: int) -> str:
    if z not in _SYM:
        _SYM[z] = Element.from_Z(int(z)).symbol
    return _SYM[z]


def _nd(x):
    """Decode ASE-json ndarray objects; pass plain lists through."""
    if isinstance(x, dict) and "array" in x:
        x = x["array"]
    if isinstance(x, dict) and "__ndarray__" in x:
        shape, dtype, data = x["__ndarray__"]
        return np.array(data, dtype=dtype).reshape(shape)
    return np.asarray(x)


def composition_info(symbols: list[str]) -> dict:
    comp = Composition(dict(pd.Series(symbols).value_counts()))
    return {
        "reduced_formula": comp.reduced_formula,
        "anon_formula": comp.anonymized_formula,
        "chemsys": "-".join(sorted(e.symbol for e in comp.elements)),
        "n_elements": len(comp.elements),
    }


def slab_geometry(lattice: np.ndarray, frac: np.ndarray) -> dict:
    """In-plane lattice, area, and slab thickness measured along the surface normal.

    Thickness is the cell height minus the largest periodic gap between atomic planes, so slabs that
    straddle the cell boundary are handled. A single atom plane gives thickness 0.
    """
    a, b, c = lattice
    cross = np.cross(a, b)
    area = float(np.linalg.norm(cross))
    n = cross / area
    height = abs(float(np.dot(c, n)))
    z = np.sort(((frac @ lattice) @ n) % height)
    gaps = np.diff(np.r_[z, z[0] + height])
    cosg = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "a": float(np.linalg.norm(a)),
        "b": float(np.linalg.norm(b)),
        "gamma": float(np.degrees(np.arccos(np.clip(cosg, -1, 1)))),
        "area": area,
        "cell_height": height,
        "z_extent": float(height - gaps.max()),
    }


# ----------------------------------------------------------------------------- stability + targets
def tensor_stability(M: np.ndarray, tol: float = 0.0) -> dict:
    """Stability from the tensor itself: symmetrised matrix must be positive definite."""
    M = np.asarray(M, float)
    if M.size == 0 or not np.all(np.isfinite(M)):
        return {"min_eig": np.nan, "asym_frac": np.nan, "stable_tensor": False}
    w = np.linalg.eigvalsh((M + M.T) / 2)
    asym = np.abs(M - M.T).max() / max(np.abs(M).max(), 1e-12)
    return {"min_eig": float(w[0]), "asym_frac": float(asym), "stable_tensor": bool(w[0] > tol)}


def add_elastic_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["Y2D"] = (df.c11 * df.c22 - df.c12**2) / df.c22        # N/m, n_layers-layer slab as computed
    df["Y2D_per_layer"] = df["Y2D"] / df["n_layers"]
    df["poisson"] = df.c12 / df.c22                           # dimensionless
    df["anisotropy"] = df.c11 / df.c22
    return df


# ----------------------------------------------------------------------------- C2DB
def _num(x):
    return np.nan if x is None else float(x)


def load_c2db(cfg: dict, rebuild: bool = False, only_with_stiffness: bool = True) -> pd.DataFrame:
    """One row per C2DB material that has results-asr.stiffness.json. Cached after first read."""
    cp = cache_path(cfg, "c2db_raw.pkl")
    if cp.exists() and not rebuild:
        return pd.read_pickle(cp)
    base = data_path(cfg, cfg["paths"]["c2db_dir"])
    if not base.exists():
        raise FileNotFoundError(f"C2DB folder not found: {base}")
    rows = []
    for root, _dirs, files in os.walk(base):
        if "data.json" not in files:
            continue
        has_stiff = "results-asr.stiffness.json" in files
        if only_with_stiffness and not has_stiff:
            continue
        with open(os.path.join(root, "data.json"), encoding="utf-8") as f:
            d = json.load(f)
        r = {k: d.get(k) for k in ["uid", "number", "lgnum", "layergroup", "thickness", "dyn_stab", "ehull",
                                   "hform", "is_magnetic", "has_inversion_symmetry", "label"]}
        r["thickness_c2db"] = r.pop("thickness")
        r["path"] = os.path.relpath(root, base)
        with open(os.path.join(root, "structure.json"), encoding="utf-8") as f:
            st = json.load(f)["1"]
        numbers = _nd(st["numbers"]).astype(int)
        lattice = _nd(st["cell"]).astype(float).reshape(3, 3)
        cart = _nd(st["positions"]).astype(float).reshape(-1, 3)
        frac = cart @ np.linalg.inv(lattice)
        symbols = [_symbol(z) for z in numbers]
        r.update(composition_info(symbols))
        r.update(slab_geometry(lattice, frac))
        r["nat"] = len(numbers)
        r["n_layers"] = 1  # C2DB entries are monolayers by construction
        if has_stiff:
            with open(os.path.join(root, "results-asr.stiffness.json"), encoding="utf-8") as f:
                j = json.load(f)
            s = j["kwargs"]["data"] if "kwargs" in j else j  # 32 older files use a flat layout
            M = np.array([[_num(s.get(f"c_{i}{j}")) for j in (1, 2, 3)] for i in (1, 2, 3)])
            # C2DB tensor order is (xx, yy, xy): c_33 is the in-plane xy shear, c_66 is always null.
            r.update(c11=M[0, 0], c12=M[0, 1], c22=M[1, 1], shear_xy=M[2, 2])
            r.update(tensor_stability(M))
        rows.append(r)
    df = add_elastic_targets(pd.DataFrame(rows).sort_values("uid").reset_index(drop=True))
    df["family"] = df.anon_formula + "|lg" + df.lgnum.astype(str)
    df.to_pickle(cp)
    return df


def clean_c2db(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Keep tensor-stable, finite, symmetric-enough tensors. Returns (clean df, step counts)."""
    log = {"with_stiffness": len(df)}
    d = df[np.isfinite(df[["c11", "c12", "c22"]]).all(axis=1)]
    log["finite_tensor"] = len(d)
    d = d[d.stable_tensor]
    log["positive_definite"] = len(d)
    d = d[d.asym_frac <= cfg["stability"]["max_asym_frac"]]
    log["asymmetry_ok"] = len(d)
    d = d[(d.Y2D > 0) & np.isfinite(d.Y2D)]
    log["Y2D_positive"] = len(d)
    if cfg["cleaning"]["max_ehull"] is not None:
        d = d[d.ehull <= cfg["cleaning"]["max_ehull"]]
        log["ehull_ok"] = len(d)
    return d.reset_index(drop=True), log


def restrict_for_target(df: pd.DataFrame, target: str, cfg: dict) -> pd.DataFrame:
    if target == "poisson":
        return df[df.poisson.abs() <= cfg["cleaning"]["poisson_abs_max"]].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- JARVIS
def _parse_jarvis_tensor(raw):
    """Return a 6x6 array, or None if missing/corrupt (corrupt rows hold NaN or denormals like 5e-324)."""
    try:
        a = np.array(ast.literal_eval(raw) if isinstance(raw, str) else raw, float)
    except Exception:
        return None
    if a.shape != (6, 6) or not np.all(np.isfinite(a)):
        return None
    if np.any((np.abs(a) > 0) & (np.abs(a) < 1e-100)):
        return None
    return a


def load_jarvis(cfg: dict) -> pd.DataFrame:
    """JARVIS monolayers that have an elastic tensor. Only the xx,yy block (indices 0,1) is used:
    the shear entries ([3,3], [5,5]) do not reproduce (C11-C12)/2 for MoS2 and are not trusted."""
    path = data_path(cfg, cfg["paths"]["jarvis_json"])
    if not path.exists():
        raise FileNotFoundError(f"JARVIS json not found: {path}")
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    rows = []
    for r in recs:
        raw = r.get("elastic_tensor")
        if raw in (None, "na", "", []):
            continue
        a = _parse_jarvis_tensor(raw)
        at = r["atoms"]
        lattice = np.array(at["lattice_mat"], float)
        coords = np.array(at["coords"], float)
        frac = coords @ np.linalg.inv(lattice) if at.get("cartesian") else coords
        row = {"jid": r["jid"], "spg_number": r.get("spg_number"), "tensor_ok": a is not None,
               "nat": len(at["elements"]), "n_layers": 1}
        row.update(composition_info(list(at["elements"])))
        row.update(slab_geometry(lattice, frac))
        if a is not None:
            row.update(c11=a[0, 0], c12=a[0, 1], c22=a[1, 1])
            det = a[0, 0] * a[1, 1] - a[0, 1] ** 2
            row["stable_tensor"] = bool(a[0, 0] > 0 and a[1, 1] > 0 and det > 0)
        else:
            row.update(c11=np.nan, c12=np.nan, c22=np.nan, stable_tensor=False)
        rows.append(row)
    return add_elastic_targets(pd.DataFrame(rows))  # corrupt rows have NaN c_ij, so their targets are NaN


def clean_jarvis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"with_tensor_string": len(df), "corrupt_tensor": int((~df.tensor_ok).sum())}
    d = df[df.tensor_ok & df.stable_tensor & (df.Y2D > 0)].reset_index(drop=True)
    log["stable_xx_yy_block"] = len(d)
    return d, log


# ----------------------------------------------------------------------------- BiDB (Task B)
BIDB_REQUIRED = ["number_of_layers", "monolayer_uid", "bilayer_uid", "binding_energy_gs", "distance",
                 "space_group", "stoichiometry"]


def iqr_filter_log(s: pd.Series, k: float) -> pd.Series:
    """Boolean keep-mask: positive, finite, and within [Q1-k*IQR, Q3+k*IQR] of log10(value)."""
    ok = s.notna() & np.isfinite(s) & (s > 0)
    ls = np.log10(s.where(ok))
    q1, q3 = ls.quantile(0.25), ls.quantile(0.75)
    return ok & ls.between(q1 - k * (q3 - q1), q3 + k * (q3 - q1))


def load_bidb(cfg: dict) -> pd.DataFrame:
    """Bilayers only. UNTESTED: written against the column names in the project brief, because the
    BiDB files are not present in this checkout. Fails loudly instead of guessing."""
    p = data_path(cfg, cfg["paths"]["bidb_dir"], cfg["paths"]["bidb_properties"])
    if not p.exists():
        raise FileNotFoundError(
            f"BiDB properties file not found: {p}\n"
            "Task B needs bidb_properties.csv (and bidb_structures.json). They are not in Dataset/ -- "
            "place them under the folder named by paths.bidb_dir in config.yaml.")
    df = pd.read_csv(p, low_memory=False)
    missing = [c for c in BIDB_REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"bidb_properties.csv is missing required columns: {missing}")
    df = df[df.number_of_layers == 2].copy()
    df["n_layers"] = 2
    infos = []
    for s in df.stoichiometry.astype(str):
        try:
            infos.append(composition_info_from_formula(s))
        except Exception:
            infos.append({})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(infos)], axis=1)
    df = df[df.chemsys.notna()].reset_index(drop=True)
    df["spg_number"] = pd.to_numeric(df.space_group, errors="coerce")
    return df


def composition_info_from_formula(formula: str) -> dict:
    comp = Composition(formula)
    return {"reduced_formula": comp.reduced_formula, "anon_formula": comp.anonymized_formula,
            "chemsys": "-".join(sorted(e.symbol for e in comp.elements)), "n_elements": len(comp.elements)}


def clean_bidb(df: pd.DataFrame, target: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    k = cfg["cleaning"]["bidb_iqr_k"]
    keep = iqr_filter_log(df[target], k)
    log = {"bilayers": len(df), "dropped_nonpositive_or_nan": int((~(df[target].notna() & (df[target] > 0))).sum()),
           "dropped_outlier_or_invalid": int((~keep).sum()), "kept": int(keep.sum())}
    return df[keep].reset_index(drop=True), log


# ----------------------------------------------------------------------------- audit CLI
def audit(cfg: dict) -> dict:
    out = {}
    c = load_c2db(cfg)
    cc, clog = clean_c2db(c, cfg)
    out["c2db"] = {"rows_with_stiffness": len(c), "clean_steps": clog,
                   "unique_chemsys_clean": int(cc.chemsys.nunique()), "unique_families_clean": int(cc.family.nunique()),
                   "Y2D_clean": cc.Y2D.describe().round(2).to_dict(),
                   "poisson_rows_abs_le_1": int((cc.poisson.abs() <= cfg["cleaning"]["poisson_abs_max"]).sum())}
    j = load_jarvis(cfg)
    jc, jlog = clean_jarvis(j)
    out["jarvis"] = {"steps": jlog, "Y2D_clean": jc.Y2D.describe().round(2).to_dict(),
                     "chemsys_shared_with_c2db_clean": int(jc.chemsys.isin(set(cc.chemsys)).sum())}
    try:
        b = load_bidb(cfg)
        out["bidb"] = {"bilayers": len(b)}
    except (FileNotFoundError, KeyError) as e:
        out["bidb"] = {"status": "unavailable", "reason": str(e).splitlines()[0]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="re-read C2DB from disk")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.rebuild:
        load_c2db(cfg, rebuild=True)
    if args.audit:
        res = audit(cfg)
        rd = ROOT / cfg["paths"]["results_dir"]
        rd.mkdir(exist_ok=True)
        (rd / "data_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        print(json.dumps(res, indent=2, default=str))
