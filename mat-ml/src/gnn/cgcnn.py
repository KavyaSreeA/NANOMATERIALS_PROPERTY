"""CGCNN-type graph network for ln(Y2D) on the Task A folds (design: results/phaseA_design.md).   (.venv-mlip environment: torch, ase, pymatgen)

  python -m src.gnn.cgcnn                     # all 15 folds (3 schemes x 5)   ~1-2 h on a laptop GPU, resumable
  python -m src.gnn.cgcnn --epochs 3 --schemes random --folds 0     # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase import Atoms
from ase.neighborlist import neighbor_list
from pymatgen.core import Element

from .. import data as D

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CUT, MAXNB, NG = 6.0, 16, 40
CENTERS = torch.linspace(0, CUT, NG)
CACHE = D.ROOT / "cache"


def element_table() -> np.ndarray:
    rows = []
    for z in range(1, 95):
        e = Element.from_Z(z)
        def f(v):
            try:
                return float(v)
            except Exception:
                return np.nan
        rows.append([z, f(e.group), f(e.row), f(e.X), f(e.atomic_radius), f(e.atomic_mass), f(e.ionization_energy), f(e.electron_affinity), f(e.mendeleev_no), f(e.molar_volume)])
    A = np.array(rows, float)
    med = np.nanmedian(A, axis=0)
    A = np.where(np.isnan(A), med, A)
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    return np.vstack([np.zeros(A.shape[1]), A]).astype(np.float32)        # index 0 unused


def build_graph(path, cfg, etab):
    base = D.data_path(cfg, cfg["paths"]["c2db_dir"])
    st = json.load(open(os.path.join(base, path, "structure.json"), encoding="utf-8"))["1"]
    z = D._nd(st["numbers"]).astype(int)
    atoms = Atoms(z, positions=D._nd(st["positions"]).astype(float).reshape(-1, 3), cell=D._nd(st["cell"]).astype(float).reshape(3, 3), pbc=[True, True, False])
    i, j, d = neighbor_list("ijd", atoms, CUT)
    keep_i, keep_j, keep_d = [], [], []
    for a in range(len(z)):
        m = np.flatnonzero(i == a)
        if len(m) == 0:                                                   # isolated within the cutoff: use the nearest neighbour out to 12 A
            ii, jj, dd = neighbor_list("ijd", atoms, 12.0)
            m2 = np.flatnonzero(ii == a)
            if len(m2) == 0:
                continue
            o = m2[np.argsort(dd[m2])[:1]]
            keep_i.append(ii[o]); keep_j.append(jj[o]); keep_d.append(dd[o])
            continue
        o = m[np.argsort(d[m])[:MAXNB]]
        keep_i.append(i[o]); keep_j.append(j[o]); keep_d.append(d[o])
    ei = np.stack([np.concatenate(keep_i), np.concatenate(keep_j)])
    return {"x": torch.tensor(etab[z]), "ei": torch.tensor(ei, dtype=torch.long), "d": torch.tensor(np.concatenate(keep_d), dtype=torch.float32)}


class Conv(nn.Module):
    def __init__(self, h, e):
        super().__init__()
        self.fc, self.bn1, self.bn2 = nn.Linear(2 * h + e, 2 * h), nn.BatchNorm1d(2 * h), nn.BatchNorm1d(h)

    def forward(self, x, ei, ea):
        i, j = ei
        gate, core = self.bn1(self.fc(torch.cat([x[i], x[j], ea], 1))).chunk(2, 1)
        agg = torch.zeros_like(x).index_add_(0, i, torch.sigmoid(gate) * F.softplus(core))
        return F.softplus(x + self.bn2(agg))


class Net(nn.Module):
    def __init__(self, nin=10, h=64, layers=3):
        super().__init__()
        self.emb = nn.Linear(nin, h)
        self.convs = nn.ModuleList([Conv(h, NG) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(h, h), nn.Softplus(), nn.Linear(h, 1))

    def forward(self, b):
        x = self.emb(b["x"])
        ea = torch.exp(-((b["d"][:, None] - CENTERS.to(b["d"].device)) ** 2) / 0.15**2)
        for c in self.convs:
            x = c(x, b["ei"], ea)
        s = torch.zeros(b["n"], x.shape[1], device=x.device).index_add_(0, b["batch"], x)
        return self.head(s / b["cnt"][:, None]).squeeze(1)


def collate(graphs, idx):
    xs, eis, ds, bat, off = [], [], [], [], 0
    for k, g in enumerate(idx):
        G = graphs[g]
        xs.append(G["x"]); eis.append(G["ei"] + off); ds.append(G["d"]); bat.append(torch.full((len(G["x"]),), k)); off += len(G["x"])
    batch = torch.cat(bat)
    return {"x": torch.cat(xs).to(DEV), "ei": torch.cat(eis, 1).to(DEV), "d": torch.cat(ds).to(DEV), "batch": batch.to(DEV), "n": len(idx),
            "cnt": torch.bincount(batch, minlength=len(idx)).float().to(DEV)}


def predict(model, graphs, idx, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            out.append(model(collate(graphs, idx[s:s + bs])).cpu())
    return torch.cat(out).numpy()


def inner_split(train, groups, rng, frac=0.1):
    """validation subset of the training fold, split by the SAME grouping rule as the outer split (random rows if groups is None)."""
    train = np.asarray(train)
    if groups is None:
        v = rng.choice(len(train), size=max(1, int(frac * len(train))), replace=False)
    else:
        g = np.asarray(groups)[train]
        u = rng.permutation(np.unique(g))
        chosen, cnt = [], 0
        for k in u:
            chosen.append(k); cnt += (g == k).sum()
            if cnt >= frac * len(train):
                break
        v = np.flatnonzero(np.isin(g, chosen))
    m = np.ones(len(train), bool); m[v] = False
    return train[m], train[v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--schemes", nargs="+", default=["random", "chemsys", "family"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = D.load_config()
    F_ = json.load(open(CACHE / "phaseA_folds.json"))
    y = np.array(F_["y"]); ly = np.log(y)
    gp = CACHE / "phaseA_graphs.pt"
    t0 = time.time()
    if gp.exists():
        graphs = torch.load(gp)
    else:
        etab = element_table()
        graphs = [build_graph(p, cfg, etab) for p in F_["path"]]
        torch.save(graphs, gp)
    print(f"{len(graphs)} graphs ready ({time.time() - t0:.0f}s); mean atoms {np.mean([len(g['x']) for g in graphs]):.1f}, mean edges {np.mean([g['ei'].shape[1] for g in graphs]):.0f}; device {DEV}", flush=True)
    outp = CACHE / f"phaseA_cgcnn_oof{args.tag}.jsonl"
    done = set()
    if outp.exists():
        done = {(r["scheme"], r["fold"]) for r in map(json.loads, open(outp)) if r["epochs_max"] == args.epochs}
    for sc in args.schemes:
        group = {"random": None, "chemsys": F_["chemsys"], "family": F_["family"]}[sc]
        for f in args.folds:
            if (sc, f) in done:
                continue
            torch.manual_seed(args.seed + f); rng = np.random.default_rng(args.seed + f)
            fo = F_["folds"][sc][f]
            tr, va = inner_split(fo["train"], group, rng)
            te = np.array(fo["test"])
            mu, sd = ly[tr].mean(), ly[tr].std()
            tgt = torch.tensor((ly - mu) / sd, dtype=torch.float32)
            model = Net().to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            best, best_state, bad, ep_used = 1e9, None, 0, 0
            for ep in range(args.epochs):
                model.train()
                perm = rng.permutation(tr)
                for s in range(0, len(perm), 128):
                    ix = perm[s:s + 128]
                    if len(ix) < 2:
                        continue
                    loss = F.l1_loss(model(collate(graphs, ix)), tgt[ix].to(DEV))
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                v = np.abs(predict(model, graphs, va) - tgt[va].numpy()).mean()
                ep_used = ep + 1
                if v < best:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            model.load_state_dict(best_state)
            pred = np.exp(predict(model, graphs, te) * sd + mu)
            rec = {"scheme": sc, "fold": f, "epochs_used": ep_used, "epochs_max": args.epochs, "val_mae_std": float(best), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                   "uid": [F_["uid"][i] for i in te], "y": y[te].tolist(), "pred": pred.tolist(), "test_mae": float(np.abs(pred - y[te]).mean()), "seed": args.seed}
            open(outp, "a").write(json.dumps(rec) + "\n")
            print(f"  {sc:<8} fold {f}: test MAE {rec['test_mae']:.2f} N/m  (epochs {ep_used}, val MAE {best:.3f} sd units)  [{time.time() - t0:.0f}s]", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
