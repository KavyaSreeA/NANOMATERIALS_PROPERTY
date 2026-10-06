"""Check the pre-registered predictions P1-P5 of results/panel_replication_design.md against the JARVIS and pooled panels.  python -m src.panel_replication_checks"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import data as D

cfg = D.load_config()
rd = D.ROOT / cfg["paths"]["results_dir"]
C, J = pd.read_csv(rd / "panel_summary.csv"), pd.read_csv(rd / "panel_summary_jarvis.csv")
rng = np.random.default_rng(1)


def corr(x, y, n=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 2 and len(set(y[i])) > 2:
            bs.append(spearmanr(x[i], y[i])[0])
    return float(spearmanr(x, y)[0]), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {"n_jarvis": len(J), "n_pooled": len(C) + len(J)}
out["P1_share_ratio_gt_1.05"] = float((J.ratio_family > 1.05).mean())
out["P1_share_ci_lo_gt_1"] = float((J.ratio_family_lo > 1).mean())
out["P1_pass"] = bool(out["P1_share_ratio_gt_1.05"] >= 0.8 and out["P1_share_ci_lo_gt_1"] >= 0.7)
out["P2_share_chemsys_ratio_lt_1.15"] = float((J.ratio_chemsys < 1.15).mean())
out["P2_pass"] = bool(out["P2_share_chemsys_ratio_lt_1.15"] >= 0.7)
r, ci = corr(J.D_family, J.R_family)
out["P3_rho_J"], out["P3_ci"] = r, ci
out["P3_pass"] = bool(not (r <= -0.6 and ci[1] < 0))
P = pd.concat([C.assign(db="C2DB"), J.assign(db="JARVIS")], ignore_index=True)
r, ci = corr(P.D_family, P.R_family)
out["P4_pooled_rho"], out["P4_pooled_ci"] = r, ci
out["P4_pass"] = bool(not (r <= -0.6 and ci[1] < 0))
r, ci = corr(P.D_family, np.log(P.ratio_family))
out["exploratory_pooled_D_vs_log_ratio"] = {"rho": r, "ci95": ci, "note": "not pre-registered as primary; secondary metric from the design"}
r, ci = corr(J.D_family, np.log(J.ratio_family))
out["exploratory_J_D_vs_log_ratio"] = {"rho": r, "ci95": ci}
shared = {"hform": "formation_energy_peratom", "gap": "optb88vdw_bandgap"}
out["P5_shared_properties_ratio"] = {k: [float(C.set_index("target").loc[k, "ratio_family"]), float(J.set_index("target").loc[v, "ratio_family"])] for k, v in shared.items()}
out["P5_pass"] = bool(all(a > 1.05 and b > 1.05 for a, b in out["P5_shared_properties_ratio"].values()))
out["jarvis_below_1.05"] = J.loc[J.ratio_family <= 1.05, ["target", "ratio_family", "ratio_family_lo", "ratio_family_hi"]].round(3).to_dict("records")
json.dump(out, open(rd / "panel_replication_checks.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print(J[["target", "n", "ratio_family", "ratio_family_lo", "ratio_family_hi", "ratio_chemsys", "R_family", "D_family"]].round(3).to_string())
