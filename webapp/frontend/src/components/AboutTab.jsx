import React, { useEffect, useState } from "react";
import { api } from "../api.js";

export default function AboutTab({ health }) {
  const [m, setM] = useState(null);
  useEffect(() => { api.models().then(setM).catch(() => setM(null)); }, []);
  const y = m?.models?.taskA_Y2D;
  return (
    <div className="stack">
      <section className="card">
        <h2>What this is</h2>
        <p>A screening tool that predicts the in-plane stiffness (Young's modulus Y₂D, N/m) and Poisson ratio of a 2D monolayer from its composition and a few geometry numbers. It runs a LightGBM model trained on 7,258 mechanically stable monolayers from C2DB, using 141 inputs (132 Magpie composition statistics plus cell geometry; no DFT outputs).</p>
        <h3>What to trust</h3>
        <ul className="notes">
          <li><strong>Ranking:</strong> good. Rank correlation 0.82 on an independent database.</li>
          <li><strong>Absolute values:</strong> typical error about 14 N/m for known structure families and about 20 N/m for new ones; about 29 N/m on chemistry far from the training data.</li>
          <li><strong>Poisson ratio:</strong> weak (R² ≈ 0.14 on new families). Rough guide only.</li>
          <li><strong>Not covered:</strong> bilayers, shear effects, magnetic or unstable layers, strongly bonded or unusual structures. Not a replacement for DFT.</li>
        </ul>
        {y && (
          <p className="hint">
            Model: {m.models.taskA_Y2D.n_features} features, {y.n_train} training materials; cross-validated error {y.cross_validated_MAE_N_per_m_5seeds.random.mean.toFixed(1)} N/m (random) and {y.cross_validated_MAE_N_per_m_5seeds.family.mean.toFixed(1)} N/m (family held out). Distance-to-training check: {m.distance_check_available ? "enabled" : "not installed on this server"}.
          </p>
        )}
      </section>

      <section className="card">
        <h2>Data, licences and credits</h2>
        <p className="hint">Raw databases are not redistributed by this app. Licence terms below must be confirmed on each provider's site before public deployment.</p>
        <div className="tablewrap">
          <table className="data">
            <thead><tr><th>Resource</th><th>Role</th><th>Credit</th></tr></thead>
            <tbody>
              <tr><td>C2DB</td><td>Training data (stiffness tensors, structures)</td><td>S. Haastrup et al., 2D Mater. 5, 042002 (2018); M. N. Gjerding et al., 2D Mater. 8, 044002 (2021). Technical University of Denmark (CAMD). Recalled as CC BY 4.0; confirm.</td></tr>
              <tr><td>JARVIS-DFT</td><td>External test of the model</td><td>K. Choudhary et al., npj Comput. Mater. 6, 173 (2020); Phys. Rev. B 98, 014107 (2018). NIST. Confirm data licence.</td></tr>
              <tr><td>BiDB</td><td>Bilayer research (not used by this app)</td><td>S. Pakdel et al., Nat. Commun. 15, 932 (2024). Licence to be obtained from the provider.</td></tr>
            </tbody>
          </table>
        </div>
        <p className="hint">Software: scikit-learn, LightGBM, pymatgen, matminer, FastAPI, React.</p>
      </section>

      <section className="card">
        <h2>Roadmap</h2>
        <ul className="notes">
          <li><strong>Next version: shear effects.</strong> Simulating the response of a sheet to shear strain is planned. It is not part of this release.</li>
          <li>Later: bilayer properties (needs validated bilayer stiffness labels), more structure-aware models, feature attribution.</li>
        </ul>
        <p className="hint">API status: {health?.status || "unknown"}.</p>
      </section>
    </div>
  );
}
