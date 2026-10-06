import React, { useState } from "react";
import { api } from "../api.js";
import ResultCard from "./ResultCard.jsx";

const PRESETS = [
  { name: "MoS₂", v: { formula: "MoS2", nat: 3, a: 3.19, b: 3.19, gamma: 120, thickness: 3.13, spg_number: 187 } },
  { name: "WSe₂", v: { formula: "WSe2", nat: 3, a: 3.28, b: 3.28, gamma: 120, thickness: 3.36, spg_number: 187 } },
  { name: "Graphene", v: { formula: "C", nat: 2, a: 2.47, b: 2.47, gamma: 120, thickness: 0, spg_number: 191 } },
  { name: "h-BN", v: { formula: "BN", nat: 2, a: 2.51, b: 2.51, gamma: 120, thickness: 0, spg_number: 187 } },
];

const FIELDS = [
  ["formula", "Formula", "text", "e.g. MoS2", "Chemical formula of the cell (reduced formula is fine)"],
  ["nat", "Atoms in cell", "number", "3", "Number of atoms in the primitive cell"],
  ["a", "a (Å)", "number", "3.19", "In-plane lattice length a"],
  ["b", "b (Å)", "number", "3.19", "In-plane lattice length b"],
  ["gamma", "γ (°)", "number", "120", "Angle between a and b"],
  ["thickness", "Thickness (Å)", "number", "3.13", "Distance between the outermost atomic planes; 0 for a flat single plane"],
  ["spg_number", "Space group #", "number", "187", "Space-group number of the sheet (1–230)"],
];

export default function PredictTab() {
  const [mode, setMode] = useState("values");
  const [form, setForm] = useState({ formula: "", nat: "", a: "", b: "", gamma: "", thickness: "", spg_number: "" });
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState(null);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  async function submit(e) {
    e.preventDefault();
    setErr(""); setBusy(true);
    try {
      if (mode === "values") {
        const payload = { formula: form.formula.trim(), nat: Number(form.nat), a: Number(form.a), b: Number(form.b), gamma: Number(form.gamma), thickness: Number(form.thickness), spg_number: Number(form.spg_number) };
        setRes(await api.predict(payload));
      } else {
        if (!file) throw new Error("Choose a CIF or POSCAR file first.");
        setRes(await api.predictStructure(file));
      }
    } catch (ex) {
      setRes(null); setErr(ex.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid2">
      <form className="card" onSubmit={submit}>
        <div className="seg" role="tablist" aria-label="Input method">
          <button type="button" role="tab" aria-selected={mode === "values"} className={mode === "values" ? "on" : ""} onClick={() => setMode("values")}>Enter values</button>
          <button type="button" role="tab" aria-selected={mode === "file"} className={mode === "file" ? "on" : ""} onClick={() => setMode("file")}>Upload structure</button>
        </div>

        {mode === "values" ? (
          <>
            <div className="chips" aria-label="Illustrative examples">
              <span className="hint">Try:</span>
              {PRESETS.map((p) => (
                <button type="button" key={p.name} className="chip" onClick={() => setForm({ ...p.v })}>{p.name}</button>
              ))}
            </div>
            <p className="hint">Example inputs are illustrative values, not the exact C2DB entries.</p>
            <div className="fields">
              {FIELDS.map(([k, label, type, ph, help]) => (
                <label key={k} className={k === "formula" ? "full" : ""}>
                  <span>{label}</span>
                  <input value={form[k]} onChange={set(k)} type={type} step="any" placeholder={ph} required aria-describedby={`h-${k}`} />
                  <small id={`h-${k}`}>{help}</small>
                </label>
              ))}
            </div>
          </>
        ) : (
          <div className="upload">
            <label className="drop">
              <input type="file" accept=".cif,.vasp,.poscar,.json,POSCAR,CONTCAR" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              <span>{file ? file.name : "Choose a CIF or POSCAR file"}</span>
            </label>
            <p className="hint">
              The sheet must be padded with vacuum along the third lattice vector. The server reads the structure and derives the lattice, thickness and space group itself. Max 2 MB, up to 200 atoms.
            </p>
          </div>
        )}

        <button className="primary" disabled={busy}>{busy ? "Predicting…" : "Predict stiffness"}</button>
        {err && <p className="error" role="alert">{err}</p>}
      </form>

      <div>
        {res ? (
          <ResultCard res={res} />
        ) : (
          <section className="card empty">
            <h2>How to read a prediction</h2>
            <ul className="notes">
              <li>The number is a <strong>screening estimate</strong> from a model trained on 7,258 C2DB monolayers.</li>
              <li>Expect an average error of about <strong>14 N/m</strong> for structure types the model has seen, and about <strong>20 N/m</strong> for new ones.</li>
              <li>It ranks materials well but is only roughly quantitative on unfamiliar chemistry.</li>
              <li>A badge tells you whether your material looks like the training data.</li>
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}
