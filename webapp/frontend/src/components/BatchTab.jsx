import React, { useState } from "react";
import { api } from "../api.js";

const TEMPLATE = "formula,nat,a,b,gamma,thickness,spg_number\nMoS2,3,3.19,3.19,120,3.13,187\nWSe2,3,3.28,3.28,120,3.36,187\n";

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

export default function BatchTab() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [out, setOut] = useState(null);

  async function run(e) {
    e.preventDefault();
    if (!file) { setErr("Choose a CSV file first."); return; }
    setErr(""); setBusy(true);
    try { setOut(await api.predictBatch(file)); } catch (ex) { setOut(null); setErr(ex.message); } finally { setBusy(false); }
  }

  function exportCsv() {
    const rows = out.results.map((r) => [r.input.formula, r.input.nat, r.input.a, r.input.b, r.input.gamma, r.input.thickness, r.input.spg_number,
      r.stiffness.value.toFixed(2), r.poisson.value.toFixed(3), r.domain.level, r.stiffness.expected_abs_error ?? ""]);
    download("predictions.csv", ["formula,nat,a,b,gamma,thickness,spg_number,stiffness_N_per_m,poisson,domain,expected_abs_error_N_per_m", ...rows.map((r) => r.join(","))].join("\n") + "\n");
  }

  return (
    <div className="stack">
      <form className="card" onSubmit={run}>
        <h2>Batch predictions</h2>
        <p className="hint">Upload a CSV with the columns <code>formula, nat, a, b, gamma, thickness, spg_number</code> (up to 500 rows, 2 MB).</p>
        <div className="row">
          <label className="drop small"><input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] || null)} /><span>{file ? file.name : "Choose CSV"}</span></label>
          <button type="button" className="ghost" onClick={() => download("template.csv", TEMPLATE)}>Download template</button>
          <button className="primary inline" disabled={busy}>{busy ? "Running…" : "Run"}</button>
        </div>
        {err && <p className="error" role="alert">{err}</p>}
      </form>

      {out && (
        <section className="card">
          <div className="row between">
            <h2>{out.n_ok} predicted{out.n_errors ? `, ${out.n_errors} with problems` : ""}</h2>
            {out.n_ok > 0 && <button className="ghost" onClick={exportCsv}>Download CSV</button>}
          </div>
          {out.errors.length > 0 && (
            <ul className="warnlist">{out.errors.map((x, i) => <li key={i}>Line {x.line}: {x.error}</li>)}</ul>
          )}
          <div className="tablewrap">
            <table className="data">
              <thead><tr><th>Line</th><th>Formula</th><th className="num">Y₂D (N/m)</th><th className="num">Typical error</th><th className="num">ν</th><th>Domain</th></tr></thead>
              <tbody>
                {out.results.map((r) => (
                  <tr key={r.line}>
                    <td>{r.line}</td><td>{r.input.formula}</td>
                    <td className="num">{r.stiffness.value.toFixed(1)}</td>
                    <td className="num">{r.stiffness.expected_abs_error != null ? `±${r.stiffness.expected_abs_error.toFixed(1)}` : `±${r.stiffness.error_known_family.toFixed(0)}–${r.stiffness.error_new_family.toFixed(0)}`}</td>
                    <td className="num">{r.poisson.value.toFixed(2)}</td>
                    <td>{r.domain.level.replace("_", " ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
