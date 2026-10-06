import React from "react";

const BADGE = {
  typical: { icon: "✓", text: "Typical material", cls: "good" },
  somewhat_unusual: { icon: "!", text: "Somewhat unusual", cls: "warn" },
  unusual: { icon: "▲", text: "Unusual material", cls: "bad" },
  unknown: { icon: "?", text: "Distance check unavailable", cls: "muted" },
};

const fmt = (v, d = 1) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d));

function PositionBar({ pos }) {
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  return (
    <div className="posbar" role="img" aria-label={`Predicted stiffness lies in the ${pos.label}`}>
      <div className="track">
        {[["p25", pos.quartiles.p25], ["median", pos.quartiles.median], ["p75", pos.quartiles.p75]].map(([k, x]) => (
          <span key={k} className="tick" style={{ left: pct(x) }} title={`${k}: ${pos.values[k]} N/m`} />
        ))}
        <span className="marker" style={{ left: pct(pos.position) }} />
      </div>
      <p className="hint">Marker = this prediction; ticks = quartiles of the C2DB training materials ({fmt(pos.values.p25)}, {fmt(pos.values.median)} and {fmt(pos.values.p75)} N/m; log scale). It falls in the <strong>{pos.label}</strong>.</p>
    </div>
  );
}

export default function ResultCard({ res }) {
  const s = res.stiffness;
  const p = res.poisson;
  const d = res.domain;
  const b = BADGE[d.level] || BADGE.unknown;
  return (
    <section className="card result" aria-live="polite">
      <div className="result-head">
        <div>
          <div className="eyebrow">Predicted in-plane stiffness Y₂D</div>
          <div className="big">
            {fmt(s.value)} <span className="unit">N/m</span>
          </div>
          <div className="err">
            {s.expected_abs_error != null ? (
              <>typical error <strong>≈ ±{fmt(s.expected_abs_error)} N/m</strong> ({s.error_basis})</>
            ) : (
              <>typical error <strong>±{fmt(s.error_known_family)} N/m</strong> for known structure families, <strong>±{fmt(s.error_new_family)} N/m</strong> for new ones</>
            )}
          </div>
        </div>
        <span className={`badge ${b.cls}`}>
          <span aria-hidden="true">{b.icon}</span> {b.text}
        </span>
      </div>

      <p className="domain">{d.message}</p>
      {d.available && (
        <p className="hint">
          Distance to nearest training material: {d.distance.toFixed(2)} (typical ≤ {d.typical_distance.toFixed(2)}, unusual ≥ {d.unusual_distance.toFixed(2)}, standardised feature space).
        </p>
      )}

      <PositionBar pos={s.position} />

      <div className="poisson">
        <div>
          <div className="eyebrow nocaps">Poisson ratio</div>
          <div className="mid">ν = {p.value.toFixed(2)}</div>
        </div>
        <div className="poisson-note">
          <span className="tag">low confidence</span>
          <p className="hint">{p.note} Typical error ±{p.error_known_family.toFixed(2)} (known families) to ±{p.error_new_family.toFixed(2)} (new families).</p>
        </div>
      </div>

      {res.warnings?.length > 0 && (
        <ul className="warnlist">
          {res.warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}

      <details>
        <summary>Inputs used</summary>
        <table className="kv">
          <tbody>
            {Object.entries(res.input).map(([k, v]) => (
              <tr key={k}><th>{k}</th><td>{typeof v === "number" ? Number(v.toFixed(4)) : v}</td></tr>
            ))}
          </tbody>
        </table>
      </details>

      <ul className="notes">
        {res.notes.map((n, i) => <li key={i}>{n}</li>)}
      </ul>
    </section>
  );
}
