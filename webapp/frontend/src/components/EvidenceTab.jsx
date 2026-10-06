import React from "react";

const FIGS = [
  ["fig3_leakage.png", "Why random splits mislead", "Share of test materials whose chemical system, formula or structure family also appears in training. Under a random split 96 % have a structural relative in the training set.", 520],
  ["fig4_cv_results.png", "Error rises when whole structure families are held out", "Mean absolute error of the stiffness model: 13.8 N/m with random splits and 13.9 N/m when chemical systems are held out, but 19.9 N/m (1.44×) when structure families are held out.", 900],
  ["fig_mechanism.png", "The penalty comes from lost relatives", "Error ratio (family held out ÷ random) by the size of the material's own family: none for materials with no relatives, about 1.5× for large families. Chemistry shows no such trend.", 900],
  ["fig6_panel.png", "The same effect across 22 properties in two databases", "Family-held-out error ratio for 12 C2DB and 10 JARVIS properties (bars: 95 % intervals). Every interval is above 1; three JARVIS transport properties are close to it.", 900],
  ["fig9_jarvis_external.png", "Checked on a different database (JARVIS-DFT)", "Trained on C2DB, tested on 186 JARVIS materials: ranks well (Spearman 0.82) but errs by about 29 N/m on chemistry unseen in training.", 900],
];

export default function EvidenceTab() {
  return (
    <div className="stack">
      <section className="card">
        <h2>Why the error band matters</h2>
        <div className="stats">
          <div><div className="mid">13.8 → 19.9</div><div className="hint">N/m average error: random split → structure families held out</div></div>
          <div><div className="mid">1.44×</div><div className="hint">error increase for new structure families (95 % CI 1.30–1.62)</div></div>
          <div><div className="mid">0.82</div><div className="hint">rank correlation on 186 JARVIS materials (use for ranking)</div></div>
        </div>
        <p>
          Most published accuracy figures use random train/test splits, in which close relatives of every test material are in the training set. This project measured what happens when they are not.
          The app therefore reports the error you should expect for the kind of material you enter, not an optimistic average.
        </p>
      </section>
      {FIGS.map(([file, title, cap, w]) => (
        <figure className="card fig" key={file}>
          <h3>{title}</h3>
          <img src={`/figures/${file}`} alt={cap} style={{ maxWidth: w }} loading="lazy" />
          <figcaption>{cap}</figcaption>
        </figure>
      ))}
    </div>
  );
}
