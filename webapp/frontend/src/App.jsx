import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import PredictTab from "./components/PredictTab.jsx";
import BatchTab from "./components/BatchTab.jsx";
import EvidenceTab from "./components/EvidenceTab.jsx";
import AboutTab from "./components/AboutTab.jsx";

const TABS = [
  ["predict", "Predict"],
  ["batch", "Batch"],
  ["evidence", "Evidence"],
  ["about", "About & credits"],
];

export default function App() {
  const [tab, setTab] = useState("predict");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "down" }));
  }, []);

  return (
    <div className="app">
      <header className="top">
        <div className="wrap top-inner">
          <div>
            <h1>2D Stiffness Predictor</h1>
            <p className="sub">Screening estimates of monolayer stiffness, with the error you should expect.</p>
          </div>
          <span className={`pill ${health?.status === "ok" ? "ok" : health ? "bad" : ""}`} role="status">
            {health?.status === "ok" ? "API online" : health ? "API offline" : "connecting…"}
          </span>
        </div>
        <nav className="wrap tabs" aria-label="Sections">
          {TABS.map(([id, label]) => (
            <button key={id} className={tab === id ? "tab active" : "tab"} onClick={() => setTab(id)} aria-current={tab === id ? "page" : undefined}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="wrap">
        {tab === "predict" && <PredictTab />}
        {tab === "batch" && <BatchTab />}
        {tab === "evidence" && <EvidenceTab />}
        {tab === "about" && <AboutTab health={health} />}
      </main>
      <footer className="wrap foot">
        Trained on C2DB monolayers (GPAW/PBE). A screening aid, not a replacement for DFT. Not validated for bilayers.
      </footer>
    </div>
  );
}
