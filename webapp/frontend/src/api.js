async function parse(res) {
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    let msg = body && body.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return body;
}

export const api = {
  health: () => fetch("/api/health").then(parse),
  models: () => fetch("/api/models").then(parse),
  predict: (payload) =>
    fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }).then(parse),
  predictStructure: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/predict/structure", { method: "POST", body: fd }).then(parse);
  },
  predictBatch: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/predict/batch", { method: "POST", body: fd }).then(parse);
  },
};
