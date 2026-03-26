/**
 * COMPASS Platform API client.
 *
 * All functions return { data, error } for consistent error handling.
 * The frontend falls back to mock data if the API is unreachable.
 */

// API key from environment — Vite exposes VITE_ prefixed vars
const API_KEY = import.meta.env.VITE_COMPASS_API_KEY || "";

async function request(url, options = {}) {
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
        ...options.headers,
      },
    });
    if (!res.ok) {
      const body = await res.text();
      return { data: null, error: `${res.status}: ${body}` };
    }
    if (res.status === 204) return { data: null, error: null };
    const data = await res.json();
    return { data, error: null };
  } catch (err) {
    return { data: null, error: err.message };
  }
}

// Health
export async function healthCheck() {
  return request("/api/health");
}

// Pipeline
export async function submitRun(name, mode, mutations, configOverrides = {}, enzymeId = null) {
  return request("/api/pipeline/run", {
    method: "POST",
    body: JSON.stringify({
      name, mode, mutations,
      config_overrides: configOverrides,
      ...(enzymeId ? { enzyme_id: enzymeId } : {}),
    }),
  });
}

// Enzymes
export async function getEnzymes() {
  return request("/api/pipeline/enzymes");
}

// Organisms
export async function getOrganisms() {
  return request("/api/pipeline/organisms");
}

export async function listJobs() {
  return request("/api/pipeline/jobs");
}

export async function getJob(jobId) {
  return request(`/api/pipeline/jobs/${jobId}`);
}

// Results
export async function getResults(jobId) {
  return request(`/api/results/${jobId}`);
}

export async function exportResults(jobId, format = "json") {
  try {
    const res = await fetch(`/api/results/${jobId}/export?format=${format}`);
    if (!res.ok) return { data: null, error: `${res.status}` };
    const blob = await res.blob();
    return { data: blob, error: null };
  } catch (err) {
    return { data: null, error: err.message };
  }
}

// UMAP embeddings
export async function getUmapData(jobId) {
  return request(`/api/results/${jobId}/umap`);
}

// Cross-reactivity matrix
export async function getCrossReactivity(jobId) {
  return request(`/api/results/${jobId}/cross-reactivity`);
}

// Spatially-addressed electrode array: pools, kinetics, specificity
export async function getPoolData(jobId) {
  return request(`/api/results/${jobId}/pools`);
}

// Figures — returns URL string, not a fetch
export function getFigureUrl(jobId, type) {
  return `/api/figures/${jobId}/${type}`;
}

// Panels
export async function listPanels() {
  return request("/api/panels");
}

export async function createPanel(name, description, mutations) {
  return request("/api/panels", {
    method: "POST",
    body: JSON.stringify({ name, description, mutations }),
  });
}

export async function deletePanel(panelId) {
  return request(`/api/panels/${panelId}`, { method: "DELETE" });
}

// Scoring
export async function listScoringModels() {
  return request("/api/scoring/models");
}

// Block 3: Optimisation
export async function getPresets() {
  return request("/api/v1/presets");
}

export async function getDiagnostics(jobId, preset = "balanced") {
  return request(`/api/v1/panel/${jobId}/diagnostics?preset=${preset}`);
}

export async function getWHOCompliance(jobId, preset = "balanced") {
  return request(`/api/v1/panel/${jobId}/who_compliance?preset=${preset}`);
}

export async function getTopK(jobId, targetLabel, k = 5) {
  return request(`/api/v1/panel/${jobId}/top_k/${encodeURIComponent(targetLabel)}?k=${k}`);
}

export async function runSweep(jobId, parameterName, values, basePreset = "balanced") {
  return request(`/api/v1/panel/${jobId}/sweep`, {
    method: "POST",
    body: JSON.stringify({ parameter_name: parameterName, values, base_preset: basePreset }),
  });
}

export async function runPareto(jobId, discValues = null, scoreValues = null) {
  return request(`/api/v1/panel/${jobId}/pareto`, {
    method: "POST",
    body: JSON.stringify({ disc_values: discValues, score_values: scoreValues }),
  });
}

// Research
export async function compareScorers(jobId, modelA = "heuristic", modelB = "compass_ml") {
  return request("/api/research/compare", {
    method: "POST",
    body: JSON.stringify({ job_id: jobId, model_a: modelA, model_b: modelB }),
  });
}

export async function getThermoProfile(jobId, targetLabel) {
  return request(`/api/research/thermo/${jobId}/${encodeURIComponent(targetLabel)}`);
}

export async function getThermoStandalone(spacer, pam = "TTTV") {
  return request(`/api/research/thermo/standalone?spacer=${encodeURIComponent(spacer)}&pam=${encodeURIComponent(pam)}`);
}

export async function getAblation() {
  return request("/api/research/ablation");
}

export async function addAblationRow(row) {
  return request("/api/research/ablation", {
    method: "POST",
    body: JSON.stringify(row),
  });
}

// Nuclease profiles
export async function getNucleaseProfiles() {
  return request("/api/research/nuclease/profiles");
}

export async function getNucleaseComparison() {
  return request("/api/research/nuclease/comparison");
}

// WebSocket for live progress
export function connectJobWS(jobId, onMessage, onClose) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/jobs/${jobId}`);
  ws.onmessage = (event) => onMessage(JSON.parse(event.data));
  ws.onclose = onClose || (() => {});
  ws.onerror = () => ws.close();
  return ws;
}
