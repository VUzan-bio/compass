import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  Brain, Check, ChevronDown, ChevronRight, Clock, Crosshair,
  Database, Droplet, Filter, Layers, Loader2, Play, Search, Settings, Zap,
  CheckCircle, Cpu, WifiOff,
} from "lucide-react";
import { T, FONT, HEADING, MONO } from "../tokens";
import { useIsMobile } from "../hooks/useIsMobile";
import { MUTATIONS, MODULES, MODULE_NAME_MAP, PROGRESS_TO_STEP, resolveStep, ORGANISMS } from "../mockData";
import { Badge, DrugBadge, Btn } from "../components/ui/index.jsx";
import { submitRun, getJob, getResults, connectJobWS } from "../api";

const DEFAULT_MUTS = [
  "rpoB_S531L", "rpoB_H526Y", "rpoB_D516V",
  "katG_S315T", "fabG1_C-15T",
  "embB_M306V", "embB_M306I",
  "pncA_H57D",
  "gyrA_D94G", "gyrA_A90V",
  "rrs_A1401G", "eis_C-14T",
];

const HomePage = ({ goTo, connected }) => {
  const mobile = useIsMobile();
  const [organism, setOrganism] = useState("mtb");
  const [runName, setRunName] = useState("COMPASS_panel_" + new Date().toISOString().slice(0, 10).replace(/-/g, ""));
  const [mode, setMode] = useState("standard");
  const [selectedModules, setSelectedModules] = useState(new Set(MODULES.map(m => m.id)));
  const [configOpen, setConfigOpen] = useState(false);
  const [organismSectionOpen, setOrganismSectionOpen] = useState(true);
  const [panelSectionOpen, setPanelSectionOpen] = useState(true);
  const [scorerSectionOpen, setScorerSectionOpen] = useState(true);
  const [scorer, setScorer] = useState(null); // "heuristic" | "compass_ml" | null
  const [enzymeId, setEnzymeId] = useState("enAsCas12a"); // "AsCas12a" | "enAsCas12a"
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState(null);

  // Active organism's mutation list
  const activeOrganism = ORGANISMS.find(o => o.id === organism) || ORGANISMS[0];
  const orgMutations = activeOrganism.mutations;

  /* ── Inline pipeline execution state ── */
  const [pipeJobId, setPipeJobId] = useState(null);
  const [pipeStep, setPipeStep] = useState(0);
  const [pipeDone, setPipeDone] = useState(false);
  const [pipeQueued, setPipeQueued] = useState(false);
  const [pipeStats, setPipeStats] = useState([]);
  const [pipeElapsed, setPipeElapsed] = useState(0);
  const [showLog, setShowLog] = useState(false);
  const [archOpen, setArchOpen] = useState(false);
  const [configCollapsed, setConfigCollapsed] = useState(false);
  const pipeStartRef = useRef(Date.now());
  const pipeStepStartRef = useRef(Date.now());
  const pipeTimerRef = useRef(null);
  const pipeWsRef = useRef(null);
  const pipePollRef = useRef(null);
  const prevPipeStep = useRef(-1);

  // Reset step timer whenever pipeStep changes (drives continuous progress bar)
  useEffect(() => { pipeStepStartRef.current = Date.now(); }, [pipeStep]);

  /* ── Preset panel definitions ── */
  const ALL_INDICES = orgMutations.map((_, i) => i);
  const CORE5_LABELS = organism === "mtb"
    ? ["rpoB_S531L", "katG_S315T", "fabG1_C-15T", "gyrA_D94G", "rrs_A1401G"]
    : orgMutations.filter(m => m.tier === 1).slice(0, 5).map(m => m.category === "gene_presence" ? m.gene : `${m.gene}_${m.ref}${m.pos}${m.alt}`);
  const mutLabel = (m) => m.category === "gene_presence" ? m.gene : `${m.gene}_${m.ref}${m.pos}${m.alt}`;
  const CORE5_INDICES = orgMutations.map((m, i) => CORE5_LABELS.includes(mutLabel(m)) ? i : -1).filter(i => i >= 0);

  const [panel, setPanel] = useState(null);        // "mdr14" | "mdr14_rnasep" | "core5" | "custom" | null
  const [selected, setSelected] = useState(new Set());
  const [targetsOpen, setTargetsOpen] = useState(false);

  const selectOrganism = (id) => {
    setOrganism(id);
    setPanel(null);
    setSelected(new Set());
  };

  const selectPanel = (p) => {
    setPanel(p);
    if (p === "mdr14" || p === "mdr14_rnasep" || p === "full") { setSelected(new Set(ALL_INDICES)); setTargetsOpen(false); }
    else if (p === "core5") { setSelected(new Set(CORE5_INDICES)); setTargetsOpen(false); }
    else { setTargetsOpen(true); }
  };

  const toggleMut = (i) => { const n = new Set(selected); n.has(i) ? n.delete(i) : n.add(i); setSelected(n); };
  const selectedDrugs = [...new Set([...selected].map(i => orgMutations[i]?.drug).filter(Boolean))];

  const launch = async () => {
    setLaunching(true);
    setError(null);
    const muts = [...selected].map(i => {
      const m = orgMutations[i];
      return m.category === "gene_presence"
        ? { gene: m.gene, mutation: "gene_presence", drug: m.drug || "OTHER" }
        : { gene: m.gene, ref_aa: m.ref, position: m.pos, alt_aa: m.alt, drug: m.drug || "OTHER" };
    });
    const apiMode = "full";
    const overrides = { ...(scorer !== "heuristic" ? { scorer } : {}), organism };
    if (connected) {
      const { data, error: err } = await submitRun(runName, apiMode, muts, overrides, enzymeId);
      if (err) { setError(err); setLaunching(false); return; }
      startInlinePipeline(data.job_id);
    } else {
      startInlinePipeline("mock-" + scorer + "-" + [...selected].join(",") + "-" + Date.now());
    }
  };

  /* ── Inline pipeline management ── */
  const cleanupPipeline = () => {
    if (pipeTimerRef.current) { clearInterval(pipeTimerRef.current); pipeTimerRef.current = null; }
    if (pipePollRef.current) { clearInterval(pipePollRef.current); pipePollRef.current = null; }
    if (pipeWsRef.current) { pipeWsRef.current.close(); pipeWsRef.current = null; }
  };

  const startInlinePipeline = (jobId) => {
    // Kill any previous run's timers/connections
    cleanupPipeline();

    setPipeJobId(jobId);
    setPipeStep(0);
    setPipeDone(false);
    setPipeQueued(false);
    setPipeStats([]);
    setPipeElapsed(0);
    prevPipeStep.current = -1;
    pipeStartRef.current = Date.now();
    pipeStepStartRef.current = Date.now();
    setLaunching(false);
    setConfigCollapsed(true);

    // Elapsed timer
    pipeTimerRef.current = setInterval(() => {
      setPipeElapsed((Date.now() - pipeStartRef.current) / 1000);
    }, 100);

    if (connected) {
      // WS for fast updates (best-effort; proxies often break this)
      try {
        const ws = connectJobWS(jobId,
          (msg) => {
            setPipeStep(prev => Math.max(prev, resolveStep(msg)));
            if (msg.status === "complete" || msg.status === "completed") {
              finishInlinePipeline(jobId);
            }
            if (msg.status === "failed") {
              setError(msg.error || "Pipeline failed");
              finishInlinePipeline(jobId);
            }
          },
          () => {}
        );
        pipeWsRef.current = ws;
      } catch { /* ignore */ }
      // Polling is the reliable path; always run it
      let pollFailCount = 0;
      pipePollRef.current = setInterval(async () => {
        const { data } = await getJob(jobId);
        if (!data) { pollFailCount++; if (pollFailCount > 5) { clearInterval(pipePollRef.current); setError("Job not found (server may have restarted)"); } return; }
        pollFailCount = 0;
        if (data.status === "pending") { setPipeQueued(true); return; }
        if (pipeQueued) setPipeQueued(false);
        setPipeStep(prev => Math.max(prev, resolveStep(data)));
        if (data.status === "complete" || data.status === "completed") {
          finishInlinePipeline(jobId);
        }
        if (data.status === "failed") {
          setError(data.error || "Pipeline failed");
          finishInlinePipeline(jobId);
        }
      }, 2000);
    } else {
      // Mock simulation
      let i = 0;
      const iv = setInterval(() => {
        if (i >= MODULES.length) { clearInterval(iv); finishInlinePipeline(jobId); return; }
        setPipeStep(i);
        i++;
      }, 800);
    }
  };

  const finishInlinePipeline = (jobId) => {
    setPipeDone(true);
    setShowLog(true);
    cleanupPipeline();

    if (connected) {
      getResults(jobId).then(({ data }) => {
        if (data?.module_stats?.length) setPipeStats(data.module_stats);
      });
    } else {
      const m5Detail = scorer === "compass_ml"
        ? "241 candidates scored: Compass-ML activity (0.125–0.608) · Compass-ML discrimination (0.288–0.959) · PAM-adjusted (0.045–0.608)"
        : "241 candidates scored: Heuristic QC (0.125–0.608) · SeqCNN calibrated T=1.1 (0.288–0.959) · PAM-adjusted (0.045–0.608)";
      setPipeStats([
        { module_id: "M1",   detail: "14 WHO catalogue mutations → genomic coordinates on H37Rv (NC_000962.3)", candidates_out: 14,  duration_ms: 1 },
        { module_id: "M2",   detail: "34,364 positions scanned → 1,797 PAM sites → 334 candidates",             candidates_out: 334, duration_ms: 98 },
        { module_id: "M3",   detail: "334 → 241 (93 removed: GC, homopolymer, Tm)",                             candidates_out: 241, duration_ms: 8 },
        { module_id: "M4",   detail: "241 → 222 (19 off-target hits, Bowtie2 ≤3 mismatches)",                   candidates_out: 222, duration_ms: 680 },
        { module_id: "M5",   detail: m5Detail,                                                                   candidates_out: 241, duration_ms: 10300 },
        { module_id: "M5.5", detail: "241 MUT/WT spacer pairs generated (84 direct, 157 proximity)",             candidates_out: 241, duration_ms: 4 },
        { module_id: "M6",   detail: "84 candidates evaluated, 66 enhanced (seed positions 2–6)",                candidates_out: 66,  duration_ms: 72 },
        { module_id: "M6.5", detail: "241 → 84 above 2× threshold (84 diagnostic-grade ≥3×)",                   candidates_out: 84,  duration_ms: 59 },
        { module_id: "M7",   detail: "241 → 14 selected (simulated annealing, 10,000 iterations)",               candidates_out: 14,  duration_ms: 2400 },
        { module_id: "M8",   detail: "14/14 primer pairs designed (6 standard, 8 AS-RPA)",                       candidates_out: 14,  duration_ms: 2400 },
        { module_id: "M8.5", detail: "AS-RPA disc: 8 scored | Dimer check: 78 flagged pairs",                    candidates_out: 14,  duration_ms: 234 },
        { module_id: "M9",   detail: "14 candidates + IS6110 species control → final 15-channel panel",          candidates_out: 15,  duration_ms: 10 },
        { module_id: "M10",  detail: "JSON + TSV + FASTA structured output",                                     candidates_out: 15,  duration_ms: 1 },
      ]);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => cleanupPipeline();
  }, []);

  /* Scorer-aware modules; M5 adapts to selected scoring model */
  const effectiveModules = useMemo(() => MODULES.map(m =>
    m.id === "M5" && scorer === "compass_ml"
      ? {
          ...m,
          name: "Compass-ML Scoring",
          execDesc: "Compass-ML (CNN + RNA-FM + RLPA) inference for efficiency and discrimination scoring",
          estSec: 180,
          substeps: [
            "Loading Compass-ML checkpoint (235K params, CNN + PAM + RNA-FM + RLPA)",
            "Downloading RNA-FM weights from HuggingFace (~1.1 GB)",
            "Computing RNA-FM embeddings (640-dim, frozen) per candidate",
            "Running multi-scale CNN branch (kernels 3/5/7)",
            "Applying R-loop propagation attention (RLPA, 34×34)",
            "Predicting efficiency + discrimination scores per candidate",
          ],
        }
      : m
  ), [scorer]);

  return (
    <div style={{ padding: mobile ? "24px 16px" : "32px 40px" }}>
      {/* Page title */}
      <div style={{ marginBottom: "24px" }}>
        <h1 style={{ fontSize: "20px", fontWeight: 600, color: T.text, margin: 0, fontFamily: HEADING, letterSpacing: "-0.01em" }}>Pipeline Configuration</h1>
      </div>

      {/* ── Run Workflow ── */}
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", marginBottom: "24px", overflow: "hidden" }}>

        {/* Collapsed header when pipeline is running */}
        {configCollapsed && pipeJobId && (
          <button onClick={() => setConfigCollapsed(!configCollapsed)} style={{
            width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "14px 24px", background: T.bgSub, border: "none", cursor: "pointer", fontFamily: FONT,
          }}>
            <div style={{ display: "flex", gap: "12px", alignItems: "center", fontSize: "13px" }}>
              <span style={{ fontWeight: 600, color: T.text }}>Pipeline Configuration</span>
              <span style={{ color: T.textSec, fontSize: "11px" }}>
                {activeOrganism.name} · {panel === "mdr14" ? "MDR-TB 14-plex" : panel === "mdr14_rnasep" ? "MDR-TB + RNaseP" : panel === "full" ? "Full panel" : panel === "core5" ? "Core 5-plex" : "Custom"} · {scorer === "compass_ml" ? "Compass-ML" : scorer === "heuristic" ? "Heuristic" : ""} · {selected.size} targets
              </span>
            </div>
            <ChevronDown size={14} color={T.textSec} style={{ transform: "rotate(0deg)", transition: "0.2s" }} />
          </button>
        )}

        <div style={{ padding: mobile ? "20px" : "32px", display: configCollapsed && pipeJobId ? "none" : "block" }}>

        {/* Expand/collapse toggle when pipeline running */}
        {pipeJobId && !configCollapsed && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "12px" }}>
            <button onClick={() => setConfigCollapsed(true)} style={{
              fontSize: "11px", color: T.textSec, background: "none", border: "none", cursor: "pointer", fontFamily: FONT,
              display: "flex", alignItems: "center", gap: "4px",
            }}>
              Collapse <ChevronDown size={12} color={T.textSec} style={{ transform: "rotate(180deg)" }} />
            </button>
          </div>
        )}

        {/* 1. Run Name; compact inline */}
        <div style={{ marginBottom: "24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <label style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: HEADING, flexShrink: 0 }}>Run Name</label>
            <input value={runName} onChange={(e) => setRunName(e.target.value)}
              style={{ flex: 1, padding: "10px 14px", border: `1px solid ${T.border}`, borderRadius: "4px", fontSize: "13px", fontFamily: MONO, color: T.text, background: T.bgSub, outline: "none", boxSizing: "border-box" }}
              placeholder="e.g. MDR-TB_14plex_v2"
            />
          </div>
        </div>

        <div style={{ height: 1, background: T.borderLight, margin: "0 0 24px" }} />

        {/* Organism Selector */}
        <div style={{ marginBottom: "16px" }}>
          <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
          <button onClick={() => setOrganismSectionOpen(!organismSectionOpen)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: "10px", padding: "14px 16px",
            background: "none", border: "none", cursor: "pointer", fontFamily: FONT,
          }}>
            <Database size={14} color={T.textSec} />
            <span style={{ fontSize: "13px", fontWeight: 600, color: T.text, flex: 1, textAlign: "left" }}>Target Organism</span>
            <span style={{ fontSize: "11px", color: T.primary, fontWeight: 600, marginRight: "4px" }}>{activeOrganism.name}</span>
            <ChevronDown size={14} color={T.textSec} style={{ transform: organismSectionOpen ? "rotate(180deg)" : "none", transition: "0.2s" }} />
          </button>
          {organismSectionOpen && (
            <div style={{ padding: "16px 20px", borderTop: `1px solid ${T.borderLight}` }}>
              <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "10px" }}>
                {ORGANISMS.map(org => (
                  <div key={org.id} onClick={() => selectOrganism(org.id)} style={{
                    padding: "14px 18px", borderRadius: "4px", cursor: "pointer",
                    border: organism === org.id ? `2px solid ${T.primary}` : `1px solid ${T.border}`,
                    background: organism === org.id ? T.primaryLight : T.bg,
                    display: "flex", flexDirection: "column", transition: "border-color 0.12s, background 0.12s",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                      <span style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, fontStyle: "italic" }}>{org.name}</span>
                      <span style={{ fontSize: "11px", fontWeight: 500, fontFamily: MONO, color: organism === org.id ? T.primary : T.textTer }}>{org.mutations.length} targets</span>
                    </div>
                    <div style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.5, marginBottom: "6px" }}>{org.description}</div>
                    <div style={{ display: "flex", gap: "12px", fontSize: "11px", color: T.textTer, borderTop: `1px solid ${organism === org.id ? T.primary + "30" : T.borderLight}`, paddingTop: "6px" }}>
                      <span>{org.reference}</span>
                      <span>GC {(org.gc * 100).toFixed(1)}%</span>
                      <span>{org.priority}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          </div>
        </div>

        {/* Diagnostic Panel */}
        <div style={{ marginBottom: "16px" }}>
          <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
          <button onClick={() => setPanelSectionOpen(!panelSectionOpen)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: "10px", padding: "14px 16px",
            background: "none", border: "none", cursor: "pointer", fontFamily: FONT,
          }}>
            <Layers size={14} color={T.textSec} />
            <span style={{ fontSize: "13px", fontWeight: 600, color: T.text, flex: 1, textAlign: "left" }}>Diagnostic Panel</span>
            <span style={{ fontSize: "11px", color: T.textTer, marginRight: "4px" }}>{panel ? (panel === "mdr14" ? "MDR-TB 14-plex" : panel === "mdr14_rnasep" ? "MDR-TB + RNaseP" : panel === "full" ? `Full ${activeOrganism.name} panel` : panel === "core5" ? "Core 5-plex" : "Custom") : "select panel"}</span>
            <ChevronDown size={14} color={T.textSec} style={{ transform: panelSectionOpen ? "rotate(180deg)" : "none", transition: "0.2s" }} />
          </button>
          {panelSectionOpen && (
            <div style={{ padding: "16px 20px", borderTop: `1px solid ${T.borderLight}` }}>
              {/* Preset cards; organism-aware */}
              <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "10px" }}>
                {[
                  ...(organism === "mtb" ? [
                    { id: "mdr14", name: "MDR-TB 14-plex", targets: ALL_INDICES.length + " targets",
                      desc: "Full WHO-catalogued first- and second-line resistance panel.",
                      meta: ["6 drug classes", "Tier 1\u20132", "High + Moderate"] },
                    { id: "mdr14_rnasep", name: "MDR-TB 14-plex + RNaseP", targets: (ALL_INDICES.length + 1) + " targets",
                      desc: "Full MDR panel plus human RNaseP (RPPH1) extraction control.",
                      meta: ["6 drug classes", "+ extraction ctrl", "CDC standard"] },
                  ] : [
                    { id: "full", name: `Full ${activeOrganism.name} panel`, targets: ALL_INDICES.length + " targets",
                      desc: `All catalogued AMR targets for ${activeOrganism.name}.`,
                      meta: [[...new Set(orgMutations.map(m => m.drug))].length + " drug classes", "All tiers"] },
                  ]),
                  { id: "core5", name: "Core 5-plex", targets: CORE5_INDICES.length + " targets",
                    desc: "High-confidence tier-1 mutations only. Point-of-care screening.",
                    meta: [[...new Set(CORE5_INDICES.map(i => orgMutations[i]?.drug))].length + " drug classes", "Tier 1", "High confidence"] },
                  { id: "custom", name: "Custom Panel", targets: panel === "custom" ? selected.size + " targets" : "",
                    desc: "Select individual mutations for targeted re-design or validation.",
                    meta: [] },
                ].map(p => (
                  <div key={p.id} onClick={() => selectPanel(p.id)} style={{
                    padding: "16px 20px", borderRadius: "4px", cursor: "pointer",
                    border: panel === p.id ? `2px solid ${T.primary}` : `1px solid ${T.border}`,
                    background: panel === p.id ? T.primaryLight : T.bg,
                    display: "flex", flexDirection: "column", transition: "border-color 0.12s, background 0.12s",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                      <span style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>{p.name}</span>
                      {p.targets && <span style={{ fontSize: "13px", fontWeight: 500, fontFamily: MONO, color: panel === p.id ? T.primary : T.textTer }}>{p.targets}</span>}
                    </div>
                    <div style={{ fontSize: "13px", color: T.textSec, lineHeight: 1.5, flex: 1, marginBottom: p.meta.length ? "8px" : "0" }}>{p.desc}</div>
                    {p.meta.length > 0 && (
                      <div style={{ display: "flex", gap: "12px", fontSize: "12px", color: T.textTer, borderTop: `1px solid ${panel === p.id ? T.primary + "30" : T.borderLight}`, paddingTop: "8px" }}>
                        {p.meta.map((label, j) => <span key={j}>{label}</span>)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          </div>
        </div>

        {/* Scoring Model */}
        <div style={{ marginBottom: "16px" }}>
          <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
          <button onClick={() => setScorerSectionOpen(!scorerSectionOpen)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: "10px", padding: "14px 16px",
            background: "none", border: "none", cursor: "pointer", fontFamily: FONT,
          }}>
            <Brain size={14} color={T.textSec} />
            <span style={{ fontSize: "13px", fontWeight: 600, color: T.text, flex: 1, textAlign: "left" }}>Scoring Model</span>
            <span style={{ fontSize: "11px", color: T.textTer, marginRight: "4px" }}>{scorer === "compass_ml" ? "Compass-ML" : scorer === "heuristic" ? "Heuristic" : "select model"}</span>
            <ChevronDown size={14} color={T.textSec} style={{ transform: scorerSectionOpen ? "rotate(180deg)" : "none", transition: "0.2s" }} />
          </button>
          {scorerSectionOpen && (
            <div style={{ padding: "16px 20px", borderTop: `1px solid ${T.borderLight}` }}>
              <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "10px" }}>
                {[
                  { id: "heuristic", label: "Heuristic", desc: "Position-weighted composite across 5 biophysical features.", tag: "Baseline" },
                  { id: "compass_ml", label: "Compass-ML", desc: "Dual-branch CNN & RNA-FM with R-loop propagation attention.", tag: "Recommended" },
                ].map(s => (
                  <button key={s.id} onClick={() => setScorer(s.id)} style={{
                    padding: "16px 20px", borderRadius: "4px", cursor: "pointer", fontFamily: FONT, textAlign: "left",
                    border: scorer === s.id ? `2px solid ${T.primary}` : `1px solid ${T.border}`,
                    background: scorer === s.id ? T.primaryLight : T.bg, transition: "all 0.15s",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                      <span style={{ fontSize: "14px", fontWeight: 600, color: scorer === s.id ? T.primaryDark : T.text, fontFamily: HEADING }}>{s.label}</span>
                      <span style={{ fontSize: "10px", fontWeight: 600, fontFamily: MONO, padding: "2px 8px", borderRadius: "3px", background: s.id === "compass_ml" ? T.primaryLight : T.bgSub, color: s.id === "compass_ml" ? T.primary : T.textTer }}>{s.tag}</span>
                    </div>
                    <div style={{ fontSize: "13px", color: scorer === s.id ? T.primaryDark : T.textSec, lineHeight: 1.5 }}>{s.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
          </div>
        </div>

        {/* Mutations selected (collapsible card) */}
        <div style={{ marginBottom: "16px" }}>
          <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
            <button onClick={() => setTargetsOpen(!targetsOpen)} style={{
              width: "100%", display: "flex", alignItems: "center", gap: "12px", padding: "14px 16px",
              background: "none", border: "none", cursor: "pointer", fontFamily: FONT,
            }}>
              <span style={{ fontSize: "13px", fontWeight: 600, color: T.text }}>{selected.size} mutations selected</span>
              <div style={{ display: "flex", gap: "4px", flex: 1 }}>
                {selectedDrugs.map(d => <DrugBadge key={d} drug={d} />)}
              </div>
              <span style={{ fontSize: "11px", color: T.textSec, marginRight: "4px" }}>View targets</span>
              <ChevronDown size={14} color={T.textSec} style={{ transform: targetsOpen ? "rotate(180deg)" : "none", transition: "0.2s" }} />
            </button>

            {/* Expanded table */}
            {targetsOpen && (
              <div style={{ borderTop: `1px solid ${T.borderLight}` }}>
                {/* Drug filter chips; only for Custom panel */}
                {panel === "custom" && (
                  <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", padding: "12px 16px", borderBottom: `1px solid ${T.borderLight}` }}>
                    <button onClick={() => setSelected(new Set(ALL_INDICES))} style={{ padding: "5px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, background: selected.size === orgMutations.length ? T.primary : T.bg, color: selected.size === orgMutations.length ? "#fff" : T.textSec, fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: FONT }}>All ({orgMutations.length})</button>
                    <button onClick={() => setSelected(new Set())} style={{ padding: "5px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, background: selected.size === 0 ? T.bgSub : T.bg, color: T.textSec, fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: FONT }}>None</button>
                    <div style={{ width: 1, background: T.border, margin: "0 4px" }} />
                    {[...new Set(orgMutations.map(m => m.drug))].map(drug => {
                      const indices = orgMutations.map((m, i) => m.drug === drug ? i : -1).filter(i => i >= 0);
                      const allSel = indices.every(i => selected.has(i));
                      return (
                        <button key={drug} onClick={() => {
                          const n = new Set(selected);
                          indices.forEach(i => allSel ? n.delete(i) : n.add(i));
                          setSelected(n);
                        }} style={{ padding: "5px 12px", borderRadius: "4px", border: `1px solid ${allSel ? T.primary : T.border}`, background: allSel ? T.primaryLight : T.bg, color: allSel ? T.primary : T.textSec, fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: FONT }}>
                          {drug} ({indices.length})
                        </button>
                      );
                    })}
                  </div>
                )}

                {/* Mutation table */}
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
                  <thead>
                    <tr style={{ background: T.bg }}>
                      {(panel === "custom" ? ["", "Gene", "Mutation", "Drug", "WHO Confidence", "Tier"] : ["Gene", "Mutation", "Drug", "WHO Confidence", "Tier"]).map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "10px 12px", fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.borderLight}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {orgMutations.map((m, i) => {
                      if (panel !== "custom" && !selected.has(i)) return null;
                      const isCustom = panel === "custom";
                      return (
                        <tr key={i} onClick={isCustom ? () => toggleMut(i) : undefined} style={{ cursor: isCustom ? "pointer" : "default", borderBottom: `1px solid ${T.borderLight}`, background: isCustom && selected.has(i) ? T.primaryLight + "40" : "transparent", transition: "background 0.1s" }}>
                          {isCustom && (
                            <td style={{ padding: "10px 12px", width: 32 }}>
                              <div style={{
                                width: 16, height: 16, borderRadius: "4px",
                                border: `2px solid ${selected.has(i) ? T.primary : T.border}`,
                                background: selected.has(i) ? T.primary : "transparent",
                                display: "flex", alignItems: "center", justifyContent: "center",
                              }}>{selected.has(i) && <Check size={10} color="#fff" strokeWidth={3} />}</div>
                            </td>
                          )}
                          <td style={{ padding: "10px 12px", fontWeight: 600, fontFamily: MONO, color: T.text, fontSize: "12px" }}>{m.gene}</td>
                          <td style={{ padding: "10px 12px", fontFamily: MONO, fontSize: "12px", color: T.textSec }}>{m.category === "gene_presence" ? "presence" : `${m.ref}${m.pos}${m.alt}`}</td>
                          <td style={{ padding: "10px 12px" }}><DrugBadge drug={m.drug} /></td>
                          <td style={{ padding: "10px 12px" }}><Badge variant={m.conf === "High" ? "success" : "warning"}>{m.conf}</Badge></td>
                          <td style={{ padding: "10px 12px" }}><Badge variant={m.tier === 1 ? "primary" : "default"}>Tier {m.tier}</Badge></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Advanced Configuration (collapsible card; matching mutations card design) */}
        <div style={{ marginBottom: "28px" }}>
          <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
          <button onClick={() => setConfigOpen(!configOpen)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: "10px", padding: "14px 16px",
            background: "none", border: "none", cursor: "pointer", fontFamily: FONT,
          }}>
            <Settings size={14} color={T.textSec} />
            <span style={{ fontSize: "13px", fontWeight: 600, color: T.text, flex: 1, textAlign: "left" }}>Advanced Configuration</span>
            <span style={{ fontSize: "11px", color: T.textTer, marginRight: "4px" }}>defaults</span>
            <ChevronDown size={14} color={T.textSec} style={{ transform: configOpen ? "rotate(180deg)" : "none", transition: "0.2s" }} />
          </button>
          {configOpen && (
            <div style={{ padding: "16px 20px", borderTop: `1px solid ${T.borderLight}` }}>
              {/* Pipeline mode toggle */}
              <div style={{ marginBottom: "16px" }}>
                <div style={{ fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "8px" }}>Pipeline Mode</div>
                <div style={{ display: "flex", gap: "6px" }}>
                  {[
                    { id: "standard", label: "Standard", tip: "All 11 modules" },
                    { id: "custom", label: "Custom", tip: "Select modules" },
                  ].map(m => (
                    <button key={m.id} onClick={() => { setMode(m.id); if (m.id === "standard") setSelectedModules(new Set(MODULES.map(x => x.id))); }}
                      style={{
                        padding: "6px 14px", borderRadius: "4px", fontSize: "12px", fontWeight: 600, fontFamily: FONT, cursor: "pointer",
                        border: `1px solid ${mode === m.id ? T.primary : T.border}`,
                        background: mode === m.id ? T.primaryLight : T.bg,
                        color: mode === m.id ? T.primaryDark : T.textSec,
                      }}
                    >{m.label}</button>
                  ))}
                </div>
              </div>
              {/* Module selection for custom mode */}
              {mode === "custom" && (
                <div style={{ marginBottom: "16px", background: T.bg, border: `1px solid ${T.borderLight}`, borderRadius: "4px", padding: "12px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <span style={{ fontSize: "12px", fontWeight: 600, color: T.text }}>Modules</span>
                    <div style={{ display: "flex", gap: "4px" }}>
                      <button onClick={() => setSelectedModules(new Set(MODULES.map(x => x.id)))} style={{ padding: "3px 8px", borderRadius: "4px", border: `1px solid ${T.border}`, background: T.bg, color: T.textSec, fontSize: "10px", fontWeight: 600, cursor: "pointer", fontFamily: FONT }}>All</button>
                      <button onClick={() => setSelectedModules(new Set())} style={{ padding: "3px 8px", borderRadius: "4px", border: `1px solid ${T.border}`, background: T.bg, color: T.textSec, fontSize: "10px", fontWeight: 600, cursor: "pointer", fontFamily: FONT }}>None</button>
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr 1fr" : "1fr 1fr 1fr 1fr", gap: "4px" }}>
                    {MODULES.map(m => {
                      const sel = selectedModules.has(m.id);
                      return (
                        <div key={m.id} onClick={() => { const n = new Set(selectedModules); sel ? n.delete(m.id) : n.add(m.id); setSelectedModules(n); }} style={{
                          display: "flex", alignItems: "center", gap: "6px", padding: "6px 8px", borderRadius: "4px", cursor: "pointer",
                          border: `1px solid ${sel ? T.primary + "50" : T.borderLight}`,
                          background: sel ? T.primaryLight + "60" : "transparent", fontSize: "11px",
                        }}>
                          <div style={{ width: 14, height: 14, borderRadius: "3px", border: `2px solid ${sel ? T.primary : T.border}`, background: sel ? T.primary : "transparent", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                            {sel && <Check size={8} color="#fff" strokeWidth={3} />}
                          </div>
                          <span style={{ fontFamily: MONO, fontWeight: 600, color: T.text, fontSize: "10px" }}>{m.id}</span>
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ marginTop: "6px", fontSize: "10px", color: T.textTer }}>{selectedModules.size}/{MODULES.length} modules</div>
                </div>
              )}
              {/* Enzyme selector */}
              <div style={{ marginBottom: "16px" }}>
                <div style={{ fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "8px" }}>Cas12a Variant</div>
                <div style={{ display: "flex", gap: "6px" }}>
                  {[
                    { id: "AsCas12a", label: "WT AsCas12a", pams: "TTTV", tip: "Canonical PAM only" },
                    { id: "enAsCas12a", label: "enAsCas12a", pams: "9 PAMs", tip: "Expanded PAM (Kleinstiver 2019)" },
                  ].map(e => (
                    <button key={e.id} onClick={() => setEnzymeId(e.id)}
                      style={{
                        flex: 1, padding: "10px 12px", borderRadius: "4px", cursor: "pointer", textAlign: "left",
                        border: `2px solid ${enzymeId === e.id ? T.primary : T.border}`,
                        background: enzymeId === e.id ? T.primaryLight : T.bg,
                        fontFamily: FONT, transition: "all 0.15s",
                      }}
                    >
                      <div style={{ fontSize: "12px", fontWeight: 600, color: enzymeId === e.id ? T.primaryDark : T.text, marginBottom: "2px" }}>{e.label}</div>
                      <div style={{ fontSize: "10px", color: T.textTer }}>{e.tip}</div>
                      <div style={{ marginTop: "4px", display: "inline-block", padding: "2px 6px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, fontFamily: MONO, background: enzymeId === e.id ? T.primary + "20" : T.bgSub, color: enzymeId === e.id ? T.primary : T.textSec }}>{e.pams}</div>
                    </button>
                  ))}
                </div>
                {enzymeId === "enAsCas12a" && (
                  <div style={{ marginTop: "8px", padding: "8px 12px", borderRadius: "4px", background: T.primaryLight + "40", border: `1px solid ${T.primary}20`, fontSize: "10px", color: T.textSec, lineHeight: 1.5 }}>
                    <strong style={{ color: T.primaryDark }}>enAsCas12a</strong> (E174R/S542R/K548R) recognizes 9 PAM variants with activity penalties from Kleinstiver et al. 2019. Non-canonical PAMs receive a multiplicative score penalty: TTTT 0.75×, TTCV 0.65×, TATV 0.55×, CTTV 0.45×, TCTV 0.40×, TGTV 0.35×, ATTV 0.30×, GTTV 0.25×. Note: TTTT is not covered by TTTV (V = A/C/G) and is a distinct expanded recognition.
                  </div>
                )}
              </div>
              {/* Parameter defaults */}
              <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "0 32px" }}>
                {[
                  ["PAM Patterns", enzymeId === "enAsCas12a" ? "TTTV + 8 expanded (incl. TTTT)" : "TTTV only"],
                  ["Spacer Lengths", "18–23 nt"], ["GC Range", "40–85% (TB-adjusted)"],
                  ["Min Discrimination", "2.0×"], ["SM Enhancement", "Enabled"],
                  ["RPA Amplicon", "80–120 bp (blood cfDNA)"],
                  ["Scoring Model", scorer === "compass_ml" ? "Compass-ML" : "Heuristic"],
                  ["PAM Penalty", enzymeId === "enAsCas12a" ? "Kleinstiver 2019" : "N/A (canonical only)"],
                ].map(([k, v]) => (
                  <div key={k}>
                    <div style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: `1px solid ${T.borderLight}`, fontSize: "12px" }}>
                      <span style={{ color: T.textSec }}>{k}</span>
                      <span style={{ fontWeight: 600, color: T.text, fontFamily: MONO, fontSize: "11px" }}>{v}</span>
                    </div>
                    {k === "RPA Amplicon" && (
                      <div style={{ display: "flex", alignItems: "flex-start", gap: "6px", padding: "6px 0 8px", fontSize: "10px", color: T.textTer, lineHeight: 1.5 }}>
                        <Droplet size={10} color={T.textTer} strokeWidth={2} style={{ flexShrink: 0, marginTop: 2 }} />
                        <span>Capped at 120 bp. cfDNA fragments in blood are ~100-160 bp. Shorter amplicons maximise template capture from fragmented circulating DNA.</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: T.border, margin: "0 0 20px" }} />

        {/* Summary + Launch */}
        {error && <div style={{ color: T.danger, fontSize: "12px", marginBottom: "12px" }}>{error}</div>}
        <div style={{ display: "flex", alignItems: mobile ? "stretch" : "center", flexDirection: mobile ? "column" : "row", justifyContent: "space-between", gap: mobile ? "12px" : "16px" }}>
          <div style={{ display: "flex", gap: "16px", fontSize: "13px", flexWrap: "wrap", alignItems: "center", color: T.textSec }}>
            <span>{selected.size} targets</span>
            <span style={{ color: T.borderStrong }}>·</span>
            <span>{[...new Set([...selected].map(i => orgMutations[i]?.drug))].length} drug classes</span>
            <span style={{ color: T.borderStrong }}>·</span>
            <span>{mode === "custom" ? selectedModules.size : MODULES.length} modules</span>
          </div>
          <Btn icon={launching ? Loader2 : Play} onClick={launch} disabled={launching || selected.size === 0 || !scorer || !!pipeJobId}>
            {launching ? "Launching…" : pipeJobId ? (pipeDone ? "Complete" : "Running…") : "Launch Pipeline"}
          </Btn>
        </div>
        </div>{/* close inner padding div */}
      </div>

      {/* ═══ INLINE PIPELINE EXECUTION ═══ */}
      {pipeJobId && (() => {
        const activeModule = effectiveModules[pipeStep] || effectiveModules[0];
        const ActiveIcon = activeModule.icon;
        const statMap = {};
        pipeStats.forEach(s => { statMap[s.module_id] = s; });
        const fmtDur = (ms) => ms >= 60000 ? `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s` : ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
        const fmtElapsed = (sec) => {
          const m = Math.floor(sec / 60);
          const s = Math.floor(sec % 60);
          return m > 0 ? `${m}m ${s}s` : `${sec.toFixed(1)}s`;
        };
        const m2Out = statMap["M2"]?.candidates_out || 0;
        const finalSize = statMap["M9"]?.candidates_out || statMap["M7"]?.candidates_out || 0;

        return (
          <div style={{
            background: T.bg,
            border: `1px solid ${T.border}`,
            borderRadius: "4px",
            marginBottom: "24px", overflow: "hidden",
          }}>
            {/* Queued state; waiting for previous run */}
            {!pipeDone && pipeQueued && (
              <div style={{ padding: "16px 20px", display: "flex", alignItems: "center", gap: "10px" }}>
                <svg width="14" height="14" viewBox="0 0 16 16" style={{ animation: "spin 1s linear infinite", flexShrink: 0 }}>
                  <circle cx="8" cy="8" r="6" fill="none" stroke={T.border} strokeWidth="2" />
                  <path d="M8 2a6 6 0 0 1 6 6" fill="none" stroke={T.textTer} strokeWidth="2" strokeLinecap="round" />
                </svg>
                <span style={{ fontSize: "13px", color: T.textSec }}>Queued, waiting for previous run to complete</span>
                <span style={{ fontFamily: MONO, fontSize: "12px", color: T.textTer, marginLeft: "auto", fontVariantNumeric: "tabular-nums" }}>{fmtElapsed(pipeElapsed)}</span>
              </div>
            )}
            {/* Running state; continuous progress bar + current module + collapsible timeline */}
            {!pipeDone && !pipeQueued && (() => {
              const stepEstSec = activeModule.estSec || 10;
              const stepElapsed = (Date.now() - pipeStepStartRef.current) / 1000;
              const subs = activeModule.substeps || [activeModule.execDesc];
              const subDur = stepEstSec / subs.length;
              const subIdx = Math.min(Math.floor(stepElapsed / Math.max(subDur, 0.5)), subs.length - 1);
              const intraStep = Math.min(0.95, 1 - Math.exp(-2.5 * stepElapsed / stepEstSec));
              // Smooth overall % based on cumulative estSec weights
              const totalEstSec = effectiveModules.reduce((s, m) => s + (m.estSec || 10), 0);
              const doneEstSec = effectiveModules.slice(0, pipeStep).reduce((s, m) => s + (m.estSec || 10), 0);
              const curEstSec = activeModule.estSec || 10;
              const pct = Math.min(99, ((doneEstSec + curEstSec * intraStep) / totalEstSec) * 100);
              const remainingStepSec = Math.max(0, stepEstSec - stepElapsed);
              const futureModulesSec = effectiveModules.slice(pipeStep + 1).reduce((s, m) => s + (m.estSec || 10), 0);
              const remainingSec = Math.ceil(remainingStepSec + futureModulesSec);
              const timeText = remainingSec >= 120 ? `~${Math.round(remainingSec / 60)} min left`
                : remainingSec >= 60 ? `~1 min left`
                : `~${remainingSec}s left`;
              return (
                <div style={{ padding: "16px 20px" }}>
                  {/* Header: elapsed + progress % + time remaining */}
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "10px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <svg width="14" height="14" viewBox="0 0 16 16" style={{ animation: "spin 1s linear infinite", flexShrink: 0 }}>
                        <circle cx="8" cy="8" r="6" fill="none" stroke={T.border} strokeWidth="2" />
                        <path d="M8 2a6 6 0 0 1 6 6" fill="none" stroke={T.primary} strokeWidth="2" strokeLinecap="round" />
                      </svg>
                      <span style={{ fontFamily: MONO, fontSize: "12px", color: T.text, fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>
                        {fmtElapsed(pipeElapsed)}
                      </span>
                      <span style={{ fontSize: "11px", color: T.textTer }}>{"\u00b7"} {pipeStep + 1}/{effectiveModules.length} modules</span>
                    </div>
                    <span style={{ fontFamily: MONO, fontSize: "11px", color: T.textTer, fontVariantNumeric: "tabular-nums" }}>
                      {Math.round(pct)}% {"\u00b7"} {timeText}
                    </span>
                  </div>
                  {/* Continuous progress bar */}
                  <div style={{ height: "6px", borderRadius: "3px", background: T.bgHover, overflow: "hidden" }}>
                    <div style={{
                      height: "100%", borderRadius: "3px",
                      background: `linear-gradient(90deg, ${T.primary}, ${T.primary}dd)`,
                      width: `${pct}%`,
                      transition: "width 400ms ease-out",
                    }} />
                  </div>
                  {/* Current module + substep */}
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "12px" }}>
                    <div style={{ width: 18, height: 18, display: "flex", alignItems: "center", justifyContent: "center", animation: "subtlePulse 2s ease-in-out infinite", flexShrink: 0 }}>
                      <ActiveIcon size={14} color={T.primary} strokeWidth={1.8} />
                    </div>
                    <div key={pipeStep} style={{ display: "flex", alignItems: "baseline", gap: "8px", animation: "stepSwipeUp 0.25s ease-out" }}>
                      <span style={{ fontFamily: MONO, fontSize: "10px", color: T.textTer }}>{activeModule.id}</span>
                      <span style={{ fontSize: "13px", fontWeight: 500, color: T.text }}>{activeModule.name}</span>
                    </div>
                    <span style={{ fontSize: "11px", color: T.textSec }}>{"\u2014"}</span>
                    <div key={`sub-${pipeStep}-${subIdx}`} style={{ fontSize: "11px", color: T.textSec, animation: "substepSwipe 0.35s ease-out" }}>
                      {subs[subIdx]}
                    </div>
                  </div>
                  {/* Collapsible module timeline */}
                  <details style={{ marginTop: "12px" }}>
                    <summary style={{
                      fontSize: "11px", color: T.textTer, cursor: "pointer", fontFamily: FONT,
                      listStyle: "none", display: "flex", alignItems: "center", gap: "4px", userSelect: "none",
                    }}>
                      <ChevronDown size={11} color={T.textTer} strokeWidth={1.8} style={{ transition: "0.2s" }} />
                      Module details
                    </summary>
                    <div style={{ marginTop: "8px", paddingTop: "8px", borderTop: `1px solid ${T.border}` }}>
                      {effectiveModules.map((m, idx) => {
                        const Icon = m.icon;
                        const isDone = idx < pipeStep;
                        const isCurrent = idx === pipeStep;
                        const isLast = idx === effectiveModules.length - 1;
                        const mSubs = m.substeps || [m.execDesc];
                        return (
                          <div key={m.id} style={{ display: "flex", gap: "0", marginBottom: isLast ? 0 : "2px" }}>
                            {/* Timeline rail */}
                            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "24px", flexShrink: 0 }}>
                              <div style={{
                                width: "20px", height: "20px", borderRadius: "5px", display: "flex", alignItems: "center", justifyContent: "center",
                                background: isDone ? T.primary + "18" : isCurrent ? T.primary + "18" : T.bgSub,
                                border: `1px solid ${isDone ? T.primary + "40" : isCurrent ? T.primary + "60" : T.border}`,
                              }}>
                                {isDone ? <Check size={10} color={T.success} strokeWidth={2.5} />
                                  : <Icon size={10} color={isCurrent ? T.primary : T.textTer} strokeWidth={1.8} />}
                              </div>
                              {!isLast && <div style={{ width: "1px", flex: 1, minHeight: "4px", background: isDone ? T.primary + "30" : T.border }} />}
                            </div>
                            {/* Content */}
                            <div style={{ flex: 1, paddingLeft: "10px", paddingBottom: isLast ? 0 : "4px" }}>
                              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                <span style={{ fontFamily: MONO, fontSize: "9px", color: T.textTer }}>{m.id}</span>
                                <span style={{ fontSize: "12px", fontWeight: isCurrent ? 600 : 400, color: isDone || isCurrent ? T.text : T.textTer }}>{m.name}</span>
                                {isCurrent && <span style={{ fontSize: "9px", color: T.primary, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px" }}>running</span>}
                              </div>
                              {isCurrent && (
                                <div style={{ marginTop: "4px", display: "flex", flexDirection: "column", gap: "2px", paddingLeft: "2px" }}>
                                  {mSubs.map((sub, si) => (
                                    <div key={si} style={{ fontSize: "10px", color: si <= subIdx ? T.textSec : T.textTer, lineHeight: 1.4, display: "flex", alignItems: "center", gap: "4px" }}>
                                      {si < subIdx ? <Check size={8} color={T.success} strokeWidth={2.5} /> : si === subIdx ? <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: T.primary, animation: "subtlePulse 2s ease-in-out infinite" }} /> : <span style={{ display: "inline-block", width: 8 }} />}
                                      {sub}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </details>
                </div>
              );
            })()}

            {/* Complete state; summary + logs toggle + CTA */}
            {pipeDone && (
              <div style={{ padding: "16px 20px" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <CheckCircle size={18} color={T.text} strokeWidth={2} />
                    <span style={{ fontSize: "13px", fontWeight: 500, color: T.text, fontFamily: FONT }}>
                      Pipeline complete
                    </span>
                    <span style={{ fontSize: "13px", color: T.textSec, fontFamily: MONO }}>
                      {fmtElapsed(pipeElapsed)}
                      {m2Out > 0 && ` · ${m2Out} candidates`}
                      {finalSize > 0 && ` · ${finalSize} selected`}
                    </span>
                  </div>
                  <button
                    onClick={() => goTo("results", { jobId: pipeJobId, scorer })}
                    style={{
                      padding: "8px 20px", borderRadius: "6px",
                      background: T.primary, color: "#fff", border: "none",
                      fontSize: "13px", fontWeight: 500, fontFamily: FONT,
                      cursor: "pointer",
                    }}
                  >
                    View Results
                  </button>
                </div>

                {/* Logs toggle */}
                <button onClick={() => setShowLog(!showLog)} style={{
                  background: "none", border: "none", cursor: "pointer", fontFamily: FONT,
                  fontSize: "11px", color: T.textTer, display: "flex", alignItems: "center", gap: "4px", padding: 0,
                }}>
                  <ChevronDown size={12} style={{ transform: showLog ? "rotate(180deg)" : "none", transition: "0.2s" }} />
                  {showLog ? "Hide" : "Show"} execution log
                </button>

                {showLog && (
                  <div style={{ marginTop: "12px", paddingTop: "12px", borderTop: `1px solid ${T.border}` }}>
                    {effectiveModules.map((m, idx) => {
                      const st = statMap[m.id];
                      const Icon = m.icon;
                      const isLast = idx === effectiveModules.length - 1;
                      const subs = m.substeps || [m.execDesc];
                      return (
                        <div key={m.id} style={{ display: "flex", gap: "0", marginBottom: isLast ? 0 : "4px" }}>
                          {/* Timeline rail */}
                          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "28px", flexShrink: 0 }}>
                            <div style={{ width: "24px", height: "24px", borderRadius: "6px", background: T.bgSub, border: `1px solid ${T.border}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <Icon size={12} color={st ? T.text : T.textTer} strokeWidth={1.8} />
                            </div>
                            {!isLast && <div style={{ width: "1px", flex: 1, minHeight: "8px", background: T.border }} />}
                          </div>
                          {/* Content */}
                          <div style={{ flex: 1, paddingLeft: "12px", paddingBottom: isLast ? 0 : "8px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "2px" }}>
                              <span style={{ fontFamily: MONO, fontSize: "10px", color: T.textTer, minWidth: "28px" }}>{m.id}</span>
                              <span style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>{m.name}</span>
                              {st && <span style={{ fontFamily: MONO, fontSize: "10px", color: T.success, marginLeft: "auto", display: "flex", alignItems: "center", gap: "4px" }}>
                                <Check size={10} color={T.success} strokeWidth={2.5} />
                                {fmtDur(st.duration_ms)}
                              </span>}
                            </div>
                            <div style={{ fontSize: "11px", color: T.textSec, marginBottom: "6px", lineHeight: 1.4 }}>{m.desc}</div>
                            {/* Substeps */}
                            <div style={{ display: "flex", flexDirection: "column", gap: "2px", paddingLeft: "2px" }}>
                              {subs.map((sub, si) => (
                                <div key={si} style={{ fontSize: "11px", color: T.textTer, lineHeight: 1.5 }}>
                                  {sub}
                                </div>
                              ))}
                            </div>
                            {st?.detail && <div style={{ fontSize: "11px", color: T.primary, marginTop: "4px", fontWeight: 500 }}>{st.detail}</div>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })()}

      {/* Methods content moved to MethodsPage */}
    </div>
  );
};

export { DEFAULT_MUTS, HomePage };
