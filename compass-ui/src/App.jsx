import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { createPortal } from "react-dom";
import {
  Activity, BarChart3, BookOpen, Check, ChevronDown, ChevronRight, Clock, Copy,
  Database, Download, ExternalLink, Eye, FileText, Filter, FlaskConical,
  Folder, GitBranch, Grid3x3, Layers, List, Loader2,
  Lock, Menu, Package, PanelLeft, PanelLeftClose, Pencil, Play, Plus, RefreshCw, Search, Settings, Target,
  Trash2, TrendingUp, X, Zap, Shield, Crosshair, Brain, Cpu, Wifi, WifiOff,
  AlertTriangle, CheckCircle, Info, Map, Droplet,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, Legend, ComposedChart, ReferenceLine,
  LineChart, Line, Area, AreaChart,
} from "recharts";
import ChipRender3D from "./ChipRender3D";
import {
  healthCheck, submitRun, getJob, getResults, exportResults,
  getFigureUrl, listPanels, createPanel, listJobs, connectJobWS,
  listScoringModels, getPresets, getDiagnostics, getWHOCompliance,
  getTopK, runSweep, runPareto,
  compareScorers, getThermoProfile, getThermoStandalone, getAblation,
  getNucleaseProfiles, getNucleaseComparison, getUmapData, getPoolData,
  getEnzymes,
} from "./api";

/* ═══════════════════════════════════════════════════════════════════
   DESIGN TOKENS — Institutional Biotech (Benchling-style)
   ═══════════════════════════════════════════════════════════════════ */
const T = {
  bg: "#FFFFFF", bgSub: "#F9FAFB", bgHover: "#F3F4F6",
  border: "#E5E7EB", borderLight: "#F3F4F6", borderStrong: "#D1D5DB",
  text: "#111827", textSec: "#6B7280", textTer: "#9CA3AF",
  primary: "#2563EB", primaryLight: "#EFF6FF", primaryDark: "#1D4ED8", primarySub: "#BFDBFE",
  success: "#059669", successLight: "#ECFDF5",
  warning: "#D97706", warningLight: "#FFFBEB",
  danger: "#DC2626", dangerLight: "#FEF2F2",
  navy: "#1E3A5F", navyLight: "#EFF6FF",
  purple: "#7C3AED", purpleLight: "#F5F3FF",
  sidebar: "#F9FAFB", sidebarActive: "#EFF6FF", sidebarHover: "#F3F4F6", sidebarText: "#6B7280",
  riskGreen: "#059669", riskGreenBg: "#ECFDF5",
  riskAmber: "#D97706", riskAmberBg: "#FFFBEB",
  riskRed: "#DC2626", riskRedBg: "#FEF2F2",
};
const FONT = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
const HEADING = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
const MONO = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace";
const NUC = { A: "#059669", T: "#DC2626", G: "#D97706", C: "#4338CA" }; // nucleotide colors (kept distinct from primary)
const BP = 768; // responsive breakpoint

/* ═══════════════════════════════════════════════════════════════════
   RESPONSIVE HOOK
   ═══════════════════════════════════════════════════════════════════ */
function useIsMobile() {
  const [mobile, setMobile] = useState(() => typeof window !== "undefined" && window.innerWidth < BP);
  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${BP - 1}px)`);
    const handler = (e) => setMobile(e.matches);
    mq.addEventListener("change", handler);
    setMobile(mq.matches);
    return () => mq.removeEventListener("change", handler);
  }, []);
  return mobile;
}

/* ═══════════════════════════════════════════════════════════════════
   MOCK DATA
   ═══════════════════════════════════════════════════════════════════ */
function seq(len) { const b = "ACGT"; return Array.from({ length: len }, () => b[Math.floor(Math.random() * 4)]).join(""); }

const WHO_REFS = {
  "rpoB_S531L": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "40–70% of RIF-R globally" },
  "rpoB_H526Y": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "10–20% of RIF-R" },
  "rpoB_D516V": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "5–10% of RIF-R" },
  "katG_S315T": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "50–80% of INH-R globally" },
  "fabG1_C-15T": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "10–25% of INH-R" },
  "embB_M306V": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "40–65% of EMB-R" },
  "embB_M306I": { who: "Interim", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "15–25% of EMB-R" },
  "pncA_H57D": { who: "Interim", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: null, freq: "<5% of PZA-R" },
  "gyrA_D94G": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "30–50% of FQ-R" },
  "gyrA_A90V": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "15–30% of FQ-R" },
  "rrs_A1401G": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "70–90% of AG-R" },
  "eis_C-14T": { who: "Associated", catalogue: "WHO Mutation Catalogue, 2nd ed. (2023)", cryptic: "CRyPTIC Consortium (2022)", freq: "5–15% of AG-R (KAN)" },
};

const MUTATIONS = [
  { gene: "rpoB", ref: "S", pos: 531, alt: "L", drug: "RIF", drugFull: "Rifampicin", conf: "High", tier: 1 },
  { gene: "rpoB", ref: "H", pos: 526, alt: "Y", drug: "RIF", drugFull: "Rifampicin", conf: "High", tier: 1 },
  { gene: "rpoB", ref: "D", pos: 516, alt: "V", drug: "RIF", drugFull: "Rifampicin", conf: "High", tier: 1 },
  { gene: "katG", ref: "S", pos: 315, alt: "T", drug: "INH", drugFull: "Isoniazid", conf: "High", tier: 1 },
  { gene: "fabG1", ref: "C", pos: -15, alt: "T", drug: "INH", drugFull: "Isoniazid", conf: "High", tier: 1 },
  { gene: "embB", ref: "M", pos: 306, alt: "V", drug: "EMB", drugFull: "Ethambutol", conf: "High", tier: 1 },
  { gene: "embB", ref: "M", pos: 306, alt: "I", drug: "EMB", drugFull: "Ethambutol", conf: "Moderate", tier: 2 },
  { gene: "pncA", ref: "H", pos: 57, alt: "D", drug: "PZA", drugFull: "Pyrazinamide", conf: "Moderate", tier: 2 },
  { gene: "gyrA", ref: "D", pos: 94, alt: "G", drug: "FQ", drugFull: "Fluoroquinolones", conf: "High", tier: 1 },
  { gene: "gyrA", ref: "A", pos: 90, alt: "V", drug: "FQ", drugFull: "Fluoroquinolones", conf: "High", tier: 1 },
  { gene: "rrs", ref: "A", pos: 1401, alt: "G", drug: "AG", drugFull: "Amikacin", conf: "High", tier: 1 },
  { gene: "eis", ref: "C", pos: -14, alt: "T", drug: "AG", drugFull: "Amikacin", conf: "High", tier: 1 },
];

const RESULTS = MUTATIONS.map((m, i) => {
  const spacer = seq(20 + (i % 4));
  const wtSpacer = spacer.split("").map((c, j) => j === 10 ? (c === "A" ? "G" : c === "T" ? "C" : c === "G" ? "A" : "T") : c).join("");
  const refKey = `${m.gene}_${m.ref}${m.pos}${m.alt}`;
  const heuristic = +(0.6 + Math.random() * 0.35).toFixed(3);
  const cnnRaw = +(0.5 + Math.random() * 0.4).toFixed(4);
  const cnnCal = +(cnnRaw * 0.8 + 0.18).toFixed(4);
  const pamPen = [1.0, 0.65, 0.55, 0.45, 1.0, 0.40, 0.35, 0.30][i % 8];
  const pamAdj = +(cnnCal * pamPen).toFixed(4);
  const discRatio = +(1.5 + Math.random() * 8).toFixed(1);
  const mutAct = +(0.5 + Math.random() * 0.45).toFixed(2);
  const wtAct = +(1.0 / Math.max(discRatio, 0.01)).toFixed(4);
  return {
    ...m, label: refKey,
    strategy: i % 3 === 0 ? "Direct" : i % 3 === 1 ? "Proximity" : "Direct",
    spacer, wtSpacer, pam: ["TTTG", "TTCA", "TATA", "CTTG", "TTTC", "TCTG", "TGTC", "ATTG"][i % 8],
    pamVariant: ["TTTV", "TTCV", "TATV", "CTTV", "TTTV", "TCTV", "TGTV", "ATTV"][i % 8],
    pamPenalty: pamPen,
    isCanonicalPam: i % 8 === 0 || i % 8 === 4,
    score: heuristic, cnnScore: cnnRaw, cnnCalibrated: cnnCal, pamAdjusted: pamAdj,
    activityQc: +(0.5 + Math.random() * 0.4).toFixed(4), discriminationQc: +(0.2 + Math.random() * 0.7).toFixed(4),
    mismatchTypeScore: i % 3 === 1 ? 0 : i % 2 === 0 ? 1.0 : 0.5, flankingGcScore: +(0.3 + Math.random() * 0.5).toFixed(4),
    mlScores: [{ model_name: "compass_ml", predicted_efficiency: cnnRaw }],
    disc: discRatio,
    discrimination: { model_name: "learned_lightgbm", ratio: discRatio, mut_activity: mutAct, wt_activity: wtAct },
    gc: +(0.35 + Math.random() * 0.3).toFixed(2),
    ot: Math.floor(Math.random() * 3), hasPrimers: i < 12, hasSM: i % 4 === 1, proximityDistance: i % 3 === 1 ? 15 + Math.floor(Math.random() * 30) : null,
    fwd: i < 12 ? seq(30) : null, rev: i < 12 ? seq(30) : null,
    amplicon: i < 12 ? 120 + Math.floor(Math.random() * 60) : null,
    mutActivity: mutAct,
    wtActivity: wtAct,
    pamDisrupted: false,
    pamDisruptionType: null,
    refs: WHO_REFS[refKey] || null,
  };
});
RESULTS.push({
  gene: "IS6110", ref: "N", pos: 0, alt: "N", drug: "OTHER", drugFull: "Other", conf: "N/A", tier: 0,
  label: "IS6110", strategy: "Direct", spacer: "AATGTCGCCGCGATCGAGCG", wtSpacer: "AATGTCGCCGCGATCGAGCG",
  pam: "TTTG", pamVariant: "TTTV", pamPenalty: 1.0, isCanonicalPam: true,
  score: 0.95, cnnScore: 0.88, cnnCalibrated: 0.91, pamAdjusted: 0.91,
  mlScores: [{ model_name: "compass_ml", predicted_efficiency: 0.88 }],
  disc: 999, discrimination: { model_name: "learned_lightgbm", ratio: 999, mut_activity: 0.95, wt_activity: 0.001 },
  gc: 0.65, ot: 0, hasPrimers: true, hasSM: false,
  fwd: seq(30), rev: seq(30), amplicon: 142, mutActivity: 0.95, wtActivity: 0.001,
  pamDisrupted: false, pamDisruptionType: null,
  refs: { who: "N/A", catalogue: "Species control", pmid: "30593580", cryptic: null, freq: "6–16 copies/genome" },
});

// ── Mock cross-reactivity matrix (14×14, biologically realistic) ──
const CROSS_REACTIVITY_LABELS = [
  "IS6110", "rpoB_S531L", "rpoB_H526Y", "rpoB_D516V", "katG_S315T", "fabG1_C-15T",
  "embB_M306V", "embB_M306I", "pncA_H57D", "gyrA_D94G", "gyrA_A90V", "rrs_A1401G", "eis_C-14T", "RNaseP",
];
const CROSS_REACTIVITY_DRUG_GROUPS = [0,1,1,1,2,2,3,3,4,5,5,6,6,7]; // group indices for separator lines
const MOCK_CROSS_REACTIVITY = (() => {
  const N = 14;
  const matrix = [];
  // Same-gene pair indices (source, target, activity, note)
  // rpoB RRDR: indices 1,2,3; embB M306: indices 6,7; gyrA QRDR: indices 9,10
  const sameGene = [
    [1, 2, 0.042, "rpoB RRDR shared amplicon, S531L\u2194H526Y 15 nt separation"],
    [2, 1, 0.058, "rpoB RRDR shared amplicon, H526Y\u2194S531L"],
    [1, 3, 0.031, "rpoB RRDR shared amplicon, S531L\u2194D516V 45 nt separation"],
    [3, 1, 0.038, "rpoB RRDR shared amplicon, D516V\u2194S531L"],
    [2, 3, 0.025, "rpoB RRDR shared amplicon, H526Y\u2194D516V 30 nt separation"],
    [3, 2, 0.033, "rpoB RRDR shared amplicon, D516V\u2194H526Y"],
    [6, 7, 0.065, "embB M306 same-codon shared amplicon"],
    [7, 6, 0.072, "embB M306 same-codon shared amplicon"],
    [9, 10, 0.028, "gyrA QRDR shared amplicon, 12 nt separation"],
    [10, 9, 0.047, "gyrA QRDR shared amplicon"],
  ];
  const sameGeneMap = {};
  sameGene.forEach(s => { sameGeneMap[`${s[0]}_${s[1]}`] = { activity: s[2], note: s[3] }; });

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const key = `${i}_${j}`;
      const sg = sameGeneMap[key];
      const activity = sg ? sg.activity : Math.random() * 0.0004;
      const risk = activity < 0.01 ? "none" : activity < 0.05 ? "low" : activity < 0.15 ? "medium" : "high";
      matrix.push({
        source: CROSS_REACTIVITY_LABELS[i],
        target: CROSS_REACTIVITY_LABELS[j],
        sourceIdx: i,
        targetIdx: j,
        activity,
        risk,
        mismatches: sg ? (3 + Math.floor(Math.random() * 2)) : (8 + Math.floor(Math.random() * 5)),
        pam_valid: sg ? true : Math.random() < 0.08,
        note: sg ? sg.note : null,
      });
    }
  }
  const highRisk = matrix.filter(m => m.risk === "high");
  const sameGenePairs = matrix.filter(m => m.note != null);
  const noneCount = matrix.filter(m => m.risk === "none").length;
  return {
    matrix,
    n_targets: N,
    n_pairs: N * (N - 1),
    high_risk_pairs: highRisk,
    same_gene_pairs: sameGenePairs,
    panel_safe: highRisk.length === 0,
    none_count: noneCount,
    interpretation: "All cross-reactive pairs occur between same-gene targets sharing overlapping amplicons. In the spatially multiplexed paper electrode array, each detection zone is physically isolated by wax-printed hydrophobic barriers \u2014 cross-reactivity between zones is impossible. These scores are relevant only for hypothetical solution-phase multiplex formats.",
  };
})();

const MODULES = [
  { id: "M1", name: "Target Resolution", desc: "WHO mutations → genomic coordinates", icon: Database, execDesc: "Resolving WHO-catalogued resistance mutations to genomic coordinates on H37Rv", estSec: 5, substeps: [
    "Loading H37Rv reference genome (NC_000962.3, 4.4 Mb)",
    "Parsing 5,959 genes from GFF3 annotation",
    "Resolving mutation coordinates to genomic positions",
    "Mapping drug resistance class associations",
  ]},
  { id: "M2", name: "PAM Scanning", desc: "Multi-PAM, multi-length spacer search", icon: Search, execDesc: "Scanning both strands for Cas12a-compatible PAM sites (TTTV canonical + relaxed)", estSec: 30, substeps: [
    "Scanning TTTV canonical PAM sites on both strands",
    "Extending to relaxed PAMs (TTCV, TATV, CTTV, TCTV)",
    "Evaluating spacer lengths 18–23 nt per PAM hit",
    "Checking seed region (positions 1–8) for mutation coverage",
    "Activating proximity scan for PAM desert targets (±200 bp)",
    "Collecting direct + proximity candidates per target",
  ]},
  { id: "M3", name: "Candidate Filtering", desc: "Biophysical constraints (GC, homopolymer, Tm)", icon: Filter, execDesc: "Applying biophysical filters: GC content, homopolymer runs, self-complementarity", estSec: 15, substeps: [
    "Checking GC content (40–85% for M. tuberculosis)",
    "Screening for homopolymer runs ≥4 nt",
    "Evaluating self-complementarity (MFE < −5.0 kcal/mol)",
    "Filtering seed region violations",
    "Compiling filter report per target",
  ]},
  { id: "M4", name: "Off-Target Screening", desc: "Bowtie2 alignment + heuristic fallback", icon: Shield, execDesc: "Bowtie2 alignment against H37Rv genome, flagging off-target binding sites", estSec: 45, substeps: [
    "Preparing spacer FASTA for Bowtie2 alignment",
    "Aligning candidates against H37Rv genome",
    "Scoring off-target binding sites (≤3 mismatches)",
    "Flagging high-risk off-target candidates",
    "Generating off-target summary report",
  ]},
  { id: "M5", name: "Heuristic Scoring", desc: "Position-weighted composite scoring", icon: BarChart3, execDesc: "Position-weighted composite scoring across 5 biophysical features", estSec: 10, substeps: [
    "Computing seed position scores (weight 0.35)",
    "Evaluating GC content scores (weight 0.20)",
    "Scoring self-complementarity (weight 0.20)",
    "Computing homopolymer penalties (weight 0.10)",
    "Calculating composite heuristic score per candidate",
  ]},
  { id: "M5.5", name: "Mismatch Pairs", desc: "WT/MUT spacer pair generation", icon: GitBranch, execDesc: "Generating wildtype spacers for each mutant candidate (MUT/WT discrimination pairs)", estSec: 5, substeps: [
    "Generating wildtype spacer for each mutant candidate",
    "Identifying mismatch type and position (e.g. C>G at pos 2)",
    "Building MUT/WT discrimination pairs",
  ]},
  { id: "M6", name: "SM Enhancement", desc: "Synthetic mismatch for enhanced discrimination", icon: Zap, execDesc: "Engineering synthetic mismatches at seed positions 1–8 for enhanced discrimination", estSec: 20, substeps: [
    "Testing synthetic mismatches at seed positions 1–8",
    "Evaluating discrimination gain per mismatch position",
    "Selecting optimal enhancement positions",
    "Applying enhancements to panel members",
  ]},
  { id: "M6.5", name: "Discrimination", desc: "MUT/WT activity ratio quantification", icon: TrendingUp, execDesc: "Quantifying MUT/WT activity ratios for diagnostic-grade discrimination assessment", estSec: 15, substeps: [
    "Computing MUT/WT activity ratios per candidate",
    "Classifying diagnostic-grade candidates (≥2× threshold)",
    "Ranking candidates by discrimination strength",
    "Identifying direct vs. proximity strategy per target",
  ]},
  { id: "M7", name: "Multiplex Optimization", desc: "Simulated annealing panel selection", icon: Grid3x3, execDesc: "Simulated annealing over candidate combinations for optimal panel selection", estSec: 30, substeps: [
    "Initializing simulated annealing (T₀ = 1.0, 10K iterations)",
    "Iterating candidate combinations for optimal panel",
    "Minimizing cross-reactivity between panel members",
    "Cooling schedule: evaluating convergence",
    "Assembling final panel with best score",
  ]},
  { id: "M8", name: "RPA Primer Design", desc: "Standard + allele-specific RPA", icon: Crosshair, execDesc: "Designing RPA primers (25–38 nt, Tm 57–72 °C) with dimer checking", estSec: 60, substeps: [
    "Designing forward primers (25–38 nt, Tm 57–72°C)",
    "Designing reverse primers with amplicon size constraints",
    "Generating allele-specific RPA primers for proximity targets",
    "Running primer dimer thermodynamic checks",
    "Co-selecting crRNA-compatible primer pairs",
    "Evaluating amplicon sizes (80–120 bp target range)",
  ]},
  { id: "M8.5", name: "Co-Selection", desc: "crRNA–primer compatibility check", icon: Check, execDesc: "Validating crRNA–primer compatibility and amplicon overlap constraints", estSec: 10, substeps: [
    "Validating crRNA–primer amplicon overlap",
    "Checking AS-RPA discrimination ratios (MUT vs WT)",
    "Running full primer dimer analysis across all oligos",
    "Scoring moderate and high-risk dimer interactions",
  ]},
  { id: "M9", name: "Panel Assembly", desc: "MultiplexPanel + IS6110 control", icon: Package, execDesc: "Assembling final panel: crRNA sequences, primer pairs, amplicon maps, discrimination predictions", estSec: 5, substeps: [
    "Assembling MultiplexPanel structure",
    "Designing IS6110 insertion element control primers",
    "Computing panel sensitivity and specificity",
    "Collecting top-5 alternatives per target",
    "Generating full panel report (JSON + TSV)",
  ]},
  { id: "M10", name: "Export", desc: "JSON, TSV, FASTA structured output", icon: Download, execDesc: "Exporting structured output: JSON, TSV, FASTA", estSec: 3, substeps: [
    "Serializing structured JSON report",
    "Computing PCA embeddings for UMAP visualization",
    "Exporting TSV summary and FASTA sequences",
  ]},
];

const MODULE_NAME_MAP = {
  "Initializing": 0, "Target Resolution": 1, "PAM Scanning": 2,
  "Candidate Filtering": 3, "Off-Target Screening": 4, "Heuristic Scoring": 5,
  "Mismatch Pairs": 6, "SM Enhancement": 7, "Discrimination Scoring": 8,
  "Multiplex Optimization": 9, "RPA Primer Design": 10, "Co-Selection Validation": 11,
  "Panel Assembly": 12, "Export": 13, "Complete": 13, "Serializing Results": 13,
};

/* Scoring feature weights — matches compass/core/constants.py HEURISTIC_WEIGHTS exactly */
const SCORING_FEATURES = [
  { name: "Seed Position", key: "seed_position", weight: 0.35, desc: "Positions 1–8 (PAM-proximal) perfect match penalty. Mismatches in seed dramatically reduce cleavage.", source: "Kim et al. 2017" },
  { name: "GC Content", key: "gc", weight: 0.20, desc: "Optimal 40–60%. Extreme GC causes self-complementarity (high) or weak binding (low).", source: "Empirical" },
  { name: "Self-Complementarity", key: "structure", weight: 0.20, desc: "Spacer self-complementarity penalty. High self-complementarity blocks Cas12a loading.", source: "SantaLucia 1998" },
  { name: "Homopolymer", key: "homopolymer", weight: 0.10, desc: "≥4 consecutive identical nucleotides penalized (includes poly-T terminator risk).", source: "Heuristic" },
  { name: "Off-Target", key: "offtarget", weight: 0.15, desc: "Bowtie2 alignment to H37Rv genome. Each hit with ≤3 mismatches reduces score.", source: "Langmead & Salzberg 2012" },
];

const DRUG_LABELS = {
  RIF: "Rifampicin", INH: "Isoniazid", EMB: "Ethambutol",
  PZA: "Pyrazinamide", FQ: "Fluoroquinolones", AG: "Amikacin",
};

/* ═══════════════════════════════════════════════════════════════════
   BIBLIOGRAPHY
   ═══════════════════════════════════════════════════════════════════ */
const BIBLIOGRAPHY = [
  // CRISPR Biology
  { id: "zetsche2015", authors: "Zetsche B, Gootenberg JS, Abudayyeh OO, et al.", year: 2015, title: "Cpf1 is a single RNA-guided endonuclease of a class 2 CRISPR-Cas system", journal: "Cell", doi: "10.1016/j.cell.2015.09.038", pmid: "26422227", category: "CRISPR Biology" },
  { id: "chen2018", authors: "Chen JS, Ma E, Harrington LB, et al.", year: 2018, title: "CRISPR-Cas12a target binding unleashes indiscriminate single-stranded DNase activity", journal: "Science", doi: "10.1126/science.aar6245", pmid: "29449511", category: "CRISPR Biology" },
  { id: "strohkendl2018", authors: "Strohkendl I, Saifuddin FA, Rybarski JR, et al.", year: 2018, title: "Kinetic basis for DNA target specificity of CRISPR-Cas12a", journal: "Molecular Cell", doi: "10.1016/j.molcel.2018.06.043", pmid: "30078724", category: "CRISPR Biology" },
  { id: "kleinstiver2019", authors: "Kleinstiver BP, Sousa AA, Walton RT, et al.", year: 2019, title: "Engineered CRISPR-Cas12a variants with increased activities and improved targeting ranges", journal: "Nature Biotechnology", doi: "10.1038/s41587-018-0011-0", pmid: "30742127", category: "CRISPR Biology" },
  { id: "strohkendl2024", authors: "Strohkendl I, Saha A, Moy C, et al.", year: 2024, title: "Cas12a domain flexibility guides R-loop formation and forces RuvC resetting", journal: "Molecular Cell", doi: "10.1016/j.molcel.2024.05.032", category: "CRISPR Biology" },
  // R-Loop Thermodynamics
  { id: "zhang2024", authors: "Zhang J, Guan X, Moon J, et al.", year: 2024, title: "Interpreting CRISPR-Cas12a enzyme kinetics through free energy change of nucleic acids", journal: "Nucleic Acids Research", doi: "10.1093/nar/gkae1124", category: "R-Loop Thermodynamics" },
  { id: "aris2025", authors: "Aris KDP, Cofsky JC, Shi H, et al.", year: 2025, title: "Dynamic basis of supercoiling-dependent DNA interrogation by Cas12a via R-loop intermediates", journal: "Nature Communications", doi: "10.1038/s41467-025-57703-y", category: "R-Loop Thermodynamics" },
  // Guide Activity Prediction (ML)
  { id: "kim2018", authors: "Kim HK, Min S, Song M, et al.", year: 2018, title: "Deep learning improves prediction of CRISPR-Cpf1 guide RNA activity", journal: "Nature Biotechnology", doi: "10.1038/nbt.4061", pmid: "29431741", category: "Guide Activity Prediction" },
  { id: "huang2024", authors: "Huang B, Mu K, Li G, et al.", year: 2024, title: "Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics", journal: "iMeta", doi: "10.1002/imt2.214", category: "Guide Activity Prediction" },
  { id: "chen2022rnafm", authors: "Chen J, Hu Z, Sun S, et al.", year: 2022, title: "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions", journal: "arXiv:2204.00300", doi: null, url: "https://arxiv.org/abs/2204.00300", category: "Guide Activity Prediction" },
  { id: "blondel2020", authors: "Blondel M, Teboul O, Berthet Q, Djolonga J", year: 2020, title: "Fast Differentiable Sorting and Ranking [Soft Spearman correlation loss used in Compass-ML training]", journal: "ICML 2020", doi: null, url: "https://arxiv.org/abs/2002.08871", category: "Guide Activity Prediction" },
  { id: "yao2025", authors: "Yao Z, Li W, He K, et al.", year: 2025, title: "Facilitating crRNA design by integrating DNA interaction features of CRISPR-Cas12a system", journal: "Advanced Science", doi: "10.1002/advs.202501269", category: "Guide Activity Prediction" },
  // Clinical Standards
  { id: "who2024tpp", authors: "World Health Organization", year: 2024, title: "Target product profiles for tuberculosis diagnosis and detection of drug resistance", journal: "WHO", doi: null, url: "https://www.who.int/publications/i/item/9789240097698", isbn: "978-92-4-009769-8", category: "Clinical Standards" },
  { id: "maclean2023", authors: "MacLean EL-H, Kohli M, Weber SF, et al.", year: 2023, title: "Updating the WHO target product profile for next-generation Mycobacterium tuberculosis drug susceptibility testing at peripheral centres", journal: "PLOS Global Public Health", doi: "10.1371/journal.pgph.0001754", category: "Clinical Standards" },
  { id: "who2023", authors: "WHO", year: 2023, title: "Catalogue of mutations in Mycobacterium tuberculosis complex and their association with drug resistance (2nd ed.)", journal: "World Health Organization", doi: null, url: "https://iris.who.int/handle/10665/374061", isbn: "978-92-4-008241-0", category: "Clinical Standards" },
  { id: "cryptic2022", authors: "CRyPTIC Consortium", year: 2022, title: "A data compendium associating the genomes of 12,289 Mycobacterium tuberculosis isolates with quantitative resistance phenotypes to 13 antibiotics", journal: "PLoS Biology", doi: "10.1371/journal.pbio.3001721", pmid: "35944069", category: "Clinical Standards" },
  // CRISPR Diagnostics
  { id: "broughton2020", authors: "Broughton JP, Deng X, Yu G, et al.", year: 2020, title: "CRISPR-Cas12-based detection of SARS-CoV-2", journal: "Nature Biotechnology", doi: "10.1038/s41587-020-0513-4", pmid: "32300245", category: "CRISPR Diagnostics" },
  { id: "ai2019", authors: "Ai JW, Zhou X, Xu T, et al.", year: 2019, title: "CRISPR-based rapid and ultra-sensitive diagnostic test for Mycobacterium tuberculosis", journal: "Emerging Microbes & Infections", doi: "10.1080/22221751.2019.1664939", pmid: "31522608", category: "CRISPR Diagnostics" },
  // Bioinformatics
  { id: "langmead2012", authors: "Langmead B, Salzberg SL", year: 2012, title: "Fast gapped-read alignment with Bowtie 2", journal: "Nature Methods", doi: "10.1038/nmeth.1923", pmid: "22388286", category: "Bioinformatics" },
  { id: "piepenburg2006", authors: "Piepenburg O, Williams CH, Stemple DL, Armes NA", year: 2006, title: "DNA detection using recombination proteins", journal: "PLoS Biology", doi: "10.1371/journal.pbio.0040204", pmid: "16756388", category: "Bioinformatics" },
];

/* ═══════════════════════════════════════════════════════════════════
   UTILITY COMPONENTS
   ═══════════════════════════════════════════════════════════════════ */
const DRUG_COLORS = {
  RIF: { bg: "transparent", text: "#1E3A5F", border: "#1E3A5F" }, INH: { bg: "transparent", text: "#D97706", border: "#D97706" },
  EMB: { bg: "transparent", text: "#059669", border: "#059669" }, FQ: { bg: "transparent", text: "#DC2626", border: "#DC2626" },
  AG: { bg: "transparent", text: "#7C3AED", border: "#7C3AED" }, PZA: { bg: "transparent", text: "#0891B2", border: "#0891B2" },
};
const DEFAULT_DRUG = { bg: "transparent", text: "#6B7280", border: "#E5E7EB" };

const Badge = ({ children, variant = "default" }) => {
  const s = {
    default: { background: "transparent", color: "#6B7280", border: `1px solid #E5E7EB` },
    primary: { background: T.primaryLight, color: T.primary, border: `1px solid ${T.primary}` },
    success: { background: "#ECFDF5", color: "#059669", border: "1px solid #059669" },
    warning: { background: "#FFFBEB", color: "#D97706", border: "1px solid #D97706" },
    danger: { background: "#FEF2F2", color: "#DC2626", border: "1px solid #DC2626" },
    purple: { background: "#F5F3FF", color: "#7C3AED", border: "1px solid #7C3AED" },
  };
  return (
    <span style={{ ...(s[variant] || s.default), padding: "2px 8px", borderRadius: "3px", fontSize: "11px", fontWeight: 500, fontFamily: FONT, display: "inline-flex", alignItems: "center", gap: "4px", whiteSpace: "nowrap" }}>{children}</span>
  );
};

const DrugBadge = ({ drug }) => {
  const c = DRUG_COLORS[drug] || DEFAULT_DRUG;
  return <span style={{ background: "transparent", color: c.text, border: `1px solid ${c.border || c.text}`, padding: "2px 8px", borderRadius: "3px", fontSize: "11px", fontWeight: 600, fontFamily: FONT, display: "inline-block" }}>{drug}</span>;
};

const Seq = ({ s: str }) => (
  <span style={{ fontFamily: MONO, fontSize: "12px", letterSpacing: "1px" }}>
    {str?.split("").map((c, i) => (
      <span key={i} style={{ color: c === "A" ? "#059669" : c === "T" ? "#DC2626" : c === "G" ? "#D97706" : "#4338CA", fontWeight: 400 }}>{c}</span>
    ))}
  </span>
);

const Btn = ({ children, variant = "primary", onClick, disabled, icon: Icon, full, size = "md" }) => {
  const styles = {
    primary: { background: T.primary, color: "#fff", border: "none" },
    secondary: { background: "#fff", color: T.text, border: `1px solid ${T.border}` },
    ghost: { background: "transparent", color: T.textSec, border: "none" },
    danger: { background: T.danger, color: "#fff", border: "none" },
  };
  const sizes = { sm: { padding: "6px 12px", fontSize: "12px" }, md: { padding: "10px 20px", fontSize: "14px" }, lg: { padding: "12px 24px", fontSize: "14px" } };
  return (
    <button onClick={onClick} disabled={disabled} style={{
      ...styles[variant], ...sizes[size], borderRadius: "6px", fontWeight: 500,
      cursor: disabled ? "not-allowed" : "pointer", display: "inline-flex",
      alignItems: "center", gap: "8px", fontFamily: FONT, opacity: disabled ? 0.5 : 1,
      width: full ? "100%" : "auto", justifyContent: "center",
      transition: "background 120ms ease-out, border-color 120ms ease-out, opacity 120ms ease-out",
    }}>{Icon && <Icon size={14} />}{children}</button>
  );
};

const tooltipStyle = { background: "#fff", border: `1px solid ${T.border}`, borderRadius: "4px", fontSize: "12px", boxShadow: "0 1px 3px rgba(0,0,0,0.05)", fontFamily: MONO };

/* Gaussian KDE for smooth density estimation */
function gaussianKDE(data, bandwidth = 0.05, nPoints = 100) {
  const min = 0, max = 1;
  const step = (max - min) / nPoints;
  const points = [];
  for (let x = min; x <= max; x += step) {
    let density = 0;
    for (const d of data) {
      const z = (x - d) / bandwidth;
      density += Math.exp(-0.5 * z * z) / (bandwidth * Math.sqrt(2 * Math.PI));
    }
    density /= data.length;
    points.push({ x: parseFloat(x.toFixed(3)), density: parseFloat(density.toFixed(4)) });
  }
  return points;
}

function stdDev(arr) {
  if (arr.length < 2) return 0;
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, v) => a + (v - m) ** 2, 0) / (arr.length - 1));
}

/* ═══════════════════════════════════════════════════════════════════
   TOAST NOTIFICATION SYSTEM
   ═══════════════════════════════════════════════════════════════════ */
const ToastContext = React.createContext(() => {});
const useToast = () => React.useContext(ToastContext);

const ToastProvider = ({ children }) => {
  const [toasts, setToasts] = useState([]);
  const addToast = useCallback((message, type = "success") => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 2500);
  }, []);
  return (
    <ToastContext.Provider value={addToast}>
      {children}
      {createPortal(
        <div style={{ position: "fixed", bottom: 24, right: 24, zIndex: 99999, display: "flex", flexDirection: "column-reverse", gap: "8px", pointerEvents: "none" }}>
          {toasts.map((t) => (
            <div key={t.id} style={{
              background: t.type === "success" ? "#065F46" : t.type === "error" ? "#DC2626" : "#111827",
              color: "#fff", padding: "8px 16px", borderRadius: "4px", fontSize: "13px", fontWeight: 500,
              fontFamily: FONT, boxShadow: "0 1px 3px rgba(0,0,0,0.1)", display: "flex", alignItems: "center", gap: "8px",
              animation: "toastIn 0.25s ease-out",
            }}>
              {t.type === "success" && <Check size={14} />}
              {t.message}
            </div>
          ))}
        </div>,
        document.body
      )}
    </ToastContext.Provider>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   API DATA TRANSFORMER — maps API CandidateResponse → flat v8 format
   ═══════════════════════════════════════════════════════════════════ */
function transformApiCandidate(c) {
  /* Handle both the detailed per-candidate shape and the TargetResult shape from /api/results */
  const sc = c.selected_candidate;
  if (sc) {
    /* TargetResult shape from /api/results/{job_id} */
    const parts = (c.mutation || "").match(/^([A-Za-z*-]*)(\d+)([A-Za-z*-]*)$/);
    return {
      gene: c.gene, ref: parts?.[1] || "", pos: parts ? parseInt(parts[2]) : 0, alt: parts?.[3] || "",
      drug: c.drug, drugFull: c.drug || "", conf: "", tier: "",
      label: c.label, strategy: c.detection_strategy === "direct" ? "Direct" : "Proximity",
      spacer: sc.spacer_seq, wtSpacer: sc.wt_spacer_seq || "", pam: sc.pam_seq,
      pamVariant: sc.pam_variant || "", pamPenalty: sc.pam_penalty ?? null, isCanonicalPam: sc.is_canonical_pam ?? null,
      score: sc.composite_score, disc: +(sc.discrimination_ratio || 0).toFixed(1), gc: sc.gc_content,
      discrimination: sc.discrimination || null, discMethod: sc.disc_method || null,
      neuralDisc: sc.neural_disc ?? null, featureDisc: sc.feature_disc ?? null,
      cnnScore: sc.cnn_score ?? null,
      cnnCalibrated: sc.cnn_calibrated ?? null,
      pamAdjusted: (sc.cnn_calibrated != null && sc.pam_penalty != null) ? +(sc.cnn_calibrated * sc.pam_penalty).toFixed(4) : sc.cnn_calibrated ?? null,
      activityQc: sc.activity_qc ?? null, discriminationQc: sc.discrimination_qc ?? null,
      mismatchTypeScore: sc.mismatch_type_score ?? null, flankingGcScore: sc.flanking_gc_score ?? null,
      mlScores: sc.ml_scores || [],
      ot: 0, hasPrimers: c.has_primers, hasSM: c.has_sm || false,
      smSpacer: c.sm_enhanced_spacer || null, smPosition: c.sm_position || null,
      smOriginalBase: c.sm_original_base || null, smReplacementBase: c.sm_replacement_base || null,
      fwd: c.fwd_primer, rev: c.rev_primer, amplicon: c.amplicon_length,
      proximityDistance: c.proximity_distance || null,
      mutActivity: sc.discrimination?.mut_activity || 0, wtActivity: sc.discrimination?.wt_activity || 0,
      asrpaDiscrimination: c.asrpa_discrimination || null,
      refs: WHO_REFS[c.label] || null,
      scoringBreakdown: null,
      isControl: false,
      readinessScore: c.readiness_score ?? null,
      readinessComponents: c.readiness_components ?? null,
      experimentalPriority: c.experimental_priority ?? null,
      riskProfile: c.risk_profile ?? null,
      priorityReason: c.priority_reason ?? null,
    };
  }
  /* Original detailed per-candidate shape */
  return {
    gene: c.gene, ref: c.ref_aa, pos: c.position, alt: c.alt_aa,
    drug: c.drug, drugFull: c.drug_full, conf: c.who_confidence, tier: c.tier,
    label: c.target_label, strategy: c.detection_strategy === "direct" ? "Direct" : "Proximity",
    spacer: c.spacer_seq, wtSpacer: c.wt_spacer_seq, pam: c.pam_seq,
    pamVariant: c.pam_variant || "", pamPenalty: c.pam_penalty ?? null, isCanonicalPam: c.is_canonical_pam ?? null,
    score: c.score, disc: +(c.discrimination?.ratio || 0).toFixed(1), gc: c.gc_content,
    discrimination: c.discrimination || null,
    cnnScore: c.cnn_score ?? null,
    cnnCalibrated: c.cnn_calibrated ?? null,
    pamAdjusted: (c.cnn_calibrated != null && c.pam_penalty != null) ? +(c.cnn_calibrated * c.pam_penalty).toFixed(4) : c.cnn_calibrated ?? null,
    activityQc: c.activity_qc ?? null, discriminationQc: c.discrimination_qc ?? null,
    mismatchTypeScore: c.mismatch_type_score ?? null, flankingGcScore: c.flanking_gc_score ?? null,
    mlScores: c.ml_scores || [],
    ot: c.offtarget_count, hasPrimers: c.has_primers, hasSM: c.has_sm,
    smSpacer: c.sm_enhanced_spacer || null, smPosition: c.sm_position || null,
    smOriginalBase: c.sm_original_base || null, smReplacementBase: c.sm_replacement_base || null,
    fwd: c.fwd_primer, rev: c.rev_primer, amplicon: c.amplicon_length,
    proximityDistance: c.proximity_distance || null,
    mutActivity: c.discrimination?.mut_activity || 0,
    wtActivity: c.discrimination?.wt_activity || 0,
    asrpaDiscrimination: c.asrpa_discrimination || null,
    refs: WHO_REFS[c.target_label] || null,
    scoringBreakdown: c.scoring_breakdown || null,
    isControl: c.is_control || false,
  };
}

/* ═══════════════════════════════════════════════════════════════════
   CANDIDATE VIEWER — Detail panel with amplicon map + mismatch
   ═══════════════════════════════════════════════════════════════════ */
const AmpliconMap = ({ r }) => {
  const W = 640, H = 100, pad = 40;
  const track = W - 2 * pad;
  const ampLen = r.amplicon || 150;
  const scale = track / ampLen;
  const mutPos = Math.floor(ampLen * 0.55);
  const spacerStart = mutPos - Math.floor(r.spacer.length / 2);
  const spacerEnd = spacerStart + r.spacer.length;
  const pamStart = spacerStart - 4;
  const fwdEnd = 30;
  const revStart = ampLen - 30;
  const x = (pos) => pad + pos * scale;
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ fontFamily: MONO }}>
      <line x1={pad} y1={45} x2={W - pad} y2={45} stroke={T.border} strokeWidth={2} />
      {r.fwd && <rect x={x(0)} y={36} width={fwdEnd * scale} height={18} rx={3} fill="#059669" fillOpacity={0.15} stroke="#059669" strokeWidth={1} />}
      {r.fwd && <text x={x(fwdEnd / 2)} y={32} textAnchor="middle" fontSize={8} fill="#059669" fontWeight={600}>FWD</text>}
      {r.rev && <rect x={x(revStart)} y={36} width={30 * scale} height={18} rx={3} fill="#4338CA" fillOpacity={0.15} stroke="#4338CA" strokeWidth={1} />}
      {r.rev && <text x={x(revStart + 15)} y={32} textAnchor="middle" fontSize={8} fill="#4338CA" fontWeight={600}>REV</text>}
      <rect x={x(pamStart)} y={36} width={4 * scale} height={18} rx={2} fill={T.warning} fillOpacity={0.3} stroke={T.warning} strokeWidth={1} />
      <text x={x(pamStart + 2)} y={72} textAnchor="middle" fontSize={8} fill={T.warning} fontWeight={600}>PAM</text>
      <rect x={x(spacerStart)} y={36} width={r.spacer.length * scale} height={18} rx={3} fill={T.primary} fillOpacity={0.25} stroke={T.primary} strokeWidth={1.5} />
      <text x={x((spacerStart + spacerEnd) / 2)} y={47} textAnchor="middle" fontSize={8} fill={T.primaryDark} fontWeight={600}>crRNA spacer</text>
      <line x1={x(mutPos)} y1={28} x2={x(mutPos)} y2={62} stroke={T.danger} strokeWidth={2} strokeDasharray="3 2" />
      <circle cx={x(mutPos)} cy={24} r={4} fill={T.danger} />
      <text x={x(mutPos)} y={78} textAnchor="middle" fontSize={8} fill={T.danger} fontWeight={600}>{r.ref}{r.pos}{r.alt}</text>
      <text x={pad} y={92} fontSize={8} fill={T.textTer}>{r.gene} locus</text>
      <text x={W - pad} y={92} textAnchor="end" fontSize={8} fill={T.textTer}>{ampLen} bp amplicon</text>
    </svg>
  );
};

const MismatchProfile = ({ spacer, wtSpacer, strategy }) => {
  if (!spacer || !wtSpacer || wtSpacer.length !== spacer.length) {
    if (strategy === "Proximity") {
      return (
        <div style={{ fontSize: "12px", color: T.purple, lineHeight: 1.6, padding: "8px 0" }}>
          <strong>Proximity detection</strong> — discrimination is provided by the AS-RPA primers, not by crRNA mismatch. The crRNA binds a conserved region near the mutation site.
        </div>
      );
    }
    return (
      <div style={{ fontSize: "12px", color: T.textTer, lineHeight: 1.6, padding: "8px 0" }}>
        WT spacer not available — mismatch profile cannot be displayed.
      </div>
    );
  }
  return (
    <div style={{ fontFamily: MONO, fontSize: "12px", lineHeight: 2 }}>
      <div style={{ display: "flex", alignItems: "center", gap: "4px", marginBottom: "2px" }}>
        <span style={{ width: 40, fontSize: "10px", color: T.textTer, fontWeight: 600 }}>MUT</span>
        {spacer.split("").map((c, i) => {
          const mm = c !== wtSpacer[i];
          return (<span key={`m${i}`} style={{ width: 18, height: 22, display: "inline-flex", alignItems: "center", justifyContent: "center", borderRadius: "3px", fontWeight: 600, fontSize: "11px", background: mm ? NUC[c] : "transparent", color: mm ? "#FFFFFF" : NUC[c], border: mm ? "none" : `1px solid ${T.borderLight}` }}>{c}</span>);
        })}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "4px", marginBottom: "2px" }}>
        <span style={{ width: 40 }} />
        {spacer.split("").map((c, i) => (<span key={`d${i}`} style={{ width: 18, height: 14, display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: "10px", color: c !== wtSpacer[i] ? T.danger : T.borderLight, fontWeight: 600 }}>{c !== wtSpacer[i] ? "▼" : "·"}</span>))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
        <span style={{ width: 40, fontSize: "10px", color: T.textTer, fontWeight: 600 }}>WT</span>
        {wtSpacer.split("").map((c, i) => {
          const mm = c !== spacer[i];
          return (<span key={`w${i}`} style={{ width: 18, height: 22, display: "inline-flex", alignItems: "center", justifyContent: "center", borderRadius: "3px", fontWeight: 600, fontSize: "11px", background: mm ? "#F3F4F6" : "transparent", color: mm ? T.textSec : NUC[c], border: mm ? `1px solid ${T.border}` : `1px solid ${T.borderLight}` }}>{c}</span>);
        })}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "4px", marginTop: "4px" }}>
        <span style={{ width: 40 }} />
        {spacer.split("").map((_, i) => (<span key={`p${i}`} style={{ width: 18, textAlign: "center", fontSize: "8px", color: [1, 2, 3, 4, 5, 6, 7, 8].includes(i + 1) ? T.primary : T.textTer, fontWeight: [1, 2, 3, 4, 5, 6, 7, 8].includes(i + 1) ? 700 : 400 }}>{i + 1}</span>))}
      </div>
      <div style={{ fontSize: "10px", color: T.textTer, marginTop: "6px" }}>Positions 1–8 (blue) = PAM-proximal seed region. Mismatches here have strongest effect on Cas12a discrimination.</div>
    </div>
  );
};

const CandidateViewer = ({ r, onClose }) => {
  if (!r) return null;
  const mobile = useIsMobile();
  const toast = useToast();
  const ref = r.refs;
  const discColor = r.disc >= 3 ? T.success : r.disc >= 2 ? T.primary : r.disc >= 1.5 ? T.warning : T.danger;
  // Use SM-enhanced spacer when available (the actual crRNA to synthesize)
  const displaySpacer = (r.hasSM && r.smSpacer) ? r.smSpacer : r.spacer;

  /* Compute per-feature scores deterministically from candidate data */
  const computeFeatures = () => {
    if (r.scoringBreakdown) {
      /* Real API data — use actual per-feature scores from pipeline HeuristicScore */
      const sb = r.scoringBreakdown;
      return [
        { ...SCORING_FEATURES[0], raw: +(1 - (sb.seed_position_score || 0)).toFixed(3), weighted: +((1 - (sb.seed_position_score || 0)) * 0.35).toFixed(4) },
        { ...SCORING_FEATURES[1], raw: +(1 - (sb.gc_penalty || 0)).toFixed(3), weighted: +((1 - (sb.gc_penalty || 0)) * 0.20).toFixed(4) },
        { ...SCORING_FEATURES[2], raw: +(1 - (sb.structure_penalty || 0)).toFixed(3), weighted: +((1 - (sb.structure_penalty || 0)) * 0.20).toFixed(4) },
        { ...SCORING_FEATURES[3], raw: +(1 - (sb.homopolymer_penalty || 0)).toFixed(3), weighted: +((1 - (sb.homopolymer_penalty || 0)) * 0.10).toFixed(4) },
        { ...SCORING_FEATURES[4], raw: +(1 - (sb.offtarget_penalty || 0)).toFixed(3), weighted: +((1 - (sb.offtarget_penalty || 0)) * 0.15).toFixed(4) },
      ];
    }
    /* Mock data — simulate deterministically from spacer */
    const seed = r.spacer.charCodeAt(0) + r.spacer.charCodeAt(1);
    return SCORING_FEATURES.map((f, i) => {
      let raw;
      if (f.key === "gc") raw = 1 - Math.abs(r.gc - 0.5) * 4;
      else if (f.key === "offtarget") raw = r.ot === 0 ? 1.0 : r.ot <= 1 ? 0.6 : 0.2;
      else raw = 0.4 + ((seed * (i + 7) * 13) % 60) / 100;
      raw = Math.max(0, Math.min(1, raw));
      return { ...f, raw: +raw.toFixed(3), weighted: +(raw * f.weight).toFixed(4) };
    });
  };
  const features = computeFeatures();
  const compositeCalc = features.reduce((a, f) => a + f.weighted, 0);

  return (
    <div style={{ position: "fixed", top: 0, right: 0, bottom: 0, width: mobile ? "100%" : 720, background: T.bg, boxShadow: "none", zIndex: 10000, overflow: "auto", borderLeft: mobile ? "none" : `1px solid ${T.border}` }}>
      <div style={{ padding: mobile ? "16px 16px" : "24px 28px", borderBottom: `1px solid ${T.border}`, display: "flex", justifyContent: "space-between", alignItems: "flex-start", position: "sticky", top: 0, background: T.bg, zIndex: 1 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: mobile ? "6px" : "10px", marginBottom: "6px", flexWrap: "wrap" }}>
            <span style={{ fontSize: mobile ? "16px" : "20px", fontWeight: 600, fontFamily: MONO, color: T.text }}>{r.gene}</span>
            <span style={{ fontSize: mobile ? "13px" : "16px", fontFamily: MONO, color: T.textSec }}>{r.ref}{r.pos}{r.alt}</span>
            <DrugBadge drug={r.drug} />
            <Badge variant={r.strategy === "Direct" ? "success" : "purple"}>{r.strategy}</Badge>
          </div>
          <div style={{ fontSize: "12px", color: T.textSec }}>{r.drugFull} resistance · WHO Tier {r.tier} · {r.conf} confidence</div>
        </div>
        <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", padding: "8px" }}><X size={20} color={T.textSec} /></button>
      </div>

      <div style={{ padding: mobile ? "16px" : "24px 28px" }}>
        {/* Key metrics */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: mobile ? "8px" : "0", background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: mobile ? "12px" : "16px", marginBottom: "24px" }}>
          {[
            { l: "Activity", v: (r.cnnCalibrated ?? r.score).toFixed(3), c: (r.cnnCalibrated ?? r.score) > 0.7 ? T.primary : (r.cnnCalibrated ?? r.score) > 0.5 ? T.warning : T.danger },
            ...(r.pamAdjusted != null && r.pamPenalty != null && r.pamPenalty < 1.0 ? [{ l: "PAM-adjusted", v: `${r.pamAdjusted.toFixed(3)} (${r.pamPenalty}×)`, c: T.textSec }] : []),
            { l: r.strategy === "Proximity" ? "Disc (AS-RPA)" : "Discrimination", v: r.strategy === "Proximity" ? (r.asrpaDiscrimination ? (r.asrpaDiscrimination.block_class === "none" ? "1× (no mismatch)" : `${r.asrpaDiscrimination.disc_ratio >= 100 ? "≥100" : r.asrpaDiscrimination.disc_ratio.toFixed(0)}× ${r.asrpaDiscrimination.terminal_mismatch}`) : "AS-RPA") : r.gene === "IS6110" ? "N/A (control)" : `${typeof r.disc === "number" ? r.disc.toFixed(1) : r.disc}×`, c: r.strategy === "Proximity" ? (r.asrpaDiscrimination?.block_class === "none" ? T.danger : T.purple) : r.gene === "IS6110" ? T.textTer : discColor },
            ...(r.strategy === "Proximity" && r.proximityDistance ? [{ l: "Distance", v: `${r.proximityDistance} bp`, c: T.purple }] : []),
            { l: "Activity QC", v: r.activityQc != null ? r.activityQc.toFixed(3) : r.score.toFixed(3), c: T.textTer },
            ...(r.discriminationQc != null ? [{ l: "Disc QC", v: r.discriminationQc.toFixed(3), c: r.discriminationQc > 0.6 ? T.success : r.discriminationQc > 0.3 ? T.warning : T.danger }] : []),
            { l: "GC%", v: `${(r.gc * 100).toFixed(0)}%`, c: T.text },
            { l: "Off-targets", v: r.ot, c: r.ot === 0 ? T.success : T.warning },
            { l: "PAM", v: r.pam, c: T.text, badge: r.pamVariant && r.pamVariant !== "TTTV" ? r.pamVariant : null, penalty: r.pamPenalty },
          ].map((s, i) => (
            <div key={s.l} style={{ flex: mobile ? "1 1 40%" : 1, textAlign: "center", borderLeft: !mobile && i > 0 ? `1px dashed ${T.border}` : "none", minWidth: mobile ? "30%" : "auto" }}>
              <div style={{ fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "4px" }}>{s.l}</div>
              <div style={{ fontSize: mobile ? "15px" : "18px", fontWeight: 600, color: s.c, fontFamily: FONT }}>{s.v}</div>
              {s.badge && <div style={{ marginTop: "3px", display: "inline-block", padding: "1px 5px", borderRadius: "4px", fontSize: "9px", fontWeight: 600, fontFamily: FONT, background: s.penalty >= 0.5 ? "#FEF3C7" : "#FEF2F2", color: s.penalty >= 0.5 ? "#1D4ED8" : "#DC2626" }}>{s.badge} {s.penalty != null ? `${s.penalty}×` : ""}</div>}
            </div>
          ))}
        </div>

        {/* PROXIMITY explanation block */}
        {r.strategy === "Proximity" && (
          <div style={{ background: T.purpleLight, border: `1px solid ${T.purple}33`, borderRadius: "4px", padding: "16px 20px", marginBottom: "24px" }}>
            <div style={{ fontSize: "13px", fontWeight: 600, color: T.purple, fontFamily: HEADING, marginBottom: "6px" }}>Proximity Detection — PAM Desert Region</div>
            <div style={{ fontSize: "12px", color: "#2563EB", lineHeight: 1.6 }}>
              <p style={{ margin: "0 0 6px" }}>
                The <strong>{r.gene} {r.ref}{r.pos}{r.alt}</strong> mutation sits in a high-GC region with no Cas12a PAM placing the SNP within any spacer.
                Instead, the crRNA binds a conserved site <strong>{r.proximityDistance ? `${r.proximityDistance} bp` : "nearby"}</strong> from the mutation.
              </p>
              <p style={{ margin: 0 }}>
                Discrimination is provided by <strong>allele-specific RPA (AS-RPA) primers</strong> whose 3′ terminal nucleotide matches only the resistance allele.
                The crRNA confirms the amplified region is the correct locus.
              </p>
            </div>
          </div>
        )}

        {/* Amplicon Map */}
        <div style={{ marginBottom: "24px" }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "8px" }}>Amplicon Map</div>
          <div style={{ background: T.bgSub, borderRadius: "4px", padding: "12px 8px", border: `1px solid ${T.borderLight}` }}>
            <AmpliconMap r={r} />
          </div>
        </div>

        {/* crRNA Spacer */}
        <div style={{ marginBottom: "24px" }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "8px" }}>crRNA Spacer</div>
          <div style={{ background: T.bgSub, borderRadius: "4px", padding: "12px 14px", border: `1px solid ${T.borderLight}`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div>
              <span style={{ fontSize: "10px", color: T.textTer, marginRight: "6px" }}>5'→</span>
              <Seq s={displaySpacer} />
              <span style={{ fontSize: "10px", color: T.textTer, marginLeft: "6px" }}>→3'</span>
            </div>
            <button onClick={() => { navigator.clipboard?.writeText(displaySpacer); toast("Spacer copied to clipboard"); }} style={{ background: "none", border: `1px solid ${T.border}`, borderRadius: "4px", padding: "4px 8px", cursor: "pointer", display: "flex", alignItems: "center", gap: "4px", fontSize: "10px", color: T.textSec }}>
              <Copy size={12} /> Copy
            </button>
          </div>
        </div>

        {/* Mismatch Profile */}
        <div style={{ marginBottom: "24px" }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>MUT vs WT Mismatch Profile</div>
          <div style={{ fontSize: "11px", color: T.textSec, marginBottom: "10px" }}>Mismatched positions between mutant and wildtype spacer alignment</div>
          <div style={{ background: T.bgSub, borderRadius: "4px", padding: "14px", border: `1px solid ${T.borderLight}`, overflowX: "auto" }}>
            <MismatchProfile spacer={displaySpacer} wtSpacer={r.wtSpacer} strategy={r.strategy} />
          </div>
        </div>

        {/* Evidence */}
        {ref && (
          <div style={{ marginBottom: "24px" }}>
            <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "10px" }}>Evidence</div>
            <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
              {[
                ["WHO Classification", ref.who, ref.who === "Associated" ? "success" : "warning"],
                ["WHO Catalogue", ref.catalogue, null],
                ["Clinical Frequency", ref.freq, null],
                ["CRyPTIC Dataset", ref.cryptic || "—", null],
              ].map(([k, v, type], i, arr) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 16px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none", fontSize: "12px" }}>
                  <span style={{ color: T.textSec }}>{k}</span>
                  {type === "success" || type === "warning" ? <Badge variant={type}>{v}</Badge>
                   : <span style={{ fontWeight: 600, color: T.text }}>{v}</span>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Oligo Sequences */}
        <div style={{ marginBottom: "24px" }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "10px" }}>Oligo Sequences</div>
          <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
            {[
              { name: `${r.gene}_${r.ref}${r.pos}${r.alt}_crRNA`, seq: `AATTTCTACTCTTGTAGAT${displaySpacer}`, note: "Direct repeat + spacer (IVT template)" },
              ...(r.fwd ? [{ name: `${r.gene}_${r.ref}${r.pos}${r.alt}_FWD`, seq: r.fwd, note: "RPA forward primer" }] : []),
              ...(r.rev ? [{ name: `${r.gene}_${r.ref}${r.pos}${r.alt}_REV`, seq: r.rev, note: "RPA reverse primer" }] : []),
            ].map((o, i, arr) => (
              <div key={o.name} style={{ padding: "10px 16px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                  <span style={{ fontSize: "11px", fontWeight: 600, fontFamily: MONO, color: T.text }}>{o.name}</span>
                  <button onClick={() => { navigator.clipboard?.writeText(o.seq); toast(`${o.name} copied`); }} style={{ background: "none", border: `1px solid ${T.border}`, borderRadius: "5px", padding: "3px 6px", cursor: "pointer", fontSize: "10px", color: T.textSec, display: "flex", alignItems: "center", gap: "3px" }}><Copy size={10} /> Copy</button>
                </div>
                <Seq s={o.seq} />
                <div style={{ fontSize: "10px", color: T.textTer, marginTop: "3px" }}>{o.note}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Scoring Breakdown — 5 real pipeline features */}
        <div style={{ marginBottom: "24px" }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>Scoring Breakdown</div>
          <div style={{ fontSize: "11px", color: T.textSec, marginBottom: "10px" }}>Per-feature contribution to composite score (heuristic model)</div>
          <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
            {features.map((f, i) => (
              <div key={f.key} style={{ display: "flex", alignItems: "center", gap: "10px", padding: "8px 14px", borderBottom: i < features.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
                <div style={{ width: 130, fontSize: "11px", fontWeight: 600, color: T.text, flexShrink: 0 }}>{f.name}</div>
                <div style={{ flex: 1, display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{ flex: 1, height: 6, background: T.bgSub, borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${f.raw * 100}%`, height: "100%", background: f.raw > 0.7 ? T.primary : f.raw > 0.4 ? T.warning : T.danger, borderRadius: 3, transition: "width 0.3s" }} />
                  </div>
                  <span style={{ fontFamily: FONT, fontSize: "10px", fontWeight: 600, color: T.textSec, width: 36, textAlign: "right" }}>{(f.raw * 100).toFixed(0)}%</span>
                </div>
                <div style={{ width: 40, textAlign: "right", fontSize: "10px", color: T.textTer, fontFamily: FONT }}>×{(f.weight * 100).toFixed(0)}%</div>
                <div style={{ width: 50, textAlign: "right", fontFamily: FONT, fontSize: "11px", fontWeight: 600, color: T.text }}>{f.weighted.toFixed(3)}</div>
              </div>
            ))}
            <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: "10px", padding: "10px 14px", background: T.bgSub }}>
              <span style={{ fontSize: "11px", fontWeight: 600, color: T.textSec }}>Composite Score</span>
              <span style={{ fontFamily: FONT, fontSize: "14px", fontWeight: 600, color: T.text }}>{compositeCalc.toFixed(3)}</span>
              <span style={{ fontSize: "10px", color: T.textTer }}>(actual: {r.score.toFixed(3)})</span>
            </div>
          </div>
        </div>

        {/* Amplicon details */}
        {r.hasPrimers && (
          <div>
            <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "8px" }}>Amplicon Details</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
              <div style={{ flex: "1 1 auto", minWidth: mobile ? "100%" : 0, background: T.bgSub, borderRadius: "4px", padding: "12px", fontSize: "12px" }}>
                <div style={{ color: T.textTer, marginBottom: "4px" }}>Amplicon length</div>
                <div style={{ fontWeight: 600, fontFamily: FONT, color: T.text }}>{r.amplicon} bp</div>
              </div>
              <div style={{ flex: "1 1 auto", minWidth: mobile ? "45%" : 0, background: T.bgSub, borderRadius: "4px", padding: "12px", fontSize: "12px" }}>
                <div style={{ color: T.textTer, marginBottom: "4px" }}>Strategy</div>
                <div style={{ fontWeight: 600, color: T.text }}>{r.strategy}</div>
              </div>
              <div style={{ flex: "1 1 auto", minWidth: mobile ? "45%" : 0, background: r.hasSM ? T.primaryLight : T.bgSub, borderRadius: "4px", padding: "12px", fontSize: "12px" }}>
                <div style={{ color: T.textTer, marginBottom: "4px" }}>Synthetic mismatch</div>
                <div style={{ fontWeight: 600, color: r.hasSM ? T.primaryDark : T.textTer }}>{r.hasSM ? "Applied" : "None"}</div>
              </div>
            </div>
            {/* Shared amplicon warning for same-codon targets */}
            {(() => {
              const codonGroups = { "rpoB_RRDR": ["rpoB_S531L", "rpoB_H526Y", "rpoB_D516V"], "embB_M306": ["embB_M306V", "embB_M306I"] };
              for (const [, group] of Object.entries(codonGroups)) {
                if (group.includes(r.label)) {
                  const siblings = group.filter(l => l !== r.label);
                  return (
                    <div style={{ marginTop: "10px", padding: "8px 12px", background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.2)", borderRadius: "4px", fontSize: "11px", color: T.textSec, lineHeight: 1.6 }}>
                      <strong style={{ color: T.warning }}>Shared amplicon:</strong> This target shares the same amplicon region with {siblings.join(", ")}. In a single-pot assay both mutations produce a positive drug-class signal, but the specific amino acid change cannot be resolved without additional crRNA reporters.
                    </div>
                  );
                }
              }
              return null;
            })()}
          </div>
        )}
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════════ */
const NAV = [
  { section: "Design", items: [
    { id: "home", label: "Home", icon: Activity },
    { id: "methods", label: "Methods", icon: BookOpen },
    { id: "results", label: "Results", icon: BarChart3 },
  ]},
  { section: "Library", items: [
    { id: "panels", label: "Panels", icon: Layers },
    { id: "mutations", label: "Mutations", icon: Database },
  ]},
  { section: "Models", items: [
    { id: "scoring", label: "Scoring", icon: Brain },
    { id: "research", label: "Research", icon: FlaskConical },
  ]},
];

const Sidebar = ({ page, setPage, connected, mobileOpen, setMobileOpen, collapsed, setCollapsed }) => {
  const mobile = useIsMobile();
  const handleNav = (id) => { setPage(id); if (mobile) setMobileOpen(false); };
  const isCollapsed = !mobile && collapsed;

  const inner = (
    <aside style={{
      width: mobile ? "280px" : (isCollapsed ? 56 : 220), background: T.sidebar,
      borderRight: mobile ? "none" : `1px solid ${T.border}`,
      display: "flex", flexDirection: "column", flexShrink: 0,
      transition: "width 0.2s ease",
      ...(mobile ? { position: "fixed", top: 0, left: 0, bottom: 0, zIndex: 9998 } : {}),
    }}>
      {/* Logo + Toggle */}
      <div style={{ padding: isCollapsed ? "16px 0" : "16px 20px", borderBottom: `1px solid ${T.border}`, display: "flex", alignItems: "center", justifyContent: isCollapsed ? "center" : "space-between", gap: "8px" }}>
        {!isCollapsed && (
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <img src="/compass-logo.png" alt="COMPASS" style={{ height: "20px", objectFit: "contain" }} />
            {!connected && (
              <span style={{ fontSize: "10px", color: T.danger, fontWeight: 600, display: "flex", alignItems: "center", gap: "3px" }}>
                <WifiOff size={10} /> offline
              </span>
            )}
          </div>
        )}
        {isCollapsed && (
          <button onClick={() => setCollapsed(false)} style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", display: "flex", borderRadius: "4px" }} title="Expand sidebar">
            <PanelLeft size={18} color={T.textSec} />
          </button>
        )}
        {mobile ? (
          <button onClick={() => setMobileOpen(false)} style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", marginLeft: "auto" }}><X size={20} color={T.textSec} /></button>
        ) : !isCollapsed && (
          <button onClick={() => setCollapsed(!collapsed)} style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", display: "flex", borderRadius: "4px" }} title="Collapse sidebar">
            <PanelLeftClose size={16} color={T.textTer} />
          </button>
        )}
      </div>

      {/* Nav groups */}
      <nav style={{ flex: 1, padding: isCollapsed ? "12px 6px" : "8px 12px", overflowY: "auto" }}>
        {NAV.map((g) => (
          <div key={g.section} style={{ marginBottom: "20px" }}>
            {!isCollapsed && <div style={{ fontSize: "11px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", padding: "0 8px", marginBottom: "4px" }}>{g.section}</div>}
            {g.items.map((it) => {
              const active = page === it.id;
              return (
                <button key={it.id} onClick={() => handleNav(it.id)} title={isCollapsed ? it.label : undefined} style={{
                  display: "flex", alignItems: "center", justifyContent: isCollapsed ? "center" : "flex-start", gap: "8px", width: "100%", padding: isCollapsed ? "8px 0" : "8px 12px",
                  borderRadius: "4px", border: "none", cursor: "pointer", fontFamily: FONT, fontSize: "13px",
                  fontWeight: active ? 500 : 400, background: active ? T.sidebarActive : "transparent",
                  color: active ? T.primary : T.sidebarText, marginBottom: "2px",
                  borderLeft: active && !isCollapsed ? `2px solid ${T.primary}` : "2px solid transparent",
                  transition: "background 120ms ease-out, color 120ms ease-out, border-color 120ms ease-out",
                }}>
                  <it.icon size={16} strokeWidth={active ? 2 : 1.5} />
                  {!isCollapsed && it.label}
                </button>
              );
            })}
          </div>
        ))}
      </nav>
      {/* Footer */}
      {!isCollapsed && (
        <div style={{ padding: "12px 16px", borderTop: `1px solid ${T.border}`, fontSize: "10px", color: T.borderStrong, textAlign: "center", fontFamily: MONO }}>
          v2.0
        </div>
      )}
    </aside>
  );

  if (mobile) {
    if (!mobileOpen) return null;
    return createPortal(
      <>
        <div onClick={() => setMobileOpen(false)} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 9997 }} />
        {inner}
      </>,
      document.body
    );
  }
  return inner;
};

/* ═══════════════════════════════════════════════════════════════════
   COLLAPSIBLE SECTION HELPER
   ═══════════════════════════════════════════════════════════════════ */
const CollapsibleSection = ({ title, children, defaultOpen = false, badge }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: "12px", border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
      <button onClick={() => setOpen(!open)} style={{
        display: "flex", alignItems: "center", gap: "8px", width: "100%", padding: "12px 16px",
        background: T.bgSub, border: "none", cursor: "pointer", fontFamily: FONT, fontSize: "13px",
        fontWeight: 600, color: T.text, textAlign: "left", justifyContent: "space-between",
        transition: "background 120ms ease-out",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <ChevronRight size={14} style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 200ms ease-out" }} />
          {title}
          {badge && (
            <span style={{ fontSize: 10, fontWeight: 600, padding: "2px 8px", borderRadius: "3px", background: badge.bg || T.primaryLight, color: badge.color || T.primary, fontFamily: FONT }}>{badge.text}</span>
          )}
        </div>
        <span style={{ fontSize: 10, color: T.textTer, fontWeight: 400 }}>{open ? "collapse" : "expand"}</span>
      </button>
      {open && <div style={{ padding: "16px", borderTop: `1px solid ${T.border}` }}>{children}</div>}
    </div>
  );
};

/* Collapsible figure wrapper for Overview tab — open by default, click to toggle */
const FigureSection = ({ title, children, defaultOpen = true }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: "24px" }}>
      <button onClick={() => setOpen(!open)} style={{
        display: "flex", alignItems: "center", gap: "6px", background: "none", border: "none",
        cursor: "pointer", padding: "4px 0", marginBottom: open ? "8px" : 0, fontFamily: FONT,
        fontSize: "11px", fontWeight: 600, color: T.textTer, textTransform: "uppercase",
        letterSpacing: "0.04em",
      }}>
        <ChevronDown size={12} style={{ transform: open ? "rotate(0deg)" : "rotate(-90deg)", transition: "transform 0.15s" }} />
        {title}
      </button>
      {open && children}
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   HOME PAGE — Run workflow + methodology blog
   ═══════════════════════════════════════════════════════════════════ */
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
  const [runName, setRunName] = useState("COMPASS_panel_" + new Date().toISOString().slice(0, 10).replace(/-/g, ""));
  const [mode, setMode] = useState("standard");
  const [selectedModules, setSelectedModules] = useState(new Set(MODULES.map(m => m.id)));
  const [configOpen, setConfigOpen] = useState(false);
  const [panelSectionOpen, setPanelSectionOpen] = useState(true);
  const [scorerSectionOpen, setScorerSectionOpen] = useState(true);
  const [scorer, setScorer] = useState(null); // "heuristic" | "compass_ml" | null
  const [enzymeId, setEnzymeId] = useState("enAsCas12a"); // "AsCas12a" | "enAsCas12a"
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState(null);

  /* ── Inline pipeline execution state ── */
  const [pipeJobId, setPipeJobId] = useState(null);
  const [pipeStep, setPipeStep] = useState(0);
  const [pipeDone, setPipeDone] = useState(false);
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
  const ALL_INDICES = MUTATIONS.map((_, i) => i);
  const CORE5_LABELS = ["rpoB_S531L", "katG_S315T", "fabG1_C-15T", "gyrA_D94G", "rrs_A1401G"];
  const CORE5_INDICES = MUTATIONS.map((m, i) => CORE5_LABELS.includes(`${m.gene}_${m.ref}${m.pos}${m.alt}`) ? i : -1).filter(i => i >= 0);

  const [panel, setPanel] = useState(null);        // "mdr14" | "mdr14_rnasep" | "core5" | "custom" | null
  const [selected, setSelected] = useState(new Set());
  const [targetsOpen, setTargetsOpen] = useState(false);

  const selectPanel = (p) => {
    setPanel(p);
    if (p === "mdr14") { setSelected(new Set(ALL_INDICES)); setTargetsOpen(false); }
    else if (p === "core5") { setSelected(new Set(CORE5_INDICES)); setTargetsOpen(false); }
    else { setTargetsOpen(true); }
  };

  const toggleMut = (i) => { const n = new Set(selected); n.has(i) ? n.delete(i) : n.add(i); setSelected(n); };
  const selectedDrugs = [...new Set([...selected].map(i => MUTATIONS[i]?.drug).filter(Boolean))];

  const launch = async () => {
    setLaunching(true);
    setError(null);
    const muts = [...selected].map(i => ({
      gene: MUTATIONS[i].gene,
      ref_aa: MUTATIONS[i].ref,
      position: MUTATIONS[i].pos,
      alt_aa: MUTATIONS[i].alt,
      drug: MUTATIONS[i].drug || "OTHER",
    }));
    const apiMode = "full";
    const overrides = scorer !== "heuristic" ? { scorer } : {};
    if (connected) {
      const { data, error: err } = await submitRun(runName, apiMode, muts, overrides, enzymeId);
      if (err) { setError(err); setLaunching(false); return; }
      startInlinePipeline(data.job_id);
    } else {
      startInlinePipeline("mock-" + scorer + "-" + [...selected].join(",") + "-" + Date.now());
    }
  };

  /* ── Inline pipeline management ── */
  const startInlinePipeline = (jobId) => {
    setPipeJobId(jobId);
    setPipeStep(0);
    setPipeDone(false);
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
      try {
        const ws = connectJobWS(jobId,
          (msg) => {
            if (msg.current_module) {
              const idx = MODULE_NAME_MAP[msg.current_module];
              if (idx !== undefined) setPipeStep(idx);
            }
            if (msg.status === "complete" || msg.status === "completed") {
              finishInlinePipeline(jobId);
            }
          },
          () => { startInlinePolling(jobId); }
        );
        pipeWsRef.current = ws;
      } catch {
        startInlinePolling(jobId);
      }
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

  const startInlinePolling = (jobId) => {
    pipePollRef.current = setInterval(async () => {
      const { data } = await getJob(jobId);
      if (data) {
        const idx = MODULE_NAME_MAP[data.current_module] || 0;
        setPipeStep(idx);
        if (data.status === "complete" || data.status === "completed") {
          finishInlinePipeline(jobId);
          clearInterval(pipePollRef.current);
        }
      }
    }, 2000);
  };

  const finishInlinePipeline = (jobId) => {
    setPipeDone(true);
    setShowLog(true);
    if (pipeTimerRef.current) clearInterval(pipeTimerRef.current);

    if (connected) {
      getResults(jobId).then(({ data }) => {
        if (data?.module_stats?.length) setPipeStats(data.module_stats);
      });
    } else {
      const m5Detail = scorer === "compass_ml"
        ? "241 candidates scored — Compass-ML activity (0.125–0.608) · Compass-ML discrimination (0.288–0.959) · PAM-adjusted (0.045–0.608)"
        : "241 candidates scored — Heuristic QC (0.125–0.608) · SeqCNN calibrated T=1.1 (0.288–0.959) · PAM-adjusted (0.045–0.608)";
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
    return () => {
      if (pipeTimerRef.current) clearInterval(pipeTimerRef.current);
      pipeWsRef.current?.close();
      if (pipePollRef.current) clearInterval(pipePollRef.current);
    };
  }, []);

  /* Scorer-aware modules — M5 adapts to selected scoring model */
  const effectiveModules = useMemo(() => MODULES.map(m =>
    m.id === "M5" && scorer === "compass_ml"
      ? {
          ...m,
          name: "Compass-ML Scoring",
          execDesc: "Compass-ML (CNN + RNA-FM + RLPA) inference for efficiency and discrimination scoring",
          substeps: [
            "Loading Compass-ML checkpoint (235K params, CNN + PAM + RNA-FM + RLPA)",
            "Computing RNA-FM embeddings (640-dim, frozen)",
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
                {panel === "mdr14" ? "MDR-TB 14-plex" : panel === "mdr14_rnasep" ? "MDR-TB 14-plex + RNaseP" : panel === "core5" ? "Core 5-plex" : "Custom"} · {scorer === "compass_ml" ? "Compass-ML" : scorer === "heuristic" ? "Heuristic" : ""} · {selected.size} targets
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

        {/* 1. Run Name — compact inline */}
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

        {/* Diagnostic Panel */}
        <div style={{ marginBottom: "16px" }}>
          <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
          <button onClick={() => setPanelSectionOpen(!panelSectionOpen)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: "10px", padding: "14px 16px",
            background: "none", border: "none", cursor: "pointer", fontFamily: FONT,
          }}>
            <Layers size={14} color={T.textSec} />
            <span style={{ fontSize: "13px", fontWeight: 600, color: T.text, flex: 1, textAlign: "left" }}>Diagnostic Panel</span>
            <span style={{ fontSize: "11px", color: T.textTer, marginRight: "4px" }}>{panel ? (panel === "mdr14" ? "MDR-TB 14-plex" : panel === "mdr14_rnasep" ? "MDR-TB + RNaseP" : panel === "core5" ? "Core 5-plex" : "Custom") : "select panel"}</span>
            <ChevronDown size={14} color={T.textSec} style={{ transform: panelSectionOpen ? "rotate(180deg)" : "none", transition: "0.2s" }} />
          </button>
          {panelSectionOpen && (
            <div style={{ padding: "16px 20px", borderTop: `1px solid ${T.borderLight}` }}>
              {/* Preset cards — 2×2 */}
              <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "10px" }}>
                {[
                  { id: "mdr14", name: "MDR-TB 14-plex", targets: ALL_INDICES.length + " targets",
                    desc: "Full WHO-catalogued first- and second-line resistance panel.",
                    meta: ["6 drug classes", "Tier 1–2", "High + Moderate"] },
                  { id: "mdr14_rnasep", name: "MDR-TB 14-plex + RNaseP", targets: (ALL_INDICES.length + 1) + " targets",
                    desc: "Full MDR panel plus human RNaseP (RPPH1) extraction control.",
                    meta: ["6 drug classes", "+ extraction ctrl", "CDC standard"] },
                  { id: "core5", name: "Core 5-plex", targets: "5 targets",
                    desc: "High-confidence tier-1 mutations only. Point-of-care screening.",
                    meta: ["4 drug classes", "Tier 1", "High confidence"] },
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
                {/* Drug filter chips — only for Custom panel */}
                {panel === "custom" && (
                  <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", padding: "12px 16px", borderBottom: `1px solid ${T.borderLight}` }}>
                    <button onClick={() => setSelected(new Set(ALL_INDICES))} style={{ padding: "5px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, background: selected.size === MUTATIONS.length ? T.primary : T.bg, color: selected.size === MUTATIONS.length ? "#fff" : T.textSec, fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: FONT }}>All ({MUTATIONS.length})</button>
                    <button onClick={() => setSelected(new Set())} style={{ padding: "5px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, background: selected.size === 0 ? T.bgSub : T.bg, color: T.textSec, fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: FONT }}>None</button>
                    <div style={{ width: 1, background: T.border, margin: "0 4px" }} />
                    {[...new Set(MUTATIONS.map(m => m.drug))].map(drug => {
                      const indices = MUTATIONS.map((m, i) => m.drug === drug ? i : -1).filter(i => i >= 0);
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
                    {MUTATIONS.map((m, i) => {
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
                          <td style={{ padding: "10px 12px", fontFamily: MONO, fontSize: "12px", color: T.textSec }}>{m.ref}{m.pos}{m.alt}</td>
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

        {/* Advanced Configuration (collapsible card — matching mutations card design) */}
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
                        <span>Capped at 120 bp — cfDNA fragments in blood are ~100–160 bp. Shorter amplicons maximise template capture from fragmented circulating DNA.</span>
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
            <span>{[...new Set([...selected].map(i => MUTATIONS[i]?.drug))].length} drug classes</span>
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
        const fmtDur = (ms) => ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
        const m2Out = statMap["M2"]?.candidates_out || 0;
        const finalSize = statMap["M9"]?.candidates_out || statMap["M7"]?.candidates_out || 0;

        return (
          <div style={{
            background: T.bg,
            border: `1px solid ${T.border}`,
            borderRadius: "4px",
            marginBottom: "24px", overflow: "hidden",
          }}>
            {/* Running state — module + dynamic substep cycling */}
            {!pipeDone && (() => {
              const stepEstSec = activeModule.estSec || 10;
              const stepElapsed = (Date.now() - pipeStepStartRef.current) / 1000;
              const subs = activeModule.substeps || [activeModule.execDesc];
              const subDur = stepEstSec / subs.length;
              const subIdx = Math.min(Math.floor(stepElapsed / Math.max(subDur, 0.5)), subs.length - 1);
              const intraStep = Math.min(0.95, 1 - Math.exp(-2.5 * stepElapsed / stepEstSec));
              const pct = ((pipeStep + intraStep) / effectiveModules.length) * 100;
              return (
                <div style={{ padding: "16px 20px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <div style={{ width: 18, height: 18, display: "flex", alignItems: "center", justifyContent: "center", animation: "subtlePulse 2s ease-in-out infinite" }}>
                      <ActiveIcon size={14} color={T.primary} strokeWidth={1.8} />
                    </div>
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "2px" }}>
                      <div key={pipeStep} style={{ display: "flex", alignItems: "baseline", gap: "8px", animation: "stepSwipeUp 0.25s ease-out" }}>
                        <span style={{ fontFamily: MONO, fontSize: "11px", color: T.textTer }}>{activeModule.id}</span>
                        <span style={{ fontSize: "14px", fontWeight: 500, color: T.text }}>{activeModule.name}</span>
                      </div>
                      <div key={`sub-${pipeStep}-${subIdx}`} style={{ fontSize: "11px", color: T.textSec, lineHeight: 1.4, animation: "substepSwipe 0.35s ease-out", minHeight: "16px" }}>
                        {subs[subIdx]}
                      </div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "2px", flexShrink: 0 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <svg width="14" height="14" viewBox="0 0 16 16" style={{ animation: "spin 1s linear infinite" }}>
                          <circle cx="8" cy="8" r="6" fill="none" stroke={T.border} strokeWidth="2" />
                          <path d="M8 2a6 6 0 0 1 6 6" fill="none" stroke={T.primary} strokeWidth="2" strokeLinecap="round" />
                        </svg>
                        <span style={{ fontFamily: MONO, fontSize: "12px", color: T.textSec, fontVariantNumeric: "tabular-nums" }}>{pipeElapsed.toFixed(1)}s</span>
                      </div>
                      <span style={{ fontSize: "10px", color: T.textTer, fontFamily: MONO }}>{pipeStep + 1} of {effectiveModules.length} modules</span>
                    </div>
                  </div>
                  {/* Segmented progress dots */}
                  <div style={{ display: "flex", gap: "3px", marginTop: "12px" }}>
                    {effectiveModules.map((_, i) => (
                      <div key={i} style={{
                        flex: 1, height: "4px", borderRadius: "2px",
                        background: i < pipeStep ? T.primary : i === pipeStep ? T.primary : T.bgHover,
                        opacity: i === pipeStep ? (0.4 + 0.6 * intraStep) : 1,
                        transition: "background 200ms ease-out, opacity 300ms ease-out",
                      }} />
                    ))}
                  </div>
                  {/* Time estimate hint — appears after 5s on same step */}
                  {stepElapsed > 5 && (() => {
                    const remainingStepSec = Math.max(0, stepEstSec - stepElapsed);
                    const futureModulesSec = effectiveModules.slice(pipeStep + 1).reduce((sum, m) => sum + (m.estSec || 10), 0);
                    const totalRemaining = Math.ceil((remainingStepSec + futureModulesSec) / 60);
                    const remainingSec = Math.ceil(remainingStepSec + futureModulesSec);
                    const timeText = remainingSec >= 60 ? `~${totalRemaining} min remaining` : `~${remainingSec}s remaining`;
                    return (
                      <div style={{ marginTop: "6px", fontSize: "11px", color: T.textTer, display: "flex", alignItems: "center", gap: "6px" }}>
                        <Clock size={11} color={T.textTer} strokeWidth={1.8} />
                        <span>{activeModule.name} can take a moment {"\u2014"} {timeText} ({effectiveModules.length - pipeStep - 1} modules after this)</span>
                      </div>
                    );
                  })()}
                </div>
              );
            })()}

            {/* Complete state — summary + logs toggle + CTA */}
            {pipeDone && (
              <div style={{ padding: "16px 20px" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <CheckCircle size={18} color={T.text} strokeWidth={2} />
                    <span style={{ fontSize: "13px", fontWeight: 500, color: T.text, fontFamily: FONT }}>
                      Pipeline complete
                    </span>
                    <span style={{ fontSize: "13px", color: T.textSec, fontFamily: MONO }}>
                      {pipeElapsed.toFixed(1)}s
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

/* ═══════════════════════════════════════════════════════════════════
   METHODS PAGE
   ═══════════════════════════════════════════════════════════════════ */
const MethodsPage = () => {
  const mobile = useIsMobile();
  const [openSections, setOpenSections] = useState({ pipeline: false, narsilml: false, architecture: false, clinical: false, discrimination: false, nuclease: false, defaults: false, limitations: false, references: false });
  const sectionRefs = useRef({});

  const toggle = (k) => setOpenSections(prev => {
    const willOpen = !prev[k];
    if (willOpen) {
      setTimeout(() => {
        sectionRefs.current[k]?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    }
    return { ...prev, [k]: willOpen };
  });

  /* ── Reusable section card — grey header bar, collapsible body ── */
  const Section = ({ id, title, subtitle, badge, children }) => {
    const isOpen = openSections[id];
    return (
      <div ref={el => { sectionRefs.current[id] = el; }} style={{ marginBottom: "12px", border: `1px solid ${T.border}`, borderRadius: "6px", overflow: "hidden", background: T.bg }}>
        <button onClick={() => toggle(id)} style={{
          display: "flex", alignItems: "center", width: "100%", padding: "14px 20px",
          background: T.bgSub, border: "none", cursor: "pointer", fontFamily: FONT,
          gap: "10px", textAlign: "left",
        }}>
          <div style={{ width: "16px", height: "16px", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            {isOpen ? <ChevronDown size={14} color={T.textSec} /> : <ChevronRight size={14} color={T.textSec} />}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>{title}</span>
              {badge && <span style={{ fontSize: "10px", fontWeight: 600, padding: "2px 8px", borderRadius: "3px", background: badge.bg || T.primaryLight, color: badge.color || T.primary, fontFamily: FONT }}>{badge.text}</span>}
            </div>
            {subtitle && <div style={{ fontSize: "12px", color: T.textSec, marginTop: "2px", lineHeight: 1.4 }}>{subtitle}</div>}
          </div>
          <span style={{ fontSize: "10px", color: T.textTer, flexShrink: 0 }}>{isOpen ? "collapse" : "expand"}</span>
        </button>
        {isOpen && <div style={{ padding: mobile ? "16px" : "20px 24px", borderTop: `1px solid ${T.border}` }}>{children}</div>}
      </div>
    );
  };

  /* ── Key-value row ── */
  const KV = ({ label, value, mono }) => (
    <div style={{ display: "flex", alignItems: "baseline", gap: "12px", padding: "8px 0", borderBottom: `1px solid ${T.borderLight}` }}>
      <span style={{ fontSize: "12px", color: T.textTer, fontWeight: 500, minWidth: mobile ? 90 : 130, flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: "12px", fontWeight: 600, color: T.text, fontFamily: mono ? MONO : FONT }}>{value}</span>
    </div>
  );

  return (
    <div style={{ padding: mobile ? "24px 16px" : "32px 40px" }}>

      {/* ═══ Page Header ═══ */}
      <div style={{ marginBottom: "28px" }}>
        <h1 style={{ fontSize: "20px", fontWeight: 600, color: T.text, fontFamily: HEADING, margin: 0, letterSpacing: "-0.01em" }}>Methods</h1>
        <p style={{ fontSize: "13px", color: T.textSec, marginTop: "4px", lineHeight: 1.5 }}>
          How COMPASS designs, scores, and validates CRISPR-Cas12a diagnostic panels for drug-resistant tuberculosis.
        </p>
      </div>

      {/* ═══════════ 1. PIPELINE ═══════════ */}
      <Section id="pipeline" title="Pipeline" subtitle="End-to-end from WHO mutation catalogue to validated diagnostic panel" badge={{ text: "4 stages", bg: T.primaryLight, color: T.primary }}>
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "repeat(4, 1fr)", gap: "0", border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
          {[
            { n: "01", icon: Map, title: "Define targets", desc: "Resolve WHO catalogue mutations to H37Rv genomic coordinates, codon context, and drug class annotations.", color: T.primary },
            { n: "02", icon: Brain, title: "Score candidates", desc: "Scan PAM sites, generate crRNAs, and predict activity with Compass-ML trained on 15K cis-cleavage measurements.", color: T.primary },
            { n: "03", icon: Zap, title: "Optimise panel", desc: "Simulated annealing over candidate assignments with co-designed AS-RPA primers and multiplex constraints.", color: T.primaryDark },
            { n: "04", icon: Shield, title: "Assess clinically", desc: "Evaluate against WHO TPP thresholds for per-drug sensitivity, specificity, and three operating modes.", color: T.success },
          ].map((c, i) => (
            <div key={c.n} style={{ padding: "20px", borderLeft: i > 0 && !mobile ? `1px solid ${T.border}` : "none", borderTop: i > 0 && mobile ? `1px solid ${T.border}` : "none" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                <div style={{ width: "32px", height: "32px", borderRadius: "6px", background: T.bgSub, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <c.icon size={16} color={c.color} strokeWidth={1.8} />
                </div>
                <span style={{ fontSize: "11px", fontWeight: 600, color: T.textTer, fontFamily: MONO }}>{c.n}</span>
              </div>
              <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "6px" }}>{c.title}</div>
              <div style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6 }}>{c.desc}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* ═══════════ 2. COMPASS-ML ═══════════ */}
      <Section id="narsilml" title="Compass-ML" subtitle="Dual-branch neural network predicting Cas12a guide efficiency and mismatch discrimination" badge={{ text: "235K params", bg: T.primaryLight, color: T.primary }}>

        {/* Metric strip */}
        <div style={{ display: "flex", border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden", marginBottom: "20px" }}>
          {[
            { label: "PARAMETERS", value: "235K" },
            { label: "VAL \u03c1", value: "0.750" },
            { label: "DISC r", value: "0.57" },
            { label: "PAM", value: "9-class" },
            { label: "INFERENCE", value: "<1ms" },
          ].map((s, i) => (
            <div key={s.label} style={{ flex: 1, padding: "14px 12px", textAlign: "center", borderLeft: i > 0 ? `1px solid ${T.border}` : "none" }}>
              <div style={{ fontSize: "10px", fontWeight: 500, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.03em", marginBottom: "4px" }}>{s.label}</div>
              <div style={{ fontSize: "18px", fontWeight: 600, color: T.text, fontFamily: MONO }}>{s.value}</div>
            </div>
          ))}
        </div>

        {/* Three branches */}
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr 1fr", gap: "12px", marginBottom: "20px" }}>
          {[
            { tag: "CNN", title: "Target DNA Branch", desc: "Multi-scale convolutions (k=3,5,7) scan the 34-nt target context for PAM quality, seed composition, and dinucleotide patterns." },
            { tag: "RNA-FM", title: "Guide RNA Branch", desc: "Pre-trained foundation model (23M sequences) captures folding stability and accessibility governing Cas12a loading." },
            { tag: "RLPA", title: "R-Loop Propagation", desc: "Directional attention encodes PAM-proximal to distal R-loop propagation. Ablation: +0.1% over CNN+PAM+RNA-FM (consistent across 3 seeds)." },
          ].map(c => (
            <div key={c.tag} style={{ padding: "16px", border: `1px solid ${T.border}`, borderRadius: "4px" }}>
              <span style={{ fontSize: "10px", fontWeight: 600, color: T.primary, padding: "2px 6px", borderRadius: "3px", background: T.primaryLight, fontFamily: MONO, display: "inline-block", marginBottom: "10px" }}>{c.tag}</span>
              <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "6px" }}>{c.title}</div>
              <div style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6 }}>{c.desc}</div>
            </div>
          ))}
        </div>

        {/* Multi-task callout */}
        <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}25`, borderRadius: "4px", padding: "12px 16px", fontSize: "13px", color: T.primaryDark, lineHeight: 1.6, marginBottom: "20px" }}>
          <strong>Multi-task learning</strong> — Efficiency and discrimination are predicted jointly. Discrimination (the MUT/WT cleavage ratio) determines whether a guide can distinguish resistant from susceptible bacteria at single-nucleotide resolution.
        </div>

        {/* Technical details table */}
        <div style={{ fontSize: "11px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.03em", marginBottom: "8px" }}>Technical details</div>
        <div style={{ marginBottom: "16px" }}>
          {[
            ["Architecture", "Dual-branch CNN + RNA-FM with RLPA + 9-class PAM encoding"],
            ["Training data", "Kim 2018 (15K cis) + flanking shuffle augmentation"],
            ["Val \u03c1 (production)", "0.750 (best of 3 seeds, Kim 2018 HT1-2)"],
            ["Training protocol", "AdamW + CosineWarmRestarts + Huber + soft Spearman"],
            ["Multi-task heads", "Efficiency (sigmoid) + Discrimination (Softplus)"],
            ["Attention", "R-Loop Propagation Attention (RLPA), directional propagation bias"],
          ].map(([k, v]) => <KV key={k} label={k} value={v} />)}
        </div>

        {/* Benchmark callout */}
        <div style={{ background: T.bgSub, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "12px 16px", fontSize: "12px", color: T.textSec, lineHeight: 1.65 }}>
          <strong style={{ color: T.text }}>Benchmark:</strong> Full architecture (CNN + PAM + RNA-FM + RLPA) achieves {"\u03c1"} = 0.750 on Kim 2018 HT1-2 validation (best of 3 seeds). Ablation: CNN-only {"\u03c1"} = 0.740, +PAM = 0.741, +RNA-FM = 0.744, +RLPA = 0.745. DeepCpf1 baseline {"\u03c1"} = 0.71.
        </div>
      </Section>

      {/* ═══════════ 3. ARCHITECTURE DETAIL ═══════════ */}
      <Section id="architecture" title="Architecture Detail" subtitle="Full layer-by-layer breakdown of the Compass-ML network">
        {[
          { label: "Branch 1", title: "Multi-Scale CNN", input: "34-nt one-hot encoded target (4 PAM + 20 protospacer + 10 flanking).", process: "Three parallel conv paths (k=3,5,7), 40 channels each, BN + dropout(0.3). Projected to 64-dim via 1\u00d71 conv.", output: "64-dim per position: dinucleotide preferences, seed complementarity, PAM patterns." },
          { label: "Branch 2", title: "RNA-FM Projection", input: "Guide RNA (20\u201323 nt). Processed by frozen RNA-FM (23M sequences, masked LM).", process: "640-dim per-nucleotide embeddings \u2192 trainable linear \u2192 64-dim. Zero-padded to 34 positions.", output: "64-dim structural embedding: folding, stability, 5\u2032 accessibility." },
          { label: "Fusion", title: "R-Loop Propagation Attention", input: "Concatenated 128-dim (64 CNN + 64 RNA-FM) at each of 34 positions.", process: "Single-head attention, 32-dim Q/K/V, directional propagation bias (PAM-proximal \u2192 distal), learnable 34\u00d734 positional bias.", output: "Attention-weighted 128-dim features re-weighted by positional importance." },
          { label: "Output", title: "Multi-Task Heads", input: "RLPA-weighted representation, globally pooled.", process: "Efficiency: 128\u219264\u219232\u21921 (sigmoid). Discrimination: 547\u219264\u219232\u21921 (Softplus).", output: "Two scalars: efficiency (0\u20131) and discrimination ratio (fold-change MUT/WT)." },
        ].map((block, idx, arr) => (
          <div key={block.title} style={{ padding: "16px 0", borderBottom: idx < arr.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
              <span style={{ fontSize: "10px", fontWeight: 600, color: T.primary, background: T.primaryLight, padding: "2px 8px", borderRadius: "3px", fontFamily: MONO, textTransform: "uppercase" }}>{block.label}</span>
              <span style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>{block.title}</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr 1fr", gap: "16px" }}>
              {[{ l: "Input", t: block.input }, { l: "Process", t: block.process }, { l: "Output", t: block.output }].map(col => (
                <div key={col.l}>
                  <div style={{ fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.03em", marginBottom: "4px" }}>{col.l}</div>
                  <div style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6 }}>{col.t}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </Section>

      {/* ═══════════ 4. CLINICAL ASSESSMENT ═══════════ */}
      <Section id="clinical" title="Clinical Assessment" subtitle="WHO Target Product Profile compliance for drug susceptibility testing">
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "12px" }}>
          {[
            { icon: TrendingUp, title: "Per-drug sensitivity", desc: "WHO TPP minimums: \u226595% RIF, \u226590% INH/FQ, \u226580% EMB/PZA/AG." },
            { icon: Shield, title: "Specificity estimate", desc: "Discrimination ratio predicts false-positive rates. \u22653\u00d7 diagnostic-grade, \u226510\u00d7 reference-lab." },
            { icon: Settings, title: "Three operating modes", desc: "High Sensitivity (field), Balanced (WHO TPP), High Specificity (reference lab)." },
            { icon: Layers, title: "Ranked alternatives", desc: "3\u20135 backup candidates per target with documented efficiency-discrimination tradeoffs." },
          ].map(c => (
            <div key={c.title} style={{ display: "flex", gap: "14px", padding: "16px", border: `1px solid ${T.border}`, borderRadius: "4px" }}>
              <div style={{ width: "32px", height: "32px", borderRadius: "6px", flexShrink: 0, background: T.bgSub, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <c.icon size={16} color={T.primary} strokeWidth={1.8} />
              </div>
              <div>
                <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "4px" }}>{c.title}</div>
                <div style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6 }}>{c.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* ═══════════ 5. DISCRIMINATION THRESHOLDS ═══════════ */}
      <Section id="discrimination" title="Discrimination Thresholds" subtitle="Mismatch discrimination determines clinical deployment tier">
        <p style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6, margin: "0 0 16px" }}>
          For Direct candidates: Cas12a cleavage ratio (MUT/WT) predicted by XGBoost on 18 thermodynamic features (r=0.57, trained on 6,136 EasyDesign pairs). For Proximity candidates: AS-RPA primer selectivity.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr 1fr" : "repeat(4, 1fr)", gap: "0", border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden", marginBottom: "16px" }}>
          {[
            { label: "Excellent", val: "\u2265 10\u00d7", desc: "Single-plex clinical use. Robust across sample types.", color: T.success },
            { label: "Good", val: "\u2265 3\u00d7", desc: "Multiplex panel. Electrochemical and lateral flow.", color: T.primary },
            { label: "Acceptable", val: "\u2265 2\u00d7", desc: "Requires confirmatory readout or dual-target.", color: T.warning },
            { label: "Insufficient", val: "< 2\u00d7", desc: "Synthetic mismatch enhancement needed.", color: T.danger },
          ].map((t, i) => (
            <div key={t.label} style={{ padding: "16px 16px", borderLeft: i > 0 && !mobile ? `1px solid ${T.border}` : "none", borderTop: i >= 2 && mobile ? `1px solid ${T.border}` : "none" }}>
              <div style={{ fontSize: "18px", fontWeight: 600, color: T.text, fontFamily: MONO, marginBottom: "2px" }}>{t.val}</div>
              <div style={{ fontSize: "11px", fontWeight: 600, color: t.color, textTransform: "uppercase", letterSpacing: "0.03em", marginBottom: "6px" }}>{t.label}</div>
              <div style={{ fontSize: "11px", color: T.textSec, lineHeight: 1.5 }}>{t.desc}</div>
            </div>
          ))}
        </div>

        {/* Synthetic mismatch enhancement */}
        <div style={{ background: T.bgSub, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "12px 16px" }}>
          <div style={{ fontSize: "12px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>Synthetic Mismatch Enhancement</div>
          <div style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6 }}>
            For candidates with insufficient discrimination, deliberate mismatches within the seed region (positions 1{"\u2013"}8) destabilize wildtype binding while preserving mutant recognition. Published improvements of 6{"\u2013"}7.5{"\u00d7"} at PAM+4 (Kohabir et al. 2024), with up to {">"}10{"\u00d7"} in favourable seed contexts (Chen et al. 2018; Teng et al. 2019).
          </div>
        </div>
      </Section>

      {/* ═══════════ 6. NUCLEASE REFERENCE ═══════════ */}
      <Section id="nuclease" title="Cas12a (Cpf1) Reference" subtitle="Class 2, Type V-A CRISPR effector for isothermal reporter-based detection">
        <p style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6, margin: "0 0 14px" }}>
          Recognises T-rich PAM upstream, generates staggered DSBs, and activates non-specific ssDNase (trans-cleavage) for isothermal reporter-based detection at 37{"\u00b0"}C.
        </p>
        <div>
          {[
            ["Variant", "WT AsCas12a / enAsCas12a (selectable)"],
            ["PAM", "TTTV (WT) + 8 expanded motifs (enAsCas12a)"],
            ["crRNA", "19 nt direct repeat + 20\u201323 nt spacer"],
            ["Trans-cleavage", "Non-specific ssDNase (reporter activation)"],
            ["Temperature", "37\u00b0C (RPA-compatible)"],
            ["Readouts", "Electrochemical (SWV on LIG) \u00b7 lateral flow \u00b7 fluorescence"],
          ].map(([k, v]) => <KV key={k} label={k} value={v} />)}
        </div>
      </Section>

      {/* ═══════════ 7. PIPELINE DEFAULTS ═══════════ */}
      <Section id="defaults" title="Pipeline Defaults" subtitle="Default parameters for guide design, filtering, and panel optimisation">
        <div>
          {[
            ["PAM", "TTTV + 8 expanded", "enAsCas12a: 9 PAM variants with activity penalties"],
            ["Spacer length", "20\u201323 nt", "20 canonical; 21\u201323 for high-GC targets"],
            ["GC range", "40\u201385%", "TB-adjusted (genome 65.6% GC)"],
            ["Max homopolymer", "4 nt", "Poly-T \u22655 causes R-loop stalling"],
            ["Off-target", "\u22643 mismatches", "Bowtie2 against full genome"],
            ["RPA amplicon", "80\u2013120 bp", "Optimised for blood cfDNA (~140 bp median)"],
            ["Discrimination min", "2.0\u00d7", "\u22653.0\u00d7 for electrochemical/LFA"],
          ].map(([param, value, rationale], i, arr) => (
            <div key={param} style={{ display: "flex", alignItems: "baseline", gap: "12px", padding: "8px 0", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
              <span style={{ fontSize: "12px", fontWeight: 600, color: T.text, minWidth: mobile ? 90 : 130, flexShrink: 0 }}>{param}</span>
              <span style={{ fontSize: "12px", fontWeight: 600, color: T.primary, minWidth: 90, flexShrink: 0, fontFamily: MONO }}>{value}</span>
              <span style={{ fontSize: "11px", color: T.textTer, flex: 1 }}>{rationale}</span>
            </div>
          ))}
        </div>
      </Section>

      {/* ═══════════ 8. LIMITATIONS ═══════════ */}
      <Section id="limitations" title="Limitations" subtitle="All predictions are in silico estimates. Experimental validation is required before diagnostic deployment.">
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "10px" }}>
          {[
            { title: "Discrimination prediction", text: "XGBoost (18 features) r=0.57 on 6,136 EasyDesign pairs. pam_to_mm_distance is top feature. Heuristic baseline r\u22480.30." },
            { title: "Training domain shift", text: "Trained on WT AsCas12a/LbCas12a, deployed on enAsCas12a. Human cell lines \u2192 M.tb (65.6% GC)." },
            { title: "AS-RPA specificity", text: "Boltzmann thermodynamic estimates, not experimentally validated. Ratios >100\u00d7 are capped." },
            { title: "Multiplex compatibility", text: "Cross-reactivity by sequence homology. Primer dimer stability predicted but not yet in SA cost function." },
            { title: "Shared amplicons", text: "Targets in same gene region may share amplicons. Cannot resolve specific amino acid changes without distinct crRNA reporters." },
            { title: "Amplicon folding", text: "No \u0394G_fold calculation. GC-rich M.tb amplicons risk stable hairpins blocking recombinase invasion." },
            { title: "Specificity estimates", text: "Proxy formula (1\u22121/disc) assumes separated distributions. Real specificity depends on signal variance." },
            { title: "Reporter independence", text: "Compass-ML predicts Cas12a trans-cleavage, not reporter chemistry. Absolute signal is platform-dependent." },
          ].map((item, i) => (
            <div key={i} style={{ padding: "14px 16px", border: `1px solid ${T.borderLight}`, borderRadius: "4px" }}>
              <div style={{ fontSize: "12px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>{item.title}</div>
              <div style={{ fontSize: "11px", color: T.textSec, lineHeight: 1.6 }}>{item.text}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* ═══════════ 9. REFERENCES ═══════════ */}
      <Section id="references" title={`References (${BIBLIOGRAPHY.length})`} subtitle="Primary literature supporting COMPASS pipeline design and validation">
        {(() => {
          const categories = [...new Set(BIBLIOGRAPHY.map(b => b.category))];
          return categories.map((cat, catIdx) => (
            <div key={cat} style={{ marginBottom: catIdx < categories.length - 1 ? "16px" : 0 }}>
              <div style={{ fontSize: "10px", fontWeight: 600, color: T.primary, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px", paddingBottom: "6px", borderBottom: `1px solid ${T.borderLight}` }}>{cat}</div>
              {BIBLIOGRAPHY.filter(b => b.category === cat).map((b, i, arr) => (
                <div key={b.id} style={{
                  padding: "8px 0",
                  borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none",
                  display: "flex", justifyContent: "space-between", alignItems: mobile ? "flex-start" : "center",
                  gap: "12px", flexDirection: mobile ? "column" : "row",
                }}>
                  <div style={{ flex: 1, fontSize: "12px", lineHeight: 1.5, color: T.textSec }}>
                    <strong style={{ color: T.text }}>{b.authors} ({b.year}).</strong>{" "}
                    {b.title}.{" "}
                    <span style={{ color: T.textTer }}>{b.journal}.</span>
                  </div>
                  <div style={{ display: "flex", gap: "6px", flexShrink: 0 }}>
                    {b.doi && <a href={`https://doi.org/${b.doi}`} target="_blank" rel="noopener noreferrer" style={{ fontSize: "10px", color: T.primary, textDecoration: "none", fontWeight: 600, padding: "2px 6px", borderRadius: "3px", background: T.primaryLight }}>DOI</a>}
                    {b.pmid && <a href={`https://pubmed.ncbi.nlm.nih.gov/${b.pmid}/`} target="_blank" rel="noopener noreferrer" style={{ fontSize: "10px", color: T.primary, textDecoration: "none", fontWeight: 600, padding: "2px 6px", borderRadius: "3px", background: T.primaryLight }}>PubMed</a>}
                    {!b.doi && !b.pmid && b.url && <a href={b.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: "10px", color: T.primary, textDecoration: "none", fontWeight: 600, padding: "2px 6px", borderRadius: "3px", background: T.primaryLight }}>Link</a>}
                  </div>
                </div>
              ))}
            </div>
          ));
        })()}
      </Section>

      <div style={{ height: "32px" }} />
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   EXECUTION THEME TOKENS (light, integrated with sidebar)
   ═══════════════════════════════════════════════════════════════════ */
/* Monotone execution palette — black, grey, white only */
const EX = {
  bg: T.bgSub,
  text: "#111111",
  textSec: "#888888",
  textTer: "#999999",
  line: "#e0e0e0",
  lineDone: "#111111",
  nodeUp: "#cccccc",
  nodeDone: "#111111",
  desc: "#666666",
};

/* ═══════════════════════════════════════════════════════════════════
   PIPELINE PAGE — Redirects to Home (execution is now inline)
   ═══════════════════════════════════════════════════════════════════ */
const PipelinePage = ({ jobId, connected, goTo }) => {
  const mobile = useIsMobile();
  useEffect(() => { goTo("home"); }, []);
  return (
    <div style={{ padding: mobile ? "16px" : "36px 40px", textAlign: "center" }}>
      <div style={{ padding: "80px 24px" }}>
        <Cpu size={28} color="#999" strokeWidth={1.5} />
        <div style={{ fontSize: "14px", color: "#999", marginTop: "12px" }}>Redirecting to Home…</div>
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   READINESS SCORING COMPONENTS
   ═══════════════════════════════════════════════════════════════════ */
const RISK_COLORS = { green: T.riskGreen, amber: T.riskAmber, red: T.riskRed };
const RISK_BG = { green: T.riskGreenBg, amber: T.riskAmberBg, red: T.riskRedBg };
/* GreenBlue colormap — single-cell omics UMAP aesthetic.
   Stops: light gray → pale green → teal → blue → deep blue. */
const gradientColor = (t) => {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    { t: 0.00, r: 180, g: 185, b: 195 },  // #b4b9c3 — soft gray (low)
    { t: 0.25, r: 171, g: 221, b: 164 },  // #abdda4 — pale green
    { t: 0.50, r: 102, g: 194, b: 165 },  // #66c2a5 — teal
    { t: 0.75, r: 50,  g: 136, b: 189 },  // #3288bd — blue
    { t: 1.00, r: 30,  g: 58,  b: 95 },   // #1E3A5F — deep navy (high)
  ];
  let lower = stops[0], upper = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].t && t <= stops[i + 1].t) { lower = stops[i]; upper = stops[i + 1]; break; }
  }
  const range = upper.t - lower.t || 1;
  const f = (t - lower.t) / range;
  const r = Math.round(lower.r + f * (upper.r - lower.r));
  const g = Math.round(lower.g + f * (upper.g - lower.g));
  const b = Math.round(lower.b + f * (upper.b - lower.b));
  return `rgb(${r},${g},${b})`;
};
const gradientCSS = "linear-gradient(90deg, #E5E7EB, #059669, #0891B2, #4338CA)";
const CHART_TEXT = "#111827";
const CHART_TEXT_SEC = "#6B7280";
const CHART_GRID = "#E5E7EB";
const PASS_GREEN = "#059669";
const AXIS_COLORS = { efficiency: "#059669", discrimination: "#2563EB", primers: "#0891B2", safety: "#D97706", gc: "#9CA3AF" };
const AXIS_LABELS = { efficiency: "Activity", discrimination: "Discrimination", primers: "Primers", safety: "Off-target", gc: "GC" };

const RiskDot = ({ level, size = 12 }) => (
  <span style={{
    display: "inline-block", width: size, height: size, borderRadius: "50%",
    backgroundColor: RISK_COLORS[level] || "#6B7280", flexShrink: 0,
  }} />
);

/* Heatmap cell for Risk Assessment Matrix — 3 discrete pastel colors */
const RISK_CELL_COLORS = { green: "#B8E6C8", amber: "#FDDCB0", red: "#F5A3A3" };
const RISK_CELL_PASS = "#a7d8b8";
const RiskHeatCell = ({ level, type = "quantitative" }) => {
  const bg = type === "binary"
    ? (level === "green" ? RISK_CELL_PASS : RISK_CELL_COLORS.red)
    : (RISK_CELL_COLORS[level] || RISK_CELL_COLORS.amber);
  return (
    <div style={{
      width: 44, height: 28, borderRadius: 4, border: "2px solid #fff",
      backgroundColor: bg, margin: "0 auto",
    }} />
  );
};

const PriorityBadge = ({ rank }) => {
  const isTop3 = rank <= 3;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", justifyContent: "center",
      width: isTop3 ? 28 : 24, height: isTop3 ? 28 : 24, borderRadius: "50%",
      fontSize: isTop3 ? "13px" : "11px", fontWeight: isTop3 ? 700 : 400,
      fontFamily: FONT,
      background: isTop3 ? T.text : "transparent",
      color: isTop3 ? "#fff" : T.textSec,
      border: isTop3 ? `2px solid ${T.text}` : `1px solid ${T.border}`,
    }}>
      {rank}
    </span>
  );
};

const ExperimentalPriorityCard = ({ results }) => {
  const top3 = [...results].filter(r => r.experimentalPriority != null)
    .sort((a, b) => a.experimentalPriority - b.experimentalPriority).slice(0, 3);
  const gaps = results.filter(r => r.riskProfile?.discrimination === "red");
  if (top3.length === 0) return null;
  return (
    <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "24px 28px", marginBottom: "24px" }}>
      <div style={{ fontSize: "11px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "16px" }}>Experimental Priorities</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
        {top3.map((r) => {
          const disc = r.strategy === "Direct" ? r.disc : (r.asrpaDiscrimination?.disc_ratio || 0);
          const eff = r.cnnCalibrated ?? r.score;
          return (
            <div key={r.label} style={{ display: "flex", gap: "14px", alignItems: "flex-start" }}>
              <PriorityBadge rank={r.experimentalPriority} />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                  <span style={{ fontSize: "14px", fontWeight: 600, color: T.text }}>{r.label}</span>
                  <span style={{ fontSize: "11px", color: T.textSec, fontFamily: FONT }}>
                    {disc > 0 && disc < 900 ? `${disc.toFixed(1)}x disc` : ""}
                    {disc > 0 && disc < 900 ? " \u00b7 " : ""}
                    {eff.toFixed(3)} eff
                    {r.drug ? ` \u00b7 ${r.drug}` : ""}
                  </span>
                  <RiskDot level={r.riskProfile?.overall || "amber"} size={10} />
                </div>
                <div style={{ fontSize: "11px", color: T.textSec, marginTop: "3px", lineHeight: 1.5 }}>
                  {r.priorityReason || ""}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      {gaps.length > 0 && (
        <div style={{ marginTop: "14px", padding: "10px 14px", background: T.riskRedBg, border: `1px solid ${T.riskRed}33`, borderRadius: "4px", fontSize: "11px", color: T.danger, lineHeight: 1.6 }}>
          <strong>Panel gap:</strong> {gaps.map(r => r.label).join(", ")} — no viable discrimination pathway. Requires alternative strategy or SM enhancement.
        </div>
      )}
    </div>
  );
};

const RISK_AXIS_TYPE = { activity: "quantitative", discrimination: "quantitative", primers: "binary", gc_risk: "quantitative", off_target: "quantitative" };
const RiskMatrix = ({ results }) => {
  const mobile = useIsMobile();
  const sorted = [...results].filter(r => r.experimentalPriority != null)
    .sort((a, b) => a.experimentalPriority - b.experimentalPriority);
  if (sorted.length === 0) return null;
  const axes = ["activity", "discrimination", "primers", "gc_risk", "off_target"];
  const axisNames = { activity: "Activity", discrimination: "Disc", primers: "Primers", gc_risk: "GC", off_target: "Off-T" };
  return (
    <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden", marginBottom: "24px" }}>
      <div style={{ padding: "16px 24px", borderBottom: `1px solid ${T.border}`, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Risk Assessment Matrix</div>
        <div style={{ fontSize: "12px", color: T.textTer }}>{sorted.length} targets</div>
      </div>
      <div style={{ overflowX: "auto", padding: "8px 0" }}>
        <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: "2px 2px", fontSize: "12px" }}>
          <thead>
            <tr>
              <th style={{ padding: "8px 16px", textAlign: "left", fontWeight: 600, color: T.textTer, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.04em", width: mobile ? 100 : 140 }}>Target</th>
              {axes.map(a => (
                <th key={a} style={{ padding: "8px 6px", textAlign: "center", fontWeight: 600, color: T.textTer, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.04em", width: 52 }}>{axisNames[a]}</th>
              ))}
              <th style={{ padding: "8px 6px", textAlign: "center", fontWeight: 600, color: T.textTer, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.04em", width: 52 }}>Overall</th>
              <th style={{ padding: "8px 14px", textAlign: "center", fontWeight: 600, color: T.textTer, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.04em", width: 50 }}>#</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(r => {
              const risk = r.riskProfile || {};
              return (
                <tr key={r.label}>
                  <td style={{ padding: "4px 16px", fontWeight: 600, fontSize: "11px", color: T.text, whiteSpace: "nowrap" }}>{r.label}</td>
                  {axes.map(a => (
                    <td key={a} style={{ padding: "3px 2px", textAlign: "center" }}><RiskHeatCell level={risk[a]} type={RISK_AXIS_TYPE[a]} /></td>
                  ))}
                  <td style={{ padding: "3px 2px", textAlign: "center" }}><RiskHeatCell level={risk.overall} /></td>
                  <td style={{ padding: "4px 14px", textAlign: "center" }}><PriorityBadge rank={r.experimentalPriority} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {/* Discrete legend */}
      <div style={{ display: "flex", alignItems: "center", gap: "16px", padding: "10px 24px 14px", flexWrap: "wrap" }}>
        {[
          { color: RISK_CELL_COLORS.green, label: "Safe" },
          { color: RISK_CELL_COLORS.amber, label: "Moderate" },
          { color: RISK_CELL_COLORS.red, label: "Risk" },
          { color: RISK_CELL_PASS, label: "Pass (binary)" },
        ].map(s => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: 16, height: 12, borderRadius: 3, backgroundColor: s.color }} />
            <span style={{ fontSize: "10px", color: T.textTer }}>{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const ReadinessChart = ({ results }) => {
  const hasReadiness = results.some(r => r.readinessScore != null);
  if (!hasReadiness) return null;
  const sorted = [...results].filter(r => r.readinessScore != null)
    .sort((a, b) => b.readinessScore - a.readinessScore);
  const axes = ["efficiency", "discrimination", "primers", "safety", "gc"];
  const chartData = sorted.map(r => {
    const c = r.readinessComponents || {};
    return { name: r.label, drug: r.drug, readiness: r.readinessScore, ...Object.fromEntries(axes.map(a => [a, +(c[a] || 0).toFixed(3)])) };
  });
  const [hovIdx, setHovIdx] = useState(null);

  // Pastel palette — matches UMAP embedding aesthetic
  const DRUG_LINE = { RIF: "#5B8BD4", INH: "#9B8EC4", EMB: "#66C2A5", PZA: "#8DA0CB", FQ: "#E78AC3", AG: "#A6D854", OTHER: "#B3B3B3", CTRL: "#B3B3B3" };

  // Full-width responsive SVG — use viewBox for scaling
  const W = 900, H = 290, padL = 100, padR = 80, padT = 20, padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const axisX = axes.map((_, i) => padL + (i / (axes.length - 1)) * plotW);

  return (
    <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "24px 24px 20px", marginBottom: "24px" }}>
      <div style={{ marginBottom: "12px", display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "8px" }}>
        <div>
          <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Diagnostic Readiness Score</div>
          <div style={{ fontSize: "11px", color: T.textSec, marginTop: "3px" }}>Each line is one candidate across 5 readiness axes. Strong candidates stay high. Colored by drug class.</div>
        </div>
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
          {Object.entries(DRUG_LINE).filter(([k]) => chartData.some(r => r.drug === k)).map(([d, c]) => (
            <div key={d} style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "10px", color: T.textSec }}>
              <span style={{ display: "inline-block", width: 14, height: 3, borderRadius: 2, backgroundColor: c }} />
              {d}
            </div>
          ))}
        </div>
      </div>

      {/* Full-width SVG — viewBox scales to container */}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }} preserveAspectRatio="xMidYMid meet">
        {/* Soft background bands for readability */}
        {[0, 1, 2, 3].map(i => (
          <rect key={i} x={padL} y={padT + i * (plotH / 4)} width={plotW} height={plotH / 4}
            fill={i % 2 === 0 ? "rgba(0,0,0,0.015)" : "transparent"} />
        ))}

        {/* Axis lines */}
        {axes.map((a, i) => (
          <g key={a}>
            <line x1={axisX[i]} y1={padT} x2={axisX[i]} y2={padT + plotH} stroke="#E5E7EB" strokeWidth={1} />
            {/* Axis label */}
            <text x={axisX[i]} y={padT + plotH + 24} textAnchor="middle" fontSize={11} fontWeight={600} fill="#6B7280" fontFamily={FONT}>{AXIS_LABELS[a]}</text>
            {/* Tick marks at 0%, 25%, 50%, 75%, 100% */}
            {[0, 0.25, 0.5, 0.75, 1].map(v => {
              const ty = padT + plotH * (1 - v);
              return (
                <g key={v}>
                  <line x1={axisX[i] - 3} y1={ty} x2={axisX[i] + 3} y2={ty} stroke="#D1D5DB" strokeWidth={0.5} />
                  {i === 0 && <text x={axisX[0] - 8} y={ty + 3} textAnchor="end" fontSize={9} fill="#9CA3AF">{Math.round(v * 100)}</text>}
                </g>
              );
            })}
          </g>
        ))}

        {/* 50% reference line */}
        <line x1={axisX[0]} y1={padT + plotH / 2} x2={axisX[axes.length - 1]} y2={padT + plotH / 2} stroke="#E5E7EB" strokeWidth={0.7} strokeDasharray="6,4" />

        {/* Candidate polylines — non-hovered first (dimmed), then hovered on top */}
        {chartData.map((row, ri) => {
          if (hovIdx != null && hovIdx === ri) return null; // draw hovered last
          const lineColor = DRUG_LINE[row.drug] || "#6B7280";
          const points = axes.map((a, i) => `${axisX[i]},${padT + plotH * (1 - row[a])}`).join(" ");
          return (
            <polyline key={row.name} points={points} fill="none"
              stroke={lineColor} strokeWidth={1.5}
              strokeOpacity={hovIdx != null ? 0.08 : 0.5}
              strokeLinejoin="round" strokeLinecap="round"
              style={{ transition: "stroke-opacity 0.25s", cursor: "pointer" }}
              onMouseEnter={() => setHovIdx(ri)} onMouseLeave={() => setHovIdx(null)}
            />
          );
        })}

        {/* Hovered line (on top, bold) */}
        {hovIdx != null && (() => {
          const row = chartData[hovIdx];
          const lineColor = DRUG_LINE[row.drug] || "#6B7280";
          const points = axes.map((a, i) => `${axisX[i]},${padT + plotH * (1 - row[a])}`).join(" ");
          return (
            <polyline points={points} fill="none" stroke={lineColor} strokeWidth={3}
              strokeOpacity={1} strokeLinejoin="round" strokeLinecap="round" />
          );
        })()}

        {/* Dots at axis intersections for hovered line */}
        {hovIdx != null && axes.map((a, i) => {
          const row = chartData[hovIdx];
          const y = padT + plotH * (1 - row[a]);
          const lineColor = DRUG_LINE[row.drug] || "#6B7280";
          return (
            <g key={a}>
              <circle cx={axisX[i]} cy={y} r={5} fill={lineColor} stroke="#fff" strokeWidth={2} />
              <text x={axisX[i] + (i === axes.length - 1 ? -10 : 10)} y={y - 8} textAnchor={i === axes.length - 1 ? "end" : "start"} fontSize={10} fontWeight={600} fill="#374151" fontFamily={FONT}>{(row[a] * 100).toFixed(0)}%</text>
            </g>
          );
        })}

        {/* Hovered candidate name + readiness score */}
        {hovIdx != null && (() => {
          const row = chartData[hovIdx];
          const lineColor = DRUG_LINE[row.drug] || "#6B7280";
          const readinessColor = row.readiness >= 0.7 ? "#66C2A5" : row.readiness >= 0.4 ? "#FFB347" : "#F4A1A1";
          const firstY = padT + plotH * (1 - row[axes[0]]);
          return (
            <g>
              {/* Left: target name */}
              <rect x={2} y={firstY - 10} width={padL - 12} height={18} rx={3} fill="#fff" stroke="#E5E7EB" strokeWidth={0.5} />
              <text x={padL - 14} y={firstY + 3} textAnchor="end" fontSize={10} fontWeight={600} fill={lineColor} fontFamily={FONT}>{row.name}</text>
              {/* Right: readiness score */}
              <rect x={axisX[axes.length - 1] + 8} y={padT + plotH * (1 - row[axes[axes.length - 1]]) - 10} width={55} height={18} rx={3} fill={readinessColor} opacity={0.15} />
              <text x={axisX[axes.length - 1] + 36} y={padT + plotH * (1 - row[axes[axes.length - 1]]) + 4} textAnchor="middle" fontSize={12} fontWeight={600} fill={readinessColor} fontFamily={FONT}>{(row.readiness * 100).toFixed(0)}</text>
            </g>
          );
        })()}

        {/* Right-side readiness scores (always visible when not hovering) */}
        <text x={axisX[axes.length - 1] + 12} y={padT - 8} textAnchor="start" fontSize={9} fontWeight={600} fill="#9CA3AF">Readiness</text>
        {hovIdx == null && chartData.map((row, ri) => {
          const y = padT + plotH * (1 - row[axes[axes.length - 1]]);
          const readinessColor = row.readiness >= 0.7 ? "#66C2A5" : row.readiness >= 0.4 ? "#FFB347" : "#F4A1A1";
          return (
            <text key={ri} x={axisX[axes.length - 1] + 14} y={y + 3} fontSize={9} fontWeight={600} fill={readinessColor} fontFamily={FONT}
              style={{ cursor: "pointer" }} onMouseEnter={() => setHovIdx(ri)} onMouseLeave={() => setHovIdx(null)}>
              {(row.readiness * 100).toFixed(0)}
            </text>
          );
        })}

        {/* Left-side candidate names (always visible when not hovering) */}
        {hovIdx == null && chartData.map((row, ri) => {
          const y = padT + plotH * (1 - row[axes[0]]);
          const lineColor = DRUG_LINE[row.drug] || "#6B7280";
          return (
            <text key={ri} x={padL - 8} y={y + 3} textAnchor="end" fontSize={8} fill={lineColor} fontFamily={FONT} opacity={0.7}
              style={{ cursor: "pointer" }} onMouseEnter={() => setHovIdx(ri)} onMouseLeave={() => setHovIdx(null)}>
              {row.name.length > 14 ? row.name.slice(0, 12) + "…" : row.name}
            </text>
          );
        })}
      </svg>

      {/* Panel-wide gap detection */}
      {(() => {
        const axisAvgs = axes.map(a => ({ axis: a, avg: chartData.reduce((s, r) => s + r[a], 0) / chartData.length }));
        const weakest = axisAvgs.reduce((min, cur) => cur.avg < min.avg ? cur : min);
        if (weakest.avg < 0.4) return (
          <div style={{ marginTop: "8px", padding: "8px 14px", background: `${T.warning}10`, border: `1px solid ${T.warning}30`, borderRadius: "4px", fontSize: "11px", color: T.textSec, lineHeight: 1.5 }}>
            Panel-wide gap: <strong style={{ color: T.text }}>{AXIS_LABELS[weakest.axis]}</strong> axis averages {(weakest.avg * 100).toFixed(0)}% — consider strengthening candidates on this dimension.
          </div>
        );
        return null;
      })()}
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   UMAP EMBEDDING PANEL
   ═══════════════════════════════════════════════════════════════════ */
const DRUG_CANVAS = { RIF: "#4338CA", INH: "#D97706", EMB: "#059669", FQ: "#DC2626", AG: "#7C3AED", PZA: "#0891B2", OTHER: "#6B7280" };

const UMAPPanel = ({ jobId }) => {
  const [umapData, setUmapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [colorBy, setColorBy] = useState("score");
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const canvasRef = useRef(null);
  const scaleRef = useRef(null);
  const mobile = useIsMobile();

  useEffect(() => {
    if (!jobId) return;
    getUmapData(jobId)
      .then(({ data }) => { if (data) setUmapData(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [jobId]);

  // Canvas drawing
  useEffect(() => {
    if (!umapData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    const displayW = canvas.clientWidth;
    const displayH = mobile ? 300 : 480;
    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;
    canvas.style.height = `${displayH}px`;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const pad = 36;

    // Clear — light background for UMAP
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, displayW, displayH);

    const points = umapData.points;
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const scX = (x) => pad + ((x - xMin) / xRange) * (displayW - 2 * pad);
    const scY = (y) => pad + ((y - yMin) / yRange) * (displayH - 2 * pad);
    scaleRef.current = { scX, scY, points };

    const getColor = (p) => {
      if (colorBy === "drug") return DRUG_CANVAS[p.drug] || DRUG_CANVAS.OTHER;
      if (colorBy === "gene") return GENE_COLORS[getGene(p.target_label)] || "#999";
      if (colorBy === "score") return gradientColor(p.score != null ? p.score : 0.5);
      if (colorBy === "gc") {
        const gc = p.gc_content != null ? p.gc_content : 0.5;
        const gcNorm = 1 - Math.min(Math.abs(gc - 0.5) / 0.25, 1);
        return gradientColor(gcNorm);
      }
      if (colorBy === "strategy") return p.detection_strategy === "direct" ? "#0891B2" : "#4338CA";
      return gradientColor(0.5);
    };

    // Dense dot cloud for unselected points (like CITEseq UMAP style)
    const unselected = points.filter(p => !p.selected);
    const dotR = unselected.length > 10000 ? 1.8 : unselected.length > 2000 ? 2.2 : 2.8;
    ctx.globalAlpha = 0.4;
    for (const p of unselected) {
      ctx.beginPath();
      ctx.arc(scX(p.x), scY(p.y), dotR, 0, Math.PI * 2);
      ctx.fillStyle = getColor(p);
      ctx.fill();
    }

    // Selected: large, opaque, bordered (no text labels)
    const selected = points.filter(p => p.selected);
    ctx.globalAlpha = 1.0;
    for (const p of selected) {
      const cx = scX(p.x), cy = scY(p.y);
      // Halo
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(79,70,229,0.12)";
      ctx.fill();
      // Dot
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = getColor(p);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.strokeStyle = "rgba(0,0,0,0.15)";
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }, [umapData, colorBy, mobile]);

  // Hover detection (selected points only for speed)
  const handleMouseMove = useCallback((e) => {
    if (!scaleRef.current || !umapData) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    const { scX, scY, points } = scaleRef.current;
    const selected = points.filter(p => p.selected);
    let found = null;
    for (const p of selected) {
      const dx = scX(p.x) - mx, dy = scY(p.y) - my;
      if (dx * dx + dy * dy < 144) { found = p; break; }
    }
    setHoveredPoint(found);
  }, [umapData]);

  if (loading) return null;
  if (!umapData) return null;

  const GENE_COLORS = { rpoB: "#e6194b", katG: "#3cb44b", fabG1: "#4363d8", embB: "#f58231", pncA: "#911eb4", gyrA: "#42d4f4", rrs: "#f032e6", eis: "#aaffc3", IS6110: "#bfef45" };
  const getGene = (label) => { if (!label) return "other"; const g = label.replace(/_.*/, ""); return GENE_COLORS[g] ? g : "other"; };
  const colorOpts = [
    { key: "drug", label: "Drug class" },
    { key: "gene", label: "Gene" },
    { key: "score", label: "Activity" },
    { key: "gc", label: "GC%" },
    { key: "strategy", label: "Strategy" },
  ];

  return (
    <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: mobile ? "20px" : "28px 32px", marginBottom: "24px", position: "relative" }}>
      <div style={{ marginBottom: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Candidate Embedding Space</div>
          <div style={{ fontSize: "11px", color: T.textSec, marginTop: "3px", lineHeight: 1.5 }}>
            UMAP of {umapData.n_total.toLocaleString()} scored candidates (Compass-ML 128-dim RLPA embeddings).
            {" "}{umapData.n_selected} panel members highlighted. Proximity in this space reflects learned sequence similarity.
          </div>
        </div>
        <div style={{ display: "flex", gap: "6px", alignItems: "center", flexShrink: 0 }}>
          <span style={{ fontSize: "10px", color: T.textTer, fontWeight: 600 }}>COLOR:</span>
          {colorOpts.map(o => (
            <button key={o.key} onClick={() => setColorBy(o.key)} style={{
              padding: "3px 10px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, fontFamily: FONT, cursor: "pointer",
              border: `1px solid ${colorBy === o.key ? T.primary + "88" : T.border}`,
              background: colorBy === o.key ? T.primaryLight : "transparent",
              color: colorBy === o.key ? T.primary : T.textSec,
            }}>{o.label}</button>
          ))}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        style={{ width: "100%", borderRadius: "4px", border: `1px solid ${T.borderLight}`, cursor: "crosshair" }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredPoint(null)}
      />

      {/* Tooltip */}
      {hoveredPoint && (
        <div style={{
          position: "absolute", left: mousePos.x + 16, top: mousePos.y + 60,
          background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px",
          padding: "10px 14px", fontSize: "11px", pointerEvents: "none", zIndex: 10,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)", minWidth: "150px",
        }}>
          <div style={{ fontWeight: 600, fontSize: "12px", color: DRUG_CANVAS[hoveredPoint.drug] || T.text, marginBottom: "4px" }}>{hoveredPoint.target_label}</div>
          {hoveredPoint.score != null && <div style={{ color: T.textSec }}>Score: <strong style={{ color: T.text }}>{hoveredPoint.score.toFixed(3)}</strong></div>}
          {hoveredPoint.gc_content != null && <div style={{ color: T.textSec }}>GC: <strong style={{ color: T.text }}>{(hoveredPoint.gc_content * 100).toFixed(1)}%</strong></div>}
          {hoveredPoint.drug && <div style={{ color: T.textSec }}>Drug: <strong style={{ color: T.text }}>{hoveredPoint.drug}</strong></div>}
        </div>
      )}

      {/* Stats bar */}
      <div style={{ display: "flex", gap: "20px", marginTop: "10px", fontSize: "11px", color: T.textTer, flexWrap: "wrap" }}>
        <span><strong style={{ color: T.text, fontFamily: FONT }}>{umapData.n_total.toLocaleString()}</strong> candidates scored</span>
        <span><strong style={{ color: T.primary, fontFamily: FONT }}>{umapData.n_selected}</strong> selected ({((umapData.n_selected / umapData.n_total) * 100).toFixed(2)}%)</span>
        {umapData.stats?.panel_spread > 0 && <span>Panel spread: <strong style={{ fontFamily: FONT }}>{umapData.stats.panel_spread.toFixed(2)}</strong></span>}
        {umapData.stats?.coverage != null && <span>Space coverage: <strong style={{ fontFamily: FONT }}>{(umapData.stats.coverage * 100).toFixed(1)}%</strong></span>}
        <span style={{ fontSize: "10px", color: T.textTer }}>{umapData.stats?.method || "UMAP"} · cosine · PCA→30d</span>
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: "12px", marginTop: "8px", flexWrap: "wrap", alignItems: "center" }}>
        {colorBy === "drug" ? (
          Object.entries(DRUG_CANVAS).filter(([d]) => d !== "OTHER" && umapData.points.some(p => p.drug === d)).map(([drug, color]) => (
            <div key={drug} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
              <span style={{ fontSize: "10px", color: T.textSec, fontWeight: 500 }}>{drug}</span>
            </div>
          ))
        ) : colorBy === "gene" ? (
          Object.entries(GENE_COLORS).filter(([g]) => umapData.points.some(p => getGene(p.target_label) === g)).map(([gene, color]) => (
            <div key={gene} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
              <span style={{ fontSize: "10px", color: T.textSec, fontWeight: 500, fontFamily: FONT }}>{gene}</span>
            </div>
          ))
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "10px", color: T.textTer }}>Low</span>
            <div style={{ width: 120, height: 8, borderRadius: 4, background: gradientCSS }} />
            <span style={{ fontSize: "10px", color: T.textTer }}>High</span>
          </div>
        )}
        <span style={{ fontSize: "10px", color: T.textTer }}>|</span>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: T.textTer, border: "1.5px solid #fff" }} />
          <span style={{ fontSize: "10px", color: T.textTer }}>Selected (panel)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <div style={{ width: 4, height: 4, borderRadius: "50%", background: T.textTer, opacity: 0.3 }} />
          <span style={{ fontSize: "10px", color: T.textTer }}>Screened</span>
        </div>
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   RESULT TABS
   ═══════════════════════════════════════════════════════════════════ */
const OverviewTab = ({ results, scorer, jobId }) => {
  const mobile = useIsMobile();

  // Detect scorer from prop (primary) or ml_scores (fallback)
  const usesCompassMl = scorer === "compass_ml" || results.some(r => r.mlScores?.some(m => (m.model_name || m.modelName) === "compass_ml"));
  const mlModelLabel = usesCompassMl ? "Compass-ML" : "Heuristic";
  const mlModelDetail = usesCompassMl ? "235K params · CNN + PAM + RNA-FM + RLPA" : "Biophysical features";

  const getResultScore = (r) => r.cnnCalibrated ?? r.score;
  const drugs = [...new Set(results.map((r) => r.drug))];
  const byDrug = drugs.map((d) => ({ drug: d, count: results.filter((r) => r.drug === d).length, avgScore: +(results.filter((r) => r.drug === d).reduce((a, r) => a + getResultScore(r), 0) / results.filter((r) => r.drug === d).length).toFixed(3) }));
  const withPrimers = results.filter((r) => r.hasPrimers).length;
  const directResults = results.filter((r) => r.strategy === "Direct" && r.disc < 900);
  const avgDisc = directResults.length ? +(directResults.reduce((a, r) => a + r.disc, 0) / directResults.length).toFixed(1) : 0;
  const highDisc = directResults.filter((r) => r.disc >= 3).length;
  const directCount = results.filter((r) => r.strategy === "Direct").length;
  const proximityCount = results.filter((r) => r.strategy === "Proximity").length;
  const avgActivity = results.length ? +(results.reduce((a, r) => a + getResultScore(r), 0) / results.length).toFixed(3) : 0;
  const minScore = results.length ? Math.min(...results.map(r => getResultScore(r))).toFixed(3) : "0";
  const maxScore = results.length ? Math.max(...results.map(r => getResultScore(r))).toFixed(3) : "0";
  const cnnResults = results.filter(r => r.cnnCalibrated != null);
  const avgCNN = cnnResults.length ? +(cnnResults.reduce((a, r) => a + r.cnnCalibrated, 0) / cnnResults.length).toFixed(3) : null;
  const pamAdjResults = results.filter(r => r.pamAdjusted != null);
  const avgPamAdj = pamAdjResults.length ? +(pamAdjResults.reduce((a, r) => a + r.pamAdjusted, 0) / pamAdjResults.length).toFixed(3) : null;
  const avgScore = avgActivity; // alias for scatter plot

  // Model agreement — Spearman ρ between heuristic and Compass-ML (PAM-adjusted)
  const modelAgreement = (() => {
    const pairs = results.filter(r => r.cnnCalibrated != null).map(r => ({ h: r.score, g: r.cnnCalibrated * (r.pamPenalty ?? 1.0) }));
    if (pairs.length < 3) return null;
    const rankArr = (arr) => {
      const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
      const ranks = new Array(arr.length);
      sorted.forEach((s, rank) => { ranks[s.i] = rank + 1; });
      return ranks;
    };
    const hRanks = rankArr(pairs.map(p => p.h));
    const gRanks = rankArr(pairs.map(p => p.g));
    const n = pairs.length;
    const dSq = hRanks.reduce((sum, hr, i) => sum + (hr - gRanks[i]) ** 2, 0);
    return +(1 - (6 * dSq) / (n * (n * n - 1))).toFixed(2);
  })();

  // Verdict-first computed values
  const assayReady = results.filter(r => r.readinessScore != null && r.readinessScore >= 0.4 && r.hasPrimers).length;
  const totalTargets = results.length;
  const sensitivity = totalTargets ? Math.round(withPrimers / totalTargets * 100) : 0;
  const missingPrimers = results.filter(r => !r.hasPrimers);
  const belowThreshold = results.filter(r => r.readinessScore != null && r.readinessScore < 0.4);
  const discModel = directResults.some(r => r.discMethod === "neural") ? "Compass-ML disc head" : directResults.some(r => (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature") ? "XGBoost · 18 features" : "position × destab";

  return (
    <div>
      {/* ── Verdict-first panel ── */}
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: mobile ? "20px 16px" : "24px 28px", marginBottom: "24px" }}>
        {/* Headline verdict */}
        <div style={{ marginBottom: "20px" }}>
          <div style={{ fontSize: "18px", fontWeight: 600, color: T.text, fontFamily: HEADING, lineHeight: 1.3 }}>
            {assayReady} of {totalTargets} targets are assay-ready
          </div>
          <div style={{ fontSize: "13px", color: T.textSec, marginTop: "4px" }}>
            {drugs.length} drug classes · {directCount} direct / {proximityCount} proximity · {sensitivity}% primer coverage
          </div>
        </div>

        {/* Three evidence columns */}
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr 1fr", gap: mobile ? "16px" : "12px" }}>
          {/* Column 1: Can we detect resistance? */}
          <div style={{ background: T.bgSub, borderRadius: "6px", padding: "16px 20px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "14px" }}>
              <Shield size={11} color={T.textTer} strokeWidth={2} />
              Can we detect resistance?
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                  <Package size={12} color={T.textTer} strokeWidth={1.8} />
                  Drug classes
                </span>
                <span style={{ fontSize: "20px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{drugs.length}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                  <Crosshair size={12} color={T.textTer} strokeWidth={1.8} />
                  Primer coverage
                </span>
                <span style={{ fontSize: "20px", fontWeight: 600, color: sensitivity === 100 ? T.success : T.warning, fontFamily: FONT }}>{sensitivity}%</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                  <GitBranch size={12} color={T.textTer} strokeWidth={1.8} />
                  Detection split
                </span>
                <span style={{ fontSize: "13px", fontWeight: 600, color: T.textSec, fontFamily: FONT }}>{directCount} direct · {proximityCount} prox</span>
              </div>
            </div>
          </div>

          {/* Column 2: How confident are we? */}
          <div style={{ background: T.bgSub, borderRadius: "6px", padding: "16px 20px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "14px" }}>
              <Activity size={11} color={T.textTer} strokeWidth={2} />
              How confident are we?
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                  <TrendingUp size={12} color={T.textTer} strokeWidth={1.8} />
                  Avg discrimination
                </span>
                <span style={{ fontSize: "20px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{avgDisc}×</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                  <Filter size={12} color={T.textTer} strokeWidth={1.8} />
                  Diagnostic-grade (≥3×)
                </span>
                <span style={{ fontSize: "20px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{highDisc}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                  <Zap size={12} color={T.textTer} strokeWidth={1.8} />
                  Avg. activity{usesCompassMl ? " (Compass-ML)" : ""}
                </span>
                <span style={{ fontSize: "20px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{avgActivity}</span>
              </div>
              {avgPamAdj != null && avgPamAdj !== avgActivity && (
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                    <BarChart3 size={12} color={T.textTer} strokeWidth={1.8} />
                    Avg. PAM-adjusted
                  </span>
                  <span style={{ fontSize: "17px", fontWeight: 600, color: T.textSec, fontFamily: FONT }}>{avgPamAdj}</span>
                </div>
              )}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                  <BarChart3 size={12} color={T.textTer} strokeWidth={1.8} />
                  Activity range
                </span>
                <span style={{ fontSize: "13px", fontWeight: 600, color: T.textSec, fontFamily: FONT }}>{minScore} – {maxScore}</span>
              </div>
              {usesCompassMl && modelAgreement != null && (
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                    <Brain size={12} color={T.textTer} strokeWidth={1.8} />
                    Model ρ (heuristic vs net)
                  </span>
                  <span style={{ fontSize: "13px", fontWeight: 600, color: T.textSec, fontFamily: FONT }}>{modelAgreement}</span>
                </div>
              )}
              <div style={{ fontSize: "10px", color: T.textTer, marginTop: "2px", paddingLeft: "16px" }}>
                Disc model: {discModel}
              </div>
            </div>
          </div>

          {/* Column 3: What's missing? */}
          <div style={{ background: T.bgSub, borderRadius: "6px", padding: "16px 20px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "14px" }}>
              <AlertTriangle size={11} color={T.textTer} strokeWidth={2} />
              What's missing?
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {missingPrimers.length > 0 ? (
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.warning, marginBottom: "4px" }}>
                    <AlertTriangle size={12} color={T.warning} strokeWidth={1.8} />
                    {missingPrimers.length} target{missingPrimers.length > 1 ? "s" : ""} need primers
                  </div>
                  <div style={{ fontSize: "11px", color: T.textTer, fontFamily: MONO, paddingLeft: "16px" }}>{missingPrimers.map(r => r.label).join(", ")}</div>
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.success }}>
                  <Check size={12} color={T.success} strokeWidth={2.5} />
                  All targets have primers
                </div>
              )}
              {belowThreshold.length > 0 ? (
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.warning, marginBottom: "4px" }}>
                    <AlertTriangle size={12} color={T.warning} strokeWidth={1.8} />
                    {belowThreshold.length} below readiness threshold
                  </div>
                  <div style={{ fontSize: "11px", color: T.textTer, fontFamily: MONO, paddingLeft: "16px" }}>{belowThreshold.map(r => r.label).join(", ")}</div>
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.success }}>
                  <Check size={12} color={T.success} strokeWidth={2.5} />
                  All targets above readiness threshold
                </div>
              )}
              {(() => {
                const lowDisc = directResults.filter(r => r.disc < 3 && r.disc > 0);
                if (lowDisc.length === 0) return null;
                return (
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.warning, marginBottom: "4px" }}>
                      <TrendingUp size={12} color={T.warning} strokeWidth={1.8} />
                      {lowDisc.length} below 3× discrimination
                    </div>
                    <div style={{ fontSize: "11px", color: T.textTer, fontFamily: MONO, paddingLeft: "16px" }}>{lowDisc.map(r => r.label).join(", ")}</div>
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      </div>

      {/* ── Interpretation box ── */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}25`, borderRadius: "4px", padding: mobile ? "16px" : "18px 24px", marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
          <Brain size={16} color={T.primary} strokeWidth={1.8} />
          <span style={{ fontSize: "13px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING }}>Panel Interpretation</span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", color: T.primaryDark, lineHeight: 1.6 }}>
          <div><strong>{assayReady}/{totalTargets} targets are assay-ready</strong> across {drugs.length} drug classes ({drugs.join(", ")}). {directCount} use direct crRNA discrimination, {proximityCount} rely on AS-RPA primer selectivity for allelic specificity.</div>
          <div>
            {avgActivity >= 0.8
              ? `Mean predicted activity of ${avgActivity} indicates strong Cas12a cis-cleavage efficiency, predicting robust trans-cleavage activation and detectable SWV peak reduction on LIG electrodes. Activity range ${minScore}\u2013${maxScore} suggests consistent signal across the panel.`
              : avgActivity >= 0.6
              ? `Mean predicted activity of ${avgActivity} is moderate. On the electrochemical platform, weaker candidates may require extended incubation (>30 min) or increased Cas12a concentration to reach the 3\u03c3 detection threshold on MB-ssDNA reporters.`
              : `Mean predicted activity of ${avgActivity} reflects the Compass-ML efficiency prediction (trained on Kim 2018, human cell cis-cleavage). Scores below 0.6 indicate predicted activity in the lower range \u2014 however, the ranking between candidates remains informative for synthesis prioritisation. Actual trans-cleavage efficiency on the LIG electrode will differ and requires experimental measurement.`
            }
            {avgPamAdj != null && avgPamAdj < avgActivity * 0.85 ? ` PAM-adjusted average (${avgPamAdj}) reflects enAsCas12a activity penalties on non-canonical PAMs (Kleinstiver et al. 2019). Canonical TTTV candidates retain full predicted activity; expanded PAMs (TTCV, TATV, CTTV) are penalised 35\u201375%.` : ""}
          </div>
          <div>
            {highDisc >= directCount * 0.8
              ? `${highDisc}/${directResults.length} direct candidates exceed the 3\u00d7 diagnostic-grade threshold (avg ${avgDisc}\u00d7, predicted by XGBoost on 18 thermodynamic features). At these ratios, the MUT electrochemical signal is clearly resolvable from WT on both SWV and lateral-flow readouts. Panel is well-suited for single-nucleotide resistance genotyping.`
              : highDisc >= directCount * 0.5
              ? `${highDisc}/${directResults.length} direct candidates meet diagnostic-grade discrimination (\u22653\u00d7). Targets below threshold should be prioritised for synthetic mismatch enhancement (seed positions 2\u20136) to improve WT suppression before electrode validation.`
              : `Only ${highDisc}/${directResults.length} direct candidates reach diagnostic-grade discrimination. Synthetic mismatch enhancement (Kohabir et al. 2024) at seed positions 2\u20136 is recommended. Alternatively, these targets may perform better as proximity candidates with AS-RPA primer selectivity.`
            }
          </div>
          {(missingPrimers.length > 0 || belowThreshold.length > 0) && (
            <div>
              {missingPrimers.length > 0 ? `${missingPrimers.length} target${missingPrimers.length > 1 ? "s" : ""} lack RPA primers \u2014 these cannot be amplified and are not deployable. ` : ""}
              {belowThreshold.length > 0 ? `${belowThreshold.length} target${belowThreshold.length > 1 ? "s" : ""} fall below readiness threshold (${belowThreshold.map(r => r.label).join(", ")}). These should still be synthesised for experimental validation \u2014 in silico readiness scores are conservative estimates. Prioritise these for the first round of electrode characterisation to calibrate the model.` : ""}
            </div>
          )}
          {withPrimers === totalTargets && belowThreshold.length === 0 && (
            <div style={{ color: T.success }}><strong>All {totalTargets} targets have primers and pass readiness threshold.</strong> Panel is ready for experimental validation on the LIG electrode array.</div>
          )}
        </div>
      </div>

      {/* Risk Assessment Matrix */}
      {results.some(r => r.riskProfile != null) && (
        <FigureSection title="Risk Assessment Matrix">
          <RiskMatrix results={results} />
        </FigureSection>
      )}

      {/* Diagnostic Readiness Score Chart */}
      <FigureSection title="Diagnostic Readiness Score">
        <ReadinessChart results={results} />
      </FigureSection>

      {/* UMAP Embedding Space */}
      {jobId && (
        <FigureSection title="Candidate Embedding Space">
          <UMAPPanel jobId={jobId} />
        </FigureSection>
      )}

      {/* Score vs Discrimination Scatter — readiness-sized dots */}
      {!mobile && (() => {
        const getScore = (r) => usesCompassMl ? (r.cnnCalibrated ?? r.score) : r.score;
        const hasReadiness = results.some(r => r.readinessScore != null);
        const scatterData = results.filter(r => r.disc > 0 && r.disc < 900).map(r => ({
          score: getScore(r), disc: Math.min(r.disc, 25), label: r.label, drug: r.drug, strategy: r.strategy, hasPrimers: r.hasPrimers, readiness: r.readinessScore || 0.5,
        }));
        const inTopRight = scatterData.filter(d => d.score >= 0.4 && d.disc >= 3).length;
        return (
          <FigureSection title="Score vs Discrimination">
          <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "28px 32px", marginBottom: "0" }}>
            <div style={{ marginBottom: "16px" }}>
              <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Score vs Discrimination</div>
              <div style={{ fontSize: "11px", color: T.textSec, marginTop: "3px" }}>
                Each candidate plotted by efficiency score (x) and discrimination ratio (y).
                Top-right quadrant = diagnostic-ready.{hasReadiness ? " Dot size reflects diagnostic readiness score (larger = higher readiness)." : " Dot size reflects primer availability."}
              </div>
            </div>
            <div style={{ position: "relative", borderRadius: 4, padding: "12px 8px 4px 0" }}>
              <ResponsiveContainer width="100%" height={360}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 25, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                  <XAxis type="number" dataKey="score" name="Score" domain={[0, 1]} tick={{ fontSize: 10, fontFamily: FONT, fill: CHART_TEXT_SEC }} label={{ value: "Efficiency Score", position: "insideBottom", offset: -12, fontSize: 11, fill: CHART_TEXT }} />
                  <YAxis type="number" dataKey="disc" name="Discrimination" domain={[0, "auto"]} tick={{ fontSize: 10, fontFamily: FONT, fill: CHART_TEXT_SEC }} label={{ value: "Discrimination (×)", angle: -90, position: "insideLeft", offset: 10, fontSize: 11, fill: CHART_TEXT }} />
                  <Tooltip content={({ payload }) => {
                    if (!payload?.length) return null;
                    const d = payload[0]?.payload;
                    if (!d) return null;
                    const ready = d.score >= 0.4 && d.disc >= 3;
                    return (
                      <div style={{ ...tooltipStyle, padding: "12px 16px" }}>
                        <div style={{ fontWeight: 600, fontSize: "12px", color: gradientColor(d.readiness), marginBottom: "4px" }}>{d.label}</div>
                        <div style={{ fontSize: "11px", color: T.textSec }}>Score: <strong style={{ color: T.text }}>{d.score.toFixed(3)}</strong></div>
                        <div style={{ fontSize: "11px", color: T.textSec }}>Discrimination: <strong style={{ color: T.text }}>{d.disc.toFixed(1)}×</strong></div>
                        <div style={{ fontSize: "11px", color: T.textSec }}>{d.drug} · {d.strategy}{hasReadiness ? ` · Readiness ${(d.readiness * 100).toFixed(0)}%` : (d.hasPrimers ? " · Primers OK" : " · No primers")}</div>
                        <div style={{ marginTop: "4px" }}><Badge variant={ready ? "success" : "warning"}>{ready ? "Diagnostic-ready" : "Needs improvement"}</Badge></div>
                      </div>
                    );
                  }} />
                  <ReferenceLine x={0.4} stroke="#333333" strokeDasharray="5 3" strokeWidth={1.5} />
                  <ReferenceLine y={3} stroke="#333333" strokeDasharray="5 3" strokeWidth={1.5} />
                  <Scatter data={scatterData} isAnimationActive={false}>
                    {scatterData.map((entry, i) => {
                      const dotR = hasReadiness ? Math.max(4, entry.readiness * 14) : (entry.hasPrimers ? 8 : 5);
                      return <Cell key={i} fill={gradientColor(entry.readiness)} r={dotR} stroke="rgba(0,0,0,0.1)" strokeWidth={1} opacity={0.85} />;
                    })}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
              {/* Quadrant labels */}
              <div style={{ position: "absolute", top: "24px", right: "28px", fontSize: "9px", fontWeight: 600, color: "#059669", opacity: 0.8, textTransform: "uppercase", letterSpacing: "0.04em" }}>Diagnostic-ready</div>
              <div style={{ position: "absolute", top: "24px", left: "60px", fontSize: "9px", fontWeight: 600, color: "#9CA3AF", opacity: 0.5, textTransform: "uppercase", letterSpacing: "0.04em" }}>Low score</div>
              <div style={{ position: "absolute", bottom: "42px", right: "28px", fontSize: "9px", fontWeight: 600, color: "#9CA3AF", opacity: 0.5, textTransform: "uppercase", letterSpacing: "0.04em" }}>Low discrimination</div>
            </div>
            {/* Legend */}
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "8px", flexWrap: "wrap" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ fontSize: "10px", color: T.textTer }}>Low readiness</span>
                <div style={{ width: 120, height: 8, borderRadius: 4, background: gradientCSS }} />
                <span style={{ fontSize: "10px", color: T.textTer }}>High readiness</span>
              </div>
              {hasReadiness ? (<>
                <span style={{ fontSize: "10px", color: T.textTer }}>|</span>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: T.textTer, opacity: 0.6 }} />
                  <span style={{ fontSize: "10px", color: T.textTer }}>Low readiness</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <div style={{ width: 14, height: 14, borderRadius: "50%", background: T.textTer, opacity: 0.6 }} />
                  <span style={{ fontSize: "10px", color: T.textTer }}>High readiness</span>
                </div>
              </>) : (<>
                <span style={{ fontSize: "10px", color: T.textTer }}>|</span>
                <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <div style={{ width: 12, height: 12, borderRadius: "50%", background: T.textTer, opacity: 0.8 }} />
                  <span style={{ fontSize: "10px", color: T.textTer }}>With primers</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: T.textTer, opacity: 0.4 }} />
                  <span style={{ fontSize: "10px", color: T.textTer }}>No primers</span>
                </div>
              </>)}
            </div>
            {/* Interpretation */}
            {(() => {
              const topRight = scatterData.filter(d => d.score >= 0.4 && d.disc >= 3);
              const bottomRight = scatterData.filter(d => d.score >= 0.4 && d.disc < 3);
              const topLeft = scatterData.filter(d => d.score < 0.4 && d.disc >= 3);
              const bestCandidate = [...scatterData].sort((a, b) => (b.score * b.disc) - (a.score * a.disc))[0];
              const worstCandidate = [...scatterData].sort((a, b) => (a.score * a.disc) - (b.score * b.disc))[0];
              const proximityCands = results.filter(r => r.strategy === "Proximity" && r.gene !== "IS6110");
              const viableProx = proximityCands.filter(r => !r.asrpaDiscrimination || r.asrpaDiscrimination.block_class !== "none");
              const nonViableProx = proximityCands.length - viableProx.length;
              return (
                <div style={{ marginTop: "14px", padding: "12px 16px", background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", fontSize: "11px", color: T.textSec, lineHeight: 1.7 }}>
                  <strong style={{ color: T.primary }}>Interpretation:</strong> {topRight.length}/{scatterData.length} Direct candidates are diagnostic-ready (score ≥ 0.4, disc ≥ 3×).
                  {bestCandidate ? ` Best overall: ${bestCandidate.label} (${bestCandidate.score.toFixed(3)}, ${bestCandidate.disc.toFixed(1)}×).` : ""}
                  {bottomRight.length > 0 ? ` ${bottomRight.length} Direct candidate${bottomRight.length > 1 ? "s have" : " has"} good scores but low Cas12a discrimination (${bottomRight.slice(0, 2).map(d => d.label).join(", ")}${bottomRight.length > 2 ? "…" : ""}) — synthetic mismatch enhancement may improve these.` : ""}
                  {topLeft.length > 0 ? ` ${topLeft.length} candidate${topLeft.length > 1 ? "s" : ""} ${topLeft.length > 1 ? "have" : "has"} strong discrimination but weak scores — alternative spacers may help.` : ""}
                  {proximityCands.length > 0 ? ` ${proximityCands.length} Proximity candidate${proximityCands.length > 1 ? "s are" : " is"} not plotted — their discrimination comes from AS-RPA primers, not crRNA mismatch. Of these, ${viableProx.length} show viable AS-RPA discrimination${nonViableProx > 0 ? ` and ${nonViableProx} ha${nonViableProx > 1 ? "ve" : "s"} no viable discrimination pathway (WC pair)` : ""}.` : ""}
                  {worstCandidate && worstCandidate !== bestCandidate ? ` Weakest Direct: ${worstCandidate.label} (${worstCandidate.score.toFixed(3)}, ${worstCandidate.disc.toFixed(1)}×).` : ""}
                </div>
              );
            })()}
          </div>
          </FigureSection>
        );
      })()}

      {/* Heuristic vs Compass-ML Scatter */}
      {!mobile && usesCompassMl && avgCNN != null && (() => {
        const scatterData = results.filter(r => r.cnnCalibrated != null).map(r => ({
          heuristic: r.score, narsilMl: r.cnnCalibrated, pamAdj: r.pamAdjusted ?? r.cnnCalibrated,
          label: r.label, drug: r.drug,
        }));
        const agreePct = (() => {
          const above = (v) => v >= 0.5;
          const agree = scatterData.filter(d => above(d.heuristic) === above(d.narsilMl)).length;
          return scatterData.length ? Math.round(agree / scatterData.length * 100) : 0;
        })();
        return (
          <FigureSection title="Scoring Model Comparison">
          <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "28px 32px", marginBottom: "0" }}>
            <div style={{ marginBottom: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Scoring Model Comparison</div>
                <div style={{ fontSize: "11px", color: T.textSec, marginTop: "3px", lineHeight: 1.5 }}>
                  Biophysical QC (x) vs Compass-ML activity (y) per candidate. Points near the diagonal indicate model agreement.
                  Candidates above the line have higher activity than their QC score suggests; below indicates biophysical concerns.
                </div>
              </div>
              <div style={{ textAlign: "right", flexShrink: 0, marginLeft: "20px" }}>
                <div style={{ display: "flex", gap: "16px" }}>
                  <div><div style={{ fontSize: "9px", color: T.textTer, fontWeight: 600 }}>AGREEMENT</div><div style={{ fontSize: "13px", fontWeight: 600, color: T.primary, fontFamily: FONT }}>{agreePct}%</div></div>
                  {modelAgreement != null && <div><div style={{ fontSize: "9px", color: T.textTer, fontWeight: 600 }}>SPEARMAN</div><div style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{modelAgreement}</div></div>}
                  <div><div style={{ fontSize: "9px", color: T.textTer, fontWeight: 600 }}>AVG QC</div><div style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{avgScore}</div></div>
                  <div><div style={{ fontSize: "9px", color: T.textTer, fontWeight: 600 }}>AVG ACTIVITY</div><div style={{ fontSize: "13px", fontWeight: 600, color: T.primary, fontFamily: FONT }}>{avgCNN}</div></div>
                </div>
              </div>
            </div>
            <div style={{ borderRadius: 4, padding: "12px 8px 4px 0" }}>
            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 25, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis type="number" dataKey="heuristic" name="Biophysical QC" domain={[0, 1]} tick={{ fontSize: 10, fontFamily: FONT, fill: CHART_TEXT_SEC }} label={{ value: "Biophysical QC (heuristic)", position: "insideBottom", offset: -12, fontSize: 11, fill: CHART_TEXT }} />
                <YAxis type="number" dataKey="narsilMl" name="Activity" domain={[0, 1]} tick={{ fontSize: 10, fontFamily: FONT, fill: CHART_TEXT_SEC }} label={{ value: "Activity (Compass-ML)", angle: -90, position: "insideLeft", offset: 10, fontSize: 11, fill: CHART_TEXT }} />
                <Tooltip content={({ payload }) => {
                  if (!payload?.length) return null;
                  const d = payload[0]?.payload;
                  if (!d) return null;
                  const diff = d.narsilMl - d.heuristic;
                  return (
                    <div style={{ ...tooltipStyle, padding: "12px 16px" }}>
                      <div style={{ fontWeight: 600, fontSize: "12px", color: gradientColor(d.narsilMl), marginBottom: "4px" }}>{d.label}</div>
                      <div style={{ fontSize: "11px", color: T.textSec }}>Activity: <strong style={{ color: T.primary }}>{d.narsilMl.toFixed(3)}</strong></div>
                      <div style={{ fontSize: "11px", color: T.textSec }}>PAM-adjusted: <strong style={{ color: T.text }}>{d.pamAdj.toFixed(3)}</strong></div>
                      <div style={{ fontSize: "11px", color: T.textSec }}>QC (heuristic): <strong style={{ color: T.textTer }}>{d.heuristic.toFixed(3)}</strong></div>
                      <div style={{ fontSize: "11px", color: diff > 0.05 ? T.success : diff < -0.05 ? T.warning : T.textTer, marginTop: "2px" }}>
                        {"\u0394"} = {diff > 0 ? "+" : ""}{diff.toFixed(3)} ({diff > 0.05 ? "Net scores higher" : diff < -0.05 ? "Heuristic scores higher" : "Models agree"})
                      </div>
                    </div>
                  );
                }} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#333333" strokeDasharray="6 4" strokeWidth={1} opacity={0.6} />
                <Scatter data={scatterData} isAnimationActive={false}>
                  {scatterData.map((entry, i) => (
                    <Cell key={i} fill={gradientColor(entry.narsilMl)} r={7} stroke="rgba(0,0,0,0.1)" strokeWidth={1} opacity={0.85} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            </div>
            {/* Legend */}
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "8px", flexWrap: "wrap" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ fontSize: "10px", color: T.textTer }}>Low</span>
                <div style={{ width: 120, height: 8, borderRadius: 4, background: gradientCSS }} />
                <span style={{ fontSize: "10px", color: T.textTer }}>High activity</span>
              </div>
              <span style={{ fontSize: "10px", color: T.textTer }}>|</span>
              <span style={{ fontSize: "10px", color: T.textTer }}>Dashed line = perfect agreement (y = x)</span>
            </div>
            {/* Interpretation */}
            {(() => {
              const aboveLine = scatterData.filter(d => d.narsilMl > d.heuristic + 0.05);
              const belowLine = scatterData.filter(d => d.heuristic > d.narsilMl + 0.05);
              const onLine = scatterData.length - aboveLine.length - belowLine.length;
              return (
                <div style={{ marginTop: "14px", padding: "12px 16px", background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", fontSize: "11px", color: T.textSec, lineHeight: 1.7 }}>
                  <strong style={{ color: T.primary }}>Interpretation:</strong> {agreePct}% of candidates are classified the same way by both models (above/below 0.5 threshold). {onLine}/{scatterData.length} score within ±0.05 of each other.
                  {aboveLine.length > 0 ? ` Compass-ML scores ${aboveLine.length} candidate${aboveLine.length > 1 ? "s" : ""} higher (${aboveLine.slice(0, 2).map(d => d.label).join(", ")}${aboveLine.length > 2 ? "\u2026" : ""}).` : ""}
                  {belowLine.length > 0 ? ` Heuristic scores ${belowLine.length} candidate${belowLine.length > 1 ? "s" : ""} higher (${belowLine.slice(0, 2).map(d => d.label).join(", ")}${belowLine.length > 2 ? "\u2026" : ""}).` : ""}
                  {modelAgreement != null ? ` Rank correlation \u03c1 = ${modelAgreement} \u2014 ${modelAgreement >= 0.7 ? "strong agreement, QC corroborates activity predictions" : modelAgreement >= 0.4 ? "moderate agreement, QC catches biophysical edge cases activity model misses" : "weak agreement, models measure different things \u2014 QC serves as independent sanity check"}.` : ""}
                </div>
              );
            })()}
          </div>
          </FigureSection>
        );
      })()}

      {/* Drug coverage table */}
      <FigureSection title="Drug Coverage">
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
        <div style={{ padding: "16px 24px", borderBottom: `1px solid ${T.border}`, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Drug Coverage</div>
          <div style={{ fontSize: "12px", color: T.textTer }}>{drugs.length} classes</div>
        </div>
        <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
          <thead>
            <tr>
              {["Drug", "Candidates", "Avg Score", "Avg Disc (Direct)", "Primers"].map((h) => (
                <th key={h} style={{ padding: "10px 24px", textAlign: "left", fontWeight: 600, color: T.textTer, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {byDrug.map((d) => {
              const rows = results.filter((r) => r.drug === d.drug);
              const primerCount = rows.filter((r) => r.hasPrimers).length;
              return (
                <tr key={d.drug} style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                  <td style={{ padding: "12px 24px" }}><DrugBadge drug={d.drug} /></td>
                  <td style={{ padding: "12px 24px", fontFamily: FONT, fontWeight: 600 }}>{d.count}</td>
                  <td style={{ padding: "12px 24px", fontFamily: FONT, color: d.avgScore > 0.75 ? T.success : d.avgScore > 0.6 ? T.text : T.warning }}>{d.avgScore}</td>
                  {(() => {
                    const directRows = rows.filter(r => r.strategy === "Direct" && r.disc > 0 && r.disc < 900);
                    const proxCount = rows.filter(r => r.strategy === "Proximity").length;
                    if (directRows.length > 0) {
                      const avg = (directRows.reduce((a, r) => a + r.disc, 0) / directRows.length).toFixed(1);
                      return <td style={{ padding: "12px 24px", fontFamily: FONT }}>{avg}×<span style={{ fontSize: "9px", color: T.textTer, marginLeft: "3px" }}>({directRows.length})</span>{proxCount > 0 && <span style={{ fontSize: "9px", color: T.textTer, marginLeft: "4px" }}>+{proxCount} AS-RPA</span>}</td>;
                    }
                    return <td style={{ padding: "12px 24px", fontSize: "10px", color: T.purple, fontWeight: 600 }}>AS-RPA only</td>;
                  })()}
                  <td style={{ padding: "12px 24px" }}>
                    <span style={{ fontFamily: FONT, fontWeight: 600 }}>{primerCount}/{d.count}</span>
                    {primerCount < d.count && <span style={{ marginLeft: "6px", fontSize: "10px", color: T.warning, fontWeight: 600 }}>{d.count - primerCount} missing</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        </div>
      </div>
      </FigureSection>
    </div>
  );
};

/* ─── Spacer Architecture ─── nucleotide-by-nucleotide crRNA SVG ─── */
const SpacerArchitecture = ({ r }) => {
  const mobile = useIsMobile();
  const toast = useToast();
  const [copied, setCopied] = useState(false);
  // Use SM-enhanced spacer when available so the SM base change is visible
  const spacer = (r.hasSM && r.smSpacer) ? r.smSpacer : r.spacer;
  const wt = r.wtSpacer && r.wtSpacer.length === spacer.length ? r.wtSpacer : null;
  const hasWt = !!wt;
  const len = spacer.length;

  // Derive per-nucleotide annotations
  const nts = spacer.split("").map((base, i) => {
    const pos = i + 1;
    return { base, pos, isSeed: pos <= 8, isSnp: false, isSynthMM: false };
  });

  // Classify mismatches: SNP vs synthetic mismatch
  const mmIndices = hasWt ? nts.map((nt, i) => spacer[i] !== wt[i] ? i : -1).filter(i => i >= 0) : [];
  if (r.hasSM && r.smPosition) {
    // We know exactly which position is the SM from the backend
    const smIdx = r.smPosition - 1; // smPosition is 1-based
    if (smIdx >= 0 && smIdx < nts.length) {
      nts[smIdx].isSynthMM = true;
    }
    mmIndices.filter(i => i !== smIdx).forEach(i => { nts[i].isSnp = true; });
  } else if (r.hasSM && mmIndices.length >= 2) {
    // Fallback: infer SM position from seed region mismatches
    const smIdx = mmIndices.find(i => nts[i].pos >= 2 && nts[i].pos <= 6);
    if (smIdx !== undefined) {
      nts[smIdx].isSynthMM = true;
      mmIndices.filter(i => i !== smIdx).forEach(i => { nts[i].isSnp = true; });
    } else {
      mmIndices.forEach(i => { nts[i].isSnp = true; });
    }
  } else {
    mmIndices.forEach(i => { nts[i].isSnp = true; });
  }

  // Bigger, more spacious cells
  const cellW = 30, cellH = 44, cellGap = 3;
  const pamW = 64, pamGap = 10, oX = 30;
  const spacerX = oX + pamW + pamGap;
  const totalNtW = len * (cellW + cellGap) - cellGap;
  const svgW = spacerX + totalNtW + 36;
  const svgH = 80;

  const cellBg = (nt) => nt.isSnp ? T.danger : nt.isSynthMM ? T.warning : nt.isSeed ? T.primaryLight : T.bgSub;
  const cellBorder = (nt) => nt.isSnp ? T.danger : nt.isSynthMM ? T.warning : nt.isSeed ? "rgba(79,70,229,0.25)" : T.borderLight;
  const letterFill = (nt) => (nt.isSnp || nt.isSynthMM) ? "#FFFFFF" : NUC[nt.base] || T.textSec;
  const posFill = (nt) => (nt.isSnp || nt.isSynthMM) ? "rgba(255,255,255,0.7)" : nt.isSeed ? T.primary : T.textTer;

  const snpNt = nts.find(n => n.isSnp);
  const smNt = nts.find(n => n.isSynthMM);
  const snpChange = snpNt ? `${wt[snpNt.pos - 1]}→${snpNt.base}` : "—";
  const smChange = smNt ? `${wt[smNt.pos - 1]}→${smNt.base}` : null;

  const handleCopy = (e) => {
    e.stopPropagation();
    if (navigator.clipboard) navigator.clipboard.writeText(spacer);
    setCopied(true);
    toast("Spacer copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ marginBottom: "28px" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>crRNA Spacer Architecture</div>
        <button onClick={handleCopy} style={{ background: "none", border: `1px solid ${T.border}`, borderRadius: "4px", padding: "4px 10px", cursor: "pointer", display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", color: copied ? T.success : T.textSec, fontFamily: FONT, transition: "color 0.15s" }}>
          {copied ? <Check size={12} /> : <Copy size={12} />} {copied ? "Copied" : "Copy spacer"}
        </button>
      </div>

      {/* SVG card — centered with generous padding */}
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "28px 24px 20px", overflowX: "auto" }}>
        <div style={{ display: "flex", justifyContent: "center" }}>
          <svg width={svgW} height={svgH} style={{ fontFamily: MONO, display: "block", minWidth: svgW }}>
            {/* 5' label */}
            <text x={oX - 6} y={28} fontSize={12} fill={T.textTer} fontWeight={600} textAnchor="end">5′</text>

            {/* PAM block */}
            <rect x={oX} y={4} width={pamW} height={cellH} rx={8} fill={T.primary} />
            <text x={oX + pamW / 2} y={17} textAnchor="middle" fontSize={9} fill="rgba(255,255,255,0.6)" fontWeight={600} style={{ letterSpacing: "0.04em" }}>PAM</text>
            <text x={oX + pamW / 2} y={36} textAnchor="middle" fontSize={16} fill="#FFFFFF" fontWeight={600} letterSpacing="2">{r.pam}</text>

            {/* Nucleotide cells */}
            {nts.map((nt, i) => {
              const x = spacerX + i * (cellW + cellGap);
              return (
                <g key={i}>
                  <rect x={x} y={4} width={cellW} height={cellH} rx={5} fill={cellBg(nt)} stroke={cellBorder(nt)} strokeWidth={1.2} />
                  <text x={x + cellW / 2} y={28} textAnchor="middle" fontSize={15} fontWeight={600} fill={letterFill(nt)}>{nt.base}</text>
                  <text x={x + cellW / 2} y={42} textAnchor="middle" fontSize={9} fontWeight={nt.isSeed ? 700 : 400} fill={posFill(nt)}>{nt.pos}</text>
                </g>
              );
            })}

            {/* 3' label */}
            <text x={spacerX + totalNtW + 12} y={28} fontSize={12} fill={T.textTer} fontWeight={600}>3′</text>

            {/* Seed bracket */}
            {(() => {
              const bx1 = spacerX;
              const bx2 = spacerX + 8 * (cellW + cellGap) - cellGap;
              const by = cellH + 14;
              return (
                <g>
                  <line x1={bx1 + 3} y1={by - 4} x2={bx1 + 3} y2={by + 1} stroke={T.primary} strokeWidth={1.2} opacity={0.5} />
                  <line x1={bx1 + 3} y1={by + 1} x2={bx2 - 3} y2={by + 1} stroke={T.primary} strokeWidth={1.2} opacity={0.5} />
                  <line x1={bx2 - 3} y1={by - 4} x2={bx2 - 3} y2={by + 1} stroke={T.primary} strokeWidth={1.2} opacity={0.5} />
                  <text x={(bx1 + bx2) / 2} y={by + 15} textAnchor="middle" fontSize={10} fill={T.primary} fontWeight={600} fontFamily={FONT}>SEED (1–8)</text>
                </g>
              );
            })()}
          </svg>
        </div>

        {/* Legend + metadata row — below the SVG, centered */}
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", alignItems: "center", gap: "20px", marginTop: "16px", paddingTop: "14px", borderTop: `1px solid ${T.borderLight}` }}>
          {/* Legend items */}
          {[
            { color: T.primary, label: "PAM" },
            { color: T.primaryLight, label: "Seed (1–8)", border: "rgba(79,70,229,0.25)" },
            { color: T.danger, label: "SNP" },
            ...(r.hasSM ? [{ color: T.warning, label: "Synth. MM" }] : []),
          ].map((item, idx) => (
            <div key={idx} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <div style={{ width: 12, height: 12, borderRadius: 3, background: item.color, border: item.border ? `1px solid ${item.border}` : "none" }} />
              <span style={{ fontSize: "11px", color: T.textSec, fontWeight: 500 }}>{item.label}</span>
            </div>
          ))}

          {/* Divider */}
          <div style={{ width: "1px", height: "16px", background: T.borderLight }} />

          {/* Metadata chips */}
          <span style={{ fontSize: "11px", color: T.textSec, fontFamily: FONT }}>{len} nt · GC {(r.gc * 100).toFixed(0)}%</span>
          {snpNt ? (
            <span style={{ fontSize: "11px", color: T.danger, fontWeight: 600, fontFamily: FONT }}>SNP @ pos {snpNt.pos} · {snpChange}</span>
          ) : r.strategy === "Proximity" ? (
            <span style={{ fontSize: "11px", color: T.purple, fontWeight: 600 }}>Mutation outside spacer{r.proximityDistance ? ` (${r.proximityDistance} bp away)` : ""}</span>
          ) : !hasWt ? (
            <span style={{ fontSize: "11px", color: T.textTer }}>WT spacer unavailable</span>
          ) : (
            <span style={{ fontSize: "11px", color: T.textTer }}>No SNP in spacer</span>
          )}
          {smNt ? (
            <span style={{ fontSize: "11px", color: T.warning, fontWeight: 600, fontFamily: FONT }}>SM: {smChange} @ pos {smNt.pos}</span>
          ) : (
            <span style={{ fontSize: "11px", color: T.textTer }}>SM: none</span>
          )}
        </div>
      </div>
    </div>
  );
};

const generateInterpretation = (r) => {
  const lines = [];
  const eff = r.cnnCalibrated ?? r.score;
  const gc = r.gc * 100;
  const disc = typeof r.disc === "number" ? r.disc : 0;
  const spacer = (r.hasSM && r.smSpacer) ? r.smSpacer : r.spacer;
  const wt = r.wtSpacer && r.wtSpacer.length === spacer.length ? r.wtSpacer : null;

  // Find SNP position
  let snpPos = null;
  let snpChange = null;
  if (wt) {
    for (let i = 0; i < spacer.length; i++) {
      if (spacer[i] !== wt[i]) {
        const isSM = r.hasSM && r.smPosition && (r.smPosition - 1) === i;
        if (!isSM) { snpPos = i + 1; snpChange = `${wt[i]}\u2192${spacer[i]}`; break; }
      }
    }
  }

  // Overall assessment
  if (eff >= 0.7) lines.push(`Strong candidate (activity score ${eff.toFixed(3)}). High predicted Cas12a trans-cleavage rate — expected to produce a clear SWV signal decrease within 15–30 min on the electrochemical platform, well above the limit of detection.`);
  else if (eff >= 0.5) lines.push(`Moderate candidate (activity score ${eff.toFixed(3)}). Predicted trans-cleavage is sufficient for detection but not optimal — the SWV signal decrease may require 30–45 min to reach a confident positive call on the electrochemical platform.`);
  else lines.push(`Weak candidate (activity score ${eff.toFixed(3)}). Low predicted trans-cleavage rate — the electrochemical signal decrease may be near the detection limit, risking false negatives. Consider alternatives from the top-K list or synthetic mismatch optimisation.`);

  // PAM quality
  const pam = (r.pam || "").toUpperCase();
  if (r.isCanonicalPam || pam.match(/^TTT[ACG]/)) {
    lines.push(`Canonical PAM (${r.pam}) \u2014 optimal Cas12a recognition, no activity penalty applied.`);
  } else {
    const penaltyStr = r.pamPenalty != null ? ` Activity penalty: ${r.pamPenalty}\u00d7 (Kleinstiver et al. 2019).` : "";
    lines.push(`Expanded PAM (${r.pam}${r.pamVariant ? `, ${r.pamVariant}` : ""}) \u2014 recognized by enAsCas12a with reduced activity vs canonical TTTV.${penaltyStr} This is the best available PAM site in the GC-rich M. tuberculosis genomic context around this mutation.`);
  }

  // PAM disruption — binary discrimination override
  if (r.pamDisrupted) {
    const disruptionDetail = r.pamDisruptionType === "wt_pam_broken"
      ? "Binary discrimination \u2014 the resistance SNP disrupts the PAM consensus in the wildtype sequence. Cas12a cannot bind WT DNA at this locus, providing effectively infinite discrimination. This is the strongest possible discrimination mechanism: all-or-nothing PAM recognition gating."
      : "Binary discrimination \u2014 the resistance SNP disrupts the PAM consensus in the mutant sequence. Cas12a cannot bind MUT DNA at this locus. This inverts the expected detection logic \u2014 signal absence indicates resistance.";
    lines.push(disruptionDetail);
  }

  // Discrimination
  const discModelName = r.discrimination?.model_name || "";
  const isNeuralDisc = r.discMethod === "neural";
  const isLearnedDisc = discModelName.includes("learned") || r.discMethod === "feature";
  const discSource = isNeuralDisc ? "neural discrimination head (Compass-ML multi-task, trained on 6,136 EasyDesign pairs)" : isLearnedDisc ? "learned model (XGBoost, 18 thermodynamic features)" : "heuristic model (position \u00D7 destabilisation)";
  if (r.pamDisrupted) {
    // Skip normal discrimination analysis — already covered above
  } else if (r.strategy === "Proximity") {
    lines.push(`Proximity detection \u2014 the resistance SNP falls outside the crRNA spacer${r.proximityDistance ? ` (${r.proximityDistance} bp away)` : ""}. Allele discrimination relies on AS-RPA primers (10\u2013100\u00D7 selectivity), not Cas12a mismatch intolerance. The Cas12a disc ratio (~${disc.toFixed(1)}\u00D7) is not relevant for this strategy.`);
  } else if (snpPos) {
    // Mismatch chemistry context
    let mmChem = "";
    if (snpChange) {
      const bases = snpChange.split("\u2192");
      if (bases.length === 2) {
        const purines = new Set(["A", "G"]);
        const b1 = bases[0].toUpperCase(), b2 = bases[1].toUpperCase();
        if (purines.has(b1) && purines.has(b2)) mmChem = " (purine\u2192purine, severely destabilising)";
        else if (!purines.has(b1) && !purines.has(b2)) mmChem = " (pyrimidine\u2192pyrimidine, moderately destabilising)";
        else if ((b1 === "G" && b2 === "T") || (b1 === "T" && b2 === "G")) mmChem = " (G:T wobble, tolerated by Cas12a)";
        else mmChem = " (transversion)";
      }
    }
    if (snpPos <= 8) {
      const rloopEffect = disc >= 10 ? "near-complete R-loop collapse" : disc >= 5 ? "substantial R-loop disruption" : disc >= 3 ? "moderate R-loop destabilization" : disc >= 2 ? "partial R-loop disruption" : "minimal R-loop disruption";
      if (disc >= 5) lines.push(`SNP at seed position ${snpPos} (${snpChange}${mmChem}) provides strong discrimination (${disc.toFixed(1)}\u00D7, ${discSource}). The mismatch in the PAM-proximal seed region (pos 1\u20138) causes ${rloopEffect} on the wildtype template, ensuring high specificity.`);
      else if (disc >= 3) lines.push(`SNP at seed position ${snpPos} (${snpChange}${mmChem}) provides diagnostic-grade discrimination (${disc.toFixed(1)}\u00D7, ${discSource}). Seed region mismatches cause ${rloopEffect}, reducing Cas12a binding on the wildtype template.`);
      else lines.push(`SNP at seed position ${snpPos} (${snpChange}${mmChem}) gives limited discrimination (${disc.toFixed(1)}\u00D7, ${discSource}) despite being in the seed region \u2014 ${rloopEffect}. The surrounding sequence context or mismatch chemistry may stabilise partial R-loop formation. Synthetic mismatch enhancement may improve this.`);
    } else {
      if (disc >= 3) lines.push(`SNP at PAM-distal position ${snpPos} (${snpChange}${mmChem}) provides ${disc.toFixed(1)}\u00D7 discrimination (${discSource}) with ${disc >= 10 ? "near-complete R-loop collapse" : disc >= 5 ? "substantial R-loop disruption" : "moderate R-loop destabilization"}. Although outside the seed, the mismatch is sufficient for diagnostic-grade allele differentiation.`);
      else lines.push(`SNP at PAM-distal position ${snpPos} (${snpChange}${mmChem}) gives limited discrimination (${disc.toFixed(1)}\u00D7, ${discSource}) \u2014 ${disc >= 2 ? "partial R-loop disruption" : "minimal R-loop disruption"}. PAM-distal mismatches are better tolerated by Cas12a \u2014 synthetic mismatch in the seed region could boost specificity.`);
    }
  } else if (r.strategy === "Direct" && !wt) {
    lines.push(`Direct detection strategy. Discrimination ratio: ${disc.toFixed(1)}\u00D7 (${discSource}). WT spacer data unavailable for positional analysis.`);
  }

  // Synthetic mismatch
  if (r.hasSM) {
    let smPos = r.smPosition || null;
    let smChange = null;
    if (smPos && wt && spacer[smPos - 1] !== wt[smPos - 1]) smChange = `${wt[smPos - 1]}\u2192${spacer[smPos - 1]}`;
    const smImprovement = r.smImprovementFactor ? ` (${r.smImprovementFactor.toFixed(1)}\u00D7 improvement)` : "";
    if (smPos && smChange) lines.push(`Synthetic mismatch at position ${smPos} (${smChange}) creates a double-mismatch penalty on the wildtype template${smImprovement}. Activity cost depends on position and mismatch type (seed-proximal SM: 10\u201340% reduction; PAM-distal SM: 5\u201315% reduction; Liang et al. 2023).`);
    else lines.push(`Synthetic mismatch applied \u2014 an engineered base substitution creates a double-mismatch penalty on the wildtype template, boosting specificity${smImprovement}. Activity cost is position-dependent (Liang et al. 2023).`);
  }

  // GC content
  if (gc > 65) lines.push(`High GC content (${gc.toFixed(0)}%) increases R-loop thermodynamic stability but also raises the energetic cost of target strand unwinding. This is typical for M. tuberculosis (genome-wide GC ~65.6%).`);
  else if (gc < 40) lines.push(`Low GC content (${gc.toFixed(0)}%) \u2014 unusual for M. tuberculosis. R-loop stability may be reduced, potentially lowering cleavage efficiency.`);

  // Off-targets
  if (r.ot > 0) lines.push(`${r.ot} potential off-target site${r.ot > 1 ? "s" : ""} detected in the H37Rv genome. Review cross-reactivity before synthesis \u2014 off-targets within the same amplicon region could generate false positives.`);

  return lines;
};

const CandidateAccordion = ({ r, onShowAlternatives }) => {
  const mobile = useIsMobile();
  const toast = useToast();
  const ref = r.refs;
  const discColor = r.disc >= 3 ? T.success : r.disc >= 2 ? T.primary : r.disc >= 1.5 ? T.warning : T.danger;
  const displaySpacer = (r.hasSM && r.smSpacer) ? r.smSpacer : r.spacer;
  const [openTab, setOpenTab] = useState(null);
  const interpretation = useMemo(() => generateInterpretation(r), [r]);

  const toggleTab = (tab) => { setOpenTab(prev => prev === tab ? null : tab); };

  const tabStyle = (tab) => ({
    flex: 1, padding: "10px 14px", background: openTab === tab ? T.bg : "transparent",
    border: `1px solid ${openTab === tab ? T.border : T.borderLight}`,
    borderBottom: openTab === tab ? "none" : `1px solid ${T.borderLight}`,
    borderRadius: openTab === tab ? "8px 8px 0 0" : "8px",
    cursor: "pointer", fontSize: "11px", fontWeight: 600,
    color: openTab === tab ? T.primary : T.textSec, fontFamily: FONT,
    display: "flex", alignItems: "center", gap: "6px", justifyContent: "center",
    transition: "all 0.15s",
  });

  return (
    <div style={{ padding: mobile ? "16px" : "20px 24px", background: T.bgSub, borderTop: `1px solid ${T.borderLight}` }}>
      {/* Key metrics row */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0", background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "14px", marginBottom: "20px" }}>
        {[
          { l: "Activity", v: (r.cnnCalibrated ?? r.score).toFixed(3), c: (r.cnnCalibrated ?? r.score) > 0.7 ? T.primary : (r.cnnCalibrated ?? r.score) > 0.5 ? T.warning : T.danger },
          ...(r.pamAdjusted != null && r.pamPenalty != null && r.pamPenalty < 1.0 ? [{ l: "PAM-adjusted", v: `${r.pamAdjusted.toFixed(3)} (${r.pamPenalty}×)`, c: T.textSec }] : []),
          { l: r.strategy === "Proximity" ? "Disc (AS-RPA)" : "Discrimination", v: r.strategy === "Proximity" ? (r.asrpaDiscrimination ? (r.asrpaDiscrimination.block_class === "none" ? "1× (no mismatch)" : `${r.asrpaDiscrimination.disc_ratio >= 100 ? "≥100" : r.asrpaDiscrimination.disc_ratio.toFixed(0)}× ${r.asrpaDiscrimination.terminal_mismatch}`) : "AS-RPA") : r.gene === "IS6110" ? "N/A (control)" : `${typeof r.disc === "number" ? r.disc.toFixed(1) : r.disc}×`, c: r.strategy === "Proximity" ? (r.asrpaDiscrimination?.block_class === "none" ? T.danger : T.purple) : r.gene === "IS6110" ? T.textTer : discColor },
          ...(r.strategy === "Proximity" && r.proximityDistance ? [{ l: "Distance", v: `${r.proximityDistance} bp`, c: T.purple }] : []),
          { l: "Activity QC", v: r.activityQc != null ? r.activityQc.toFixed(3) : r.score.toFixed(3), c: T.textTer },
          ...(r.discriminationQc != null ? [{ l: "Disc QC", v: r.discriminationQc.toFixed(3), c: r.discriminationQc > 0.6 ? T.success : r.discriminationQc > 0.3 ? T.warning : T.danger }] : []),
          { l: "GC%", v: `${(r.gc * 100).toFixed(0)}%`, c: T.text },
          { l: "Off-targets", v: r.ot, c: r.ot === 0 ? T.success : T.warning },
          { l: "Strategy", v: r.strategy, c: r.strategy === "Direct" ? T.success : T.purple },
        ].map((s, i) => (
          <div key={s.l} style={{ flex: 1, textAlign: "center", borderLeft: i > 0 ? `1px dashed ${T.border}` : "none", minWidth: mobile ? "30%" : "auto" }}>
            <div style={{ fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "4px" }}>{s.l}</div>
            <div style={{ fontSize: "16px", fontWeight: 600, color: s.c, fontFamily: FONT }}>{s.v}</div>
          </div>
        ))}
      </div>

      {/* PROXIMITY explanation block */}
      {r.strategy === "Proximity" && (
        <div style={{ background: T.purpleLight, border: `1px solid ${T.purple}33`, borderRadius: "4px", padding: "12px 16px", marginBottom: "16px" }}>
          <div style={{ fontSize: "12px", fontWeight: 600, color: T.purple, fontFamily: HEADING, marginBottom: "4px" }}>Proximity Detection — PAM Desert</div>
          <div style={{ fontSize: "11px", color: "#2563EB", lineHeight: 1.5 }}>
            crRNA binds a conserved site {r.proximityDistance ? `${r.proximityDistance} bp` : "near"} the mutation. Discrimination via AS-RPA primers.
          </div>
        </div>
      )}

      {/* crRNA Spacer Architecture — full width, with Show Alternatives top-right */}
      <div style={{ position: "relative" }}>
        {onShowAlternatives && (
          <button onClick={(e) => { e.stopPropagation(); onShowAlternatives(); }} style={{
            position: "absolute", top: 0, right: 0, zIndex: 2,
            background: T.primaryLight, border: `1px solid ${T.primary}44`,
            borderRadius: "4px", padding: "6px 14px", cursor: "pointer",
            fontSize: "11px", fontWeight: 600, color: T.primaryDark, fontFamily: FONT,
            display: "flex", alignItems: "center", gap: "5px", transition: "all 0.15s",
          }}
          onMouseEnter={e => { e.currentTarget.style.background = T.primary; e.currentTarget.style.color = "#fff"; }}
          onMouseLeave={e => { e.currentTarget.style.background = T.primaryLight; e.currentTarget.style.color = T.primaryDark; }}
          >
            <Layers size={12} /> Show alternatives
          </button>
        )}
        <SpacerArchitecture r={r} />
      </div>

      {/* Dynamic Interpretation Box */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", padding: "16px 20px", marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
          <Info size={14} color={T.primary} />
          <span style={{ fontSize: "12px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING }}>Interpretation</span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {interpretation.map((line, i) => (
            <div key={i} style={{ fontSize: "11.5px", color: T.primaryDark, lineHeight: 1.65, paddingLeft: "20px", position: "relative" }}>
              <span style={{ position: "absolute", left: 0, top: "1px", width: "14px", height: "14px", borderRadius: "50%", background: T.primarySub, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "8px", fontWeight: 600, color: T.primary }}>{i + 1}</span>
              {line}
            </div>
          ))}
        </div>
      </div>

      {/* Collapsible detail tabs */}
      <div style={{ display: "flex", gap: "6px", marginBottom: openTab ? "0" : "0" }}>
        <div style={tabStyle("amplicon")} onClick={(e) => { e.stopPropagation(); toggleTab("amplicon"); }}>
          <Map size={12} /> Amplicon & Mismatch
        </div>
        <div style={tabStyle("oligos")} onClick={(e) => { e.stopPropagation(); toggleTab("oligos"); }}>
          <Copy size={12} /> Oligo Sequences
        </div>
        <div style={tabStyle("evidence")} onClick={(e) => { e.stopPropagation(); toggleTab("evidence"); }}>
          <FileText size={12} /> Evidence & Metadata
        </div>
      </div>

      {/* Tab content */}
      {openTab && (
        <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderTop: "none", borderRadius: "0 0 4px 4px", padding: "16px", animation: "fadeIn 0.15s ease-out" }}>

          {/* Amplicon & Mismatch tab */}
          {openTab === "amplicon" && (
            <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "16px" }}>
              <div>
                <div style={{ fontSize: "11px", fontWeight: 600, color: T.textSec, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Amplicon Map</div>
                <div style={{ background: T.bgSub, borderRadius: "4px", padding: "8px 4px", border: `1px solid ${T.borderLight}` }}>
                  <AmpliconMap r={r} />
                </div>
              </div>
              <div>
                <div style={{ fontSize: "11px", fontWeight: 600, color: T.textSec, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>MUT vs WT Mismatch</div>
                <div style={{ background: T.bgSub, borderRadius: "4px", padding: "12px", border: `1px solid ${T.borderLight}`, overflowX: "auto" }}>
                  <MismatchProfile spacer={displaySpacer} wtSpacer={r.wtSpacer} strategy={r.strategy} />
                </div>
              </div>
              {r.hasPrimers && (
                <div style={{ gridColumn: mobile ? "1" : "1 / -1", display: "flex", gap: "8px" }}>
                  <div style={{ flex: 1, background: T.bgSub, borderRadius: "4px", padding: "10px", border: `1px solid ${T.borderLight}`, fontSize: "11px" }}>
                    <div style={{ color: T.textTer, marginBottom: "2px" }}>Amplicon</div>
                    <div style={{ fontWeight: 600, fontFamily: FONT, color: T.text }}>{r.amplicon} bp</div>
                  </div>
                  <div style={{ flex: 1, background: r.isCanonicalPam === false ? "#FEF3C7" : T.bgSub, borderRadius: "4px", padding: "10px", border: `1px solid ${r.isCanonicalPam === false ? "#F59E0B40" : T.borderLight}`, fontSize: "11px" }}>
                    <div style={{ color: T.textTer, marginBottom: "2px" }}>PAM</div>
                    <div style={{ fontWeight: 600, fontFamily: MONO, color: T.text }}>{r.pam}</div>
                    {r.pamVariant && <div style={{ marginTop: "2px", fontSize: "9px", fontWeight: 600, color: r.isCanonicalPam ? T.success : "#4338CA" }}>{r.pamVariant}{r.pamPenalty != null && r.pamPenalty < 1.0 ? ` · ${r.pamPenalty}× activity` : ""}</div>}
                  </div>
                  <div style={{ flex: 1, background: r.hasSM ? T.primaryLight : T.bgSub, borderRadius: "4px", padding: "10px", border: `1px solid ${T.borderLight}`, fontSize: "11px" }}>
                    <div style={{ color: T.textTer, marginBottom: "2px" }}>SM</div>
                    <div style={{ fontWeight: 600, color: r.hasSM ? T.primaryDark : T.textTer }}>{r.hasSM ? "Yes" : "No"}</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Oligo Sequences tab */}
          {openTab === "oligos" && (
            <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
              {[
                { name: `${r.label}_crRNA`, seq: `AATTTCTACTCTTGTAGAT${displaySpacer}`, note: "Direct repeat + spacer" },
                ...(r.fwd ? [{ name: `${r.label}_FWD`, seq: r.fwd, note: r.strategy === "Direct" ? "Standard RPA forward" : "AS-RPA forward (allele-specific)" }] : []),
                ...(r.rev ? [{ name: `${r.label}_REV`, seq: r.rev, note: r.strategy === "Direct" ? "Standard RPA reverse" : "AS-RPA reverse (allele-specific)" }] : []),
              ].map((o, i, arr) => (
                <div key={o.name} style={{ padding: "10px 14px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                    <span style={{ fontSize: "10px", fontWeight: 600, fontFamily: MONO, color: T.text }}>{o.name}</span>
                    <button onClick={(e) => { e.stopPropagation(); navigator.clipboard?.writeText(o.seq); toast(`${o.name} copied`); }} style={{ background: "none", border: `1px solid ${T.border}`, borderRadius: "5px", padding: "3px 8px", cursor: "pointer", fontSize: "9px", color: T.textSec, display: "flex", alignItems: "center", gap: "3px" }}><Copy size={9} /> Copy</button>
                  </div>
                  <div style={{ background: T.bg, borderRadius: "4px", padding: "8px 10px", border: `1px solid ${T.borderLight}`, marginBottom: "4px" }}>
                    <Seq s={o.seq} />
                  </div>
                  <div style={{ fontSize: "9px", color: T.textTer }}>{o.note} — {o.seq.length} nt</div>
                </div>
              ))}
            </div>
          )}

          {/* Evidence & Metadata tab */}
          {openTab === "evidence" && (
            <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "16px" }}>
              {ref && (
                <div>
                  <div style={{ fontSize: "11px", fontWeight: 600, color: T.textSec, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Clinical Evidence</div>
                  <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
                    {[
                      ["WHO Classification", ref.who],
                      ["Catalogue", ref.catalogue],
                      ["Clinical Frequency", ref.freq],
                      ["CRyPTIC", ref.cryptic || "—"],
                    ].map(([k, v], i) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "9px 14px", borderBottom: i < 3 ? `1px solid ${T.borderLight}` : "none", fontSize: "11px" }}>
                        <span style={{ color: T.textSec }}>{k}</span>
                        <span style={{ fontWeight: 600, color: T.text, textAlign: "right", maxWidth: "60%" }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div>
                <div style={{ fontSize: "11px", fontWeight: 600, color: T.textSec, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Assay Parameters</div>
                <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
                  {[
                    ["Drug Class", r.drug],
                    ["Gene", r.gene],
                    ["Strategy", r.strategy],
                    ["PAM Sequence", `${r.pam}${r.pamVariant ? ` (${r.pamVariant})` : ""}${r.pamPenalty != null && r.pamPenalty < 1.0 ? ` — ${r.pamPenalty}× activity` : ""}`],
                    ["Spacer Length", `${(r.spacer || "").length} nt`],
                    ["GC Content", `${(r.gc * 100).toFixed(1)}%`],
                    ...(r.amplicon ? [["Amplicon Size", `${r.amplicon} bp`]] : []),
                    ["Synthetic Mismatch", r.hasSM ? `Yes (pos ${r.smPosition || "?"})` : "No"],
                    ["Off-targets", `${r.ot}`],
                  ].map(([k, v], i, arr) => (
                    <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "7px 14px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none", fontSize: "11px" }}>
                      <span style={{ color: T.textSec }}>{k}</span>
                      <span style={{ fontWeight: 600, color: T.text, fontFamily: MONO }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const CandidatesTab = ({ results, jobId, connected, scorer }) => {
  const mobile = useIsMobile();
  const [search, setSearch] = useState("");
  const defaultSort = "cnnCalibrated";
  const [sortKey, setSortKey] = useState(defaultSort);
  const [sortDir, setSortDir] = useState(-1);
  const [drugFilter, setDrugFilter] = useState("ALL");
  const [expanded, setExpanded] = useState(null);
  const [topKData, setTopKData] = useState({});
  const [topKLoading, setTopKLoading] = useState({});

  const buildLocalTopK = useCallback((targetLabel) => {
    // Build alternatives from other candidates targeting the same gene
    const target = results.find(r => r.label === targetLabel);
    if (!target) return { target_label: targetLabel, alternatives: [] };
    const sameGene = results.filter(r => r.gene === target.gene && r.label !== targetLabel);
    const alts = sameGene.slice(0, 5).map((r, i) => ({
      rank: i + 2, spacer_seq: r.spacer, score: +(r.cnnCalibrated ?? r.score).toFixed(3),
      discrimination: +(r.disc || 0).toFixed(1), has_primers: r.hasPrimers,
      tradeoff: r.score > target.score ? "Higher score" : r.disc > target.disc ? "Higher discrimination" : "Alternative spacer",
    }));
    return { target_label: targetLabel, selected: { rank: 1, spacer_seq: target.spacer, score: +(target.cnnCalibrated ?? target.score).toFixed(3), discrimination: +(target.disc || 0).toFixed(1) }, alternatives: alts };
  }, [results]);

  const loadTopK = useCallback((targetLabel) => {
    if (topKData[targetLabel] || topKLoading[targetLabel]) return;
    setTopKLoading(prev => ({ ...prev, [targetLabel]: true }));
    if (connected && jobId) {
      getTopK(jobId, targetLabel, 5).then(({ data }) => {
        if (data) setTopKData(prev => ({ ...prev, [targetLabel]: data }));
        else setTopKData(prev => ({ ...prev, [targetLabel]: buildLocalTopK(targetLabel) }));
        setTopKLoading(prev => ({ ...prev, [targetLabel]: false }));
      });
    } else {
      setTopKData(prev => ({ ...prev, [targetLabel]: buildLocalTopK(targetLabel) }));
      setTopKLoading(prev => ({ ...prev, [targetLabel]: false }));
    }
  }, [topKData, topKLoading, connected, jobId, buildLocalTopK]);

  const drugs = ["ALL", ...new Set(results.map((r) => r.drug))];

  const filtered = useMemo(() => {
    let arr = [...results];
    if (drugFilter !== "ALL") arr = arr.filter((r) => r.drug === drugFilter);
    if (search) {
      const q = search.toLowerCase();
      arr = arr.filter((r) => r.label.toLowerCase().includes(q) || r.gene.toLowerCase().includes(q) || r.spacer.toLowerCase().includes(q));
    }
    const getSortVal = (r) => {
      if (sortKey === "riskOverall") { const v = r.riskProfile?.overall; return v === "green" ? 2 : v === "amber" ? 1 : 0; }
      if (sortKey === "experimentalPriority") return r.experimentalPriority ?? 999;
      return r[sortKey] ?? 0;
    };
    arr.sort((a, b) => (getSortVal(a) > getSortVal(b) ? 1 : -1) * sortDir);
    return arr;
  }, [results, search, sortKey, sortDir, drugFilter]);

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir((d) => d * -1);
    else { setSortKey(key); setSortDir(-1); }
  };

  const hasCompassMl = scorer === "compass_ml" || results.some(r => r.mlScores?.some(m => (m.model_name || m.modelName) === "compass_ml"));
  const hasML = hasCompassMl || results.some(r => r.cnnCalibrated != null);

  const hasReadiness = filtered.some(r => r.readinessScore != null);
  const hasPamAdj = filtered.some(r => r.pamAdjusted != null && r.pamAdjusted !== r.cnnCalibrated);
  // Columns: #, Target, Drug, Spacer, Activity, PAM-adj, Disc, QC, Readiness
  const cols = [
    ...(hasReadiness ? [{ key: "experimentalPriority", label: "#", w: 36 }] : []),
    { key: "label", label: "Target", w: 140 },
    { key: "drug", label: "Drug", w: 54 },
    { key: "spacer", label: "Spacer", w: 200 },
    { key: "cnnCalibrated", label: "Activity", w: 66 },
    ...(hasPamAdj ? [{ key: "pamAdjusted", label: "PAM-adj", w: 66 }] : []),
    { key: "disc", label: "Disc", w: 64 },
    { key: "score", label: "QC", w: 48 },
    ...(hasReadiness ? [{ key: "readinessScore", label: "Readiness", w: 90 }] : []),
  ];

  // Hover state for spacer color reveal
  const [hoveredRow, setHoveredRow] = useState(null);

  return (
    <div>
      {/* Explainer box — blue */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", padding: mobile ? "14px" : "16px 20px", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <List size={16} color={T.primary} strokeWidth={1.8} />
          <span style={{ fontSize: "13px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING }}>Candidate Scoring</span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", color: T.primaryDark, lineHeight: 1.6 }}>
          <div><span style={{ color: T.primary }}>Activity</span> — Compass-ML predicted Cas12a on-target efficiency (0–1). <span style={{ color: T.primary }}>PAM-adj</span> = Activity × PAM penalty (actual signal strength).</div>
          <div><span style={{ color: T.primary }}>Disc</span> — MUT/WT fold-difference. ≥3× diagnostic-grade. {"<"}2× insufficient.</div>
          <div><span style={{ color: T.primary }}>QC</span> — biophysical heuristic (GC, homopolymer, off-target, self-comp). Sanity check, not a ranking score.</div>
          <div><span style={{ color: T.primary }}>Readiness</span> — composite score (0–100) combining efficiency, discrimination, primer coverage, safety, and GC content. ≥40 is assay-ready.</div>
          <div style={{ color: T.textSec }}>Click any row to expand full details, scored sequence, primers, and alternatives.</div>
        </div>
      </div>

      {/* Toolbar */}
      <div style={{ display: "flex", flexDirection: mobile ? "column" : "row", gap: "10px", marginBottom: "16px", alignItems: mobile ? "stretch" : "center" }}>
        <div style={{ position: "relative", flex: 1 }}>
          <Search size={14} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: T.textTer }} />
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search targets, genes, spacers…" style={{ width: "100%", padding: "9px 12px 9px 34px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "12px", outline: "none", boxSizing: "border-box" }} />
        </div>
        <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
          {drugs.map((d) => (
            <button key={d} onClick={() => setDrugFilter(d)} style={{
              padding: "6px 12px", borderRadius: "4px", border: `1px solid ${drugFilter === d ? T.text : T.border}`,
              background: drugFilter === d ? T.text : T.bg, color: drugFilter === d ? "#fff" : T.textSec,
              fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: FONT,
            }}>{d}</button>
          ))}
        </div>
      </div>

      {/* Candidates — cards on mobile, table on desktop */}
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
       {mobile ? (
        /* ── Mobile card layout — monochrome ── */
        <div>
          {filtered.map((r) => {
            const isExpanded = expanded === r.label;
            const scoreVal = r.cnnCalibrated ?? r.score;
            const discColor = r.gene === "IS6110" ? T.textTer : r.strategy === "Proximity" ? T.textSec : r.disc >= 3 ? T.success : r.disc >= 2 ? T.warning : T.danger;
            const riskLevel = r.riskProfile?.overall;
            return (
              <div key={r.label}>
                <div onClick={() => setExpanded(isExpanded ? null : r.label)} style={{ padding: "14px 16px", cursor: "pointer", borderBottom: isExpanded ? "none" : `1px solid ${T.borderLight}`, background: isExpanded ? T.bgSub : "transparent" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      {isExpanded ? <ChevronDown size={14} color={T.text} /> : <ChevronRight size={14} color={T.textTer} />}
                      <span style={{ fontWeight: 600, fontFamily: MONO, fontSize: "12px", color: T.text }}>{r.label}</span>
                      <span style={{ fontSize: "10px", color: T.textTer, fontFamily: MONO }}>{r.strategy === "Proximity" ? "P" : "D"}</span>
                    </div>
                    <span style={{ fontSize: "11px", color: T.textSec, fontFamily: MONO }}>{r.drug}</span>
                  </div>
                  <div style={{ display: "flex", gap: "16px", fontSize: "11px" }}>
                    <div>
                      <span style={{ color: T.textTer }}>Activity </span>
                      <span style={{ fontFamily: FONT, fontWeight: 600, color: T.text }}>{scoreVal.toFixed(3)}</span>
                    </div>
                    <div>
                      <span style={{ color: T.textTer }}>Disc </span>
                      <span style={{ fontFamily: FONT, fontWeight: 600, color: discColor }}>
                        {r.strategy === "Proximity" ? "AS-RPA" : r.gene === "IS6110" ? "N/A" : `${typeof r.disc === "number" ? r.disc.toFixed(1) : r.disc}×`}
                      </span>
                    </div>
                    {r.readinessScore != null && (
                      <div>
                        <span style={{ color: T.textTer }}>Ready </span>
                        <span style={{ fontFamily: FONT, fontWeight: 600, color: r.readinessScore >= 0.7 ? T.success : r.readinessScore >= 0.4 ? T.warning : T.danger }}>{(r.readinessScore * 100).toFixed(0)}</span>
                      </div>
                    )}
                  </div>
                </div>
                {isExpanded && (
                  <>
                    <CandidateAccordion r={r} onShowAlternatives={() => loadTopK(r.label)} />
                    {/* Top-K Alternatives inline */}
                    {topKLoading[r.label] && <div style={{ padding: "8px 16px", fontSize: "11px", color: T.textTer, background: T.bgSub }}><Loader2 size={12} style={{ animation: "spin 1s linear infinite", display: "inline-block", verticalAlign: "middle", marginRight: "4px" }} />Loading alternatives…</div>}
                    {topKData[r.label]?.alternatives && (
                      <div style={{ margin: "0 16px 16px", border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden", background: T.bg }}>
                        <div style={{ padding: "8px 12px", background: T.primaryLight, fontSize: "11px", fontWeight: 600, color: T.primaryDark, borderBottom: `1px solid ${T.border}` }}>Top-K Alternatives for {r.label}</div>
                        {topKData[r.label].alternatives.map((alt, ai) => (
                          <div key={ai} style={{ padding: "8px 12px", borderBottom: ai < topKData[r.label].alternatives.length - 1 ? `1px solid ${T.borderLight}` : "none", fontSize: "11px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <div><span style={{ fontFamily: FONT, fontWeight: 600 }}>#{alt.rank}</span> <Seq s={alt.spacer_seq?.slice(0, 16)} /></div>
                            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                              <span style={{ fontFamily: FONT }}>{alt.score}</span>
                              <span style={{ fontFamily: FONT }}>{r.strategy === "Proximity" ? <span style={{ fontSize: "10px", color: T.purple }}>AS-RPA</span> : `${alt.discrimination}×`}</span>
                              {alt.has_primers ? <Badge variant="success">P</Badge> : <Badge variant="danger">—</Badge>}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </div>
       ) : (
        /* ── Desktop table layout — monochrome by default, color encodes meaning ── */
        <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
          <thead>
            <tr style={{ background: T.bgSub }}>
              <th style={{ padding: "10px 8px", borderBottom: `1px solid ${T.border}`, width: 28 }} />
              {cols.map((c) => (
                <th key={c.key} onClick={() => toggleSort(c.key)} style={{ padding: "10px 12px", textAlign: "left", fontWeight: 600, color: T.textTer, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.border}`, cursor: "pointer", width: c.w, userSelect: "none" }}>
                  {c.label} {sortKey === c.key ? (sortDir > 0 ? "↑" : "↓") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => {
              const isExpanded = expanded === r.label;
              const isHov = hoveredRow === r.label;
              const riskLevel = r.riskProfile?.overall;
              const riskBorderColor = riskLevel === "red" ? T.danger : riskLevel === "amber" ? T.warning : "transparent";
              const discColor = r.gene === "IS6110" ? T.textTer : r.strategy === "Proximity" ? T.textSec : r.disc >= 3 ? T.success : r.disc >= 2 ? T.warning : T.danger;
              const activityVal = r.cnnCalibrated ?? r.score;
              const pamAdjVal = r.pamAdjusted ?? activityVal;
              const stratIcon = r.strategy === "Proximity" ? "P" : "D";
              return (
                <React.Fragment key={r.label}>
                  <tr
                    onClick={() => setExpanded(isExpanded ? null : r.label)}
                    onMouseEnter={() => setHoveredRow(r.label)}
                    onMouseLeave={() => setHoveredRow(null)}
                    style={{
                      cursor: "pointer",
                      borderBottom: `1px solid ${isExpanded ? "transparent" : T.borderLight}`,
                      background: isExpanded ? T.bgSub : isHov ? T.bgHover : "transparent",
                      transition: "background 0.15s, transform 0.15s",
                      ...(isHov && !isExpanded ? { background: T.bgSub } : {}),
                    }}>
                    <td style={{ padding: "10px 8px", textAlign: "center" }}>
                      {isExpanded ? <ChevronDown size={14} color={T.text} /> : <ChevronRight size={14} color={T.textTer} />}
                    </td>
                    {/* # (priority) */}
                    {hasReadiness && <td style={{ padding: "10px 6px", textAlign: "center" }}>{r.experimentalPriority != null && <PriorityBadge rank={r.experimentalPriority} />}</td>}
                    {/* Target — includes strategy icon */}
                    <td style={{ padding: "10px 12px" }}>
                      <span style={{ fontWeight: 600, fontFamily: MONO, fontSize: "11px", color: T.text }}>{r.label}</span>
                      <span style={{ fontSize: "9px", color: T.textTer, marginLeft: "6px", fontFamily: MONO, fontWeight: 500 }}>{stratIcon}</span>
                    </td>
                    {/* Drug — plain text, monochrome */}
                    <td style={{ padding: "10px 12px", fontFamily: MONO, fontSize: "11px", color: T.textSec, fontWeight: 500 }}>{r.drug}</td>
                    {/* Spacer — muted monospace, colored nucleotides on hover only */}
                    <td style={{ padding: "10px 12px" }}>
                      {isHov ? (
                        <span style={{ fontFamily: MONO, fontSize: "11px", letterSpacing: "1px" }}>
                          {r.spacer?.slice(0, 24).split("").map((c, i) => (
                            <span key={i} style={{ color: c === "A" ? "#059669" : c === "T" ? "#DC2626" : c === "G" ? "#D97706" : "#4338CA", fontWeight: 500 }}>{c}</span>
                          ))}
                        </span>
                      ) : (
                        <span style={{ fontFamily: MONO, fontSize: "11px", letterSpacing: "1px", color: T.textTer }}>{r.spacer?.slice(0, 24)}</span>
                      )}
                    </td>
                    {/* Activity — Compass-ML calibrated, colored */}
                    <td style={{ padding: "10px 12px", fontFamily: FONT, fontWeight: 600, fontSize: "11px", color: activityVal > 0.7 ? T.primary : activityVal > 0.5 ? T.warning : T.danger }}>{activityVal.toFixed(3)}</td>
                    {/* PAM-adj — activity × PAM penalty, dimmer */}
                    {hasPamAdj && <td style={{ padding: "10px 12px", fontFamily: FONT, fontSize: "11px", color: T.textSec }}>{pamAdjVal.toFixed(3)}{r.pamPenalty != null && r.pamPenalty < 1.0 ? <span style={{ fontSize: "9px", color: T.textTer, marginLeft: "2px" }}>({r.pamPenalty}×)</span> : ""}</td>}
                    {/* Disc — colored by threshold (the one meaningful color) */}
                    <td style={{ padding: "10px 12px", fontFamily: FONT, fontWeight: 600, fontSize: "11px", color: r.pamDisrupted ? "#7c3aed" : discColor }}>
                      {r.pamDisrupted ? <span style={{ display: "inline-block", padding: "1px 6px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, background: "#7c3aed18", color: "#7c3aed", border: "1px solid #7c3aed33", letterSpacing: "0.04em" }}>PAM</span> : r.gene === "IS6110" ? <span style={{ fontSize: "10px" }}>N/A</span> : r.strategy === "Proximity" ? <span style={{ fontSize: "10px" }}>AS-RPA</span> : `${typeof r.disc === "number" ? r.disc.toFixed(1) : r.disc}×`}
                    </td>
                    {/* QC — heuristic, small gray */}
                    <td style={{ padding: "10px 8px", fontFamily: FONT, fontSize: "10px", color: T.textTer }}>{r.score.toFixed(2)}</td>
                    {/* Readiness — gradient fill (keep strongest visual element) */}
                    {hasReadiness && (
                      <td style={{ padding: "10px 8px" }}>
                        {r.readinessScore != null ? (
                          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <div style={{ flex: 1, height: "6px", borderRadius: "3px", background: T.borderLight, overflow: "hidden", minWidth: "40px" }}>
                              <div style={{ height: "100%", width: `${r.readinessScore * 100}%`, borderRadius: "3px",
                                background: r.readinessScore >= 0.7 ? T.success : r.readinessScore >= 0.4 ? T.warning : T.danger,
                              }} />
                            </div>
                            <span style={{ fontFamily: FONT, fontWeight: 600, fontSize: "11px", minWidth: "24px", color: r.readinessScore >= 0.7 ? T.success : r.readinessScore >= 0.4 ? T.warning : T.danger }}>
                              {(r.readinessScore * 100).toFixed(0)}
                            </span>
                          </div>
                        ) : "—"}
                      </td>
                    )}
                  </tr>
                  {isExpanded && (
                    <tr>
                      <td colSpan={cols.length + 1} style={{ padding: 0 }}>
                        <CandidateAccordion r={r} onShowAlternatives={() => loadTopK(r.label)} />
                        {/* Top-K Alternatives inline */}
                        {topKLoading[r.label] && <div style={{ padding: "8px 24px", fontSize: "11px", color: T.textTer, background: T.bgSub }}><Loader2 size={12} style={{ animation: "spin 1s linear infinite", display: "inline-block", verticalAlign: "middle", marginRight: "4px" }} />Loading alternatives…</div>}
                        {topKData[r.label]?.alternatives && (
                          <div style={{ margin: "0 24px 16px", border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden", background: T.bg }}>
                            <div style={{ padding: "10px 14px", background: T.bgSub, fontSize: "11px", fontWeight: 600, color: T.text, borderBottom: `1px solid ${T.border}`, display: "flex", alignItems: "center", gap: "6px" }}>
                              <Layers size={12} /> Top-K Alternatives for {r.label}
                            </div>
                            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
                              <thead>
                                <tr style={{ background: T.bgSub }}>
                                  {["Rank", "Spacer", "Score", "Disc", "Primers", "Tradeoff"].map(h => (
                                    <th key={h} style={{ padding: "6px 12px", textAlign: "left", fontWeight: 600, color: T.textTer, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.borderLight}` }}>{h}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {topKData[r.label].alternatives.map((alt, ai) => (
                                  <tr key={ai} style={{ borderBottom: ai < topKData[r.label].alternatives.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
                                    <td style={{ padding: "7px 12px", fontFamily: FONT, fontWeight: 600, color: T.textSec }}>#{alt.rank}</td>
                                    <td style={{ padding: "7px 12px", fontFamily: MONO, fontSize: "11px", color: T.textTer }}>{alt.spacer_seq?.slice(0, 20)}</td>
                                    <td style={{ padding: "7px 12px", fontFamily: FONT, fontWeight: 600, color: T.text }}>{alt.score}</td>
                                    <td style={{ padding: "7px 12px", fontFamily: FONT, fontWeight: 600, color: r.strategy === "Proximity" ? T.textSec : alt.discrimination >= 3 ? T.success : alt.discrimination >= 2 ? T.warning : T.danger }}>{r.strategy === "Proximity" ? <span style={{ fontSize: "10px" }}>AS-RPA</span> : `${alt.discrimination}×`}</td>
                                    <td style={{ padding: "7px 12px", fontFamily: FONT, fontSize: "10px", color: alt.has_primers ? T.success : T.textTer }}>{alt.has_primers ? "Yes" : "—"}</td>
                                    <td style={{ padding: "7px 12px", fontSize: "10px", color: T.textSec }}>{alt.tradeoff || "—"}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
        </div>
       )}
        <div style={{ padding: "12px 16px", fontSize: "11px", color: T.textTer, borderTop: `1px solid ${T.border}`, background: T.bgSub }}>
          Showing {filtered.length} of {results.length} candidates
        </div>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════
// Cross-Reactivity Matrix (14×14 heatmap)
// ═══════════════════════════════════════════════════════════════
const CrossReactivityMatrix = () => {
  const [hovCell, setHovCell] = useState(null);
  const [hovPos, setHovPos] = useState({ x: 0, y: 0 });
  const [showPamFilter, setShowPamFilter] = useState(false);

  const data = MOCK_CROSS_REACTIVITY;
  const labels = CROSS_REACTIVITY_LABELS;
  const N = labels.length;
  const cellSize = 28;
  const gap = 1;
  const labelW = 85;
  const totalW = labelW + N * (cellSize + gap);
  const totalH = labelW + N * (cellSize + gap);

  const getCellColor = (pair) => {
    if (!pair) return "#1e293b"; // diagonal
    if (showPamFilter && !pair.pam_valid) return "#F8F8F6";
    if (pair.risk === "none") return "#F5F3EE";
    if (pair.risk === "low") return "#FEF3C7";
    if (pair.risk === "medium") return "#FB923C";
    if (pair.risk === "high") return "#EF4444";
    return "#F5F3EE";
  };

  const getOnTargetScore = (idx) => {
    const label = labels[idx];
    const r = RESULTS.find(x => x.label === label);
    return r ? ((r.cnnCalibrated ?? r.score) || 0).toFixed(2) : "—";
  };

  const pairMap = {};
  data.matrix.forEach(m => { pairMap[`${m.sourceIdx}_${m.targetIdx}`] = m; });

  // Group separators
  const groupBounds = [];
  let prevGroup = -1;
  CROSS_REACTIVITY_DRUG_GROUPS.forEach((g, i) => {
    if (g !== prevGroup) { groupBounds.push(i); prevGroup = g; }
  });

  return (
    <div>
      {/* Toggle */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12, alignItems: "center", justifyContent: "center" }}>
        <button onClick={() => setShowPamFilter(!showPamFilter)} style={{
          fontSize: 10, fontWeight: 600, padding: "4px 12px", borderRadius: 4,
          border: `1px solid ${showPamFilter ? T.primary : T.border}`, cursor: "pointer", fontFamily: MONO,
          background: showPamFilter ? `${T.primary}15` : T.bg, color: showPamFilter ? T.primary : T.textSec,
        }}>
          {showPamFilter ? "✓ " : ""}PAM-only filter
        </button>
        <span style={{ fontSize: 10, color: T.textTer }}>
          {showPamFilter ? "Showing only pairs with valid PAM site on off-target amplicon" : "Showing all pairs"}
        </span>
      </div>

      {/* Heatmap */}
      <div style={{ overflowX: "auto", marginBottom: 16, position: "relative", display: "flex", justifyContent: "center" }}>
        <svg width={totalW + 10} height={totalH + 10} style={{ display: "block", maxWidth: "100%" }}
          onMouseLeave={() => setHovCell(null)}>
          {/* Column headers (rotated 45°) */}
          {labels.map((l, j) => (
            <text key={`ch-${j}`} x={labelW + j * (cellSize + gap) + cellSize / 2} y={labelW - 6}
              fontSize="7" fill={T.textSec} fontFamily={MONO} textAnchor="end"
              transform={`rotate(-45, ${labelW + j * (cellSize + gap) + cellSize / 2}, ${labelW - 6})`}>
              {l}
            </text>
          ))}
          {/* Row headers */}
          {labels.map((l, i) => (
            <text key={`rh-${i}`} x={labelW - 4} y={labelW + i * (cellSize + gap) + cellSize / 2 + 3}
              fontSize="7" fill={T.textSec} fontFamily={MONO} textAnchor="end">
              {l}
            </text>
          ))}
          {/* Cells */}
          {labels.map((_, i) => labels.map((_, j) => {
            const pair = i === j ? null : pairMap[`${i}_${j}`];
            const x = labelW + j * (cellSize + gap);
            const y = labelW + i * (cellSize + gap);
            const color = getCellColor(pair);
            return (
              <g key={`c-${i}-${j}`}
                onMouseEnter={(e) => {
                  setHovCell(i === j ? { diag: true, idx: i } : pair);
                  const rect = e.currentTarget.closest("svg").getBoundingClientRect();
                  setHovPos({ x: e.clientX - rect.left + 10, y: e.clientY - rect.top - 10 });
                }}
                onMouseMove={(e) => {
                  const rect = e.currentTarget.closest("svg").getBoundingClientRect();
                  setHovPos({ x: e.clientX - rect.left + 10, y: e.clientY - rect.top - 10 });
                }}>
                <rect x={x} y={y} width={cellSize} height={cellSize} fill={color} rx={2}
                  stroke={hovCell && !hovCell.diag && hovCell.sourceIdx === i && hovCell.targetIdx === j ? T.text : "none"}
                  strokeWidth={1} />
                {i === j && (
                  <text x={x + cellSize / 2} y={y + cellSize / 2 + 3} fontSize="6" fill="#fff" textAnchor="middle" fontFamily={MONO}>
                    {getOnTargetScore(i)}
                  </text>
                )}
              </g>
            );
          }))}
          {/* Group separator lines */}
          {groupBounds.slice(1).map((b, i) => (
            <g key={`sep-${i}`}>
              <line x1={labelW + b * (cellSize + gap) - 1} y1={labelW} x2={labelW + b * (cellSize + gap) - 1} y2={labelW + N * (cellSize + gap)} stroke="#fff" strokeWidth="2" />
              <line x1={labelW} y1={labelW + b * (cellSize + gap) - 1} x2={labelW + N * (cellSize + gap)} y2={labelW + b * (cellSize + gap) - 1} stroke="#fff" strokeWidth="2" />
            </g>
          ))}
        </svg>

        {/* Tooltip */}
        {hovCell && (
          <div style={{
            position: "absolute", left: hovPos.x, top: hovPos.y, pointerEvents: "none", zIndex: 20,
            background: T.bg, border: `1px solid ${T.border}`, borderRadius: 4, padding: "8px 12px",
            boxShadow: "0 1px 3px rgba(0,0,0,0.06)", maxWidth: 280,
          }}>
            {hovCell.diag ? (
              <div style={{ fontSize: 10, fontFamily: MONO, color: T.text }}>
                <strong>{labels[hovCell.idx]}</strong> — on-target (S_eff = {getOnTargetScore(hovCell.idx)})
              </div>
            ) : (
              <>
                <div style={{ fontSize: 10, fontWeight: 600, color: T.text, fontFamily: MONO }}>
                  {hovCell.source} crRNA → {hovCell.target} amplicon
                </div>
                <div style={{ fontSize: 9, color: T.textSec, marginTop: 3, lineHeight: 1.6, fontFamily: MONO }}>
                  Activity: {(hovCell.activity * 100).toFixed(2)}% of on-target (<span style={{ fontWeight: 600, color: hovCell.risk === "none" ? T.success : hovCell.risk === "low" ? "#D97706" : hovCell.risk === "medium" ? "#EA580C" : T.danger }}>{hovCell.risk.toUpperCase()}</span>)<br />
                  Mismatches: {hovCell.mismatches}<br />
                  PAM valid: {hovCell.pam_valid ? "yes" : "no"}
                  {hovCell.note && <><br /><span style={{ color: T.primary }}>{hovCell.note}</span></>}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Color legend */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap", alignItems: "center", justifyContent: "center" }}>
        {[
          { label: "On-target", color: "#1e293b" },
          { label: "None (<1%)", color: "#F5F3EE" },
          { label: "Low (1–5%)", color: "#FEF3C7" },
          { label: "Medium (5–15%)", color: "#FB923C" },
          { label: "High (>15%)", color: "#EF4444" },
        ].map(l => (
          <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 9, color: T.textSec, fontFamily: MONO }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: l.color, display: "inline-block", border: `1px solid ${T.borderLight}` }} />
            {l.label}
          </div>
        ))}
      </div>

      {/* Summary panel */}
      <div style={{ background: T.bgSub, border: `1px solid ${T.border}`, borderRadius: 4, padding: "16px 20px", marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: T.text, fontFamily: MONO, lineHeight: 1.8 }}>
          <strong>{data.n_pairs} / {data.n_pairs}</strong> pairs tested<br />
          <strong>{data.none_count}</strong> pairs: no cross-reactivity (&lt; 1%)<br />
          <strong>{data.same_gene_pairs.length}</strong> pairs: low{"\u2013"}medium cross-reactivity (same-gene overlapping amplicons)
        </div>
        <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 4, fontSize: 10, fontWeight: 600,
            background: data.panel_safe ? "#ECFDF5" : "#FEF2F2",
            color: data.panel_safe ? "#059669" : "#DC2626",
            fontFamily: MONO,
          }}>
            {data.panel_safe ? "\u2705 SAFE" : "\u26A0 REVIEW"} for spatially multiplexed electrode array
          </span>
        </div>
        {data.same_gene_pairs.length > 0 && (
          <div style={{ marginTop: 10, fontSize: 10, color: T.textSec, lineHeight: 1.7 }}>
            <strong>Same-gene pairs with residual cross-reactivity:</strong>
            {data.same_gene_pairs.filter(p => p.sourceIdx < p.targetIdx || !data.same_gene_pairs.find(q => q.sourceIdx === p.targetIdx && q.targetIdx === p.sourceIdx && q.sourceIdx < q.targetIdx)).map(p => (
              <div key={`${p.source}-${p.target}`} style={{ fontFamily: MONO, marginLeft: 8, fontSize: 9 }}>
                {"\u2022"} {p.source} {"\u2194"} {p.target}: {(p.activity * 100).toFixed(1)}% ({p.risk.toUpperCase()}) {"\u2014"} {p.note}
              </div>
            ))}
            <div style={{ marginTop: 6, fontSize: 9, color: T.textTer }}>
              Managed by spatial separation: each crRNA contacts only its own amplicon within its physically isolated detection zone.
            </div>
          </div>
        )}
      </div>

      {/* Interpretation */}
      <div style={{ background: T.bg, border: `1px solid ${T.borderLight}`, borderRadius: 4, padding: "12px 16px", fontSize: 10, color: T.textSec, lineHeight: 1.7 }}>
        <div style={{ fontWeight: 600, color: T.text, marginBottom: 4 }}>Interpretation</div>
        Cross-reactivity is assessed by scoring each crRNA against all non-self amplicons in the 14-target panel.
        For the paper-based spatially multiplexed electrode array (Bezinge et al., <em>Adv. Mater.</em> 2023),
        each detection zone is isolated by wax-printed hydrophobic barriers \u2014 each crRNA physically contacts only its own zone's amplicon, making inter-zone cross-reactivity impossible.
        <br /><br />
        This analysis validates that same-gene targets with overlapping amplicons (e.g., rpoB_S531L and rpoB_H526Y, which share the rpoB RRDR amplicon) do not produce false positives even in a hypothetical shared-solution format, and identifies any targets where crRNA redesign would improve panel orthogonality.
        <br /><br />
        <strong>PAM-level filtering:</strong> Cas12a requires a 5\u2032-TTTV PAM for activation. Off-target sites without a valid PAM are scored as zero regardless of spacer complementarity, as PAM recognition is an absolute prerequisite for R-loop initiation (Suea-Ngam et al., <em>Chem. Sci.</em> 2021, Fig. 4).
      </div>
    </div>
  );
};

const DiscriminationTab = ({ results }) => {
  const mobile = useIsMobile();
  const nonControl = results.filter((r) => r.disc < 900);
  const directCands = nonControl.filter((r) => r.strategy === "Direct");
  const proximityCands = nonControl.filter((r) => r.strategy === "Proximity");
  const data = directCands.map((r) => ({ name: r.label, disc: +r.disc, score: r.score, drug: r.drug }));
  const excellent = directCands.filter((r) => r.disc >= 10).length;
  const good = directCands.filter((r) => r.disc >= 3 && r.disc < 10).length;
  const acceptable = directCands.filter((r) => r.disc >= 2 && r.disc < 3).length;
  const insufficient = directCands.filter((r) => r.disc < 2).length;

  return (
    <div>
      {/* Blue explainer */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", padding: mobile ? "16px" : "20px 24px", marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <TrendingUp size={16} color={T.primary} strokeWidth={1.8} />
          <span style={{ fontSize: "13px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING }}>Can this guide tell apart resistant from normal?</span>
        </div>
        <div style={{ fontSize: "13px", color: T.primaryDark, lineHeight: 1.7 }}>
          <p style={{ margin: "0 0 8px" }}>
            Each crRNA is designed to perfectly match the resistance mutation (MUT). When it encounters normal/wildtype DNA (WT),
            mismatches at the mutation site reduce Cas12a cleavage. The discrimination ratio is how many times stronger the signal
            is on resistant DNA versus normal DNA — for example, "5×" means the guide produces 5 times more signal on a resistant sample.
          </p>
          <p style={{ margin: 0 }}>
            A ratio ≥ 3× is considered diagnostic-grade — reliable enough for clinical use with electrochemical (SWV) or lateral-flow readout.
            ≥ 2× is the minimum for any detection method. Below 2× the guide cannot reliably distinguish resistant from susceptible bacteria
            and requires synthetic mismatch enhancement.
          </p>
        </div>
      </div>

      {/* Threshold cards — glass style */}
      <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr 1fr" : "repeat(4, 1fr)", gap: "10px", marginBottom: "24px" }}>
        {[
          { label: "Excellent", val: "≥ 10×", count: excellent, color: "#16a34a", desc: "Single-plex clinical use. Robust across sample types." },
          { label: "Good", val: "≥ 3×", count: good, color: T.primary, desc: "Multiplex panel. Electrochemical and lateral flow." },
          { label: "Acceptable", val: "≥ 2×", count: acceptable, color: "#d97706", desc: "Requires confirmatory readout or dual-target." },
          { label: "Insufficient", val: "< 2×", count: insufficient, color: "#dc2626", desc: "Synthetic mismatch enhancement needed." },
        ].map(t => (
          <div key={t.label} style={{ background: T.bg, borderRadius: "4px", padding: "16px 18px", border: `1px solid ${T.border}` }}>
            <div style={{ fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>{t.label}</div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <div style={{ fontSize: "11px", fontWeight: 600, color: T.textSec }}>{t.val}</div>
              <div style={{ fontSize: "18px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{t.count}</div>
            </div>
            <div style={{ fontSize: "10px", color: T.textTer, marginTop: "6px" }}>{t.desc}</div>
            <div style={{ marginTop: "8px", height: "3px", borderRadius: "2px", background: T.borderLight, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${directCands.length ? (t.count / directCands.length) * 100 : 0}%`, borderRadius: "2px", background: t.color, transition: "width 0.3s" }} />
            </div>
          </div>
        ))}
      </div>

      {/* Discrimination chart — horizontal lollipop */}
      {(() => {
        const DRUG_DC = { RIF: "#1E3A5F", INH: "#4338CA", EMB: "#059669", FQ: "#DC2626", AG: "#3730A3", PZA: "#059669", OTHER: "#9CA3AF" };
        const sorted = [...directCands].sort((a, b) => b.disc - a.disc);
        const discChart = sorted.map((r) => ({ name: r.label, disc: +r.disc, score: r.score, drug: r.drug }));
        const diagGrade = discChart.filter(d => d.disc >= 3).length;
        const maxDisc = Math.max(...discChart.map(d => d.disc), 12);
        const thresholds = [
          { val: 2, label: "2× min", color: T.danger },
          { val: 3, label: "3× diagnostic", color: T.warning },
          { val: 10, label: "10× excellent", color: T.success },
        ].filter(t => t.val <= maxDisc);
        return (
          <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: mobile ? "20px" : "28px 32px", marginBottom: "24px" }}>
            <div style={{ marginBottom: "20px" }}>
              <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Discrimination Ratio — Direct Detection</div>
              <div style={{ fontSize: "11px", color: T.textSec, marginTop: "3px" }}>
                {directCands.length} candidates using crRNA mismatch discrimination. Sorted highest to lowest.
              </div>
              <div style={{ fontSize: "10px", color: T.textTer, marginTop: "2px", display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: directCands.some(r => r.discMethod === "neural") ? "#3b82f6" : directCands.some(r => (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature") ? "#22c55e" : T.warning }} />
                {directCands.some(r => r.discMethod === "neural")
                  ? "Predicted by Compass-ML neural discrimination head (multi-task, 235K params, trained on 6,136 EasyDesign pairs)"
                  : directCands.some(r => (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature")
                  ? "Predicted by learned model (XGBoost on 18 thermodynamic features, trained on 6,136 EasyDesign pairs)"
                  : "Predicted by heuristic model (position sensitivity \u00D7 mismatch destabilisation)"
                }
              </div>
            </div>
            {/* Horizontal lollipop chart */}
            <div style={{ position: "relative", paddingLeft: "120px", paddingRight: "52px" }}>
              {/* Threshold vertical lines */}
              {thresholds.map(t => {
                const pct = (t.val / maxDisc) * 100;
                return (
                  <div key={t.val} style={{ position: "absolute", left: `calc(120px + (100% - 172px) * ${pct / 100})`, top: 0, bottom: 0, width: 0, borderLeft: `1.5px dashed ${t.color}33`, zIndex: 0, pointerEvents: "none" }}>
                    <span style={{ position: "absolute", top: -16, left: "50%", transform: "translateX(-50%)", fontSize: "9px", fontWeight: 600, color: `${t.color}99`, whiteSpace: "nowrap" }}>{t.label}</span>
                  </div>
                );
              })}
              {/* Rows */}
              <div style={{ display: "flex", flexDirection: "column", gap: "2px", position: "relative", zIndex: 1, paddingTop: "10px" }}>
                {discChart.map((d) => {
                  const pct = Math.min((d.disc / maxDisc) * 100, 100);
                  const status = d.disc >= 10 ? "excellent" : d.disc >= 3 ? "good" : d.disc >= 2 ? "acceptable" : "insufficient";
                  const statusColor = status === "excellent" ? T.success : status === "good" ? T.primary : status === "acceptable" ? T.warning : T.danger;
                  return (
                    <div key={d.name} style={{ display: "flex", alignItems: "center", height: "30px", marginLeft: "-120px", marginRight: "-52px" }}>
                      <div style={{ width: "120px", flexShrink: 0, fontSize: "10px", fontFamily: MONO, fontWeight: 600, color: T.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", paddingRight: "12px", textAlign: "right" }}>{d.name}</div>
                      <div style={{ flex: 1, position: "relative", height: "30px", display: "flex", alignItems: "center" }}>
                        {/* Track line */}
                        <div style={{ position: "absolute", left: 0, right: 0, height: "1px", background: T.borderLight }} />
                        {/* Gradient stem */}
                        <div style={{ position: "absolute", left: 0, width: `${pct}%`, height: "3px", borderRadius: "1.5px",
                          background: `${statusColor}44`,
                        }} />
                        {/* Dot */}
                        <div style={{
                          position: "absolute", left: `${pct}%`,
                          width: "14px", height: "14px", borderRadius: "50%",
                          background: statusColor,
                          border: "2px solid #fff",
                          boxShadow: "none",
                          transform: "translateX(-7px)",
                          transition: "transform 0.15s ease",
                        }} />
                      </div>
                      {/* Value */}
                      <div style={{ width: "52px", flexShrink: 0, textAlign: "right", fontSize: "11px", fontFamily: FONT, fontWeight: 600, color: statusColor }}>
                        {d.disc.toFixed(1)}×
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            {/* Legend */}
            <div style={{ display: "flex", alignItems: "center", gap: "16px", marginTop: "16px", flexWrap: "wrap", paddingTop: "12px", borderTop: `1px solid ${T.borderLight}` }}>
              {[...new Set(directCands.map(r => r.drug))].map(d => (
                <div key={d} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: DRUG_DC[d] || DRUG_DC.OTHER }} />
                  <span style={{ fontSize: "10px", color: T.textSec, fontWeight: 500 }}>{d}</span>
                </div>
              ))}
              <div style={{ width: "1px", height: "12px", background: T.borderLight }} />
              {[{ c: T.success, l: "≥ 10× Excellent" }, { c: T.primary, l: "≥ 3× Good" }, { c: T.warning, l: "≥ 2× Acceptable" }, { c: T.danger, l: "< 2× Insufficient" }].map(s => (
                <div key={s.l} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <div style={{ width: 8, height: 2, borderRadius: 1, background: s.c }} />
                  <span style={{ fontSize: "9px", color: T.textTer }}>{s.l}</span>
                </div>
              ))}
            </div>
            {(() => {
              const bestDisc = discChart[0];
              const worstDisc = discChart[discChart.length - 1];
              const below2 = discChart.filter(d => d.disc < 2);
              const avgDisc = discChart.length ? +(discChart.reduce((a, d) => a + d.disc, 0) / discChart.length).toFixed(1) : 0;
              return (
                <div style={{ marginTop: "14px", padding: "12px 16px", background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", fontSize: "11px", color: T.textSec, lineHeight: 1.7 }}>
                  <strong style={{ color: T.primary }}>Interpretation:</strong> {diagGrade}/{directCands.length} candidates reach diagnostic-grade (≥ 3×), panel avg {avgDisc}×{directCands.some(r => r.discMethod === "neural") ? " (neural disc head)" : directCands.some(r => (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature") ? " (learned model)" : " (heuristic)"}.
                  {bestDisc ? ` Highest: ${bestDisc.name} at ${bestDisc.disc.toFixed(1)}× — likely a seed-region mismatch (positions 1–8).` : ""}
                  {worstDisc ? ` Lowest: ${worstDisc.name} at ${worstDisc.disc.toFixed(1)}×${worstDisc.disc < 2 ? " — insufficient for any detection method, SM enhancement required." : worstDisc.disc < 3 ? " — acceptable but not diagnostic-grade." : "."}` : ""}
                  {below2.length > 0 ? ` ${below2.length} candidate${below2.length > 1 ? "s" : ""} (${below2.map(d => d.name).slice(0, 3).join(", ")}${below2.length > 3 ? "…" : ""}) fall below the 2× minimum — these have PAM-distal mismatches and require synthetic mismatch engineering.` : " All candidates meet the 2× minimum detection threshold."}
                  {excellent > 0 ? ` ${excellent} candidate${excellent > 1 ? "s" : ""} ${excellent > 1 ? "achieve" : "achieves"} excellent (≥ 10×) discrimination, suitable for lateral-flow deployment.` : ""}
                </div>
              );
            })()}
          </div>
        );
      })()}

      {/* Ranking table — Direct only */}
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden", marginBottom: "24px" }}>
        <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, padding: "16px 20px", borderBottom: `1px solid ${T.border}` }}>Discrimination Ranking — Direct Detection</div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
          <thead>
            <tr style={{ background: T.bgSub }}>
              {["Rank", "Target", "Drug", "Discrimination", "Model", "Activity", "Status"].map(h => (
                <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 600, color: T.textSec, borderBottom: `1px solid ${T.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...directCands].sort((a, b) => b.disc - a.disc).map((r, i) => (
              <tr key={r.label} style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: T.textTer }}>{i + 1}</td>
                <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600, fontSize: "11px" }}>{r.label}</td>
                <td style={{ padding: "10px 14px" }}><DrugBadge drug={r.drug} /></td>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: r.pamDisrupted ? "#7c3aed" : r.disc >= 3 ? T.success : r.disc >= 2 ? T.warning : T.danger }}>
                  {r.pamDisrupted ? <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}><span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, background: "#7c3aed18", color: "#7c3aed", border: "1px solid #7c3aed33" }}>PAM \u221e</span></span> : <>{typeof r.disc === "number" ? r.disc.toFixed(1) : r.disc}×</>}
                </td>
                <td style={{ padding: "10px 14px", fontSize: "10px", color: r.pamDisrupted ? "#7c3aed" : r.discMethod === "neural" ? "#3b82f6" : (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature" ? T.success : T.textTer }}>
                  {r.pamDisrupted ? "PAM gating" : r.discMethod === "neural" ? "Neural" : (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature" ? "Learned" : "Heuristic"}
                </td>
                <td style={{ padding: "10px 14px", fontFamily: FONT }}>{(r.cnnCalibrated ?? r.score).toFixed(3)}</td>
                <td style={{ padding: "10px 14px" }}>
                  {r.pamDisrupted ? (
                    <Badge variant="success">PAM-disrupted</Badge>
                  ) : (
                    <Badge variant={r.disc >= 3 ? "success" : r.disc >= 2 ? "warning" : "danger"}>
                      {r.disc >= 10 ? "Excellent" : r.disc >= 3 ? "Good" : r.disc >= 2 ? "Acceptable" : "Insufficient"}
                    </Badge>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Proximity / AS-RPA section */}

      {proximityCands.length > 0 && (
        <div style={{ background: T.bg, border: `1px solid ${T.purple}33`, borderRadius: "4px", overflow: "hidden" }}>
          <div style={{ fontSize: "14px", fontWeight: 600, color: T.purple, fontFamily: HEADING, padding: "16px 20px", borderBottom: `1px solid ${T.purple}33` }}>
            AS-RPA Discrimination — Proximity Detection
            <span style={{ fontSize: "11px", fontWeight: 400, color: T.textTer, marginLeft: "10px" }}>{proximityCands.length} candidates (primer-based discrimination)</span>
          </div>
          <div style={{ padding: "16px 20px 8px", fontSize: "12px", color: T.textSec, lineHeight: 1.6 }}>
            These candidates use <strong>allele-specific RPA primers</strong> for discrimination — the crRNA binds outside the mutation site.
            Discrimination is provided by preferential primer extension on the mutant template.
            {proximityCands.some(r => r.asrpaDiscrimination) && (
              <span> Thermodynamic estimates below are based on 3′ terminal mismatch identity and penultimate mismatch design.</span>
            )}
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
            <thead>
              <tr style={{ background: T.bgSub }}>
                {["Target", "Drug", "Distance", "Activity", "3′ Mismatch", "Penult. MM", "Disc. Ratio", "Block", "Primers"].map(h => (
                  <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 600, color: T.textSec, borderBottom: `1px solid ${T.borderLight}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {proximityCands.map((r) => {
                const d = r.asrpaDiscrimination;
                const blockColor = d?.block_class === "strong" ? T.success : d?.block_class === "moderate" ? T.warning : T.danger;
                return (
                  <tr key={r.label} style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                    <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600, fontSize: "11px" }}>{r.label}</td>
                    <td style={{ padding: "10px 14px" }}><DrugBadge drug={r.drug} /></td>
                    <td style={{ padding: "10px 14px", fontFamily: FONT, color: T.purple }}>{r.proximityDistance ? `${r.proximityDistance} bp` : "—"}</td>
                    <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: (r.cnnCalibrated ?? r.score) >= 0.7 ? T.success : (r.cnnCalibrated ?? r.score) >= 0.4 ? T.warning : T.danger }}>{(r.cnnCalibrated ?? r.score).toFixed(3)}</td>
                    <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600 }}>{d?.terminal_mismatch || "—"}</td>
                    <td style={{ padding: "10px 14px", fontSize: "11px" }}>{d ? (d.has_penultimate_mm ? <span style={{ color: T.success, fontWeight: 600 }}>Yes</span> : <span style={{ color: T.textTer }}>No</span>) : "—"}</td>
                    <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: d ? (d.block_class === "none" ? T.danger : d.disc_ratio >= 50 ? T.success : d.disc_ratio >= 10 ? T.warning : T.danger) : T.textTer }}>
                      {d ? (d.block_class === "none" ? "1× (WC)" : d.disc_ratio >= 100 ? "≥100×" : `${d.disc_ratio.toFixed(0)}×`) : "—"}
                    </td>
                    <td style={{ padding: "10px 14px" }}>
                      {d ? (d.block_class === "none"
                        ? <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, background: "#FEF2F2", color: T.danger, textTransform: "uppercase" }}>NO DISC</span>
                        : <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, background: blockColor + "20", color: blockColor, textTransform: "uppercase" }}>{d.block_class}</span>
                      ) : "—"}
                    </td>
                    <td style={{ padding: "10px 14px" }}>
                      {d?.block_class === "none"
                        ? <Badge variant="danger">Not viable</Badge>
                        : <Badge variant={r.hasPrimers ? "success" : "danger"}>{r.hasPrimers ? "AS-RPA" : "No primers"}</Badge>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {proximityCands.some(r => r.asrpaDiscrimination?.block_class === "none") && (
            <div style={{ padding: "12px 20px", fontSize: "11px", color: T.danger, background: "#FEF2F2", borderTop: `1px solid #FECACA` }}>
              <strong>Panel gap:</strong> {proximityCands.filter(r => r.asrpaDiscrimination?.block_class === "none").map(r => r.label).join(", ")} — primer 3′ base forms a Watson-Crick pair with the WT template (no mismatch = no discrimination).
              These targets require primer strand reversal, alternative SNP base selection, or a different discrimination strategy.
            </div>
          )}
          {proximityCands.some(r => r.asrpaDiscrimination) && (
            <div style={{ padding: "12px 20px", fontSize: "10px", color: T.textTer, fontStyle: "italic", borderTop: `1px solid ${T.purple}15` }}>
              Thermodynamic estimates — not experimentally validated. Ratios from Boltzmann conversion exp(ΔΔG/RT) at 37 °C, capped at 100× (empirical AS-RPA discrimination typically 10–100×; Ye et al. 2019).
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const PrimersTab = ({ results }) => {
  const mobile = useIsMobile();
  const [hoveredRow, setHoveredRow] = useState(null);
  const withPrimers = results.filter((r) => r.hasPrimers);
  const withoutPrimers = results.filter((r) => !r.hasPrimers && r.gene !== "IS6110");
  const directWithPrimers = withPrimers.filter((r) => r.strategy === "Direct");
  const proximityWithPrimers = withPrimers.filter((r) => r.strategy === "Proximity");

  return (
    <div>
      {/* RPA Explanation */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", padding: mobile ? "16px" : "20px 24px", marginBottom: "24px", display: "flex", gap: "14px", alignItems: "flex-start" }}>
        <Crosshair size={20} color={T.primaryDark} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <div style={{ fontSize: "14px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING, marginBottom: "4px" }}>Recombinase Polymerase Amplification (RPA)</div>
          <p style={{ fontSize: "13px", color: T.primaryDark, lineHeight: 1.6, margin: 0, opacity: 0.85 }}>
            RPA is an isothermal amplification method (37°C) that replaces PCR thermocycling. Each crRNA target needs a pair of
            30–35 nt primers flanking an 80–120 bp amplicon containing the crRNA binding site. The amplified product is then
            detected by Cas12a <em>trans</em>-cleavage of MB-ssDNA reporters on the electrochemical platform (SWV signal decrease on LIG electrodes).
          </p>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", marginTop: "10px", padding: "8px 12px", background: "rgba(255,255,255,0.5)", borderRadius: "4px", border: `1px solid ${T.primary}22` }}>
            <Droplet size={14} color={T.primaryDark} strokeWidth={2} style={{ flexShrink: 0, marginTop: 1 }} />
            <span style={{ fontSize: "11px", color: T.primaryDark, lineHeight: 1.5, opacity: 0.85 }}>
              Capped at 120 bp — cfDNA fragments in blood are ~100–160 bp (median ~140 bp). Shorter amplicons maximise template capture from fragmented circulating DNA.
            </span>
          </div>
        </div>
      </div>

      {/* Standard vs AS-RPA info cards */}
      <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "12px", marginBottom: "24px" }}>
        <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: T.success }} />
            <span style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Standard RPA</span>
            <Badge variant="success">{directWithPrimers.length} targets</Badge>
          </div>
          <p style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6, margin: 0 }}>
            Symmetric flanking primers for <strong>DIRECT detection</strong> candidates. The crRNA spacer overlaps the mutation site,
            so allele discrimination comes from Cas12a mismatch intolerance — not from primers. Primers simply amplify the region
            containing the crRNA binding site. Discrimination ratios are {results.some(r => r.discMethod === "neural") ? "predicted by Compass-ML neural discrimination head (multi-task, trained on 6,136 EasyDesign pairs)" : results.some(r => (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature") ? "predicted by a learned model (XGBoost, 18 thermodynamic features)" : "estimated by position × destabilisation heuristic"}.
          </p>
        </div>
        <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: T.purple }} />
            <span style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Allele-Specific RPA (AS-RPA)</span>
            <Badge variant="purple">{proximityWithPrimers.length} targets</Badge>
          </div>
          <p style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6, margin: 0 }}>
            For <strong>PROXIMITY detection</strong> candidates where the mutation falls outside the crRNA footprint.
            The forward primer's 3' terminal nucleotide is locked to the mutant allele, so only mutant DNA is amplified.
            A deliberate mismatch at the penultimate position further suppresses wildtype amplification (Ye et al., 2019).
          </p>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", marginTop: "10px", padding: "8px 12px", background: `${T.purple}0A`, borderRadius: "4px", border: `1px solid ${T.purple}22` }}>
            <Info size={14} color={T.purple} strokeWidth={2} style={{ flexShrink: 0, marginTop: 1 }} />
            <span style={{ fontSize: "11px", color: T.textSec, lineHeight: 1.5 }}>
              <strong style={{ color: T.purple }}>PAM desert.</strong> These targets lack a TTTV PAM within the spacer window overlapping the SNP — common in M. tuberculosis (65.6% GC).
              Discrimination is shifted entirely to primer-level allele specificity (estimated ≥100× selectivity), not Cas12a mismatch intolerance.
            </span>
          </div>
        </div>
      </div>

      {/* Missing primers warning */}
      {withoutPrimers.length > 0 && (
        <div style={{ background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: "4px", padding: "16px 20px", marginBottom: "24px", display: "flex", gap: "12px", alignItems: "flex-start" }}>
          <Shield size={18} color="#DC2626" style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "#DC2626", fontFamily: HEADING, marginBottom: "4px" }}>
              {withoutPrimers.length} target{withoutPrimers.length > 1 ? "s" : ""} missing RPA primers
            </div>
            <p style={{ fontSize: "12px", color: "#DC2626", lineHeight: 1.5, margin: "0 0 8px", opacity: 0.85 }}>
              These targets could not have primers designed, typically due to extreme GC content in flanking regions
              (M. tuberculosis is 65.6% GC) preventing primers from meeting the 60–65°C Tm constraint.
              Tight amplicon constraint (≤120 bp for blood cfDNA) may further limit primer placement in GC-rich regions.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
              {withoutPrimers.map(r => (
                <span key={r.label} style={{ fontFamily: MONO, fontSize: "10px", padding: "3px 8px", borderRadius: "4px", background: "#FEF2F2", color: "#DC2626", fontWeight: 600 }}>{r.label}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Primer table */}
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
        <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, padding: "16px 20px", borderBottom: `1px solid ${T.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>RPA Primer Pairs</span>
          <span style={{ fontSize: "12px", fontWeight: 600, color: T.textSec }}>{withPrimers.length} of {results.length} targets</span>
        </div>
        <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", minWidth: 700 }}>
          <thead>
            <tr style={{ background: T.bgSub }}>
              {["Target", "Type", "Disc", "Forward Primer", "Reverse Primer", "Amplicon", "GC%", "SM"].map((h) => (
                <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 600, color: T.textSec, borderBottom: `1px solid ${T.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {withPrimers.map((r) => {
              const isHov = hoveredRow === r.label;
              const discVal = r.strategy === "Proximity"
                ? (() => { const sp = r.asrpaDiscrimination?.estimated_specificity; return sp != null ? `≥${(1 / Math.max(1 - sp, 0.001)).toFixed(0)}×` : "≥100×"; })()
                : r.gene === "IS6110" ? "N/A"
                : r.disc > 0 ? `${r.disc.toFixed(1)}×${r.hasSM ? " (post-SM)" : ""}` : "—";
              return (
              <tr key={r.label} style={{ borderBottom: `1px solid ${T.borderLight}`, transition: "background 0.15s", background: isHov ? `${T.primary}08` : "transparent" }}
                onMouseEnter={() => setHoveredRow(r.label)} onMouseLeave={() => setHoveredRow(null)}>
                <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600, fontSize: "11px" }}>{r.label}</td>
                <td style={{ padding: "10px 14px" }}>
                  <Badge variant={r.strategy === "Direct" ? "success" : "purple"}>
                    {r.strategy === "Direct" ? "Standard" : "AS-RPA"}
                  </Badge>
                </td>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, fontSize: "11px", color: r.gene === "IS6110" ? T.textTer : r.strategy === "Proximity" ? T.purple : r.disc >= 3 ? T.success : r.disc >= 2 ? T.warning : T.danger }}>
                  {discVal}
                </td>
                <td style={{ padding: "10px 14px" }}>
                  {isHov ? <Seq s={r.fwd} /> : <span style={{ fontFamily: MONO, fontSize: "11.5px", letterSpacing: "1.2px", color: T.textTer }}>{r.fwd}</span>}
                </td>
                <td style={{ padding: "10px 14px" }}>
                  {isHov ? <Seq s={r.rev} /> : <span style={{ fontFamily: MONO, fontSize: "11.5px", letterSpacing: "1.2px", color: T.textTer }}>{r.rev}</span>}
                </td>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: r.amplicon <= 100 ? T.success : r.amplicon <= 120 ? T.warning : T.danger }}>{r.amplicon} bp</td>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontSize: "11px", color: (r.ampliconGc || r.gc || 0.656) >= 0.70 ? T.warning : T.textSec }}>{((r.ampliconGc || r.gc || 0.656) * 100).toFixed(0)}%{(r.ampliconGc || r.gc || 0.656) >= 0.70 ? <span style={{ fontSize: "9px", color: T.warning, marginLeft: "3px" }} title="High GC may cause hairpins blocking RPA recombinase invasion">{"\u26a0"}</span> : ""}</td>
                <td style={{ padding: "10px 14px" }}><Badge variant={r.hasSM ? "primary" : "default"}>{r.hasSM ? "Yes" : "No"}</Badge></td>
              </tr>
              );
            })}
          </tbody>
        </table>
        </div>
      </div>
    </div>
  );
};

const MultiplexTab = ({ results, panelData, jobId, connected }) => {
  const mobile = useIsMobile();
  const drugs = [...new Set(results.map((r) => r.drug))];
  const controlIncluded = results.some((r) => r.gene === "IS6110");
  const directCount = results.filter(r => r.strategy === "Direct").length;
  const proximityCount = results.filter(r => r.strategy === "Proximity").length;
  const withPrimers = results.filter(r => r.hasPrimers && !(r.asrpaDiscrimination?.block_class === "none")).length;

  // Electrode grid color-by mode
  const [colorBy, setColorBy] = useState("drug");
  // Hover state for electrode pads
  const [hoveredPad, setHoveredPad] = useState(null);

  // Kinetics data from API or fallback
  const [poolData, setPoolData] = useState(null);
  useEffect(() => {
    if (connected && jobId) {
      getPoolData(jobId).then(({ data }) => { if (data) setPoolData(data); }).catch(() => {});
    }
  }, [connected, jobId]);

  // ═══════════ PREDICTED ELECTROCHEMICAL READOUT — State ═══════════
  const [echemCandidate, setEchemCandidate] = useState("rpoB_S531L");
  const [echemTechnique, setEchemTechnique] = useState("SWV");
  const [echemTime, setEchemTime] = useState(30);       // minutes
  const [echemBloodTiter, setEchemBloodTiter] = useState(100); // cp/mL
  const [echemKtrans, setEchemKtrans] = useState(0.02); // s⁻¹ (mid-range estimate for LIG-AuNP surface trans-cleavage, 0.01–0.1 s⁻¹)
  const [echemAdvanced, setEchemAdvanced] = useState(false);
  const [echemGamma0, setEchemGamma0] = useState(1.5e11); // molecules/cm²
  const [echemPorosity, setEchemPorosity] = useState(3);
  const [echemIscale, setEchemIscale] = useState(3.0);    // μA
  const [echemShowFwdRev, setEchemShowFwdRev] = useState(false); // SWV i_fwd/i_rev toggle
  const [echemArch, setEchemArch] = useState("C"); // Reporter architecture: A=pAP/ALP, B=Silver, C=MB

  const kinetics = poolData?.kinetics || {
    phases: [
      { phase: "crRNA rehydration", solution_bound: "N/A", on_electrode: "2\u20135 min", description: "Dried crRNA dissolves from LIG surface into assay buffer", is_bottleneck: false },
      { phase: "RNP formation", solution_bound: "0.5\u20131 min", on_electrode: "2\u20135 min", description: "Cas12a + crRNA \u2192 active RNP (in situ complexation)", is_bottleneck: false },
      { phase: "Target recognition", solution_bound: "~10 sec", on_electrode: "1\u20133 min", description: "RNP binds cognate amplicon, R-loop formation, cis-cleavage activation", is_bottleneck: false },
      { phase: "Surface trans-cleavage", solution_bound: "~5 min", on_electrode: "10\u201320 min", description: "Activated Cas12a cleaves MB-ssDNA reporters tethered to LIG electrode", is_bottleneck: true },
    ],
    totals: {
      detection_solution: "~6\u20138 min", detection_electrode: "15\u201330 min",
      rpa_time: "15\u201320 min", total_solution: "~23\u201328 min", total_electrode: "30\u201350 min",
      who_tpp_target: "< 120 min", who_tpp_pass: true,
    },
    parameters: [
      { param: "k_form (RNP association)", value: "1.75 \u00d7 10\u2075 M\u207b\u00b9s\u207b\u00b9", source: "Lesinski et al. 2024", note: "SPR measurement — analogous to on-pad scenario but different surface than LIG" },
      { param: "k_off (RNP dissociation)", value: "1.87 \u00d7 10\u207b\u2074 s\u207b\u00b9", source: "Lesinski et al. 2024", note: null },
      { param: "k_cis (cis-cleavage)", value: "0.03 s\u207b\u00b9", source: "Lesinski et al. 2024", note: null },
      { param: "k_trans (solution, free ssDNA)", value: "~2.0 s\u207b\u00b9", source: "Nalefski et al. 2021", note: "Free ssDNA in solution. NOT applicable to surface-tethered reporters." },
      { param: "k_trans (surface, estimated)", value: "0.01\u20130.1 s\u207b\u00b9", source: "Estimated", note: "Key experimental unknown." },
      { param: "[Cas12a]", value: "50 nM", source: "Design parameter", note: null },
      { param: "[crRNA] on pad", value: "~200 nM equivalent", source: "Design parameter", note: "Effective concentration after rehydration unknown." },
      { param: "MB-ssDNA probe density", value: "~10\u00b9\u2070\u201310\u00b9\u00b9 molecules/cm\u00b2", source: "Estimated for LIG", note: "Geometric density \u2014 effective density is higher due to LIG porosity (3\u201310\u00d7 surface area). Directly affects signal magnitude and time-to-detection." },
    ],
    insights: [
      { title: "Rate-limiting step", text: "Surface trans-cleavage of tethered MB-ssDNA reporters dominates detection time \u2014 not RNP formation or target recognition." },
      { title: "In situ complexation", text: "Lesinski et al. 2024: reduces effective [RNP] during the first ~5 minutes by ~10-fold vs pre-complexed format. This prevents Cas12a from destroying target amplicons before detection begins." },
      { title: "Experimental unknowns", text: "k_trans on LIG-tethered MB-ssDNA and crRNA rehydration kinetics have not been measured. These are key characterisation priorities." },
      { title: "Capacitive background", text: "SWV simulation models Faradaic current only. Real LIG electrodes have capacitive (non-Faradaic) baseline from double-layer charging on high-surface-area graphene foam. Signal-to-noise ratio in practice depends on the Faradaic-to-capacitive current ratio." },
    ],
    target_ranking: [
      { target: "IS6110", efficiency: 0.95, is_weak: false },
      { target: "rpoB_S531L", efficiency: 0.88, is_weak: false },
      { target: "katG_S315T", efficiency: 0.85, is_weak: false },
      { target: "gyrA_D94G", efficiency: 0.83, is_weak: false },
      { target: "rrs_A1401G", efficiency: 0.81, is_weak: false },
      { target: "fabG1_C-15T", efficiency: 0.80, is_weak: false },
      { target: "rpoB_H526Y", efficiency: 0.78, is_weak: false },
      { target: "gyrA_A90V", efficiency: 0.77, is_weak: false },
      { target: "embB_M306V", efficiency: 0.76, is_weak: false },
      { target: "rpoB_D516V", efficiency: 0.73, is_weak: false },
      { target: "eis_C-14T", efficiency: 0.70, is_weak: false },
      { target: "embB_M306I", efficiency: 0.65, is_weak: false },
      { target: "pncA_H57D", efficiency: 0.55, is_weak: true },
      { target: "RNaseP", efficiency: 0.90, is_weak: false },
    ],
  };

  // Electrode layout — 7×2 grid (14-plex: 12 resistance mutations + IS6110 + RNaseP)
  const electrodeLayout = [
    ["IS6110","rpoB_S531L","rpoB_H526Y","rpoB_D516V","katG_S315T","fabG1_C-15T","embB_M306V"],
    ["embB_M306I","pncA_H57D","gyrA_D94G","gyrA_A90V","rrs_A1401G","eis_C-14T","RNaseP"],
  ];

  // Drug colors for pads
  const PAD_DRUG_COLORS = { RIF: "#1E3A5F", INH: "#D97706", EMB: "#059669", PZA: "#0891B2", FQ: "#DC2626", AG: "#7C3AED", CTRL: "#9CA3AF" };
  const PAD_DRUG_BG = { RIF: "#EEF2FF", INH: "#FFFBEB", EMB: "#ecf8f4", PZA: "#f2f9ee", FQ: "#FEF2F2", AG: "#FFFBEB", CTRL: "#F3F4F6" };

  const targetDrug = (t) => {
    const r = results.find(x => x.label === t);
    if (r) return r.drug || "OTHER";
    if (t === "IS6110" || t === "RNaseP") return "CTRL";
    if (t.startsWith("rpoB")) return "RIF";
    if (t.startsWith("katG") || t.startsWith("fabG1")) return "INH";
    if (t.startsWith("embB")) return "EMB";
    if (t.startsWith("pncA")) return "PZA";
    if (t.startsWith("gyrA")) return "FQ";
    if (t.startsWith("rrs") || t.startsWith("eis")) return "AG";
    return "OTHER";
  };
  const targetStrategy = (t) => { const r = results.find(x => x.label === t); return r ? r.strategy : "Direct"; };
  const targetScore = (t) => { const r = results.find(x => x.label === t); return r ? ((r.cnnCalibrated ?? r.score) || 0) : 0; };
  const coAmpliconGroups = [["rpoB_S531L","rpoB_H526Y","rpoB_D516V"],["embB_M306V","embB_M306I"],["gyrA_D94G","gyrA_A90V"]];
  const isCoAmplicon = (t) => coAmpliconGroups.some(g => g.includes(t));
  const coAmpliconPartner = (t) => { const g = coAmpliconGroups.find(g => g.includes(t)); return g ? g.filter(x => x !== t)[0] : null; };

  // Efficiency lookup from kinetics data or results
  const getEfficiency = (target) => {
    const kr = (kinetics.target_ranking || []).find(x => x.target === target);
    if (kr) return kr.efficiency;
    const r = results.find(x => x.label === target);
    return r ? ((r.cnnCalibrated ?? r.score) || 0.75) : 0.75;
  };

  // pncA_H57D status check
  const pncAResult = results.find(r => r.label === "pncA_H57D");
  const pncAResolved = pncAResult?.asrpaDiscrimination?.block_class === "strong" ||
    (pncAResult?.asrpaDiscrimination?.terminal_mismatch && pncAResult.asrpaDiscrimination.terminal_mismatch.includes("C:C"));

  // === ELECTROCHEMICAL SIMULATION COMPUTATIONS ===
  const V_blood_mL = 1.0;
  const extraction_yield = 0.6;
  const V_eluate_uL = 50;
  const V_pad_uL = 50 / 14;
  const P_rpa = 0.95;
  const P_signal = 1.0;
  const IS6110_copy_number = 10;

  const drug_targets = {
    RIF: ["rpoB_S531L","rpoB_H526Y","rpoB_D516V"],
    INH: ["katG_S315T","fabG1_C-15T"],
    EMB: ["embB_M306V","embB_M306I"],
    PZA: ["pncA_H57D"],
    FQ: ["gyrA_D94G","gyrA_A90V"],
    AG: ["rrs_A1401G","eis_C-14T"],
  };

  const WHO_thresholds = { RIF: 0.95, INH: 0.90, FQ: 0.90, EMB: 0.80, PZA: 0.80, AG: 0.80 };
  const DRUG_LINE_COLORS = { RIF: "#1E3A5F", INH: "#4338CA", EMB: "#059669", PZA: "#059669", FQ: "#DC2626", AG: "#3730A3", IS6110: "#6B7280" };

  // ═══════════ PREDICTED ELECTROCHEMICAL READOUT — Physics Engine ═══════════
  // Architecture-specific electrochemistry configurations
  const ARCH_CONFIGS = {
    A: {
      label: "ALP/pAP", species: "solution-phase (diffusion-controlled)",
      reference: "Bezinge 2023",
      E0: 0.15, n: 1, // pAP irreversible oxidation, 1e⁻ rate-determining step
      E_start: -0.20, E_end: 0.25,
      E_sw: 0.025, E_pulse: 0.050, frequency: 2.5, step: 0.005,
      scan_rate: 0.05, signal_direction: "off",
      peak_label: "pAP oxidation",
      peak_shape: "asymmetric", // Nicholson-Shain irreversible
      alpha: 0.5, // transfer coefficient for irreversible oxidation
      I_scale_base: 0.8, // µA range for pAP
    },
    B: {
      label: "Silver", species: "metallic deposit (stripping voltammetry)",
      reference: "Suea-Ngam 2021",
      E0: 0.16, n: 1, // Ag⁰ → Ag⁺ + e⁻
      E_start: -0.30, E_end: 0.50,
      E_sw: 0.060, E_pulse: 0.060, frequency: 200, step: 0.010,
      scan_rate: 0.05, signal_direction: "off",
      peak_label: "Ag stripping",
      peak_shape: "stripping", // asymmetric Gaussian (sharp onset, broader tail)
      sigma_onset: 0.020, sigma_tail: 0.050,
      I_scale_base: 3.0, // µA range for silver stripping
    },
    C: {
      label: "MB", species: "surface-confined (Laviron 1979)",
      reference: "Laviron 1979",
      E0: -0.22, n: 2, // MB: 2e⁻, 2H⁺ reduction
      E_start: -0.05, E_end: -0.40,
      E_sw: 0.025, E_pulse: 0.050, frequency: 50, step: 0.004,
      scan_rate: 0.05, signal_direction: "off",
      peak_label: "MB reduction",
      peak_shape: "sech2", // Laviron surface-confined
      I_scale_base: 1.0,
    },
  };
  const archCfg = ARCH_CONFIGS[echemArch];

  const ECHEM = {
    E0: archCfg.E0, n: archCfg.n, F: 96485, R: 8.314, Temp: 310.15,
    A_geo: 0.0707, // cm²
    E_sw: archCfg.E_sw, E_pulse: archCfg.E_pulse,
    I_scale_DPV: 1.5 * archCfg.I_scale_base,
    I_scale_SWV: 3.0 * archCfg.I_scale_base,
    scan_rate: archCfg.scan_rate,
    k_form: 0.003, Cas12a_nM: 50, Cas12a_ref: 50,
    intra_device_rsd: 0.05,
    V_blood_mL: 1.0, extraction_yield: 0.6, V_eluate_uL: 50, V_pad_uL: 50 / 14,
  };

  // Γ₀ in mol/cm² from molecules/cm²
  const echemGamma0_mol = echemGamma0 / 6.022e23;
  const echemAeff = ECHEM.A_geo * echemPorosity;

  // Γ(t) — exact integral with in situ RNP formation
  const computeGamma = useCallback((t_s, S_eff, k_trans) => {
    const { k_form, Cas12a_nM, Cas12a_ref } = ECHEM;
    const integral = t_s + (1 / k_form) * (Math.exp(-k_form * t_s) - 1);
    const exponent = -k_trans * S_eff * (Cas12a_nM / Cas12a_ref) * integral;
    return echemGamma0_mol * Math.exp(exponent);
  }, [echemGamma0_mol]);

  // Γ_WT for direct detection (reduced by discrimination ratio)
  const computeGammaWT = useCallback((t_s, S_eff, D, k_trans, isProximity) => {
    if (isProximity) return echemGamma0_mol; // no amplification → no cleavage
    return computeGamma(t_s, S_eff / D, k_trans);
  }, [computeGamma, echemGamma0_mol]);

  // ── Architecture-specific peak shape helpers ──
  // Arch C (MB): Laviron sech² for surface-confined species
  const peakShape_sech2 = (E, E0, nFRT, E_half_width) => {
    const xi_plus = nFRT * (E + E_half_width - E0);
    const xi_minus = nFRT * (E - E_half_width - E0);
    return -(1 / (1 + Math.exp(xi_plus)) - 1 / (1 + Math.exp(xi_minus)));
  };
  // Arch A (pAP): Nicholson-Shain asymmetric irreversible oxidation peak
  const peakShape_asymmetric = (E, E0, alpha, n_alpha, F_c, R_c, T) => {
    const b = (alpha * n_alpha * F_c) / (R_c * T);
    const x = b * (E - E0);
    const expx = Math.exp(Math.max(-20, Math.min(20, x)));
    return 4 * expx / ((1 + expx) * (1 + expx));
  };
  // Arch B (Ag): Asymmetric Gaussian stripping peak (sharp onset, broader tail)
  const peakShape_stripping = (E, E0, sigma_onset, sigma_tail) => {
    const sigma = E <= E0 ? sigma_onset : sigma_tail;
    const x = (E - E0) / sigma;
    return Math.exp(-0.5 * x * x);
  };

  // Unified SWV compute — architecture-aware
  const computeSWV = useCallback((E_array, Gamma) => {
    const { n, F, R, Temp, E0, E_sw } = ECHEM;
    const nFRT = n * F / (R * Temp);
    const scale = echemIscale * echemAeff * archCfg.I_scale_base;
    const ratio = Gamma / echemGamma0_mol;
    if (archCfg.peak_shape === "asymmetric") {
      // Arch A: pAP oxidation — asymmetric irreversible peak
      return E_array.map(E => scale * ratio * peakShape_asymmetric(E, E0, archCfg.alpha, 1, F, R, Temp));
    }
    if (archCfg.peak_shape === "stripping") {
      // Arch B: Ag stripping — asymmetric Gaussian
      return E_array.map(E => scale * ratio * peakShape_stripping(E, E0, archCfg.sigma_onset, archCfg.sigma_tail));
    }
    // Arch C: MB surface-confined — Laviron sech² (positive peaks for SWV net current)
    return E_array.map(E => scale * ratio * peakShape_sech2(E, E0, nFRT, E_sw));
  }, [echemIscale, echemAeff, echemGamma0_mol, echemArch]);

  // SWV forward/reverse components for toggle display (Arch C only — other architectures use net only)
  const computeSWVComponents = useCallback((E_array, Gamma) => {
    const { n, F, R, Temp, E0, E_sw } = ECHEM;
    const nFRT = n * F / (R * Temp);
    const scale = echemIscale * echemAeff * archCfg.I_scale_base * 0.6;
    const ratio = Gamma / echemGamma0_mol;
    return E_array.map(E => {
      const xi_fwd = nFRT * (E + E_sw - E0);
      const xi_rev = nFRT * (E - E_sw - E0);
      const i_fwd = scale * ratio / (1 + Math.exp(xi_fwd));
      const i_rev = scale * ratio / (1 + Math.exp(xi_rev));
      return { i_fwd, i_rev };
    });
  }, [echemIscale, echemAeff, echemGamma0_mol, echemArch]);

  // Unified DPV compute — architecture-aware
  const computeDPV = useCallback((E_array, Gamma) => {
    const { n, F, R, Temp, E0, E_pulse } = ECHEM;
    const nFRT = n * F / (R * Temp);
    const scale = ECHEM.I_scale_DPV * echemAeff;
    const ratio = Gamma / echemGamma0_mol;
    if (archCfg.peak_shape === "asymmetric") {
      return E_array.map(E => scale * ratio * peakShape_asymmetric(E, E0, archCfg.alpha, 1, F, R, Temp));
    }
    if (archCfg.peak_shape === "stripping") {
      return E_array.map(E => scale * ratio * peakShape_stripping(E, E0, archCfg.sigma_onset, archCfg.sigma_tail));
    }
    // Arch C: MB — DPV uses difference of Nernst equilibria, same sech² family
    return E_array.map(E => scale * ratio * peakShape_sech2(E, E0, nFRT, E_pulse / 2));
  }, [echemAeff, echemGamma0_mol, echemArch]);

  // CV voltammogram — architecture-aware, scale-based (positive peaks for consistent peak detection)
  const computeCV = useCallback((E_array, Gamma) => {
    const { n, F, R, Temp, E0 } = ECHEM;
    const ratio = Gamma / echemGamma0_mol;
    const peakScale = echemIscale * echemAeff * archCfg.I_scale_base * 0.7;
    if (archCfg.peak_shape === "sech2") {
      const sigma = (R * Temp) / (n * F) * 2.0;
      return E_array.map(E => peakScale * ratio / (Math.cosh((E - E0) / sigma) ** 2));
    }
    if (archCfg.peak_shape === "asymmetric") {
      return E_array.map(E => peakScale * ratio * peakShape_asymmetric(E, E0, archCfg.alpha, 1, F, R, Temp));
    }
    return E_array.map(E => peakScale * ratio * peakShape_stripping(E, E0, archCfg.sigma_onset, archCfg.sigma_tail));
  }, [echemIscale, echemAeff, echemGamma0_mol, echemArch]);

  // CV duck-shape: forward + reverse scan — architecture-aware with visible Faradaic peaks
  const computeCVDuck = useCallback((Gamma) => {
    const { n, F, R, Temp, E0 } = ECHEM;
    const ratio = Gamma / echemGamma0_mol;
    const peakScale = echemIscale * echemAeff * archCfg.I_scale_base * 0.7;
    const i_cap = peakScale * 0.12; // capacitive envelope (constant, independent of Γ)
    const N = 200;
    const E_start = archCfg.E_start, E_end = archCfg.E_end;
    const E_range = E_end - E_start;
    const forward = [], reverse = [];

    if (archCfg.peak_shape === "sech2") {
      // Arch C (MB): surface-confined — symmetric cathodic/anodic sech² peaks
      const sigma = (R * Temp) / (n * F) * 2.0;
      const deltaEp = 0.010; // near-ideal surface-confined ΔEp
      const E_pc = E0 - deltaEp / 2, E_pa = E0 + deltaEp / 2;
      for (let i = 0; i <= N; i++) {
        const E = E_start + E_range * (i / N);
        forward.push({ E, I: -peakScale * ratio / (Math.cosh((E - E_pc) / sigma) ** 2) - i_cap });
      }
      for (let i = 0; i <= N; i++) {
        const E = E_end + (-E_range) * (i / N);
        reverse.push({ E, I: peakScale * ratio / (Math.cosh((E - E_pa) / sigma) ** 2) + i_cap });
      }
      return { forward, reverse, deltaEp, E_pc, E_pa };
    }

    if (archCfg.peak_shape === "asymmetric") {
      // Arch A (pAP): irreversible oxidation — forward peak, no reverse peak
      for (let i = 0; i <= N; i++) {
        const E = E_start + E_range * (i / N);
        const fara = peakScale * ratio * peakShape_asymmetric(E, E0, archCfg.alpha, 1, F, R, Temp);
        forward.push({ E, I: fara + i_cap });
      }
      for (let i = 0; i <= N; i++) {
        const E = E_end + (-E_range) * (i / N);
        const tail = peakScale * ratio * 0.12 * Math.exp(-Math.abs(E - E0) / 0.08);
        reverse.push({ E, I: -tail - i_cap });
      }
      return { forward, reverse, deltaEp: 0.060, E_pc: null, E_pa: E0 };
    }

    // Arch B (Ag stripping): sharp anodic stripping + broad cathodic re-deposition
    for (let i = 0; i <= N; i++) {
      const E = E_start + E_range * (i / N);
      const fara = peakScale * ratio * peakShape_stripping(E, E0, archCfg.sigma_onset, archCfg.sigma_tail);
      forward.push({ E, I: fara + i_cap });
    }
    for (let i = 0; i <= N; i++) {
      const E = E_end + (-E_range) * (i / N);
      const E_dep = E0 - 0.15;
      const fara = peakScale * ratio * 0.4 * Math.exp(-0.5 * ((E - E_dep) / 0.06) ** 2);
      reverse.push({ E, I: -fara - i_cap });
    }
    return { forward, reverse, deltaEp: 0.150, E_pc: E0 - 0.15, E_pa: E0 };
  }, [echemIscale, echemAeff, echemGamma0_mol, echemArch]);

  // Potential array — architecture-dependent range
  const echemE = useMemo(() => {
    const start = archCfg.E_start, end = archCfg.E_end;
    const range = end - start;
    const N = Math.round(Math.abs(range) / 0.001);
    return Array.from({ length: N + 1 }, (_, i) => start + range * (i / N));
  }, [echemArch]);

  // Get candidate data for electrochemistry
  const echemCandidateData = useMemo(() => {
    const r = results.find(x => x.label === echemCandidate);
    const eff = getEfficiency(echemCandidate);
    const disc = r?.disc && r.disc < 900 ? r.disc : 1000;
    const strategy = targetStrategy(echemCandidate);
    const drug = targetDrug(echemCandidate);
    const isProximity = strategy === "Proximity";
    return { label: echemCandidate, efficiency: eff, discrimination: disc, strategy, drug, isProximity, isIS6110: echemCandidate === "IS6110", copyNumber: echemCandidate === "IS6110" ? 10 : 1 };
  }, [echemCandidate, results]);

  // Panel A: Voltammogram curves
  const echemVoltammogram = useMemo(() => {
    const cd = echemCandidateData;
    const t_s = echemTime * 60;
    const G_base = echemGamma0_mol;
    const G_after = computeGamma(t_s, cd.efficiency, echemKtrans);
    const deltaI_pct = ((1 - G_after / G_base) * 100);

    let base, after;
    if (echemTechnique === "SWV") {
      base = computeSWV(echemE, G_base);
      after = computeSWV(echemE, G_after);
    } else if (echemTechnique === "DPV") {
      base = computeDPV(echemE, G_base);
      after = computeDPV(echemE, G_after);
    } else {
      base = computeCV(echemE, G_base);
      after = computeCV(echemE, G_after);
    }

    // All techniques now return positive peak values
    const peakBase = Math.max(...base);
    const peakAfter = Math.max(...after);

    // Also compute SWV forward/reverse if applicable
    let fwdRevBase = null, fwdRevAfter = null;
    if (echemTechnique === "SWV") {
      fwdRevBase = computeSWVComponents(echemE, G_base);
      fwdRevAfter = computeSWVComponents(echemE, G_after);
    }

    return echemE.map((E, i) => ({
      E: +E.toFixed(3),
      baseline: +base[i].toFixed(4),
      after: +after[i].toFixed(4),
      ...(fwdRevBase ? {
        base_fwd: +fwdRevBase[i].i_fwd.toFixed(4),
        base_rev: +fwdRevBase[i].i_rev.toFixed(4),
        after_fwd: +fwdRevAfter[i].i_fwd.toFixed(4),
        after_rev: +fwdRevAfter[i].i_rev.toFixed(4),
      } : {}),
    })).concat({
      _meta: true, peakBase: +peakBase.toFixed(3), peakAfter: +peakAfter.toFixed(3), deltaI: +deltaI_pct.toFixed(1),
    });
  }, [echemCandidateData, echemTime, echemKtrans, echemTechnique, echemGamma0_mol, echemE, computeGamma, computeSWV, computeSWVComponents, computeDPV, computeCV]);

  const echemMetaRaw = echemVoltammogram[echemVoltammogram.length - 1];
  // Display absolute peak values but note they are negative (cathodic)
  const echemMeta = {
    ...echemMetaRaw,
    peakBaseAbs: Math.abs(echemMetaRaw.peakBase).toFixed(3),
    peakAfterAbs: Math.abs(echemMetaRaw.peakAfter).toFixed(3),
  };
  const echemPlotData = echemVoltammogram.slice(0, -1);

  // Panel B: ΔI% time course with RNP formation lag phase
  // Lag phase: ~5-10 min for in situ RNP assembly before significant trans-cleavage begins
  const echemTimeCourse = useMemo(() => {
    const cd = echemCandidateData;
    const lagMin = 7; // lag phase in minutes (RNP formation + cis-cleavage initiation)
    const points = [];
    for (let t = 0; t <= 60; t += 1) {
      // Effective cleavage time accounts for lag
      const t_eff = Math.max(0, t - lagMin);
      const t_s = t_eff * 60;
      const G_mut = computeGamma(t_s, cd.efficiency, echemKtrans);
      const G_wt = computeGammaWT(t_s, cd.efficiency, cd.discrimination, echemKtrans, cd.isProximity);
      const G_mutHi = computeGamma(t_s, cd.efficiency, echemKtrans * 2);
      const G_mutLo = computeGamma(t_s, cd.efficiency, echemKtrans * 0.5);
      points.push({
        time: t,
        MUT: +((1 - G_mut / echemGamma0_mol) * 100).toFixed(1),
        WT: +((1 - G_wt / echemGamma0_mol) * 100).toFixed(1),
        MUT_hi: +((1 - G_mutHi / echemGamma0_mol) * 100).toFixed(1),
        MUT_lo: +((1 - G_mutLo / echemGamma0_mol) * 100).toFixed(1),
        neg: 0,
      });
    }
    // Compute detection times (3σ threshold)
    const threshold = 3 * ECHEM.intra_device_rsd * 100; // 15%
    const timeMut = points.find(p => p.MUT >= threshold)?.time ?? null;
    const timeWt = points.find(p => p.WT >= threshold)?.time ?? null;
    return { points, threshold, timeMut, timeWt, lagMin };
  }, [echemCandidateData, echemKtrans, echemGamma0_mol, computeGamma, computeGammaWT]);

  // Panel C: MUT vs WT discrimination overlay
  const echemDiscOverlay = useMemo(() => {
    const cd = echemCandidateData;
    const t_s = echemTime * 60;
    const G_base = echemGamma0_mol;
    const G_mut = computeGamma(t_s, cd.efficiency, echemKtrans);
    const G_wt = computeGammaWT(t_s, cd.efficiency, cd.discrimination, echemKtrans, cd.isProximity);

    let baseCurve, mutCurve, wtCurve;
    if (echemTechnique === "SWV") {
      baseCurve = computeSWV(echemE, G_base);
      mutCurve = computeSWV(echemE, G_mut);
      wtCurve = computeSWV(echemE, G_wt);
    } else if (echemTechnique === "DPV") {
      baseCurve = computeDPV(echemE, G_base);
      mutCurve = computeDPV(echemE, G_mut);
      wtCurve = computeDPV(echemE, G_wt);
    } else {
      baseCurve = computeCV(echemE, G_base);
      mutCurve = computeCV(echemE, G_mut);
      wtCurve = computeCV(echemE, G_wt);
    }

    // All techniques return positive peak values
    const peakBase = Math.max(...baseCurve);
    const peakMut = Math.max(...mutCurve);
    const peakWt = Math.max(...wtCurve);
    const diMut = ((1 - peakMut / peakBase) * 100);
    const diWt = ((1 - peakWt / peakBase) * 100);
    const measuredDisc = diWt > 0.1 ? diMut / diWt : (diMut > 0.1 ? Infinity : 1);

    const data = echemE.map((E, i) => ({
      E: +E.toFixed(3),
      baseline: +baseCurve[i].toFixed(4),
      MUT: +mutCurve[i].toFixed(4),
      WT: +wtCurve[i].toFixed(4),
    }));

    return { data, peakBase: +peakBase.toFixed(3), peakMut: +peakMut.toFixed(3), peakWt: +peakWt.toFixed(3), diMut: +diMut.toFixed(1), diWt: +diWt.toFixed(1), measuredDisc: +measuredDisc.toFixed(1), narsilDisc: cd.discrimination };
  }, [echemCandidateData, echemTime, echemKtrans, echemTechnique, echemGamma0_mol, echemE, computeGamma, computeGammaWT, computeSWV, computeDPV, computeCV]);

  // CV duck-shape data for Panels A & C when technique=CV
  const cvDuckData = useMemo(() => {
    if (echemTechnique !== "CV") return null;
    const cd = echemCandidateData;
    const t_s = echemTime * 60;
    const G_base = echemGamma0_mol;
    const G_mut = computeGamma(t_s, cd.efficiency, echemKtrans);
    const G_wt = computeGammaWT(t_s, cd.efficiency, cd.discrimination, echemKtrans, cd.isProximity);
    return {
      baseline: computeCVDuck(G_base), mut: computeCVDuck(G_mut), wt: computeCVDuck(G_wt),
      deltaI: +((1 - G_mut / G_base) * 100).toFixed(1),
    };
  }, [echemTechnique, echemCandidateData, echemTime, echemKtrans, echemGamma0_mol, computeGamma, computeGammaWT, computeCVDuck]);

  // SVG path builder for CV duck loops
  const cvSvgPath = (duck, xS, yS) => {
    const pts = [...duck.forward, ...duck.reverse];
    return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xS(p.E).toFixed(1)} ${yS(p.I).toFixed(1)}`).join(' ') + ' Z';
  };

  return (
    <div>
      {/* ── Explainer box ── */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}25`, borderRadius: "4px", padding: mobile ? "14px" : "16px 20px", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <Grid3x3 size={16} color={T.primary} strokeWidth={1.8} />
          <span style={{ fontSize: "13px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING }}>Multiplex Engineering</span>
        </div>
        <div style={{ fontSize: "13px", color: T.primaryDark, lineHeight: 1.6 }}>
          Spatially-addressed 14-plex electrode array with per-pad crRNA, predicted electrochemical readout (SWV/DPV/CV), cross-reactivity analysis, and in situ RNP formation kinetics. Each detection zone is physically isolated by wax-printed hydrophobic barriers {"\u2014"} enabling simultaneous detection of {drugs.length} drug resistance classes from a single blood sample.
        </div>
      </div>

      {/* ═══════════ SECTION 0: 3D Interactive Chip Render ═══════════ */}
      <CollapsibleSection title="Device Architecture · 3D" defaultOpen={true} badge={{ text: "interactive", bg: T.primaryLight, color: T.primary }}>
        <ChipRender3D
          electrodeLayout={electrodeLayout}
          targetDrug={targetDrug}
          targetStrategy={targetStrategy}
          getEfficiency={getEfficiency}
          results={results}
          computeGamma={computeGamma}
          echemTime={echemTime}
          echemKtrans={echemKtrans}
          echemGamma0_mol={echemGamma0_mol}
          HEADING={HEADING}
          MONO={MONO}
        />
      </CollapsibleSection>

      {/* ═══════════ SECTION 5b: Panel Cross-Reactivity Analysis ═══════════ */}
      <CollapsibleSection title="Panel Cross-Reactivity Analysis" defaultOpen={false} badge={{ text: `${MOCK_CROSS_REACTIVITY.same_gene_pairs.length} pairs`, bg: "#FEF3C720", color: "#D97706" }}>
        <CrossReactivityMatrix />
      </CollapsibleSection>

      {/* ═══════════ SECTION 6: In Situ RNP Formation Kinetics ═══════════ */}
      <CollapsibleSection title="In Situ RNP Formation Kinetics" defaultOpen={false} badge={{ text: kinetics.totals?.total_electrode || "30\u201350 min", bg: "#22c55e20", color: "#22c55e" }}>
        <div style={{ padding: "0", marginBottom: "24px" }}>
          <p style={{ fontSize: "12px", color: T.textSec, marginBottom: "16px", lineHeight: 1.6 }}>
            In situ RNP formation is integral to the per-pad one-pot architecture. Cas12a protein arrives in the sample buffer
            and encounters pad-specific lyophilized crRNA upon rehydration. Gradual RNP formation (Lesinski et al. 2024, <em>Anal. Chem.</em>)
            limits early cis-cleavage competition with RPA — critical at the low template concentrations expected from blood cfDNA.
            RPA amplicons designed at ≤120 bp to fit within cfDNA fragment size distribution (median ~140 bp, range 100–160 bp).
          </p>

          <div style={{ overflowX: "auto", marginBottom: "16px" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
              <thead>
                <tr style={{ borderBottom: `2px solid ${T.border}` }}>
                  <th style={{ textAlign: "left", padding: "8px 12px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>Phase</th>
                  <th style={{ textAlign: "center", padding: "8px 12px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>Solution (lower bound)</th>
                  <th style={{ textAlign: "center", padding: "8px 12px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>On-electrode (estimated)</th>
                </tr>
              </thead>
              <tbody>
                {(kinetics.phases || []).map((phase, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                    <td style={{ padding: "8px 12px", fontWeight: phase.is_bottleneck ? 500 : 400 }}>
                      {phase.phase}
                      {phase.is_bottleneck && <span style={{ fontSize: "9px", fontWeight: 500, color: T.textSec, marginLeft: "8px", padding: "1px 6px", background: T.bgSub, borderRadius: "4px", border: `1px solid ${T.borderLight}` }}>RATE-LIMITING</span>}
                    </td>
                    <td style={{ textAlign: "center", padding: "8px 12px", color: T.textSec, fontFamily: FONT }}>{phase.solution_bound}</td>
                    <td style={{ textAlign: "center", padding: "8px 12px", fontWeight: 600, fontFamily: FONT }}>{phase.on_electrode}</td>
                  </tr>
                ))}
                <tr style={{ borderTop: `2px solid ${T.border}`, fontWeight: 600 }}>
                  <td style={{ padding: "8px 12px" }}>Detection total</td>
                  <td style={{ textAlign: "center", padding: "8px 12px", fontFamily: FONT }}>{kinetics.totals?.detection_solution || "~6\u20138 min"}</td>
                  <td style={{ textAlign: "center", padding: "8px 12px", fontFamily: FONT }}>{kinetics.totals?.detection_electrode || "15\u201330 min"}</td>
                </tr>
                <tr style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                  <td style={{ padding: "8px 12px", color: T.textSec }}>+ RPA amplification</td>
                  <td style={{ textAlign: "center", padding: "8px 12px", color: T.textTer, fontFamily: FONT }}>{kinetics.totals?.rpa_time || "15\u201320 min"}</td>
                  <td style={{ textAlign: "center", padding: "8px 12px", color: T.textTer, fontFamily: FONT }}>{kinetics.totals?.rpa_time || "15\u201320 min"}</td>
                </tr>
                <tr style={{ borderTop: `2px solid ${T.border}`, fontWeight: 600, fontSize: "14px" }}>
                  <td style={{ padding: "10px 12px" }}>Assay total</td>
                  <td style={{ textAlign: "center", padding: "10px 12px", fontFamily: FONT }}>{kinetics.totals?.total_solution || "~23\u201328 min"}</td>
                  <td style={{ textAlign: "center", padding: "10px 12px", fontFamily: FONT }}>{kinetics.totals?.total_electrode || "30\u201350 min"}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div style={{ marginBottom: "16px", padding: "8px 14px", borderRadius: "4px", background: "#22c55e0a", fontSize: "12px", color: T.textSec }}>
            WHO TPP target: {kinetics.totals?.who_tpp_target || "< 120 min"}. Estimated total: {kinetics.totals?.total_electrode || "30\u201350 min"} — <strong style={{ color: T.success }}>within target</strong> with 2-4× margin.
          </div>

          <div style={{ background: T.bgSub, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "12px 16px" }}>
            <div style={{ fontSize: "11px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "8px" }}>Key Insights</div>
            {(kinetics.insights || []).map((ins, i) => (
              <div key={i} style={{ marginBottom: i < (kinetics.insights || []).length - 1 ? "8px" : 0 }}>
                <div style={{ fontSize: "11px", fontWeight: 600, color: T.text }}>{i + 1}. {ins.title}</div>
                <p style={{ fontSize: "11px", color: T.textSec, lineHeight: 1.6, margin: "2px 0 0" }}>{ins.text}</p>
              </div>
            ))}
          </div>
        </div>
      </CollapsibleSection>


      {/* ═══════════ SECTION 2: Predicted Electrochemical Readout ═══════════ */}
      <CollapsibleSection title="Predicted Electrochemical Readout" defaultOpen={true} badge={{ text: "computed", bg: T.primaryLight, color: T.primary }}>
        <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: mobile ? "16px" : "24px", marginBottom: "24px" }}>
          {/* Header description — architecture-dependent */}
          <p style={{ fontSize: "12px", color: T.textSec, marginBottom: "20px", lineHeight: 1.6 }}>
            <strong>SWV, DPV, and CV curves computed from COMPASS pipeline predictions and analytical electrochemistry for {
              echemArch === "A" ? "enzymatic pAP generation (diffusion-controlled, Bezinge 2023)"
              : echemArch === "B" ? "silver anodic stripping voltammetry (Suea-Ngam 2021)"
              : "surface-confined MB (Laviron 1979)"
            }.</strong> {echemArch === "C"
              ? "Peak shapes follow Laviron theory (1979) for adsorbed redox couples."
              : echemArch === "A"
              ? "Peak shapes follow Nicholson-Shain theory for irreversible diffusion-controlled oxidation."
              : "Peak shapes follow anodic stripping voltammetry dissolution kinetics."
            } Relative peak heights between candidates and between MUT/WT alleles are determined by Compass-ML efficiency and discrimination scores (trained on 15K real measurements). Absolute peak currents and detection times depend on electrode-specific parameters (surface trans-cleavage rate, reporter density) provided as adjustable sliders {"\u2014"} to be locked to experimental values after the first electrode characterisation.
          </p>

          {/* ── Row 1: Candidate + Technique ── */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", alignItems: "center", marginBottom: "12px" }}>
            <div style={{ flex: "1 1 300px" }}>
              <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING, textTransform: "uppercase", letterSpacing: "0.5px" }}>Candidate</label>
              <select
                value={echemCandidate}
                onChange={e => setEchemCandidate(e.target.value)}
                style={{ width: "100%", marginTop: "4px", padding: "8px 10px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: MONO, fontSize: "11px", background: T.bg, color: T.text }}
              >
                {electrodeLayout.flat().map(t => {
                  const eff = getEfficiency(t);
                  const r = results.find(x => x.label === t);
                  const disc = r?.disc && r.disc < 900 ? r.disc : null;
                  const strat = targetStrategy(t);
                  const drug = targetDrug(t);
                  return (
                    <option key={t} value={t}>
                      {t} {"\u00b7"} {drug} {"\u00b7"} S_eff={eff.toFixed(3)} {disc ? `\u00b7 D=${disc.toFixed(1)}\u00d7` : ""} {"\u00b7"} {strat}
                    </option>
                  );
                })}
              </select>
            </div>
            <div>
              <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING, textTransform: "uppercase", letterSpacing: "0.5px", display: "block" }}>Technique</label>
              <div style={{ display: "flex", gap: "0", marginTop: "4px" }}>
                {["SWV", "DPV", "CV"].map((tech, i) => (
                  <button
                    key={tech}
                    onClick={() => setEchemTechnique(tech)}
                    style={{
                      padding: "8px 16px", fontSize: "11px", fontWeight: 600, fontFamily: MONO, cursor: "pointer",
                      background: echemTechnique === tech ? T.primary : T.bg,
                      color: echemTechnique === tech ? "#fff" : T.textSec,
                      border: `1px solid ${echemTechnique === tech ? T.primary : T.border}`,
                      borderRadius: i === 0 ? "6px 0 0 6px" : i === 2 ? "0 6px 6px 0" : "0",
                      borderLeft: i > 0 ? "none" : undefined,
                    }}
                  >{tech}</button>
                ))}
              </div>
            </div>
            <div>
              <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING, textTransform: "uppercase", letterSpacing: "0.5px", display: "block" }}>Reporter Architecture</label>
              <div style={{ display: "flex", gap: "0", marginTop: "4px" }}>
                {[
                  { key: "A", label: "A: ALP/pAP" },
                  { key: "B", label: "B: Silver" },
                  { key: "C", label: "C: MB" },
                ].map((arch, i) => (
                  <button
                    key={arch.key}
                    onClick={() => setEchemArch(arch.key)}
                    style={{
                      padding: "8px 12px", fontSize: "10px", fontWeight: 600, fontFamily: MONO, cursor: "pointer",
                      background: echemArch === arch.key ? "#7c3aed" : T.bg,
                      color: echemArch === arch.key ? "#fff" : T.textSec,
                      border: `1px solid ${echemArch === arch.key ? "#7c3aed" : T.border}`,
                      borderRadius: i === 0 ? "6px 0 0 6px" : i === 2 ? "0 6px 6px 0" : "0",
                      borderLeft: i > 0 ? "none" : undefined,
                    }}
                  >{arch.label}</button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Row 2: Main sliders ── */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "16px", marginBottom: "12px" }}>
            <div style={{ flex: "1 1 180px" }}>
              <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>Incubation time: <span style={{ color: T.text, fontFamily: FONT }}>{echemTime} min</span></label>
              <input type="range" min={0} max={60} step={1} value={echemTime} onChange={e => setEchemTime(+e.target.value)}
                style={{ width: "100%", marginTop: "4px", accentColor: T.primary }} />
            </div>
            <div style={{ flex: "1 1 180px" }}>
              <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>Blood cfDNA: <span style={{ color: T.text, fontFamily: FONT }}>{echemBloodTiter} cp/mL</span></label>
              <input type="range" min={0} max={3} step={0.01} value={Math.log10(echemBloodTiter)} onChange={e => setEchemBloodTiter(Math.round(Math.pow(10, +e.target.value)))}
                style={{ width: "100%", marginTop: "4px", accentColor: T.primary }} />
            </div>
            <div style={{ flex: "1 1 180px" }}>
              <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>Surface k_trans: <span style={{ color: T.text, fontFamily: FONT }}>{echemKtrans.toFixed(3)} s{"\u207b\u00b9"}</span></label>
              <input type="range" min={Math.log10(0.0005)} max={Math.log10(0.05)} step={0.01} value={Math.log10(echemKtrans)} onChange={e => setEchemKtrans(+(Math.pow(10, +e.target.value)).toFixed(4))}
                style={{ width: "100%", marginTop: "4px", accentColor: T.primary }} />
            </div>
          </div>

          {/* ── Row 3: Advanced calibration (collapsed) ── */}
          <div style={{ marginBottom: "16px" }}>
            <button onClick={() => setEchemAdvanced(!echemAdvanced)} style={{ background: "none", border: "none", cursor: "pointer", fontSize: "11px", color: T.textSec, fontFamily: HEADING, fontWeight: 600, display: "flex", alignItems: "center", gap: "4px", padding: "4px 0" }}>
              {echemAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              Advanced Calibration
            </button>
            {echemAdvanced && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "16px", marginTop: "8px", padding: "12px 16px", background: T.bgSub, borderRadius: "4px", border: `1px solid ${T.borderLight}` }}>
                <div style={{ flex: "1 1 160px" }}>
                  <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>{"\u0393\u2080"}: <span style={{ fontFamily: FONT, color: T.text }}>{echemGamma0.toExponential(1)} mol/cm{"\u00b2"}</span></label>
                  <input type="range" min={9} max={12} step={0.1} value={Math.log10(echemGamma0)} onChange={e => setEchemGamma0(Math.pow(10, +e.target.value))}
                    style={{ width: "100%", marginTop: "4px", accentColor: T.primary }} />
                </div>
                <div style={{ flex: "1 1 160px" }}>
                  <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>Porosity {"\u03b7"}: <span style={{ fontFamily: FONT, color: T.text }}>{echemPorosity.toFixed(1)}</span></label>
                  <input type="range" min={1} max={8} step={0.1} value={echemPorosity} onChange={e => setEchemPorosity(+e.target.value)}
                    style={{ width: "100%", marginTop: "4px", accentColor: T.primary }} />
                </div>
                <div style={{ flex: "1 1 160px" }}>
                  <label style={{ fontSize: "10px", fontWeight: 600, color: T.textSec, fontFamily: HEADING }}>Peak scale: <span style={{ fontFamily: FONT, color: T.text }}>{echemIscale.toFixed(1)} {"\u03bcA"}</span></label>
                  <input type="range" min={0.1} max={10} step={0.1} value={echemIscale} onChange={e => setEchemIscale(+e.target.value)}
                    style={{ width: "100%", marginTop: "4px", accentColor: T.primary }} />
                </div>
                <div style={{ fontSize: "10px", color: T.textTer, width: "100%" }}>
                  These only affect absolute current, NOT {"\u0394"}I% predictions. {"\u0394"}I% depends only on COMPASS data + k_trans + time.
                </div>
              </div>
            )}
          </div>

          {/* ── 3-Panel Grid: A (voltammogram), B (time course), C (discrimination) ── */}
          {(() => {
            // Electrochemistry panel color palette
            const EC = { green: "#10b981", greenLight: "#10b98120", purple: "#8b5cf6", purpleLight: "#8b5cf610", pink: "#ec4899", pinkLight: "#ec489915", gray: "#9CA3AF", blue: "#4338CA", orange: "#4338CA" };
            // Architecture-aware E-axis helpers
            const eS = archCfg.E_start, eE = archCfg.E_end;
            const eRange = eE - eS;
            const eTickCount = Math.max(4, Math.min(9, Math.round(Math.abs(eRange) / 0.05)));
            const eTicks = Array.from({ length: eTickCount + 1 }, (_, i) => +(eS + eRange * (i / eTickCount)).toFixed(2));
            const echemE0 = ECHEM.E0;
            return (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "16px" }}>

            {/* ═══ PANEL A: Voltammogram — negative (cathodic) MB peaks ═══ */}
            <div style={{ borderRadius: "4px", padding: "16px", background: "#FAFAFA" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>
                  A {"\u00b7"} {echemTechnique} Voltammogram
                </div>
                {echemTechnique === "SWV" && (
                  <button onClick={() => setEchemShowFwdRev(!echemShowFwdRev)} style={{
                    background: echemShowFwdRev ? EC.purple + "18" : "transparent", border: `1px solid ${echemShowFwdRev ? EC.purple : T.border}`,
                    borderRadius: "4px", padding: "2px 8px", fontSize: "9px", fontFamily: MONO, fontWeight: 600,
                    color: echemShowFwdRev ? EC.purple : T.textSec, cursor: "pointer",
                  }}>i_fwd / i_rev</button>
                )}
              </div>
              <div style={{ fontSize: "11px", color: T.textSec, marginBottom: "8px", fontFamily: FONT }}>
                <strong style={{ color: T.text }}>{echemCandidateData.label}</strong>
                {" \u00b7 "}{"\u0394"}I% = <span style={{ fontWeight: 600, color: EC.purple }}>{echemMeta.deltaI}%</span>
                {" \u00b7 "}{echemArch === "C" ? (Math.abs(echemMeta.peakBase) * 1000).toFixed(1) : Math.abs(echemMeta.peakBase).toFixed(3)} {echemArch === "C" ? "nA" : "\u03bcA"} {"\u2192"} {echemArch === "C" ? (Math.abs(echemMeta.peakAfter) * 1000).toFixed(1) : Math.abs(echemMeta.peakAfter).toFixed(3)} {echemArch === "C" ? "nA" : "\u03bcA"}
                {echemCandidateData.discrimination <= 2.0 && echemCandidateData.discrimination < 900 && (
                  <div style={{ marginTop: "4px", padding: "3px 8px", background: "#FEF3C7", borderRadius: "3px", fontSize: "9px", color: "#92400E", lineHeight: 1.5, fontFamily: MONO }}>
                    {"\u26A0"} D = {echemCandidateData.discrimination.toFixed(1)}{"\u00d7"} {"\u2014"} WT allele {"\u0394"}I% {"\u2248"} MUT {"\u0394"}I% (S_eff_WT = {(echemCandidateData.efficiency / echemCandidateData.discrimination).toFixed(3)}). Clinical discrimination relies entirely on AS-RPA primer selectivity, not crRNA alone.
                  </div>
                )}
              </div>
              <div style={{ width: "100%", height: 280 }}>
                {(() => {
                  const mg = { top: 24, right: 20, bottom: 38, left: 54 };
                  const w = 480, h = 280, pw = w - mg.left - mg.right, ph = h - mg.top - mg.bottom;

                  if (echemTechnique === "CV" && cvDuckData) {
                    const allI = [cvDuckData.baseline, cvDuckData.mut].flatMap(d => [...d.forward, ...d.reverse]).map(p => p.I);
                    const iMin = Math.min(...allI) * 1.2, iMax = Math.max(...allI) * 1.2;
                    const xS = e => mg.left + ((e - eS) / eRange) * pw;
                    const yS = i => mg.top + ((iMax - i) / (iMax - iMin)) * ph;
                    const zeroY = yS(0);
                    return (
                      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height="100%" style={{ fontFamily: MONO }}>
                        <rect width={w} height={h} fill="#FAFAFA" rx="8" />
                        {[0.25, 0.5, 0.75].map(f => <line key={`gx${f}`} x1={mg.left + pw * f} y1={mg.top} x2={mg.left + pw * f} y2={mg.top + ph} stroke="#E8E8E8" strokeWidth="0.5" />)}
                        {[0.25, 0.5, 0.75].map(f => <line key={`gy${f}`} x1={mg.left} y1={mg.top + ph * f} x2={mg.left + pw} y2={mg.top + ph * f} stroke="#E8E8E8" strokeWidth="0.5" />)}
                        <line x1={mg.left} y1={zeroY} x2={mg.left + pw} y2={zeroY} stroke="#BBB" strokeWidth="1" strokeDasharray="4,3" />
                        <line x1={mg.left} y1={mg.top} x2={mg.left} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                        <line x1={mg.left} y1={mg.top + ph} x2={mg.left + pw} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                        <path d={cvSvgPath(cvDuckData.baseline, xS, yS)} fill="none" stroke={EC.green} strokeWidth="2" strokeDasharray="7,4" opacity="0.75" />
                        <path d={cvSvgPath(cvDuckData.mut, xS, yS)} fill={EC.purpleLight} stroke={EC.purple} strokeWidth="2.5" />
                        {/* ΔEp annotation */}
                        {cvDuckData.baseline.E_pc != null && cvDuckData.baseline.E_pa != null && (
                          <>
                            <line x1={xS(cvDuckData.baseline.E_pc)} y1={mg.top + ph * 0.15} x2={xS(cvDuckData.baseline.E_pa)} y2={mg.top + ph * 0.15} stroke="#666" strokeWidth="1" markerEnd="url(#arrowR)" markerStart="url(#arrowL)" />
                            <text x={(xS(cvDuckData.baseline.E_pc) + xS(cvDuckData.baseline.E_pa)) / 2} y={mg.top + ph * 0.12} textAnchor="middle" fill="#666" fontSize="8">{"\u0394"}Ep = {(cvDuckData.baseline.deltaEp * 1000).toFixed(0)} mV</text>
                          </>
                        )}
                        <defs>
                          <marker id="arrowR" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="#666" /></marker>
                          <marker id="arrowL" markerWidth="6" markerHeight="4" refX="1" refY="2" orient="auto"><path d="M6,0 L0,2 L6,4" fill="#666" /></marker>
                        </defs>
                        <line x1={xS(echemE0)} y1={mg.top} x2={xS(echemE0)} y2={mg.top + ph} stroke="#999" strokeWidth="0.8" strokeDasharray="3,3" />
                        <text x={xS(echemE0)} y={mg.top - 5} textAnchor="middle" fill="#999" fontSize="8">E{"\u00b0"} = {echemE0 < 0 ? "\u2212" : ""}{Math.abs(echemE0).toFixed(2)} V</text>
                        <text x={mg.left + pw / 2} y={h - 3} textAnchor="middle" fill="#444" fontSize="10">E (V vs Ag/AgCl)</text>
                        <text x={14} y={mg.top + ph / 2} textAnchor="middle" fill="#444" fontSize="10" transform={`rotate(-90, 14, ${mg.top + ph / 2})`}>I ({"\u03bc"}A)</text>
                        {eTicks.map(e => <text key={e} x={xS(e)} y={mg.top + ph + 14} textAnchor="middle" fill="#666" fontSize="8">{e.toFixed(2)}</text>)}
                        {/* Legend */}
                        <line x1={mg.left + 10} y1={mg.top + 10} x2={mg.left + 30} y2={mg.top + 10} stroke={EC.green} strokeWidth="2" strokeDasharray="5,3" />
                        <text x={mg.left + 34} y={mg.top + 14} fill={EC.green} fontSize="9" fontWeight="600">Baseline ({"\u0393\u2080"})</text>
                        <line x1={mg.left + 10} y1={mg.top + 24} x2={mg.left + 30} y2={mg.top + 24} stroke={EC.purple} strokeWidth="2.5" />
                        <text x={mg.left + 34} y={mg.top + 28} fill={EC.purple} fontSize="9" fontWeight="600">After cleavage ({"\u0394"}I={cvDuckData.deltaI}%)</text>
                      </svg>
                    );
                  }

                  // SWV / DPV — architecture-aware positive peaks
                  const data = echemPlotData;
                  const allY = data.flatMap(d => {
                    const vals = [d.baseline, d.after];
                    if (echemShowFwdRev && d.base_fwd != null) vals.push(d.base_fwd, d.base_rev, d.after_fwd, d.after_rev);
                    return vals;
                  });
                  const yMin = Math.min(0, Math.min(...allY) * 1.1 - 0.02);
                  const yMax = Math.max(...allY) * 1.15 + 0.02;
                  const xS = e => mg.left + ((e - eS) / eRange) * pw;
                  const yS = v => mg.top + ((yMax - v) / (yMax - yMin)) * ph;
                  const zeroY = yS(0);
                  const pathD = (key) => data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xS(d.E).toFixed(1)} ${yS(d[key]).toFixed(1)}`).join(' ');
                  // Fill area between baseline and after curves (shaded ΔI region)
                  const fillPath = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xS(d.E).toFixed(1)} ${yS(d.baseline).toFixed(1)}`).join(' ')
                    + data.slice().reverse().map((d, i) => `L${xS(d.E).toFixed(1)} ${yS(d.after).toFixed(1)}`).join(' ') + ' Z';

                  // Y-axis ticks
                  const yRange = yMax - yMin;
                  const yTickStep = yRange > 2 ? 1 : yRange > 1 ? 0.5 : yRange > 0.5 ? 0.2 : yRange > 0.1 ? 0.05 : 0.02;
                  const yTicks = [];
                  for (let v = Math.ceil(yMin / yTickStep) * yTickStep; v <= yMax; v += yTickStep) yTicks.push(+v.toFixed(3));

                  // Current unit: nA for MB architecture, µA for others
                  const currentUnit = echemArch === "C" ? "nA" : "\u03bcA";
                  const currentScale = echemArch === "C" ? 1000 : 1; // multiply for display

                  return (
                    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height="100%" style={{ fontFamily: MONO }}>
                      <rect width={w} height={h} fill="#FAFAFA" rx="8" />
                      {/* Grid */}
                      {[0.25, 0.5, 0.75].map(f => <line key={`gx${f}`} x1={mg.left + pw * f} y1={mg.top} x2={mg.left + pw * f} y2={mg.top + ph} stroke="#E8E8E8" strokeWidth="0.5" />)}
                      {yTicks.map(v => <line key={`gy${v}`} x1={mg.left} y1={yS(v)} x2={mg.left + pw} y2={yS(v)} stroke="#E8E8E8" strokeWidth="0.5" />)}
                      {/* Zero line */}
                      <line x1={mg.left} y1={zeroY} x2={mg.left + pw} y2={zeroY} stroke="#BBB" strokeWidth="1" strokeDasharray="4,3" />
                      {/* Axes */}
                      <line x1={mg.left} y1={mg.top} x2={mg.left} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                      <line x1={mg.left} y1={mg.top + ph} x2={mg.left + pw} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                      {/* Shaded ΔI region between baseline and after */}
                      <path d={fillPath} fill={EC.purple} opacity="0.08" />
                      {/* SWV forward/reverse components — blue (fwd), orange (rev) */}
                      {echemShowFwdRev && echemTechnique === "SWV" && data[0].base_fwd != null && (
                        <>
                          <path d={pathD("base_fwd")} fill="none" stroke={EC.blue} strokeWidth="1.5" opacity="0.6" />
                          <path d={pathD("base_rev")} fill="none" stroke={EC.orange} strokeWidth="1.5" opacity="0.6" />
                          <path d={pathD("after_fwd")} fill="none" stroke={EC.blue} strokeWidth="1.2" opacity="0.35" strokeDasharray="4,2" />
                          <path d={pathD("after_rev")} fill="none" stroke={EC.orange} strokeWidth="1.2" opacity="0.35" strokeDasharray="4,2" />
                        </>
                      )}
                      {/* Main curves — i_net (purple) */}
                      <path d={pathD("baseline")} fill="none" stroke={EC.green} strokeWidth="2.2" strokeDasharray="7,4" opacity="0.85" />
                      <path d={pathD("after")} fill="none" stroke={EC.purple} strokeWidth="2.5" />
                      {/* E° reference */}
                      <line x1={xS(echemE0)} y1={mg.top} x2={xS(echemE0)} y2={mg.top + ph} stroke="#999" strokeWidth="0.8" strokeDasharray="3,3" />
                      <text x={xS(echemE0)} y={mg.top - 5} textAnchor="middle" fill="#999" fontSize="8">E{"\u00b0"} = {echemE0 < 0 ? "\u2212" : ""}{Math.abs(echemE0).toFixed(2)} V</text>
                      {/* Peak annotations with arrows */}
                      {(() => {
                        const peakIdx = data.reduce((best, d, i) => Math.abs(d.baseline) > Math.abs(data[best].baseline) ? i : best, 0);
                        const peakE = data[peakIdx].E;
                        const bY = yS(data[peakIdx].baseline);
                        const aY = yS(data[peakIdx].after);
                        return (
                          <>
                            {/* ΔI bracket */}
                            <line x1={xS(peakE) + 8} y1={bY} x2={xS(peakE) + 8} y2={aY} stroke={EC.purple} strokeWidth="1.5" />
                            <line x1={xS(peakE) + 4} y1={bY} x2={xS(peakE) + 12} y2={bY} stroke={EC.purple} strokeWidth="1" />
                            <line x1={xS(peakE) + 4} y1={aY} x2={xS(peakE) + 12} y2={aY} stroke={EC.purple} strokeWidth="1" />
                            <text x={xS(peakE) + 16} y={(bY + aY) / 2 + 3} fill={EC.purple} fontSize="9" fontWeight="700">{"\u0394"}I={echemMeta.deltaI}%</text>
                          </>
                        );
                      })()}
                      {/* X ticks */}
                      {eTicks.map(e => <text key={e} x={xS(e)} y={mg.top + ph + 14} textAnchor="middle" fill="#666" fontSize="8">{e.toFixed(2)}</text>)}
                      {/* Y ticks */}
                      {yTicks.map(v => <text key={v} x={mg.left - 5} y={yS(v) + 3} textAnchor="end" fill="#666" fontSize="8">{(v * currentScale).toFixed(currentScale > 1 ? 0 : 1)}</text>)}
                      {/* Axis labels */}
                      <text x={mg.left + pw / 2} y={h - 3} textAnchor="middle" fill="#444" fontSize="10">E (V vs Ag/AgCl)</text>
                      <text x={14} y={mg.top + ph / 2} textAnchor="middle" fill="#444" fontSize="10" transform={`rotate(-90, 14, ${mg.top + ph / 2})`}>I ({currentUnit})</text>
                      {/* Legend */}
                      <line x1={mg.left + 10} y1={mg.top + 10} x2={mg.left + 30} y2={mg.top + 10} stroke={EC.green} strokeWidth="2" strokeDasharray="5,3" />
                      <text x={mg.left + 34} y={mg.top + 14} fill={EC.green} fontSize="9" fontWeight="600">Baseline ({"\u0393\u2080"})</text>
                      <line x1={mg.left + 10} y1={mg.top + 24} x2={mg.left + 30} y2={mg.top + 24} stroke={EC.purple} strokeWidth="2.5" />
                      <text x={mg.left + 34} y={mg.top + 28} fill={EC.purple} fontSize="9" fontWeight="600">{echemShowFwdRev ? "i_net" : `+${echemCandidateData.label}`} ({echemTime} min)</text>
                      {echemShowFwdRev && echemTechnique === "SWV" && (
                        <>
                          <line x1={mg.left + 10} y1={mg.top + 38} x2={mg.left + 30} y2={mg.top + 38} stroke={EC.blue} strokeWidth="1.5" />
                          <text x={mg.left + 34} y={mg.top + 42} fill={EC.blue} fontSize="8" fontWeight="600">i_fwd</text>
                          <line x1={mg.left + 70} y1={mg.top + 38} x2={mg.left + 90} y2={mg.top + 38} stroke={EC.orange} strokeWidth="1.5" />
                          <text x={mg.left + 94} y={mg.top + 42} fill={EC.orange} fontSize="8" fontWeight="600">i_rev</text>
                        </>
                      )}
                      {/* FWHM + peak shape annotation */}
                      <text x={w - mg.right - 4} y={h - 22} textAnchor="end" fill="#999" fontSize="7">
                        {archCfg.peak_shape === "sech2"
                          ? `FWHM \u2248 ${echemTechnique === "SWV" ? "62" : "63"} mV (${echemTechnique === "SWV" ? "3.53 RT/nF" : "broadened by \u0394E = 50 mV"})`
                          : archCfg.peak_shape === "asymmetric"
                          ? `Irreversible oxidation (\u03b1 = ${archCfg.alpha})`
                          : `Stripping (\u03c3\u2090 = ${(archCfg.sigma_onset * 1000).toFixed(0)} mV, \u03c3\u209c = ${(archCfg.sigma_tail * 1000).toFixed(0)} mV)`
                        }
                      </text>
                    </svg>
                  );
                })()}
              </div>
              {/* Interpretation block — Panel A */}
              <div style={{ marginTop: "8px", padding: "10px 14px", borderRadius: "4px", background: T.primaryLight, fontSize: "11px", color: T.primaryDark, lineHeight: 1.6 }}>
                <strong>Interpretation:</strong>{" "}
                {echemMeta.deltaI > 50
                  ? `Strong signal reduction (${"\u0394"}I = ${echemMeta.deltaI}%). Trans-cleavage of MB reporters produces a clearly resolvable voltammetric shift at ${echemTime} min. The ${echemCandidateData.label} crRNA yields sufficient on-target activity for unambiguous electrochemical detection.`
                  : echemMeta.deltaI > 15
                  ? `Moderate signal reduction (${"\u0394"}I = ${echemMeta.deltaI}%). The ${echemCandidateData.label} crRNA generates a detectable but sub-optimal peak shift at ${echemTime} min. Extending incubation time or increasing k_trans via electrode surface optimization would improve signal-to-noise.`
                  : `Weak signal reduction (${"\u0394"}I = ${echemMeta.deltaI}%). At ${echemTime} min, the trans-cleavage signal for ${echemCandidateData.label} is near the detection limit. Consider increasing reporter density, incubation time, or optimizing surface chemistry to enhance k_trans.`
                }
                {echemCandidateData.discrimination <= 2.0 && echemCandidateData.discrimination < 900 &&
                  ` Note: D = ${echemCandidateData.discrimination.toFixed(1)}\u00d7 indicates poor crRNA-level discrimination \u2014 allelic specificity depends on AS-RPA primer selectivity.`
                }
              </div>
            </div>

            {/* ═══ PANEL B: ΔI% Time Course with lag phase + threshold zones ═══ */}
            <div style={{ borderRadius: "4px", padding: "16px", background: "#FAFAFA" }}>
              <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "4px" }}>
                B {"\u00b7"} {"\u0394"}I% Time Course
              </div>
              <div style={{ fontSize: "10px", color: T.textSec, marginBottom: "8px", fontFamily: FONT }}>
                MUT: <span style={{ fontWeight: 600, color: EC.purple }}>{echemTimeCourse.timeMut != null ? `t_det \u2248 ${echemTimeCourse.timeMut} min` : ">60 min"}</span>
                {" \u00b7 "}WT: <span style={{ fontWeight: 600, color: EC.pink }}>{echemTimeCourse.timeWt != null ? `~${echemTimeCourse.timeWt} min` : "below threshold"}</span>
                {echemTimeCourse.timeMut != null && echemTimeCourse.timeWt != null && echemTimeCourse.timeWt > echemTimeCourse.timeMut && (
                  <span> {"\u00b7"} <span style={{ fontWeight: 600, color: EC.green }}>Window: {echemTimeCourse.timeMut}{"\u2013"}{echemTimeCourse.timeWt} min</span></span>
                )}
              </div>
              <div style={{ width: "100%", height: 280 }}>
                {(() => {
                  const mg = { top: 20, right: 24, bottom: 38, left: 48 };
                  const w = 480, h = 280, pw = w - mg.left - mg.right, ph = h - mg.top - mg.bottom;
                  const pts = echemTimeCourse.points;
                  const xS = t => mg.left + (t / 60) * pw;
                  const yS = v => mg.top + ((100 - v) / 100) * ph;
                  const pathD = (key) => pts.filter(p => p[key] != null).map((p, i) => `${i === 0 ? 'M' : 'L'}${xS(p.time).toFixed(1)} ${yS(p[key]).toFixed(1)}`).join(' ');
                  // Uncertainty band fill
                  const bandPath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xS(p.time).toFixed(1)} ${yS(p.MUT_hi).toFixed(1)}`).join(' ')
                    + pts.slice().reverse().map(p => `L${xS(p.time).toFixed(1)} ${yS(p.MUT_lo).toFixed(1)}`).join(' ') + ' Z';
                  const thr = echemTimeCourse.threshold;
                  return (
                    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height="100%" style={{ fontFamily: MONO }}>
                      <rect width={w} height={h} fill="#FAFAFA" rx="8" />
                      {/* Threshold zones */}
                      {/* Positive zone: >30% — green tint */}
                      <rect x={mg.left} y={yS(100)} width={pw} height={yS(30) - yS(100)} fill={EC.green} opacity="0.04" />
                      {/* Indeterminate zone: 5-30% — yellow tint */}
                      <rect x={mg.left} y={yS(30)} width={pw} height={yS(5) - yS(30)} fill="#f59e0b" opacity="0.05" />
                      {/* Negative zone: <5% — red tint */}
                      <rect x={mg.left} y={yS(5)} width={pw} height={yS(0) - yS(5)} fill="#ef4444" opacity="0.04" />
                      {/* Zone labels */}
                      <text x={w - mg.right - 2} y={yS(65)} textAnchor="end" fill={EC.green} fontSize="7" fontWeight="600" opacity="0.7">POSITIVE</text>
                      <text x={w - mg.right - 2} y={yS(17)} textAnchor="end" fill="#f59e0b" fontSize="7" fontWeight="600" opacity="0.7">INDETERMINATE</text>
                      <text x={w - mg.right - 2} y={yS(2)} textAnchor="end" fill="#ef4444" fontSize="7" fontWeight="600" opacity="0.7">NEGATIVE</text>
                      {/* Grid */}
                      {[10, 20, 30, 40, 50].map(t => <line key={`gx${t}`} x1={xS(t)} y1={mg.top} x2={xS(t)} y2={mg.top + ph} stroke="#E8E8E8" strokeWidth="0.5" />)}
                      {[20, 40, 60, 80].map(v => <line key={`gy${v}`} x1={mg.left} y1={yS(v)} x2={mg.left + pw} y2={yS(v)} stroke="#E8E8E8" strokeWidth="0.5" />)}
                      {/* Axes */}
                      <line x1={mg.left} y1={mg.top} x2={mg.left} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                      <line x1={mg.left} y1={mg.top + ph} x2={mg.left + pw} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                      {/* Threshold lines */}
                      <line x1={mg.left} y1={yS(30)} x2={mg.left + pw} y2={yS(30)} stroke={EC.green} strokeWidth="0.8" strokeDasharray="4,3" opacity="0.5" />
                      <line x1={mg.left} y1={yS(5)} x2={mg.left + pw} y2={yS(5)} stroke="#ef4444" strokeWidth="0.8" strokeDasharray="4,3" opacity="0.5" />
                      {/* 3σ threshold */}
                      <line x1={mg.left} y1={yS(thr)} x2={mg.left + pw} y2={yS(thr)} stroke="#C0392B" strokeWidth="1.2" strokeDasharray="6,3" />
                      <text x={mg.left + 4} y={yS(thr) - 4} fill="#C0392B" fontSize="8" fontWeight="600">3{"\u03c3"} = {thr.toFixed(1)}%</text>
                      {/* Lag phase indicator */}
                      <rect x={mg.left} y={mg.top} width={xS(7) - mg.left} height={ph} fill="#666" opacity="0.03" />
                      <text x={(mg.left + xS(7)) / 2} y={mg.top + ph - 6} textAnchor="middle" fill="#999" fontSize="7">RNP lag</text>
                      {/* Uncertainty band */}
                      <path d={bandPath} fill={EC.purple} opacity="0.1" />
                      {/* Main curves */}
                      <path d={pathD("MUT")} fill="none" stroke={EC.purple} strokeWidth="2.5" />
                      <path d={pathD("WT")} fill="none" stroke={EC.pink} strokeWidth="2" opacity="0.85" />
                      <path d={pathD("neg")} fill="none" stroke="#C0C0C0" strokeWidth="1" strokeDasharray="6,4" />
                      {/* Current time marker */}
                      <line x1={xS(echemTime)} y1={mg.top} x2={xS(echemTime)} y2={mg.top + ph} stroke="#666" strokeWidth="1" strokeDasharray="4,2" />
                      <text x={xS(echemTime)} y={mg.top - 4} textAnchor="middle" fill="#666" fontSize="8">t={echemTime}</text>
                      {/* Detection time markers */}
                      {echemTimeCourse.timeMut != null && (
                        <circle cx={xS(echemTimeCourse.timeMut)} cy={yS(pts.find(p => p.time === echemTimeCourse.timeMut)?.MUT || thr)} r="4" fill={EC.purple} stroke="#fff" strokeWidth="1.5" />
                      )}
                      {echemTimeCourse.timeWt != null && (
                        <circle cx={xS(echemTimeCourse.timeWt)} cy={yS(pts.find(p => p.time === echemTimeCourse.timeWt)?.WT || thr)} r="3.5" fill={EC.pink} stroke="#fff" strokeWidth="1.5" />
                      )}
                      {/* X ticks */}
                      {[0, 10, 20, 30, 40, 50, 60].map(t => <text key={t} x={xS(t)} y={mg.top + ph + 14} textAnchor="middle" fill="#666" fontSize="8">{t}</text>)}
                      {/* Y ticks */}
                      {[0, 20, 40, 60, 80, 100].map(v => <text key={v} x={mg.left - 5} y={yS(v) + 3} textAnchor="end" fill="#666" fontSize="8">{v}</text>)}
                      {/* Axis labels */}
                      <text x={mg.left + pw / 2} y={h - 3} textAnchor="middle" fill="#444" fontSize="10">Time (min)</text>
                      <text x={14} y={mg.top + ph / 2} textAnchor="middle" fill="#444" fontSize="10" transform={`rotate(-90, 14, ${mg.top + ph / 2})`}>{"\u0394"}I%</text>
                      {/* Legend */}
                      <rect x={mg.left + pw - 130} y={mg.top + 4} width="126" height="44" rx="4" fill="#fff" stroke="#E8E8E8" strokeWidth="0.5" opacity="0.9" />
                      <line x1={mg.left + pw - 124} y1={mg.top + 16} x2={mg.left + pw - 108} y2={mg.top + 16} stroke={EC.purple} strokeWidth="2.5" />
                      <text x={mg.left + pw - 104} y={mg.top + 19} fill={EC.purple} fontSize="8" fontWeight="600">MUT (k_trans={echemKtrans.toFixed(3)})</text>
                      <line x1={mg.left + pw - 124} y1={mg.top + 28} x2={mg.left + pw - 108} y2={mg.top + 28} stroke={EC.pink} strokeWidth="2" />
                      <text x={mg.left + pw - 104} y={mg.top + 31} fill={EC.pink} fontSize="8" fontWeight="600">WT (S_eff/D)</text>
                      <line x1={mg.left + pw - 124} y1={mg.top + 40} x2={mg.left + pw - 108} y2={mg.top + 40} stroke="#C0C0C0" strokeWidth="1" strokeDasharray="4,2" />
                      <text x={mg.left + pw - 104} y={mg.top + 43} fill="#999" fontSize="8">Neg ctrl</text>
                    </svg>
                  );
                })()}
              </div>
              {/* Interpretation block — Panel B */}
              <div style={{ marginTop: "8px", padding: "10px 14px", borderRadius: "4px", background: "#F0FDF4", fontSize: "11px", color: "#166534", lineHeight: 1.6 }}>
                <strong>Interpretation:</strong>{" "}
                {echemTimeCourse.timeMut != null && echemTimeCourse.timeMut <= 20
                  ? `Rapid detection: MUT signal crosses the 3\u03c3 threshold at ~${echemTimeCourse.timeMut} min, well within the WHO TPP target of <120 min. `
                  : echemTimeCourse.timeMut != null
                  ? `Detection at ~${echemTimeCourse.timeMut} min. Signal accumulation is slower than ideal \u2014 increasing k_trans or extending RPA amplification could accelerate time-to-result. `
                  : `MUT signal does not reach the 3\u03c3 threshold within 60 min at current k_trans. Surface optimization required. `
                }
                {echemTimeCourse.timeWt != null && echemTimeCourse.timeMut != null && echemTimeCourse.timeWt > echemTimeCourse.timeMut
                  ? `A ${echemTimeCourse.timeWt - echemTimeCourse.timeMut}-min discrimination window separates MUT from WT detection, enabling time-gated allelic discrimination.`
                  : echemTimeCourse.timeWt == null
                  ? `WT remains below threshold throughout, providing clean allelic discrimination at all time points.`
                  : `WT crosses threshold near MUT \u2014 allelic discrimination relies on AS-RPA primer specificity rather than crRNA kinetics.`
                }
              </div>
            </div>

            {/* ═══ PANEL C: MUT vs WT Discrimination Overlay ═══ */}
            <div style={{ borderRadius: "4px", padding: "16px", background: "#FAFAFA" }}>
              <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "4px" }}>
                C {"\u00b7"} MUT vs WT Allelic Discrimination
              </div>
              <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", fontSize: "11px", color: T.textSec, marginBottom: "10px", fontFamily: FONT }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <span style={{ width: 10, height: 3, borderRadius: 2, background: EC.purple, display: "inline-block" }} />
                  <span>MUT: <strong style={{ color: EC.purple }}>{echemArch === "C" ? (Math.abs(echemDiscOverlay.peakMut) * 1000).toFixed(1) : Math.abs(echemDiscOverlay.peakMut).toFixed(3)} {echemArch === "C" ? "nA" : "\u03bcA"}</strong> ({"\u0394"}I={echemDiscOverlay.diMut}%)</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <span style={{ width: 10, height: 3, borderRadius: 2, background: EC.pink, display: "inline-block" }} />
                  <span>WT: <strong style={{ color: EC.pink }}>{echemArch === "C" ? (Math.abs(echemDiscOverlay.peakWt) * 1000).toFixed(1) : Math.abs(echemDiscOverlay.peakWt).toFixed(3)} {echemArch === "C" ? "nA" : "\u03bcA"}</strong> ({"\u0394"}I={echemDiscOverlay.diWt}%)</span>
                </div>
                <div>
                  Disc: <strong style={{ color: echemDiscOverlay.measuredDisc < 1 ? "#ef4444" : echemDiscOverlay.measuredDisc < 2 ? "#f59e0b" : EC.purple }}>{echemDiscOverlay.measuredDisc === Infinity ? "\u221e" : `${echemDiscOverlay.measuredDisc}\u00d7`}</strong>
                  <span style={{ color: T.textTer }}> (COMPASS: {echemDiscOverlay.narsilDisc >= 900 ? "\u221e" : `${echemDiscOverlay.narsilDisc}\u00d7`})</span>
                </div>
                {echemDiscOverlay.narsilDisc < 1 && <div style={{ color: "#ef4444", fontWeight: 600 }}>{"\u26a0"} D {"<"} 1: WT activates more than MUT</div>}
                {echemDiscOverlay.measuredDisc < 2 && echemDiscOverlay.measuredDisc !== Infinity && echemDiscOverlay.narsilDisc >= 1 && <div style={{ color: "#f59e0b", fontWeight: 600 }}>{"\u26a0"} poor discrimination</div>}
              </div>
              <div style={{ width: "100%", height: 280 }}>
                {(() => {
                  const mg = { top: 24, right: 24, bottom: 38, left: 54 };
                  const w = mobile ? 480 : 920, h = 280, pw = w - mg.left - mg.right, ph = h - mg.top - mg.bottom;

                  if (echemTechnique === "CV" && cvDuckData) {
                    const allI = [cvDuckData.baseline, cvDuckData.mut, cvDuckData.wt].flatMap(d => [...d.forward, ...d.reverse]).map(p => p.I);
                    const iMin = Math.min(...allI) * 1.15, iMax = Math.max(...allI) * 1.15;
                    const xS = e => mg.left + ((e - eS) / eRange) * pw;
                    const yS = i => mg.top + ((iMax - i) / (iMax - iMin)) * ph;
                    return (
                      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height="100%" style={{ fontFamily: MONO }}>
                        <rect width={w} height={h} fill="#FAFAFA" rx="8" />
                        {[0.25, 0.5, 0.75].map(f => <line key={`g${f}`} x1={mg.left + pw * f} y1={mg.top} x2={mg.left + pw * f} y2={mg.top + ph} stroke="#E8E8E8" strokeWidth="0.5" />)}
                        <line x1={mg.left} y1={yS(0)} x2={mg.left + pw} y2={yS(0)} stroke="#BBB" strokeWidth="1" strokeDasharray="4,3" />
                        <line x1={mg.left} y1={mg.top} x2={mg.left} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                        <line x1={mg.left} y1={mg.top + ph} x2={mg.left + pw} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                        <path d={cvSvgPath(cvDuckData.baseline, xS, yS)} fill="none" stroke={EC.green} strokeWidth="2" strokeDasharray="6,4" opacity="0.6" />
                        <path d={cvSvgPath(cvDuckData.wt, xS, yS)} fill={EC.pinkLight} stroke={EC.pink} strokeWidth="2" opacity="0.85" />
                        <path d={cvSvgPath(cvDuckData.mut, xS, yS)} fill={EC.purpleLight} stroke={EC.purple} strokeWidth="2.5" />
                        <line x1={xS(echemE0)} y1={mg.top} x2={xS(echemE0)} y2={mg.top + ph} stroke="#999" strokeWidth="0.8" strokeDasharray="3,3" />
                        <text x={mg.left + pw / 2} y={h - 3} textAnchor="middle" fill="#444" fontSize="10">E (V vs Ag/AgCl)</text>
                        <text x={14} y={mg.top + ph / 2} textAnchor="middle" fill="#444" fontSize="10" transform={`rotate(-90, 14, ${mg.top + ph / 2})`}>I ({"\u03bc"}A)</text>
                        <line x1={mg.left + 10} y1={mg.top + 10} x2={mg.left + 30} y2={mg.top + 10} stroke={EC.green} strokeWidth="2" strokeDasharray="5,3" />
                        <text x={mg.left + 34} y={mg.top + 14} fill={EC.green} fontSize="9" fontWeight="600">Baseline</text>
                        <line x1={mg.left + 10} y1={mg.top + 24} x2={mg.left + 30} y2={mg.top + 24} stroke={EC.pink} strokeWidth="2" />
                        <text x={mg.left + 34} y={mg.top + 28} fill={EC.pink} fontSize="9" fontWeight="600">WT ({"\u0394"}I={echemDiscOverlay.diWt}%)</text>
                        <line x1={mg.left + 10} y1={mg.top + 38} x2={mg.left + 30} y2={mg.top + 38} stroke={EC.purple} strokeWidth="2.5" />
                        <text x={mg.left + 34} y={mg.top + 42} fill={EC.purple} fontSize="9" fontWeight="600">MUT ({"\u0394"}I={echemDiscOverlay.diMut}%)</text>
                      </svg>
                    );
                  }

                  // SWV / DPV discrimination overlay — architecture-aware
                  const data = echemDiscOverlay.data;
                  const allY = data.flatMap(d => [d.baseline, d.MUT, d.WT]);
                  const yMin = Math.min(0, Math.min(...allY) * 1.1 - 0.02);
                  const yMax = Math.max(...allY) * 1.15 + 0.02;
                  const xS = e => mg.left + ((e - eS) / eRange) * pw;
                  const yS = v => mg.top + ((yMax - v) / (yMax - yMin)) * ph;
                  const pathD = (key) => data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xS(d.E).toFixed(1)} ${yS(d[key]).toFixed(1)}`).join(' ');
                  // Shaded area between MUT and WT peaks (discrimination region)
                  const discFill = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xS(d.E).toFixed(1)} ${yS(d.MUT).toFixed(1)}`).join(' ')
                    + data.slice().reverse().map(d => `L${xS(d.E).toFixed(1)} ${yS(d.WT).toFixed(1)}`).join(' ') + ' Z';
                  // Shaded area between baseline and WT
                  const wtFill = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xS(d.E).toFixed(1)} ${yS(d.baseline).toFixed(1)}`).join(' ')
                    + data.slice().reverse().map(d => `L${xS(d.E).toFixed(1)} ${yS(d.WT).toFixed(1)}`).join(' ') + ' Z';
                  // Y ticks
                  const yRange = yMax - yMin;
                  const yTickStep = yRange > 2 ? 1 : yRange > 1 ? 0.5 : yRange > 0.5 ? 0.2 : yRange > 0.1 ? 0.05 : 0.02;
                  const yTicks = [];
                  for (let v = Math.ceil(yMin / yTickStep) * yTickStep; v <= yMax; v += yTickStep) yTicks.push(+v.toFixed(3));

                  return (
                    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height="100%" style={{ fontFamily: MONO }}>
                      <rect width={w} height={h} fill="#FAFAFA" rx="8" />
                      {/* Grid */}
                      {[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875].map(f => <line key={`gx${f}`} x1={mg.left + pw * f} y1={mg.top} x2={mg.left + pw * f} y2={mg.top + ph} stroke="#E8E8E8" strokeWidth="0.5" />)}
                      {yTicks.map(v => <line key={`gy${v}`} x1={mg.left} y1={yS(v)} x2={mg.left + pw} y2={yS(v)} stroke="#E8E8E8" strokeWidth="0.5" />)}
                      <line x1={mg.left} y1={yS(0)} x2={mg.left + pw} y2={yS(0)} stroke="#BBB" strokeWidth="1" strokeDasharray="4,3" />
                      {/* Axes */}
                      <line x1={mg.left} y1={mg.top} x2={mg.left} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                      <line x1={mg.left} y1={mg.top + ph} x2={mg.left + pw} y2={mg.top + ph} stroke="#444" strokeWidth="1.5" />
                      {/* Shaded discrimination region between MUT and WT */}
                      <path d={discFill} fill={EC.purple} opacity="0.1" />
                      {/* Shaded WT region between baseline and WT */}
                      <path d={wtFill} fill={EC.pink} opacity="0.06" />
                      {/* Curves */}
                      <path d={pathD("baseline")} fill="none" stroke={EC.green} strokeWidth="2" strokeDasharray="6,4" opacity="0.7" />
                      <path d={pathD("WT")} fill="none" stroke={EC.pink} strokeWidth="2.2" opacity="0.85" />
                      <path d={pathD("MUT")} fill="none" stroke={EC.purple} strokeWidth="2.5" />
                      {/* E° reference */}
                      <line x1={xS(echemE0)} y1={mg.top} x2={xS(echemE0)} y2={mg.top + ph} stroke="#999" strokeWidth="0.8" strokeDasharray="3,3" />
                      <text x={xS(echemE0)} y={mg.top - 5} textAnchor="middle" fill="#999" fontSize="8">E{"\u00b0"}</text>
                      {/* Peak annotations */}
                      {(() => {
                        const peakIdx = data.reduce((best, d, i) => Math.abs(d.baseline) > Math.abs(data[best].baseline) ? i : best, 0);
                        const peakE = data[peakIdx].E;
                        const bY = yS(data[peakIdx].baseline);
                        const mY = yS(data[peakIdx].MUT);
                        const wY = yS(data[peakIdx].WT);
                        const x = xS(peakE);
                        return (
                          <>
                            {/* Vertical peak lines */}
                            <line x1={x} y1={bY} x2={x} y2={Math.min(mY, wY)} stroke="#ddd" strokeWidth="0.5" />
                            {/* MUT ΔI bracket */}
                            <line x1={x + 10} y1={bY} x2={x + 10} y2={mY} stroke={EC.purple} strokeWidth="1.5" />
                            <line x1={x + 6} y1={bY} x2={x + 14} y2={bY} stroke={EC.purple} strokeWidth="1" />
                            <line x1={x + 6} y1={mY} x2={x + 14} y2={mY} stroke={EC.purple} strokeWidth="1" />
                            <text x={x + 18} y={(bY + mY) / 2 + 3} fill={EC.purple} fontSize="9" fontWeight="700">{"\u0394"}I_MUT={echemDiscOverlay.diMut}%</text>
                            {/* WT ΔI bracket */}
                            <line x1={x - 10} y1={bY} x2={x - 10} y2={wY} stroke={EC.pink} strokeWidth="1.5" />
                            <line x1={x - 14} y1={bY} x2={x - 6} y2={bY} stroke={EC.pink} strokeWidth="1" />
                            <line x1={x - 14} y1={wY} x2={x - 6} y2={wY} stroke={EC.pink} strokeWidth="1" />
                            <text x={x - 18} y={(bY + wY) / 2 + 3} textAnchor="end" fill={EC.pink} fontSize="9" fontWeight="700">{"\u0394"}I_WT={echemDiscOverlay.diWt}%</text>
                            {/* Discrimination ratio */}
                            <rect x={x - 40} y={Math.min(mY, wY) - 22} width="80" height="16" rx="3" fill={EC.purple} opacity="0.12" />
                            <text x={x} y={Math.min(mY, wY) - 10} textAnchor="middle" fill={EC.purple} fontSize="9" fontWeight="800">
                              Disc = {echemDiscOverlay.measuredDisc === Infinity ? "\u221e" : `${echemDiscOverlay.measuredDisc}\u00d7`}
                            </text>
                          </>
                        );
                      })()}
                      {/* X ticks */}
                      {eTicks.map(e => <text key={e} x={xS(e)} y={mg.top + ph + 14} textAnchor="middle" fill="#666" fontSize="8">{e.toFixed(2)}</text>)}
                      {/* Y ticks */}
                      {yTicks.map(v => <text key={v} x={mg.left - 5} y={yS(v) + 3} textAnchor="end" fill="#666" fontSize="8">{v.toFixed(1)}</text>)}
                      {/* Axis labels */}
                      <text x={mg.left + pw / 2} y={h - 3} textAnchor="middle" fill="#444" fontSize="10">E (V vs Ag/AgCl)</text>
                      <text x={14} y={mg.top + ph / 2} textAnchor="middle" fill="#444" fontSize="10" transform={`rotate(-90, 14, ${mg.top + ph / 2})`}>I ({"\u03bc"}A)</text>
                      {/* Legend */}
                      <rect x={w - mg.right - 170} y={mg.top + 4} width="166" height="46" rx="4" fill="#fff" stroke="#E8E8E8" strokeWidth="0.5" opacity="0.9" />
                      <line x1={w - mg.right - 164} y1={mg.top + 16} x2={w - mg.right - 148} y2={mg.top + 16} stroke={EC.green} strokeWidth="2" strokeDasharray="5,3" />
                      <text x={w - mg.right - 144} y={mg.top + 19} fill={EC.green} fontSize="8" fontWeight="600">Baseline (no target)</text>
                      <line x1={w - mg.right - 164} y1={mg.top + 28} x2={w - mg.right - 148} y2={mg.top + 28} stroke={EC.pink} strokeWidth="2" />
                      <text x={w - mg.right - 144} y={mg.top + 31} fill={EC.pink} fontSize="8" fontWeight="600">WT allele</text>
                      <line x1={w - mg.right - 164} y1={mg.top + 40} x2={w - mg.right - 148} y2={mg.top + 40} stroke={EC.purple} strokeWidth="2.5" />
                      <text x={w - mg.right - 144} y={mg.top + 43} fill={EC.purple} fontSize="8" fontWeight="600">MUT allele ({echemCandidateData.label})</text>
                    </svg>
                  );
                })()}
              </div>
              {/* Interpretation block — Panel C */}
              <div style={{ marginTop: "8px", padding: "10px 14px", borderRadius: "4px", background: "#FDF4FF", fontSize: "11px", color: "#6B21A8", lineHeight: 1.6 }}>
                <strong>Interpretation:</strong>{" "}
                {echemDiscOverlay.measuredDisc >= 3
                  ? `Diagnostic-grade allelic discrimination (D = ${echemDiscOverlay.measuredDisc === Infinity ? "\u221e" : echemDiscOverlay.measuredDisc + "\u00d7"}). The voltammetric \u0394I% difference between MUT and WT alleles is clearly resolvable, enabling reliable genotyping from electrochemical signal alone.`
                  : echemDiscOverlay.measuredDisc >= 1.5
                  ? `Moderate allelic discrimination (D = ${echemDiscOverlay.measuredDisc}\u00d7). The MUT/WT peak height difference is detectable but marginal \u2014 AS-RPA primer specificity provides additional discrimination at the amplification stage.`
                  : echemDiscOverlay.measuredDisc >= 1
                  ? `Poor discrimination (D = ${echemDiscOverlay.measuredDisc}\u00d7). crRNA alone cannot distinguish alleles electrochemically. Clinical specificity depends entirely on AS-RPA allele-specific primer blocking.`
                  : `Inverted discrimination (D < 1): WT allele activates Cas12a more efficiently than MUT. This crRNA design requires redesign or relies exclusively on AS-RPA primer selectivity for correct genotyping.`
                }
              </div>
            </div>
          </div>
            );
          })()}
        </div>
      </CollapsibleSection>

      {/* ═══════════ SECTION 8: AS-RPA Thermodynamic Discrimination ═══════════ */}
      {results.some(r => r.asrpaDiscrimination) && (
      <CollapsibleSection title="AS-RPA Thermodynamic Discrimination" defaultOpen={false} badge={{ text: `${proximityCount} proximity`, bg: T.primaryLight, color: T.primary }}>
        <div style={{ padding: "0", marginBottom: "24px" }}>
          <p style={{ fontSize: "12px", color: T.textSec, marginBottom: "12px", lineHeight: 1.6 }}>
            Proximity candidates use allele-specific RPA primers for discrimination. The 3\u2032 terminal mismatch
            identity determines extension blocking strength. AS-RPA discrimination happens during <strong>amplification</strong>,
            not on the electrode pad. The pad only detects whether amplicon was produced.
          </p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
              <thead>
                <tr style={{ borderBottom: `2px solid ${T.border}` }}>
                  {["Target", "Pad", "Mismatch", "\u0394\u0394G", "Disc. Ratio", "Block", "Specificity"].map((h, i) => (
                    <th key={h} style={{ textAlign: i === 0 ? "left" : "center", padding: "8px 12px", fontWeight: 600, fontFamily: HEADING, color: T.textSec, fontSize: "11px" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.filter(r => r.asrpaDiscrimination).map(r => {
                  const d = r.asrpaDiscrimination;
                  const blockColor = d.block_class === "strong" ? T.success : d.block_class === "moderate" ? T.warning : T.danger;
                  return (
                    <tr key={r.label} style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                      <td style={{ padding: "8px 12px", fontWeight: 600, fontSize: "11px", fontFamily: MONO }}>{r.label}</td>
                      <td style={{ padding: "8px 12px", textAlign: "center", fontFamily: MONO, fontSize: "10px" }}>
                        {electrodeLayout.flat().indexOf(r.label) + 1 > 0 ? `P${electrodeLayout.flat().indexOf(r.label) + 1}` : "\u2014"}
                      </td>
                      <td style={{ padding: "8px 12px", textAlign: "center", fontFamily: MONO, fontWeight: 600, fontSize: "11px" }}>{d.terminal_mismatch}</td>
                      <td style={{ padding: "8px 12px", textAlign: "center", fontFamily: FONT, fontSize: "11px" }}>{d.ddg_kcal.toFixed(1)} kcal/mol</td>
                      <td style={{ padding: "8px 12px", textAlign: "center", fontFamily: FONT, fontWeight: 600, fontSize: "11px", color: d.disc_ratio >= 50 ? T.success : d.disc_ratio >= 10 ? T.warning : T.danger }}>{d.disc_ratio >= 100 ? "\u2265100" : d.disc_ratio.toFixed(0)}\u00d7</td>
                      <td style={{ padding: "8px 12px", textAlign: "center" }}>
                        <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 500, background: blockColor + "15", color: blockColor, textTransform: "lowercase" }}>{d.block_class}</span>
                      </td>
                      <td style={{ padding: "8px 12px", textAlign: "center", fontFamily: FONT, fontSize: "11px" }}>{(d.estimated_specificity * 100).toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: "10px", color: T.textTer, marginTop: "8px" }}>
            Discrimination ratios computed via Boltzmann conversion: exp({"\u0394\u0394"}G / RT) at 37 {"\u00b0"}C. Ratios &gt; 100{"\u00d7"} capped {"\u2014"} kinetic effects dominate at high {"\u0394\u0394"}G.
          </div>
        </div>
      </CollapsibleSection>
      )}


    </div>
  );
};


/* ═══════════════════════════════════════════════════════════════════
   DIAGNOSTICS TAB — Block 3 Sensitivity-Specificity Optimization
   ═══════════════════════════════════════════════════════════════════ */
class TabErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error(`${this.props.label || "Tab"} crash:`, error, info); }
  render() {
    if (this.state.error) return (
      <div style={{ padding: "24px", background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: "4px", margin: "16px 0" }}>
        <div style={{ fontWeight: 600, color: "#DC2626", marginBottom: "8px" }}>{this.props.label || "Tab"} failed to render</div>
        <div style={{ fontSize: "12px", color: "#7F1D1D", fontFamily: MONO, whiteSpace: "pre-wrap" }}>{this.state.error.message}{"\n"}{this.state.error.stack}</div>
        <button onClick={() => this.setState({ error: null })} style={{ marginTop: "12px", padding: "6px 16px", border: "1px solid #FECACA", borderRadius: "4px", cursor: "pointer", background: "white", fontSize: "12px" }}>Retry</button>
      </div>
    );
    return this.props.children;
  }
}
const DiagnosticsErrorBoundary = (props) => <TabErrorBoundary label="Diagnostics" {...props} />;
const DiagnosticsTab = ({ results, jobId, connected, scorer }) => {
  const mobile = useIsMobile();

  // State
  const [presets, setPresets] = useState([]);
  const [activePreset, setActivePreset] = useState("balanced");
  const [diagnostics, setDiagnostics] = useState(null);
  const [whoCompliance, setWhoCompliance] = useState(null);
  const [sweepData, setSweepData] = useState(null);
  const [paretoData, setParetoData] = useState(null);
  const [loadingDiag, setLoadingDiag] = useState(false);
  const [sweepLoading, setSweepLoading] = useState(false);
  const [paretoLoading, setParetoLoading] = useState(false);
  const [expandedTargets, setExpandedTargets] = useState({});
  const [topKData, setTopKData] = useState({});

  // Detect which scorer produced the results
  const scorerInfo = useMemo(() => {
    if (scorer === "compass_ml") return { name: "Compass-ML", level: 3 };
    if (!results?.length) return { name: "Heuristic", level: 1 };
    const first = results.find(r => r.mlScores?.length > 0);
    if (first) {
      const model = first.mlScores[0].model_name || first.mlScores[0].modelName;
      if (model === "compass_ml") return { name: "Compass-ML", level: 3 };
    }
    return { name: "Heuristic", level: 1 };
  }, [results, scorer]);

  // Compute diagnostics client-side from results prop + preset thresholds
  const computeLocalDiagnostics = useCallback((preset, res) => {
    try {
      if (!res || !res.length) return;
      const p = presets.find(x => x.name === preset) || { efficiency_threshold: 0.4, discrimination_threshold: 3.0 };
      const effT = p.efficiency_threshold || 0.4;
      const discT = p.discrimination_threshold || 3.0;

      const perTarget = res.map(r => {
        const eff = r.cnnCalibrated ?? r.score ?? 0;
        const disc = r.disc != null && r.disc < 900 ? r.disc : 0;
        const asrpaViable = r.strategy !== "Proximity" || !r.asrpaDiscrimination || r.asrpaDiscrimination.block_class !== "none";
        const ready = r.hasPrimers && eff >= effT && asrpaViable && (r.strategy === "Proximity" || disc >= discT);
        return { target_label: r.label || "unknown", drug: r.drug || "", efficiency: eff, discrimination: disc, is_assay_ready: ready, has_primers: !!r.hasPrimers, strategy: r.strategy || "Direct", asrpaViable };
      });

      // Exclude species control (IS6110) from resistance metrics
      const resistanceTargets = perTarget.filter(t => t.drug !== "OTHER");
      const assayReady = resistanceTargets.filter(t => t.is_assay_ready).length;
      const directTargets = resistanceTargets.filter(t => t.strategy === "Direct" && t.discrimination > 0);
      const meanDisc = directTargets.length ? directTargets.reduce((a, t) => a + t.discrimination, 0) / directTargets.length : 0;
      // Panel specificity: Direct targets use Cas12a disc, Proximity use computed AS-RPA estimate (fallback 0.95)
      const readyResistance = resistanceTargets.filter(t => t.is_assay_ready);
      const specValues = readyResistance.map(t => {
        if (t.strategy === "Proximity") {
          const orig = res.find(r => r.label === t.target_label);
          if (orig?.asrpaDiscrimination?.estimated_specificity != null) return orig.asrpaDiscrimination.estimated_specificity;
          return 0.95;
        }
        return Math.max(0, 1 - 1 / Math.max(t.discrimination, 1.01));
      });
      const specificity = specValues.length ? specValues.reduce((a, v) => a + v, 0) / specValues.length : 0;
      const sensitivity = resistanceTargets.length ? assayReady / resistanceTargets.length : 0;

      const meanEff = res.length > 0 ? res.reduce((a, r) => a + (r.cnnCalibrated ?? r.score ?? 0), 0) / res.length : 0;

      setDiagnostics({
        sensitivity, specificity, coverage: assayReady, total_targets: resistanceTargets.length,
        assay_ready: assayReady, mean_efficiency: +meanEff.toFixed(3),
        mean_discrimination: +meanDisc.toFixed(1), per_target: perTarget,
      });

      // WHO compliance by drug class (exclude species control)
      const drugs = [...new Set(res.map(r => r.drug))].filter(d => d && d !== "OTHER");
      const whoComp = {};
      for (const drug of drugs) {
        const drugTargets = perTarget.filter(t => t.drug === drug);
        const covered = drugTargets.filter(t => t.is_assay_ready).length;
        // Per-drug specificity: only viable targets (exclude WC-pair AS-RPA with no discrimination)
        const drugSpecs = drugTargets.filter(t => t.is_assay_ready).map(t => {
          if (t.strategy === "Proximity") {
            const orig = res.find(r => r.label === t.target_label);
            if (orig?.asrpaDiscrimination?.estimated_specificity != null) return orig.asrpaDiscrimination.estimated_specificity;
            return 0.95;
          }
          return t.discrimination > 0 ? Math.max(0, 1 - 1 / Math.max(t.discrimination, 1.01)) : 0;
        }).filter(v => v > 0);
        const drugSens = drugTargets.length ? covered / drugTargets.length : 0;
        const drugSpec = drugSpecs.length ? drugSpecs.reduce((a, v) => a + v, 0) / drugSpecs.length : 0;
        const tppSens = drug === "RIF" ? 0.95 : ["INH", "FQ"].includes(drug) ? 0.90 : 0.80;
        whoComp[drug] = { sensitivity: +drugSens.toFixed(3), specificity: +drugSpec.toFixed(3), meets_tpp: drugSens >= tppSens && drugSpec >= 0.98, meets_sensitivity: drugSens >= tppSens, meets_specificity: drugSpec >= 0.98, targets_covered: covered, targets_total: drugTargets.length };
      }
      setWhoCompliance({ preset, panel_sensitivity: +sensitivity.toFixed(3), panel_specificity: +specificity.toFixed(3), who_compliance: whoComp });
    } catch (err) {
      console.error("computeLocalDiagnostics error:", err);
      // Set minimal diagnostics so the page isn't blank
      setDiagnostics({
        sensitivity: 0, specificity: 0, coverage: 0, total_targets: res?.length || 0,
        assay_ready: 0, mean_efficiency: 0, mean_discrimination: 0, per_target: [],
      });
    }
  }, [presets]);

  // Load presets on mount — try API first, fall back to hardcoded
  useEffect(() => {
    const fallbackPresets = [
      { name: "high_sensitivity", description: "Field screening — maximise coverage, tolerate lower discrimination.", efficiency_threshold: 0.2, discrimination_threshold: 1.5 },
      { name: "balanced", description: "WHO TPP-aligned — clinical diagnostic deployment.", efficiency_threshold: 0.4, discrimination_threshold: 3.0 },
      { name: "high_specificity", description: "Confirmatory — minimise false calls, reference lab use.", efficiency_threshold: 0.6, discrimination_threshold: 5.0 },
    ];
    if (!connected) { setPresets(fallbackPresets); return; }
    getPresets().then(({ data }) => { setPresets(data && data.length ? data : fallbackPresets); });
  }, [connected]);

  // Clear sweep/pareto when preset changes so stale charts don't persist
  useEffect(() => {
    setSweepData(null);
    setParetoData(null);
  }, [activePreset]);

  // Load diagnostics + WHO compliance — try API, fall back to client-side computation
  useEffect(() => {
    if (!results || !results.length) return;
    let cancelled = false;
    setLoadingDiag(true);
    if (connected && jobId) {
      Promise.all([
        getDiagnostics(jobId, activePreset),
        getWHOCompliance(jobId, activePreset),
      ]).then(([diagRes, whoRes]) => {
        if (cancelled) return;
        if (diagRes.data && whoRes.data) {
          // Normalize API response: API returns panel_sensitivity/panel_specificity
          // but the render expects sensitivity/specificity at the top level
          const d = diagRes.data;
          const perTarget = (d.per_target || []).map(t => ({
            target_label: t.label || t.target_label,
            drug: t.drug_class || t.drug || "",
            efficiency: t.score ?? t.efficiency ?? 0,
            discrimination: t.discrimination ?? 0,
            is_assay_ready: t.assay_ready ?? t.is_assay_ready ?? false,
            has_primers: t.has_primers ?? true,
            strategy: (t.strategy || t.detection_strategy || "direct") === "direct" ? "Direct" : "Proximity",
          }));
          const resistanceTargets = perTarget.filter(t => t.drug !== "species_control" && t.drug !== "OTHER");
          const assayReady = resistanceTargets.filter(t => t.is_assay_ready).length;
          setDiagnostics({
            sensitivity: d.sensitivity ?? d.panel_sensitivity ?? 0,
            specificity: d.specificity ?? d.panel_specificity ?? 0,
            coverage: assayReady,
            total_targets: resistanceTargets.length,
            assay_ready: assayReady,
            mean_efficiency: d.mean_efficiency ?? 0,
            mean_discrimination: d.mean_discrimination ?? 0,
            per_target: perTarget,
          });
          // Normalize WHO compliance: API returns meets_minimal/meets_optimal,
          // frontend expects meets_tpp, specificity, targets_covered, targets_total
          const w = whoRes.data;
          const DRUG_MAP = { rifampicin: "RIF", isoniazid: "INH", fluoroquinolone: "FQ", ethambutol: "EMB", pyrazinamide: "PZA", aminoglycoside: "AG" };
          const normalizedWho = {};
          if (w.who_compliance) {
            for (const [drug, entry] of Object.entries(w.who_compliance)) {
              // Skip species control and unknown
              if (drug === "species_control" || drug === "unknown") continue;
              const drugKey = DRUG_MAP[drug] || drug.toUpperCase();
              const tppSens = drugKey === "RIF" ? 0.95 : ["INH", "FQ"].includes(drugKey) ? 0.90 : 0.80;
              const sens = entry.sensitivity ?? 0;
              const spec = entry.specificity ?? 0;
              normalizedWho[drugKey] = {
                sensitivity: sens,
                specificity: spec,
                meets_tpp: entry.meets_tpp ?? entry.meets_minimal ?? false,
                meets_sensitivity: sens >= tppSens,
                meets_specificity: spec >= 0.98,
                targets_covered: entry.n_covered ?? 0,
                targets_total: entry.n_targets ?? 0,
              };
            }
          }
          setWhoCompliance({
            preset: w.preset,
            panel_sensitivity: w.panel_sensitivity ?? 0,
            panel_specificity: w.panel_specificity ?? 0,
            who_compliance: normalizedWho,
          });
        } else {
          computeLocalDiagnostics(activePreset, results);
        }
        setLoadingDiag(false);
      }).catch(() => {
        if (cancelled) return;
        computeLocalDiagnostics(activePreset, results);
        setLoadingDiag(false);
      });
    } else {
      computeLocalDiagnostics(activePreset, results);
      setLoadingDiag(false);
    }
    return () => { cancelled = true; };
  }, [jobId, activePreset, connected, results, computeLocalDiagnostics]);

  // Run sweep — try API, fall back to client-side
  const handleSweep = (paramName) => {
    setSweepLoading(true);
    const values = paramName === "efficiency_threshold"
      ? [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      : [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0];

    const computeLocalSweep = () => {
      if (!results?.length) { setSweepLoading(false); return; }
      const resistanceResults = results.filter(r => r.drug !== "OTHER");
      const baseP = presets.find(x => x.name === activePreset) || { efficiency_threshold: 0.4, discrimination_threshold: 3.0 };
      const points = values.map(v => {
        const effT = paramName === "efficiency_threshold" ? v : baseP.efficiency_threshold;
        const discT = paramName === "discrimination_threshold" ? v : baseP.discrimination_threshold;
        const ready = resistanceResults.filter(r => {
          const eff = r.cnnCalibrated ?? r.score;
          const disc = r.disc != null && r.disc < 900 ? r.disc : 0;
          return r.hasPrimers && eff >= effT && (r.strategy === "Proximity" || disc >= discT);
        }).length;
        const directOk = resistanceResults.filter(r => r.strategy === "Direct" && r.disc < 900 && r.disc >= discT);
        const spec = directOk.length ? directOk.reduce((a, r) => a + Math.max(0, 1 - 1 / r.disc), 0) / directOk.length : 0;
        return { value: v, sensitivity: +(ready / resistanceResults.length).toFixed(3), specificity: +spec.toFixed(3), coverage: ready, assay_ready: ready };
      });
      setSweepData({ parameter_name: paramName, points });
      setSweepLoading(false);
    };

    if (connected && jobId) {
      runSweep(jobId, paramName, values, activePreset).then(({ data }) => {
        if (data) { setSweepData(data); setSweepLoading(false); }
        else computeLocalSweep();
      });
    } else {
      computeLocalSweep();
    }
  };

  // Run Pareto — try API, fall back to client-side
  const handlePareto = () => {
    setParetoLoading(true);

    const computeLocalPareto = () => {
      if (!results?.length) { setParetoLoading(false); return; }
      const frontier = [];
      const resistanceResults = results.filter(r => r.drug !== "OTHER");
      const discGrid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0];
      const effGrid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
      for (const dT of discGrid) {
        for (const eT of effGrid) {
          const ready = resistanceResults.filter(r => {
            const eff = r.cnnCalibrated ?? r.score;
            const disc = r.disc != null && r.disc < 900 ? r.disc : 0;
            return r.hasPrimers && eff >= eT && (r.strategy === "Proximity" || disc >= dT);
          }).length;
          const directOk = resistanceResults.filter(r => r.strategy === "Direct" && r.disc < 900 && r.disc >= dT);
          const spec = directOk.length ? directOk.reduce((a, r) => a + Math.max(0, 1 - 1 / r.disc), 0) / directOk.length : 0;
          const sens = ready / resistanceResults.length;
          frontier.push({ sensitivity: +sens.toFixed(3), specificity: +spec.toFixed(3), efficiency_threshold: eT, discrimination_threshold: dT, coverage: ready });
        }
      }
      // Keep only Pareto-optimal points
      const pareto = frontier.filter((p, _, arr) => !arr.some(q => q.sensitivity > p.sensitivity && q.specificity > p.specificity));
      const unique = [...new Map(pareto.map(p => [`${p.sensitivity}-${p.specificity}`, p])).values()];
      unique.sort((a, b) => a.specificity - b.specificity);
      setParetoData({ n_points: unique.length, frontier: unique });
      setParetoLoading(false);
    };

    if (connected && jobId) {
      runPareto(jobId).then(({ data }) => {
        if (data) { setParetoData(data); setParetoLoading(false); }
        else computeLocalPareto();
      });
    } else {
      computeLocalPareto();
    }
  };

  const presetObj = presets.find(p => p.name === activePreset);
  const PRESET_LABELS = { high_sensitivity: "High Sensitivity", balanced: "Balanced (WHO TPP)", high_specificity: "High Specificity" };

  return (
    <div>
      {/* ── Explainer box ── */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}25`, borderRadius: "4px", padding: mobile ? "14px" : "16px 20px", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <Settings size={16} color={T.primary} strokeWidth={1.8} />
          <span style={{ fontSize: "13px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING }}>Sensitivity-Specificity Optimization</span>
        </div>
        <div style={{ fontSize: "13px", color: T.primaryDark, lineHeight: 1.6 }}>
          Evaluate panel performance against WHO Target Product Profile thresholds across three operating modes. Per-drug sensitivity is computed from primer coverage and readiness; specificity is estimated from discrimination ratios using 1{"\u2212"}1/D (theoretical upper bound assuming separated signal distributions). Real specificity on LIG electrodes depends on intra-device CV ({"\u2248"}5% RSD) and electrode-to-electrode variability across the 14-plex array. Adjust the optimization profile to balance field deployment (high sensitivity) versus reference laboratory (high specificity) requirements.
        </div>
      </div>

      {/* A: Preset Selector */}
      <div style={{ marginBottom: "24px" }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: T.primary, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "10px" }}>Optimization Profile</div>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          {presets.map(p => (
            <button key={p.name} onClick={() => setActivePreset(p.name)} style={{
              padding: "8px 16px", borderRadius: "4px", cursor: "pointer", fontFamily: FONT, fontSize: "13px", fontWeight: 600,
              transition: "all 0.15s", border: activePreset === p.name ? `2px solid ${T.primary}` : `1px solid ${T.border}`,
              background: activePreset === p.name ? T.primaryLight : T.bg, color: activePreset === p.name ? T.primaryDark : T.text,
            }}>
              <div>{PRESET_LABELS[p.name] || p.name}</div>
              <div style={{ fontSize: "10px", fontWeight: 400, color: T.textTer, marginTop: "2px", maxWidth: 280 }}>{p.description}</div>
            </button>
          ))}
        </div>
        {presetObj && (
          <div style={{ marginTop: "16px", fontSize: "11px", color: T.textSec, display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
            <span>Thresholds: efficiency ≥ {presetObj.efficiency_threshold}, discrimination ≥ {presetObj.discrimination_threshold}×</span>
            <span style={{
              display: "inline-flex", alignItems: "center", gap: "5px",
              padding: "2px 10px", borderRadius: "3px", fontSize: "10px", fontWeight: 600,
              background: scorerInfo.level >= 3 ? "rgba(16,185,129,0.1)" : T.bgSub,
              color: scorerInfo.level >= 3 ? T.success : T.textSec,
              border: `1px solid ${scorerInfo.level >= 3 ? T.success + "33" : T.borderLight}`,
            }}>
              <Cpu size={10} />
              Scored by: {scorerInfo.name}
            </span>
          </div>
        )}
      </div>

      {loadingDiag && (
        <div style={{ textAlign: "center", padding: "32px", color: T.textTer }}>
          <Loader2 size={20} style={{ animation: "spin 1s linear infinite" }} />
          <div style={{ marginTop: "6px", fontSize: "12px" }}>Computing diagnostics…</div>
        </div>
      )}

      {!loadingDiag && !diagnostics && (
        <div style={{ textAlign: "center", padding: "32px", color: T.textTer }}>
          <AlertTriangle size={20} color={T.warning} style={{ marginBottom: "8px" }} />
          <div style={{ fontSize: "13px", marginBottom: "8px" }}>Diagnostics data could not be computed.</div>
          <button onClick={() => { setLoadingDiag(true); setTimeout(() => { computeLocalDiagnostics(activePreset, results); setLoadingDiag(false); }, 50); }} style={{ padding: "6px 16px", border: `1px solid ${T.border}`, borderRadius: "4px", cursor: "pointer", background: T.bg, fontSize: "12px", fontFamily: FONT }}>Retry</button>
        </div>
      )}

      {!loadingDiag && diagnostics && (
        <>
          {/* B: Summary Cards */}
          <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr 1fr" : "repeat(4, 1fr)", gap: "12px", marginBottom: "24px" }}>
            {[
              { label: "Sensitivity", value: `${(diagnostics.sensitivity * 100).toFixed(1)}%`, color: diagnostics.sensitivity >= 0.85 ? T.success : diagnostics.sensitivity >= 0.7 ? T.warning : T.danger, icon: TrendingUp },
              { label: "Specificity", value: `${(diagnostics.specificity * 100).toFixed(1)}%`, color: diagnostics.specificity >= 0.80 ? T.success : diagnostics.specificity >= 0.6 ? T.warning : T.danger, icon: Shield },
              { label: "Coverage", value: `${diagnostics.coverage || diagnostics.assay_ready}/${diagnostics.total_targets}`, color: T.primary, icon: Target },
              { label: "Assay-Ready", value: diagnostics.assay_ready, color: T.purple, icon: CheckCircle },
            ].map(card => (
              <div key={card.label} style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "16px 20px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "8px" }}>
                  <card.icon size={14} color={card.color} />
                  <span style={{ fontSize: "11px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em" }}>{card.label}</span>
                </div>
                <div style={{ fontSize: "18px", fontWeight: 600, color: card.color, fontFamily: FONT, lineHeight: 1 }}>{card.value}</div>
              </div>
            ))}
          </div>

          {/* MUT vs WT Activity Distribution */}
          {!mobile && (() => { try {
            const p = presets.find(x => x.name === activePreset) || { efficiency_threshold: 0.4, discrimination_threshold: 3.0 };
            const effT = p.efficiency_threshold || 0.4;
            const discT = p.discrimination_threshold || 3.0;
            // Filter to candidates that pass the active preset's thresholds
            const filtered = results.filter(r => {
              const eff = r.cnnCalibrated ?? r.score;
              if (eff < effT) return false;
              if (r.gene === "IS6110") return false; // species control
              if (r.strategy === "Proximity") return !(r.asrpaDiscrimination?.block_class === "none");
              return (r.disc > 0 && r.disc < 900) ? r.disc >= discT : false;
            });
            if (filtered.length < 2) return (
              <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "24px 28px", marginBottom: "24px" }}>
                <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "8px" }}>MUT vs WT Predicted Activity</div>
                <div style={{ fontSize: "12px", color: T.textSec, padding: "20px 0", textAlign: "center" }}>
                  Only {filtered.length} candidate{filtered.length === 1 ? "" : "s"} pass the <strong>{PRESET_LABELS[activePreset] || activePreset}</strong> thresholds (eff ≥ {effT}, disc ≥ {discT}×). Need ≥ 2 to plot distribution. Try a less stringent preset.
                </div>
              </div>
            );
            const plotResults = filtered;
            const mutScores = plotResults.map(r => r.cnnCalibrated ?? r.score);
            const wtScores = plotResults.map(r => {
              const eff = r.cnnCalibrated ?? r.score;
              if (r.strategy === "Proximity") {
                // Proximity: discrimination comes from AS-RPA primers, not Cas12a
                // WT sample → AS-RPA primers don't extend → no amplicon → no Cas12a cleavage → ΔI ≈ 0
                const asrpaSpec = r.asrpaDiscrimination?.estimated_specificity;
                const asrpaDisc = asrpaSpec != null ? 1 / Math.max(1 - asrpaSpec, 0.001) : 100;
                return eff / asrpaDisc; // typically ≈ 0.008 (near zero)
              }
              const disc = r.disc > 0 && r.disc < 900 ? r.disc : 1.5;
              return eff / disc;
            });
            const kdeMut = gaussianKDE(mutScores, 0.05, 100);
            const kdeWt = gaussianKDE(wtScores, 0.05, 100);
            const combined = kdeMut.map((p, i) => {
              const mutD = p.density;
              const wtD = kdeWt[i]?.density || 0;
              return { x: p.x, mut: mutD, wt: wtD, overlap: Math.min(mutD, wtD) };
            });
            const meanMut = +(mutScores.reduce((a, b) => a + b, 0) / mutScores.length).toFixed(3);
            const meanWt = +(wtScores.reduce((a, b) => a + b, 0) / wtScores.length).toFixed(3);
            const separation = +(meanMut - meanWt).toFixed(3);
            // Compute overlap coefficient (proportion of area that overlaps)
            const totalMut = kdeMut.reduce((a, p) => a + p.density, 0);
            const overlapArea = combined.reduce((a, p) => a + p.overlap, 0);
            const overlapPct = totalMut > 0 ? Math.round((overlapArea / totalMut) * 100) : 0;
            return (
              <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "24px 28px", marginBottom: "24px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px" }}>
                  <div>
                    <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>MUT vs WT Predicted Activity</div>
                    <div style={{ fontSize: "11px", color: T.textSec, marginTop: "3px", lineHeight: 1.5, maxWidth: "540px" }}>
                      Density from <strong>{plotResults.length}</strong>/{results.filter(r => r.gene !== "IS6110").length} candidates passing <strong>{PRESET_LABELS[activePreset] || activePreset}</strong> thresholds (eff ≥ {effT}, disc ≥ {discT}×). Greater separation = better discrimination. Direct targets: A<sub>WT</sub> = A<sub>MUT</sub> / Cas12a disc. Proximity targets: A<sub>WT</sub> = A<sub>MUT</sub> / AS-RPA disc (WT not amplified → near-zero signal).
                    </div>
                  </div>
                  <Badge variant={separation >= 0.15 ? "success" : separation >= 0.08 ? "warning" : "danger"}>
                    {separation >= 0.15 ? "Good separation" : separation >= 0.08 ? "Moderate" : "Poor separation"}
                  </Badge>
                </div>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={combined} margin={{ top: 10, right: 15, bottom: 25, left: 15 }}>
                    <defs>
                      <linearGradient id="mutAreaFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#4338CA" stopOpacity={0.35} />
                        <stop offset="100%" stopColor="#4338CA" stopOpacity={0.03} />
                      </linearGradient>
                      <linearGradient id="wtAreaFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#9CA3AF" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#9CA3AF" stopOpacity={0.03} />
                      </linearGradient>
                      <linearGradient id="overlapAreaFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#059669" stopOpacity={0.15} />
                        <stop offset="100%" stopColor="#059669" stopOpacity={0.03} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="x" type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: T.textTer, fontFamily: MONO }} tickCount={11} axisLine={{ stroke: T.border }} tickLine={false} label={{ value: "Predicted cleavage activity", position: "insideBottom", offset: -12, fontSize: 10, fill: T.textSec }} />
                    <YAxis hide domain={[0, "auto"]} />
                    <Tooltip content={({ payload, label }) => {
                      if (!payload?.length) return null;
                      return (
                        <div style={{ ...tooltipStyle, padding: "10px 14px" }}>
                          <div style={{ fontWeight: 600, fontSize: "11px", color: T.text, marginBottom: "4px" }}>Activity: {label}</div>
                          {payload.map(p => p.dataKey !== "overlap" && (
                            <div key={p.dataKey} style={{ fontSize: "11px", color: p.dataKey === "mut" ? "#4338CA" : "#9CA3AF" }}>
                              {p.dataKey === "mut" ? "Mutant" : "Wildtype"}: {p.value?.toFixed(4)}
                            </div>
                          ))}
                        </div>
                      );
                    }} />
                    <ReferenceLine x={meanMut} stroke="#4338CA" strokeDasharray="3 3" strokeWidth={1} label={{ value: "μ MUT", position: "insideTopRight", fontSize: 9, fill: "#4338CA", fontWeight: 600 }} />
                    <ReferenceLine x={meanWt} stroke="#9CA3AF" strokeDasharray="3 3" strokeWidth={1} label={{ value: "μ WT", position: "insideTopRight", fontSize: 9, fill: "#9CA3AF", fontWeight: 600 }} />
                    <Area type="monotone" dataKey="overlap" stroke="none" fill="url(#overlapAreaFill)" isAnimationActive={false} />
                    <Area type="monotone" dataKey="mut" stroke="#4338CA" strokeWidth={2.5} fill="url(#mutAreaFill)" isAnimationActive={false} />
                    <Area type="monotone" dataKey="wt" stroke="#9CA3AF" strokeWidth={2} fill="url(#wtAreaFill)" isAnimationActive={false} strokeDasharray="6 3" />
                  </AreaChart>
                </ResponsiveContainer>
                {/* Custom legend + stats */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "8px", flexWrap: "wrap", gap: "12px" }}>
                  <div style={{ display: "flex", gap: "16px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                      <div style={{ width: "16px", height: "3px", background: "#4338CA", borderRadius: "2px" }} />
                      <span style={{ fontSize: "10px", color: T.textSec, fontWeight: 500 }}>Mutant (A<sub>MUT</sub>)</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                      <div style={{ width: "16px", height: "3px", background: "#9CA3AF", borderRadius: "2px", borderBottom: "1px dashed #A8A29E" }} />
                      <span style={{ fontSize: "10px", color: T.textSec, fontWeight: 500 }}>Wildtype (A<sub>WT</sub>)</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                      <div style={{ width: "10px", height: "10px", background: "rgba(102,194,165,0.15)", borderRadius: "2px" }} />
                      <span style={{ fontSize: "10px", color: T.textSec, fontWeight: 500 }}>Overlap zone ({overlapPct}%)</span>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: "20px" }}>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "9px", color: T.textTer, fontWeight: 600 }}>μ MUT</div>
                      <div style={{ fontSize: "14px", fontWeight: 600, color: "#4338CA", fontFamily: FONT }}>{meanMut}</div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "9px", color: T.textTer, fontWeight: 600 }}>μ WT</div>
                      <div style={{ fontSize: "14px", fontWeight: 600, color: "#9CA3AF", fontFamily: FONT }}>{meanWt}</div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "9px", color: T.textTer, fontWeight: 600 }}>SEPARATION</div>
                      <div style={{ fontSize: "14px", fontWeight: 600, color: separation >= 0.15 ? T.success : T.warning, fontFamily: FONT }}>{separation}</div>
                    </div>
                  </div>
                </div>
                {/* Interpretation */}
                {(() => {
                  const mutSorted = [...mutScores].sort((a, b) => b - a);
                  const wtSorted = [...wtScores].sort((a, b) => b - a);
                  const bestMutIdx = mutScores.indexOf(mutSorted[0]);
                  const bestMutLabel = plotResults[bestMutIdx]?.label || "top target";
                  const worstMutIdx = mutScores.indexOf(mutSorted[mutSorted.length - 1]);
                  const worstMutLabel = plotResults[worstMutIdx]?.label || "weakest target";
                  const clinicalRisk = overlapPct > 30 ? "high" : overlapPct > 15 ? "moderate" : "low";
                  return (
                    <div style={{ marginTop: "14px", padding: "12px 16px", background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", fontSize: "11px", color: T.textSec, lineHeight: 1.7 }}>
                      <strong style={{ color: T.primary }}>Interpretation:</strong> Mutant mean activity ({meanMut}) vs wildtype ({meanWt}) gives a separation of <strong style={{ color: separation >= 0.15 ? T.success : T.warning }}>{separation}</strong>.
                      {separation >= 0.15 ? " Good separation — the panel reliably distinguishes resistant from susceptible samples at the aggregate level." : separation >= 0.08 ? " Moderate separation — borderline samples may produce ambiguous calls; consider tightening the panel to high-discrimination targets only." : " Poor separation — the panel cannot reliably distinguish MUT from WT; review target selection and consider dropping low-discrimination candidates."}
                      {` Overlap zone: ${overlapPct}% — this is the aggregate overlap; individual targets with high discrimination (e.g., disc ≥10×) have near-zero overlap. In practice each target is read independently, so per-target separation matters more than panel-level aggregate.`}
                      {` Strongest MUT signal: ${bestMutLabel} (${mutSorted[0].toFixed(3)}). Weakest: ${worstMutLabel} (${mutSorted[mutSorted.length - 1].toFixed(3)}).`}
                      {plotResults.length === results.filter(r => r.gene !== "IS6110").length && activePreset !== "balanced" && ` Note: all candidates exceed the ${PRESET_LABELS[activePreset] || activePreset} thresholds — this profile produces identical results to a less stringent profile for the current panel.`}
                    </div>
                  );
                })()}
              </div>
            );
          } catch (e) { console.error("MUT vs WT chart error:", e); return null; } })()}

          {/* B2: Understanding Discrimination Scores — collapsible explainer */}
          <CollapsibleSection title="Understanding Discrimination Scores" defaultOpen={false}>
            <div style={{ fontSize: "13px", color: T.textSec, lineHeight: 1.7 }}>
              <div style={{ marginBottom: "14px" }}>
                <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>Mismatch position matters most</div>
                Cas12a reads DNA directionally from PAM toward the spacer end. The seed region spans positions 1–8 (PAM-proximal). Mismatches at positions 1–4 block R-loop formation almost completely, giving discrimination ratios of 10–50×. Positions 5–8 give 3–10×. Mismatches far from the PAM (positions 15–20) are tolerated, giving ratios of 1–2×. Each mutation's position in the spacer determines its baseline discrimination.
              </div>
              <div style={{ marginBottom: "14px" }}>
                <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>Not all mismatches are equal</div>
                A purine-to-pyrimidine change (e.g., A→C) disrupts the R-loop more than a purine-to-purine change (e.g., A→G). The geometry of the mismatch affects how much Cas12a distinguishes mutant from wildtype.
              </div>
              <div style={{ marginBottom: "14px" }}>
                <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>High GC content reduces discrimination</div>
                <em>M. tuberculosis</em> has 65.8% GC content. GC-rich sequences around a mismatch stabilise the R-loop through additional hydrogen bonds, partially compensating for the mismatch. This is why some targets (EMB, PZA) show low predicted discrimination — their mutations sit in GC-rich regions at PAM-distal positions.
              </div>
              <div style={{ marginBottom: "14px" }}>
                <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>Prediction model</div>
                {results.some(r => r.discMethod === "neural")
                  ? "Discrimination ratios are predicted by Compass-ML's neural discrimination head — a multi-task extension (235K params) trained end-to-end on efficiency and discrimination simultaneously. The disc head takes paired encoder representations [mut, wt, mut\u2212wt, mut\u00D7wt] from the shared CNN+RNA-FM+RLPA backbone and outputs a predicted MUT/WT ratio via Softplus. Trained on 6,136 paired trans-cleavage measurements from EasyDesign (Huang et al. 2024, LbCas12a). 3-fold CV: r = 0.440."
                  : results.some(r => (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature")
                  ? "Discrimination ratios are predicted by a gradient-boosted model (XGBoost) trained on 6,136 paired MUT/WT trans-cleavage measurements from the EasyDesign dataset (Huang et al. 2024, LbCas12a). The model uses 18 thermodynamic features including R-loop cumulative \u0394G, mismatch \u0394\u0394G penalties, and position sensitivity. Val: RMSE = 0.520, r = 0.565. Top feature: pam_to_mm_distance (0.148 importance)."
                  : "Discrimination ratios are predicted by a heuristic model using position sensitivity \u00D7 mismatch destabilisation scores. A trained model (XGBoost on 18 thermodynamic features) is available but was not loaded for this run."
                }
              </div>
              <div style={{ fontSize: "11px", color: T.textTer, fontStyle: "italic", borderTop: `1px solid ${T.borderLight}`, paddingTop: "10px" }}>
                These are in silico predictions. Experimental validation on the electrochemical platform will provide measured discrimination ratios through the active learning loop.
              </div>
            </div>
          </CollapsibleSection>

          {/* C: WHO Compliance Table */}
          {whoCompliance && whoCompliance.who_compliance && (() => {
            // Filter out species_control/UNKNOWN from WHO table — it's not a resistance drug class
            const whoEntries = Object.entries(whoCompliance.who_compliance).filter(([drug]) => !["UNKNOWN", "OTHER", "SPECIES_CONTROL", "species_control"].includes(drug));
            const WHO_TPP_SENS = { RIF: 0.95, INH: 0.90, FQ: 0.90, EMB: 0.80, PZA: 0.80, AG: 0.80 };
            const sensPassing = whoEntries.filter(([, d]) => d.meets_sensitivity).length;
            const specPassing = whoEntries.filter(([, d]) => d.meets_specificity).length;
            return (
            <div style={{ marginBottom: "24px", border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
              <div style={{ background: T.bgSub, padding: "12px 16px", borderBottom: `1px solid ${T.border}`, display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                <Shield size={14} color={T.primary} />
                <span style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>WHO TPP Compliance by Drug Class</span>
                <Badge variant={sensPassing === whoEntries.length ? "success" : "warning"}>
                  Sens: {sensPassing}/{whoEntries.length}
                </Badge>
                <Badge variant={specPassing === whoEntries.length ? "success" : specPassing > 0 ? "warning" : "neutral"}>
                  Spec: {specPassing}/{whoEntries.length}
                </Badge>
              </div>
              <div style={{ padding: "12px 18px", fontSize: "11px", color: T.textSec, lineHeight: 1.6, borderBottom: `1px solid ${T.borderLight}`, background: T.bg }}>
                WHO Target Product Profile (TPP) 2024 defines minimum sensitivity and specificity thresholds per drug class for diagnostic deployment. Sensitivity = fraction of resistance-conferring mutations detected (pass/fail per drug class). Specificity = approximate in silico estimate: Direct targets use 1−1/disc (assumes perfectly separated signal distributions — actual specificity depends on signal variance and threshold selection). Proximity targets use thermodynamic AS-RPA mismatch penalty. ≥98% required — marked "Pending" when below threshold as experimental validation is needed. {results.some(r => (r.discrimination?.model_name || "").includes("learned")) ? "Discrimination ratios used here are from the learned model (XGBoost, 18 thermodynamic features)." : "Discrimination ratios used here are from the heuristic model."}
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: FONT, fontSize: "12px" }}>
                  <thead>
                    <tr style={{ background: T.bgSub }}>
                      {["Drug Class", "Sensitivity", "WHO Target", "Coverage", "Avg Disc", "Specificity", "Sens. Status", "Spec. Status"].map(h => (
                        <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 600, color: T.textSec, fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.border}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {whoEntries.map(([drug, data]) => {
                      const tppTarget = WHO_TPP_SENS[drug] || 0.80;
                      // Compute avg discrimination for this drug class from diagnostics per_target
                      const drugTargets = (diagnostics.per_target || []).filter(t => {
                        const tDrug = (t.drug || "").toUpperCase().replace("RIFAMPICIN", "RIF").replace("ISONIAZID", "INH").replace("FLUOROQUINOLONE", "FQ").replace("ETHAMBUTOL", "EMB").replace("PYRAZINAMIDE", "PZA").replace("AMINOGLYCOSIDE", "AG");
                        return tDrug === drug;
                      });
                      const discTargets = drugTargets.filter(t => t.discrimination > 0 && t.strategy === "Direct");
                      const avgDisc = discTargets.length ? discTargets.reduce((a, t) => a + t.discrimination, 0) / discTargets.length : 0;
                      const sensPercent = (data.sensitivity * 100);
                      const tppPercent = (tppTarget * 100);
                      const gap = sensPercent - tppPercent;
                      return (
                      <tr key={drug} style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                        <td style={{ padding: "10px 14px" }}><DrugBadge drug={drug} /></td>
                        <td style={{ padding: "10px 14px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <span style={{ fontFamily: FONT, fontWeight: 600, fontSize: "13px", color: data.sensitivity >= tppTarget ? T.success : data.sensitivity >= tppTarget * 0.8 ? T.warning : T.danger }}>{sensPercent.toFixed(1)}%</span>
                            {gap !== 0 && <span style={{ fontSize: "10px", fontFamily: FONT, fontWeight: 600, color: gap >= 0 ? T.success : T.danger }}>{gap >= 0 ? "+" : ""}{gap.toFixed(0)}pp</span>}
                          </div>
                        </td>
                        <td style={{ padding: "10px 14px", fontFamily: FONT, fontSize: "11px", color: T.textTer }}>≥ {tppPercent.toFixed(0)}%</td>
                        <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600 }}>{data.targets_covered}/{data.targets_total}</td>
                        <td style={{ padding: "10px 14px", fontFamily: FONT, fontSize: "11px", color: avgDisc >= 3 ? T.success : avgDisc >= 2 ? T.warning : T.textTer }}>{avgDisc > 0 ? `${avgDisc.toFixed(1)}×` : "—"}</td>
                        <td style={{ padding: "10px 14px" }}>
                          {data.specificity != null ? (
                            <div>
                              <span style={{ fontFamily: FONT, fontWeight: 600, fontSize: "12px", color: data.specificity >= 0.98 ? T.success : data.specificity >= 0.90 ? T.warning : T.textTer }}>{(data.specificity * 100).toFixed(1)}%</span>
                              {data.n_excluded_specificity > 0 && <div style={{ fontSize: "9px", color: T.textTer, marginTop: "2px" }}>{data.n_excluded_specificity} excluded</div>}
                            </div>
                          ) : <span style={{ color: T.textTer }}>—</span>}
                        </td>
                        <td style={{ padding: "10px 14px" }}>
                          {data.meets_sensitivity ? (
                            <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", padding: "3px 10px", borderRadius: "3px", background: "rgba(16,185,129,0.1)", color: T.success, fontWeight: 600, fontSize: "11px" }}><CheckCircle size={12} /> Pass</span>
                          ) : (
                            <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", padding: "3px 10px", borderRadius: "3px", background: "rgba(239,68,68,0.08)", color: T.danger, fontWeight: 600, fontSize: "11px" }}><AlertTriangle size={12} /> Fail</span>
                          )}
                        </td>
                        <td style={{ padding: "10px 14px" }}>
                          {data.meets_specificity ? (
                            <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", padding: "3px 10px", borderRadius: "3px", background: "rgba(16,185,129,0.1)", color: T.success, fontWeight: 600, fontSize: "11px" }}><CheckCircle size={12} /> Pass</span>
                          ) : (
                            <span style={{ display: "inline-flex", alignItems: "center", gap: "3px", padding: "3px 10px", borderRadius: "3px", background: "rgba(245,158,11,0.08)", color: T.warning, fontWeight: 600, fontSize: "11px" }}><AlertTriangle size={12} /> Pending</span>
                          )}
                        </td>
                      </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {/* Interpretation */}
              {(() => {
                const sensFailing = whoEntries.filter(([, d]) => !d.meets_sensitivity);
                const specFailing = whoEntries.filter(([, d]) => !d.meets_specificity);
                const worstSens = sensFailing.length ? sensFailing.sort((a, b) => a[1].sensitivity - b[1].sensitivity)[0] : null;
                return (
                  <div style={{ padding: "12px 18px", background: T.primaryLight, borderTop: `1px solid ${T.borderLight}`, fontSize: "11px", color: T.textSec, lineHeight: 1.7 }}>
                    <strong style={{ color: T.primary }}>Interpretation:</strong>{" "}
                    <strong>Sensitivity:</strong> {sensPassing}/{whoEntries.length} drug classes meet WHO TPP minimal sensitivity thresholds.
                    {worstSens && ` ${worstSens[0]} is the weakest (${(worstSens[1].sensitivity * 100).toFixed(0)}% vs ${((WHO_TPP_SENS[worstSens[0]] || 0.80) * 100).toFixed(0)}% required).`}
                    {sensFailing.length > 1 && ` ${sensFailing.length} classes need additional mutation coverage.`}
                    {sensPassing === whoEntries.length && " All drug classes pass sensitivity."}
                    {" "}<strong>Specificity:</strong> {specPassing}/{whoEntries.length} classes meet the ≥98% threshold (approximate in silico proxy — actual specificity requires experimental determination with clinical samples).
                    {specFailing.length > 0 && ` ${specFailing.length} class${specFailing.length > 1 ? "es" : ""} pending — specificity estimates require experimental validation on the electrochemical platform.`}
                    {specPassing === whoEntries.length && " All classes pass specificity."}
                    {" "}<em>Note: coverage denominators reflect panel targets only, not the full WHO mutation catalogue. Clinical sensitivity for a drug class depends on the epidemiological frequency of included mutations (e.g., INH: katG S315T covers ~60% of INH-resistant isolates; adding fabG1 C-15T raises coverage to ~85%).</em>
                  </div>
                );
              })()}
            </div>
            );
          })()}

          {/* D: Per-Target Breakdown with Top-K */}
          {diagnostics.per_target && diagnostics.per_target.length > 0 && (
            <CollapsibleSection title={`Per-Target Breakdown (${diagnostics.per_target.length} targets)`} defaultOpen={false}>
              <div style={{ padding: "10px 14px", marginBottom: "12px", background: T.primaryLight, borderRadius: "4px", fontSize: "11px", color: T.primaryDark, lineHeight: 1.6 }}>
                <strong>Per-target assay readiness assessment.</strong> Each row shows the selected candidate's predicted efficiency and discrimination ratio against the active profile thresholds.
                Click any row to expand the <strong>Top-K alternative candidates</strong> — ranked alternatives with tradeoff annotations for experimental fallback planning.
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: FONT, fontSize: "12px" }}>
                  <thead>
                    <tr style={{ background: T.bgSub }}>
                      {["", "Target", "Drug", "Strategy", "Efficiency", "Discrimination", "Primers", "Status", ...(results.some(r => r.riskProfile) ? ["Risk", "#"] : [])].map(h => (
                        <th key={h} style={{ padding: "10px 12px", textAlign: "left", fontWeight: 600, color: T.textSec, fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.border}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {diagnostics.per_target.map(t => {
                      const isExpanded = expandedTargets[t.target_label];
                      const topK = topKData[t.target_label];
                      const eff = typeof t.efficiency === "number" ? t.efficiency : 0;
                      const disc = typeof t.discrimination === "number" ? t.discrimination : 0;
                      const effColor = eff >= 0.7 ? T.success : eff >= 0.5 ? T.warning : T.danger;
                      const discColor = disc >= 3 ? T.success : disc >= 2 ? T.warning : T.danger;
                      const drugDisplay = (t.drug || "").toUpperCase().replace("RIFAMPICIN", "RIF").replace("ISONIAZID", "INH").replace("FLUOROQUINOLONE", "FQ").replace("ETHAMBUTOL", "EMB").replace("PYRAZINAMIDE", "PZA").replace("AMINOGLYCOSIDE", "AG").replace("SPECIES_CONTROL", "CTRL");
                      const stratDisplay = (t.strategy || "").charAt(0).toUpperCase() + (t.strategy || "").slice(1);
                      const toggleExpand = () => {
                        setExpandedTargets(prev => ({ ...prev, [t.target_label]: !prev[t.target_label] }));
                        if (!topKData[t.target_label]) {
                          if (!connected || !jobId) {
                            setTopKData(prev => ({ ...prev, [t.target_label]: [] }));
                          } else {
                            const timeout = setTimeout(() => {
                              setTopKData(prev => prev[t.target_label] === undefined ? { ...prev, [t.target_label]: [] } : prev);
                            }, 5000);
                            getTopK(jobId, t.target_label, 5).then(({ data }) => {
                              clearTimeout(timeout);
                              setTopKData(prev => ({ ...prev, [t.target_label]: (data?.alternatives || data) || [] }));
                            });
                          }
                        }
                      };
                      return (
                        <React.Fragment key={t.target_label}>
                          <tr style={{ borderBottom: `1px solid ${T.borderLight}`, cursor: "pointer", transition: "background 0.1s" }} onClick={toggleExpand} onMouseEnter={e => e.currentTarget.style.background = T.bgSub} onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                            <td style={{ padding: "10px 8px", width: "24px" }}>{isExpanded ? <ChevronDown size={13} color={T.primary} /> : <ChevronRight size={13} color={T.textTer} />}</td>
                            <td style={{ padding: "10px 12px", fontWeight: 600, fontFamily: MONO, fontSize: "11px", color: T.text }}>{t.target_label}</td>
                            <td style={{ padding: "10px 12px" }}><DrugBadge drug={drugDisplay} /></td>
                            <td style={{ padding: "10px 12px" }}>
                              <span style={{ fontSize: "10px", fontWeight: 600, padding: "2px 8px", borderRadius: "4px", background: stratDisplay === "Direct" ? "rgba(37,99,235,0.08)" : "rgba(37,99,235,0.08)", color: stratDisplay === "Direct" ? T.primary : T.purple }}>{stratDisplay}</span>
                            </td>
                            <td style={{ padding: "10px 12px" }}>
                              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                <span style={{ fontFamily: FONT, fontWeight: 600, fontSize: "12px", color: effColor }}>{eff.toFixed(3)}</span>
                                <div style={{ width: "40px", height: "4px", background: T.borderLight, borderRadius: "2px", overflow: "hidden" }}>
                                  <div style={{ width: `${Math.min(eff * 100, 100)}%`, height: "100%", background: effColor, borderRadius: "2px" }} />
                                </div>
                              </div>
                            </td>
                            <td style={{ padding: "10px 12px" }}>
                              {t.strategy === "Proximity" ? (() => {
                                const orig = results.find(r => r.label === t.target_label);
                                const ad = orig?.asrpaDiscrimination;
                                if (ad) {
                                  const c = ad.block_class === "strong" ? T.success : ad.block_class === "moderate" ? T.warning : T.danger;
                                  return <span style={{ fontSize: "10px", fontWeight: 600, color: c }} title={`AS-RPA ${ad.terminal_mismatch} — ${ad.block_class}`}>{ad.disc_ratio >= 100 ? "≥100" : ad.disc_ratio.toFixed(0)}× <span style={{ fontWeight: 500, color: T.purple }}>AS-RPA</span></span>;
                                }
                                return <span style={{ fontSize: "10px", color: T.purple, fontWeight: 600 }}>AS-RPA</span>;
                              })() : (
                                <span style={{ fontFamily: FONT, fontWeight: 600, fontSize: "12px", color: discColor }}>{disc > 0 ? `${disc.toFixed(1)}×` : "—"}</span>
                              )}
                            </td>
                            <td style={{ padding: "10px 12px" }}>{t.has_primers ? <CheckCircle size={14} color={T.success} /> : <span style={{ color: T.textTer }}>—</span>}</td>
                            <td style={{ padding: "10px 12px" }}>
                              {t.is_assay_ready ? (
                                <span style={{ display: "inline-flex", alignItems: "center", gap: "3px", fontSize: "10px", fontWeight: 600, padding: "3px 10px", borderRadius: "3px", background: "rgba(16,185,129,0.1)", color: T.success }}>Ready</span>
                              ) : (
                                <span style={{ display: "inline-flex", alignItems: "center", gap: "3px", fontSize: "10px", fontWeight: 600, padding: "3px 10px", borderRadius: "3px", background: T.bgSub, color: T.textTer }}>Not ready</span>
                              )}
                            </td>
                            {(() => { const orig = results.find(r => r.label === t.target_label); return orig?.riskProfile ? (<>
                              <td style={{ padding: "10px 8px", textAlign: "center" }}><RiskDot level={orig.riskProfile.overall} /></td>
                              <td style={{ padding: "10px 8px", textAlign: "center" }}>{orig.experimentalPriority != null && <PriorityBadge rank={orig.experimentalPriority} />}</td>
                            </>) : null; })()}
                          </tr>
                          {isExpanded && (
                            <tr>
                              <td colSpan={results.some(r => r.riskProfile) ? 10 : 8} style={{ padding: 0, background: T.bgSub }}>
                                <div style={{ padding: "16px 20px 16px 44px" }}>
                                  {!topK ? (
                                    <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px", color: T.textTer, padding: "8px 0" }}><Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />Loading alternative candidates…</div>
                                  ) : topK.length === 0 ? (
                                    <div style={{ fontSize: "11px", color: T.textTer, padding: "8px 0", fontStyle: "italic" }}>No alternative candidates available for this target.</div>
                                  ) : (
                                    <div>
                                      {/* Clean ranked table */}
                                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px", fontFamily: FONT }}>
                                        <thead>
                                          <tr>
                                            {["Rank", "Score", "Disc", "OT", "Spacer (20-nt)", "Tradeoff vs selected"].map(h => (
                                              <th key={h} style={{ padding: "6px 10px", textAlign: "left", fontWeight: 600, color: T.textTer, fontSize: "9px", textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.borderLight}` }}>{h}</th>
                                            ))}
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {topK.slice(0, 5).map((alt, i) => {
                                            const s = alt.efficiency ?? alt.score ?? alt.composite_score ?? 0;
                                            const aDisc = alt.discrimination_ratio ?? alt.discrimination ?? 0;
                                            const spacer = alt.spacer_seq || alt.spacer || "";
                                            const notes = alt.tradeoff_summary || alt.tradeoff_note || "";
                                            const deltaEff = alt.delta_efficiency;
                                            const isSelected = i === 0;
                                            const sColor = s >= 0.7 ? T.success : s >= 0.5 ? T.warning : T.danger;
                                            const isProximity = t.strategy === "Proximity";
                                            return (
                                              <tr key={i} style={{ borderBottom: `1px solid ${T.borderLight}`, background: isSelected ? T.primaryLight : "transparent" }}>
                                                <td style={{ padding: "7px 10px", fontFamily: FONT, fontWeight: 600, color: isSelected ? T.primary : T.textSec }}>{isSelected ? "#1 ●" : `#${i + 1}`}</td>
                                                <td style={{ padding: "7px 10px" }}>
                                                  <span style={{ fontFamily: FONT, fontWeight: 600, color: sColor }}>{s.toFixed(3)}</span>
                                                  {deltaEff != null && !isSelected && <span style={{ fontSize: "9px", fontFamily: FONT, fontWeight: 600, color: deltaEff >= 0 ? T.success : T.danger, marginLeft: "4px" }}>{deltaEff >= 0 ? "+" : ""}{deltaEff.toFixed(3)}</span>}
                                                </td>
                                                <td style={{ padding: "7px 10px", fontFamily: FONT, color: T.textSec }}>{isProximity ? <span style={{ fontSize: "10px", color: T.purple }}>AS-RPA</span> : aDisc > 0 ? `${aDisc.toFixed(1)}×` : "—"}</td>
                                                <td style={{ padding: "7px 10px", fontFamily: FONT, color: alt.offtarget_count === 0 ? T.success : alt.offtarget_count != null ? T.warning : T.textTer }}>{alt.offtarget_count ?? "—"}</td>
                                                <td style={{ padding: "7px 10px", fontFamily: MONO, fontSize: "10px", color: T.textTer, letterSpacing: "0.3px" }}>{spacer ? `${spacer.slice(0, 10)} ${spacer.slice(10, 20)}` : "—"}</td>
                                                <td style={{ padding: "7px 10px", fontSize: "10px", color: T.textTer, fontStyle: isSelected ? "normal" : "italic" }}>
                                                  {isSelected ? <span style={{ fontWeight: 600, color: T.primary, fontStyle: "normal" }}>Selected candidate</span> : (notes || "comparable")}
                                                </td>
                                              </tr>
                                            );
                                          })}
                                        </tbody>
                                      </table>
                                    </div>
                                  )}
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CollapsibleSection>
          )}

          {/* E: Parameter Sweep Charts */}
          <CollapsibleSection title="Parameter Sweep" defaultOpen={false}>
            <div style={{ display: "flex", gap: "8px", marginBottom: "16px", flexWrap: "wrap" }}>
              <Btn variant="secondary" size="sm" icon={TrendingUp} onClick={() => handleSweep("efficiency_threshold")} disabled={sweepLoading}>
                Sweep Efficiency Threshold
              </Btn>
              <Btn variant="secondary" size="sm" icon={TrendingUp} onClick={() => handleSweep("discrimination_threshold")} disabled={sweepLoading}>
                Sweep Discrimination Threshold
              </Btn>
            </div>
            {sweepLoading && (
              <div style={{ textAlign: "center", padding: "24px", color: T.textTer }}>
                <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} />
                <div style={{ marginTop: "6px", fontSize: "12px" }}>Running sweep…</div>
              </div>
            )}
            {!sweepLoading && sweepData && (
              <div>
                <div style={{ fontSize: "12px", fontWeight: 600, color: T.text, marginBottom: "8px" }}>
                  Sweep: {sweepData.parameter_name?.replace(/_/g, " ")}
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={sweepData.points} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={T.borderLight} />
                    <XAxis dataKey="value" fontSize={11} fontFamily={MONO} label={{ value: sweepData.parameter_name?.replace(/_/g, " "), position: "insideBottom", offset: -2, fontSize: 11 }} />
                    <YAxis fontSize={11} fontFamily={MONO} domain={[0, 1]} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Line type="monotone" dataKey="sensitivity" stroke={T.primary} strokeWidth={2} dot={{ r: 3 }} name="Sensitivity" />
                    <Line type="monotone" dataKey="specificity" stroke={T.success} strokeWidth={2} dot={{ r: 3 }} name="Specificity" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </CollapsibleSection>

          {/* F: Pareto Frontier */}
          <CollapsibleSection title="Pareto Frontier" defaultOpen={false}>
            <div style={{ marginBottom: "16px" }}>
              <Btn variant="secondary" size="sm" icon={Zap} onClick={handlePareto} disabled={paretoLoading}>
                Compute Pareto Frontier
              </Btn>
            </div>
            {paretoLoading && (
              <div style={{ textAlign: "center", padding: "24px", color: T.textTer }}>
                <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} />
                <div style={{ marginTop: "6px", fontSize: "12px" }}>Computing frontier…</div>
              </div>
            )}
            {!paretoLoading && paretoData && paretoData.frontier && (
              <div>
                <div style={{ fontSize: "12px", fontWeight: 600, color: T.text, marginBottom: "8px" }}>
                  {paretoData.n_points} Pareto-optimal configurations
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={T.borderLight} />
                    <XAxis type="number" dataKey="specificity" name="Specificity" domain={[0, 1]} fontSize={11} fontFamily={MONO} label={{ value: "Specificity", position: "insideBottom", offset: -10, fontSize: 11 }} />
                    <YAxis type="number" dataKey="sensitivity" name="Sensitivity" domain={[0, 1]} fontSize={11} fontFamily={MONO} label={{ value: "Sensitivity", angle: -90, position: "insideLeft", offset: 10, fontSize: 11 }} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(val, name) => [typeof val === "number" ? val.toFixed(3) : val, name]} />
                    <Scatter data={paretoData.frontier} fill={T.primary} strokeWidth={1} stroke={T.primaryDark}>
                      {paretoData.frontier.map((_, i) => (
                        <Cell key={i} fill={T.primary} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )}
          </CollapsibleSection>
        </>
      )}
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   RESULTS PAGE — Tab container with accordion candidates
   ═══════════════════════════════════════════════════════════════════ */
const RESULT_TABS = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "candidates", label: "Candidates", icon: List },
  { id: "discrimination", label: "Discrimination", icon: TrendingUp },
  { id: "primers", label: "Primers", icon: Crosshair },
  { id: "multiplex", label: "Multiplex", icon: Grid3x3 },
  { id: "diagnostics", label: "Diagnostics", icon: Shield },
];

const ResultsPage = ({ connected, jobId, scorer: scorerProp, goTo }) => {
  const mobile = useIsMobile();
  const toast = useToast();
  const [tab, setTab] = useState("overview");
  const [results, setResults] = useState(null);
  const [panelData, setPanelData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [activeJob, setActiveJob] = useState(jobId || null);
  const [exportOpen, setExportOpen] = useState(false);
  const exportRef = useRef(null);

  /* Sync activeJob when jobId prop changes (e.g. navigating from pipeline) */
  useEffect(() => {
    if (jobId) setActiveJob(jobId);
  }, [jobId]);

  /* Load jobs list */
  useEffect(() => {
    if (!connected) return;
    listJobs().then(({ data }) => {
      if (data) setJobs(data.filter((j) => j.status === "complete" || j.status === "completed"));
    });
  }, [connected]);

  /* Load results for active job */
  useEffect(() => {
    if (!activeJob) return;
    if (connected) {
      setLoading(true);
      getResults(activeJob).then(({ data }) => {
        if (data?.targets) {
          setResults(data.targets.map(transformApiCandidate));
        } else if (data?.candidates) {
          setResults(data.candidates.map(transformApiCandidate));
        }
        setPanelData({
          primer_dimer_matrix: data?.primer_dimer_matrix || null,
          primer_dimer_labels: data?.primer_dimer_labels || null,
          primer_dimer_report: data?.primer_dimer_report || null,
        });
        setLoading(false);
      });
    } else if (activeJob.startsWith("mock-")) {
      /* Mock mode — adapt mock data to scorer + panel encoded in job ID */
      const isHeuristic = activeJob.includes("-heuristic-");
      // Extract selected mutation indices from job ID (format: mock-scorer-0,1,2,...-timestamp)
      const parts = activeJob.split("-");
      const indicesStr = parts.length >= 4 ? parts.slice(2, -1).join("-") : "";
      const selectedIndices = indicesStr ? indicesStr.split(",").map(Number).filter(n => !isNaN(n)) : null;
      // Filter RESULTS to only selected mutations (+ IS6110 control always included)
      let filtered = RESULTS;
      if (selectedIndices && selectedIndices.length > 0 && selectedIndices.length < RESULTS.length) {
        const selectedLabels = new Set(selectedIndices.map(i => MUTATIONS[i] ? `${MUTATIONS[i].gene}_${MUTATIONS[i].ref}${MUTATIONS[i].pos}${MUTATIONS[i].alt}` : null).filter(Boolean));
        filtered = RESULTS.filter(r => selectedLabels.has(r.label) || r.gene === "IS6110");
      }
      if (isHeuristic) {
        setResults(filtered.map(r => ({
          ...r,
          cnnScore: undefined, cnnCalibrated: undefined,
          pamAdjusted: undefined, mlScores: [],
        })));
      } else {
        setResults(filtered);
      }
    }
  }, [connected, activeJob]);

  const handleExport = async (fmt) => {
    setExportOpen(false);
    if (connected && activeJob) {
      const { data } = await exportResults(activeJob, fmt);
      if (data) {
        const url = URL.createObjectURL(data);
        const a = document.createElement("a");
        a.href = url;
        a.download = `compass_results.${fmt}`;
        a.click();
        URL.revokeObjectURL(url);
        toast(`Exported as ${fmt.toUpperCase()}`);
      }
    }
  };

  /* Close export dropdown on outside click */
  useEffect(() => {
    if (!exportOpen) return;
    const handler = (e) => { if (exportRef.current && !exportRef.current.contains(e.target)) setExportOpen(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [exportOpen]);

  const hasResults = results && results.length > 0;

  return (
    <div style={{ padding: mobile ? "16px" : "32px 40px" }}>
      {/* Header */}
      <div style={{ display: "flex", flexDirection: mobile ? "column" : "row", justifyContent: "space-between", alignItems: mobile ? "stretch" : "center", gap: "12px", marginBottom: "24px" }}>
        <div>
          <h2 style={{ fontSize: "20px", fontWeight: 600, color: T.text, margin: 0, fontFamily: HEADING }}>
            Panel Results
          </h2>
          {hasResults && (
            <p style={{ fontSize: "13px", color: T.textSec, marginTop: "4px" }}>
              {results.length} candidates · {new Set(results.map((r) => r.drug)).size} drug classes · {results.filter(r => r.hasPrimers).length} with primers
            </p>
          )}
          {!hasResults && !loading && (
            <p style={{ fontSize: "13px", color: T.textTer, marginTop: "4px" }}>No results yet</p>
          )}
        </div>
        {hasResults && (
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            {connected && jobs.length > 0 && (
              <select value={activeJob || ""} onChange={(e) => setActiveJob(e.target.value)} style={{ padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: MONO, fontSize: "11px", outline: "none", background: T.bg }}>
                <option value="">Select job…</option>
                {jobs.map((j) => <option key={j.job_id} value={j.job_id}>{j.name || j.job_id}</option>)}
              </select>
            )}
            <div ref={exportRef} style={{ position: "relative" }}>
              <Btn variant="secondary" size="sm" icon={Download} onClick={() => setExportOpen(!exportOpen)}>Export</Btn>
              {exportOpen && (
                <div style={{ position: "absolute", top: "100%", right: 0, marginTop: "4px", background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", boxShadow: "0 1px 3px rgba(0,0,0,0.06)", zIndex: 100, minWidth: 160, overflow: "hidden" }}>
                  {[
                    { fmt: "json", label: "JSON", desc: "Full structured data" },
                    { fmt: "tsv", label: "TSV", desc: "Tab-separated values" },
                    { fmt: "csv", label: "CSV", desc: "Comma-separated values" },
                    { fmt: "fasta", label: "FASTA", desc: "Spacer sequences" },
                  ].map((opt, i, arr) => (
                    <button key={opt.fmt} onClick={() => handleExport(opt.fmt)} style={{ display: "block", width: "100%", padding: "10px 14px", background: "none", border: "none", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none", cursor: "pointer", textAlign: "left", fontFamily: FONT }} onMouseEnter={(e) => { e.currentTarget.style.background = T.bgHover; }} onMouseLeave={(e) => { e.currentTarget.style.background = "none"; }}>
                      <div style={{ fontSize: "12px", fontWeight: 600, color: T.text }}>{opt.label}</div>
                      <div style={{ fontSize: "10px", color: T.textTer }}>{opt.desc}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: "48px", color: T.textTer }}>
          <Loader2 size={24} style={{ animation: "spin 1s linear infinite" }} />
          <div style={{ marginTop: "8px", fontSize: "13px" }}>Loading results…</div>
        </div>
      )}

      {!loading && !hasResults && (
        <div style={{ textAlign: "center", padding: mobile ? "48px 24px" : "80px 24px" }}>
          <div style={{ width: 64, height: 64, borderRadius: "4px", background: T.bgSub, display: "inline-flex", alignItems: "center", justifyContent: "center", marginBottom: "20px" }}>
            <BarChart3 size={28} color={T.textTer} strokeWidth={1.5} />
          </div>
          <div style={{ fontSize: "18px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "8px" }}>No pipeline results yet</div>
          <p style={{ fontSize: "13px", color: T.textSec, lineHeight: 1.6, maxWidth: 420, margin: "0 auto 24px" }}>
            Run the COMPASS pipeline from the Home page to design crRNA candidates. Results will appear here once the pipeline completes.
          </p>
          <Btn icon={Play} onClick={() => goTo("home")}>Launch Pipeline</Btn>
        </div>
      )}

      {!loading && hasResults && (
        <>
          {/* Tabs */}
          <div style={{ display: "flex", gap: "0", marginBottom: "28px", borderBottom: `1px solid ${T.border}`, overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
            {RESULT_TABS.map((t) => (
              <button key={t.id} onClick={() => setTab(t.id)} style={{
                display: "flex", alignItems: "center", gap: "6px", padding: mobile ? "10px 14px" : "12px 20px", whiteSpace: "nowrap", flexShrink: 0,
                border: "none", cursor: "pointer", fontFamily: FONT, fontSize: "13px",
                fontWeight: tab === t.id ? 600 : 400, color: tab === t.id ? T.text : T.textTer,
                background: "transparent", borderBottom: tab === t.id ? `2px solid ${T.primary}` : "2px solid transparent",
                marginBottom: "-1px", transition: "color 0.12s",
              }}>
                <t.icon size={14} strokeWidth={tab === t.id ? 2 : 1.5} />{t.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          {tab === "overview" && <OverviewTab results={results} scorer={scorerProp} jobId={activeJob} />}
          {tab === "candidates" && <CandidatesTab results={results} jobId={activeJob} connected={connected} scorer={scorerProp} />}
          {tab === "discrimination" && <DiscriminationTab results={results} />}
          {tab === "primers" && <PrimersTab results={results} />}
          {tab === "multiplex" && <TabErrorBoundary label="Multiplex"><MultiplexTab results={results} panelData={panelData} jobId={activeJob} connected={connected} /></TabErrorBoundary>}
          {tab === "diagnostics" && <DiagnosticsErrorBoundary><DiagnosticsTab results={results} jobId={activeJob} connected={connected} scorer={scorerProp} /></DiagnosticsErrorBoundary>}
        </>
      )}

    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   PANELS PAGE
   ═══════════════════════════════════════════════════════════════════ */
const DEFAULT_PANELS = [
  {
    id: "mdr14",
    name: "MDR-TB 14-plex",
    description: "Complete WHO-catalogued first- and second-line resistance panel. Covers 6 drug classes with 14 target mutations for comprehensive drug-susceptibility profiling.",
    mutations: MUTATIONS.map(m => `${m.gene}_${m.ref}${m.pos}${m.alt}`),
    created_at: "2025-01-15T00:00:00Z",
  },
  {
    id: "core5",
    name: "Core 5-plex",
    description: "High-confidence tier-1 mutations only. Targets the most clinically actionable resistance determinants for rapid point-of-care screening.",
    mutations: ["rpoB_S531L", "katG_S315T", "fabG1_C-15T", "gyrA_D94G", "rrs_A1401G"],
    created_at: "2025-01-15T00:00:00Z",
  },
  {
    id: "rif",
    name: "Rifampicin Panel",
    description: "Focused panel for rifampicin mono-resistance detection. Covers the rpoB RRDR hotspot mutations conferring >95% of phenotypic RIF resistance.",
    mutations: ["rpoB_S531L", "rpoB_H526Y", "rpoB_D516V"],
    created_at: "2025-01-15T00:00:00Z",
  },
];

const PanelsPage = ({ connected }) => {
  const mobile = useIsMobile();
  const [panels, setPanels] = useState(DEFAULT_PANELS);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newMuts, setNewMuts] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [editMuts, setEditMuts] = useState("");

  useEffect(() => {
    if (connected) {
      listPanels().then(({ data }) => { if (data) setPanels(data); });
    }
  }, [connected]);

  const handleCreate = async () => {
    const muts = newMuts.split(",").map((s) => s.trim()).filter(Boolean);
    if (connected) {
      const { data, error } = await createPanel(newName, newDesc, muts);
      if (data) setPanels((p) => [...p, data]);
    } else {
      setPanels((p) => [...p, { id: Date.now(), name: newName, description: newDesc, mutations: muts, created_at: new Date().toISOString() }]);
    }
    setShowNew(false);
    setNewName("");
    setNewDesc("");
    setNewMuts("");
  };

  const startEdit = (p) => {
    setEditingId(p.id || p.name);
    setEditName(p.name);
    setEditDesc(p.description || "");
    setEditMuts((p.mutations || []).join(", "));
  };

  const saveEdit = (panelId) => {
    const muts = editMuts.split(",").map((s) => s.trim()).filter(Boolean);
    setPanels((prev) => prev.map((p) => (p.id || p.name) === panelId ? { ...p, name: editName, description: editDesc, mutations: muts } : p));
    setEditingId(null);
  };

  const deletePanel = (panelId) => {
    setPanels((prev) => prev.filter((p) => (p.id || p.name) !== panelId));
  };

  return (
    <div style={{ padding: mobile ? "24px 16px" : "32px 40px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
        <div>
          <div style={{ fontSize: "11px", fontWeight: 600, color: T.primary, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Library</div>
          <h2 style={{ fontSize: "20px", fontWeight: 600, color: T.text, margin: 0, letterSpacing: "-0.02em", fontFamily: HEADING }}>Mutation Panels</h2>
        </div>
        <Btn icon={Plus} size="sm" onClick={() => setShowNew(!showNew)}>New Panel</Btn>
      </div>

      {showNew && (
        <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "24px", marginBottom: "24px" }}>
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Panel Name</label>
            <input value={newName} onChange={(e) => setNewName(e.target.value)} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "13px", outline: "none", boxSizing: "border-box" }} />
          </div>
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Description</label>
            <input value={newDesc} onChange={(e) => setNewDesc(e.target.value)} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "13px", outline: "none", boxSizing: "border-box" }} />
          </div>
          <div style={{ marginBottom: "16px" }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Mutations (comma-separated)</label>
            <textarea value={newMuts} onChange={(e) => setNewMuts(e.target.value)} rows={3} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: MONO, fontSize: "12px", outline: "none", resize: "vertical", boxSizing: "border-box" }} />
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            <Btn onClick={handleCreate} disabled={!newName.trim()} size="sm">Create</Btn>
            <Btn variant="ghost" size="sm" onClick={() => setShowNew(false)}>Cancel</Btn>
          </div>
        </div>
      )}

      {panels.length === 0 && !showNew && (
        <div style={{ textAlign: "center", padding: "64px 24px", color: T.textTer }}>
          <Layers size={40} strokeWidth={1} />
          <div style={{ fontSize: "14px", marginTop: "12px" }}>No panels yet. Create one to get started.</div>
        </div>
      )}

      <div style={{ display: "grid", gap: "12px" }}>
        {panels.map((p) => {
          const pid = p.id || p.name;
          const isEditing = editingId === pid;
          return (
            <div key={pid} style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "20px" }}>
              {isEditing ? (
                <div>
                  <div style={{ marginBottom: "12px" }}>
                    <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Panel Name</label>
                    <input value={editName} onChange={(e) => setEditName(e.target.value)} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "13px", outline: "none", boxSizing: "border-box" }} />
                  </div>
                  <div style={{ marginBottom: "12px" }}>
                    <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Description</label>
                    <input value={editDesc} onChange={(e) => setEditDesc(e.target.value)} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "13px", outline: "none", boxSizing: "border-box" }} />
                  </div>
                  <div style={{ marginBottom: "16px" }}>
                    <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Mutations (comma-separated)</label>
                    <textarea value={editMuts} onChange={(e) => setEditMuts(e.target.value)} rows={3} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: MONO, fontSize: "12px", outline: "none", resize: "vertical", boxSizing: "border-box" }} />
                  </div>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <Btn onClick={() => saveEdit(pid)} disabled={!editName.trim()} size="sm">Save</Btn>
                    <Btn variant="ghost" size="sm" onClick={() => setEditingId(null)}>Cancel</Btn>
                  </div>
                </div>
              ) : (
                <>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>{p.name}</div>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <Badge variant="primary">{(p.mutations || []).length} mutations</Badge>
                      <button onClick={() => startEdit(p)} title="Edit panel" style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", display: "flex", alignItems: "center", borderRadius: "4px" }}>
                        <Pencil size={14} color={T.textTer} />
                      </button>
                      <button onClick={() => deletePanel(pid)} title="Delete panel" style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", display: "flex", alignItems: "center", borderRadius: "4px" }}>
                        <Trash2 size={14} color={T.textTer} />
                      </button>
                    </div>
                  </div>
                  {p.description && <div style={{ fontSize: "12px", color: T.textSec, marginBottom: "8px" }}>{p.description}</div>}
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                    {(p.mutations || []).slice(0, 10).map((m) => (
                      <span key={m} style={{ fontFamily: MONO, fontSize: "10px", padding: "3px 8px", borderRadius: "4px", background: T.bgSub, border: `1px solid ${T.borderLight}`, color: T.text }}>{m}</span>
                    ))}
                    {(p.mutations || []).length > 10 && <span style={{ fontSize: "10px", color: T.textTer, padding: "3px" }}>+{p.mutations.length - 10} more</span>}
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   MUTATIONS PAGE
   ═══════════════════════════════════════════════════════════════════ */
const MutationsPage = () => {
  const mobile = useIsMobile();
  const [search, setSearch] = useState("");
  const [drugFilter, setDrugFilter] = useState("ALL");
  const drugs = ["ALL", ...new Set(MUTATIONS.map((m) => m.drug))];

  const filtered = useMemo(() => {
    let arr = [...MUTATIONS];
    if (drugFilter !== "ALL") arr = arr.filter((m) => m.drug === drugFilter);
    if (search) {
      const q = search.toLowerCase();
      arr = arr.filter((m) => m.gene.toLowerCase().includes(q) || `${m.ref}${m.pos}${m.alt}`.toLowerCase().includes(q));
    }
    return arr;
  }, [search, drugFilter]);

  return (
    <div style={{ padding: mobile ? "24px 16px" : "32px 40px" }}>
      <div style={{ marginBottom: "24px" }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: T.primary, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Library</div>
        <h2 style={{ fontSize: "20px", fontWeight: 600, color: T.text, margin: 0, letterSpacing: "-0.02em", fontFamily: HEADING }}>WHO Mutation Catalogue</h2>
        <p style={{ fontSize: "13px", color: T.textSec, marginTop: "4px" }}>{MUTATIONS.length} target mutations from WHO 2023 v2 catalogue</p>
      </div>

      <div style={{ display: "flex", flexDirection: mobile ? "column" : "row", gap: "10px", marginBottom: "20px" }}>
        <div style={{ position: "relative", flex: 1 }}>
          <Search size={14} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: T.textTer }} />
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search genes, mutations…" style={{ width: "100%", padding: "9px 12px 9px 34px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "12px", outline: "none", boxSizing: "border-box" }} />
        </div>
        <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
          {drugs.map((d) => (
            <button key={d} onClick={() => setDrugFilter(d)} style={{
              padding: "6px 12px", borderRadius: "4px", fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: FONT,
              border: `1px solid ${drugFilter === d ? T.primary : T.border}`,
              background: drugFilter === d ? T.primaryLight : T.bg, color: drugFilter === d ? T.primaryDark : T.textSec,
            }}>{d}</button>
          ))}
        </div>
      </div>

      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
        <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", minWidth: 600 }}>
          <thead>
            <tr style={{ background: T.bgSub }}>
              {["Gene", "Mutation", "Drug", "Confidence", "Tier", "WHO Freq"].map((h) => (
                <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 600, color: T.textSec, borderBottom: `1px solid ${T.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((m) => {
              const key = `${m.gene}_${m.ref}${m.pos}${m.alt}`;
              const ref = WHO_REFS[key];
              return (
                <tr key={key} style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                  <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600 }}>{m.gene}</td>
                  <td style={{ padding: "10px 14px", fontFamily: MONO }}>{m.ref}{m.pos}{m.alt}</td>
                  <td style={{ padding: "10px 14px" }}><DrugBadge drug={m.drug} /></td>
                  <td style={{ padding: "10px 14px" }}><Badge variant={m.conf === "High" ? "success" : "warning"}>{m.conf}</Badge></td>
                  <td style={{ padding: "10px 14px", fontFamily: FONT }}>{m.tier}</td>
                  <td style={{ padding: "10px 14px", fontSize: "11px", color: T.textSec }}>{ref?.freq || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        </div>
        <div style={{ padding: "12px 16px", fontSize: "11px", color: T.textTer, borderTop: `1px solid ${T.border}`, background: T.bgSub }}>
          Showing {filtered.length} of {MUTATIONS.length} mutations
        </div>
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   SCORING PAGE
   ═══════════════════════════════════════════════════════════════════ */
const ScoringPage = ({ connected }) => {
  const mobile = useIsMobile();
  const [models, setModels] = useState([]);
  const [showApiRef, setShowApiRef] = useState(false);
  const [openBlocks, setOpenBlocks] = useState({ heuristic: true, narsilml: false, discrimination: false, bjepa: false });
  const scoringRefs = useRef({});

  const toggleBlock = (k) => setOpenBlocks(prev => {
    const willOpen = !prev[k];
    if (willOpen) {
      setTimeout(() => {
        scoringRefs.current[k]?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    }
    return { ...prev, [k]: willOpen };
  });

  useEffect(() => {
    if (connected) {
      listScoringModels().then(({ data }) => { if (data) setModels(data); });
    }
  }, [connected]);

  /* Reusable collapsible block header */
  const ScoringBlock = ({ id, icon: Icon, title, badge, badgeVariant, dashed, dimmed, children }) => {
    const isOpen = openBlocks[id];
    return (
      <div ref={el => { scoringRefs.current[id] = el; }} style={{ background: T.bg, border: `1px ${dashed ? "dashed" : "solid"} ${T.border}`, borderRadius: "4px", marginBottom: "12px", overflow: "hidden", opacity: dimmed ? 0.85 : 1 }}>
        <button onClick={() => toggleBlock(id)} style={{
          width: "100%", display: "flex", alignItems: "center", gap: "10px", padding: "16px 20px",
          background: T.bgSub, border: "none", cursor: "pointer", fontFamily: FONT, textAlign: "left",
        }}>
          <div style={{ width: "16px", height: "16px", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            {isOpen ? <ChevronDown size={14} color={T.textSec} /> : <ChevronRight size={14} color={T.textSec} />}
          </div>
          <Icon size={18} color={T.primary} />
          <span style={{ fontSize: "15px", fontWeight: 600, color: T.text, fontFamily: HEADING, flex: 1 }}>{title}</span>
          {badge && <span style={{ fontSize: "10px", fontWeight: 600, padding: "2px 8px", borderRadius: "3px", background: badgeVariant === "success" ? "#dcfce7" : T.primaryLight, color: badgeVariant === "success" ? "#059669" : T.primary, fontFamily: FONT }}>{badge}</span>}
          <span style={{ fontSize: "10px", color: T.textTer, flexShrink: 0 }}>{isOpen ? "collapse" : "expand"}</span>
        </button>
        {isOpen && <div style={{ padding: mobile ? "16px" : "20px 24px", borderTop: `1px solid ${T.border}` }}>{children}</div>}
      </div>
    );
  };

  return (
    <div style={{ padding: mobile ? "24px 16px" : "32px 40px" }}>
      <div style={{ marginBottom: "24px" }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: T.primary, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Models</div>
        <h2 style={{ fontSize: "20px", fontWeight: 600, color: T.text, margin: 0, letterSpacing: "-0.02em", fontFamily: HEADING }}>Scoring Models</h2>
        <p style={{ fontSize: "13px", color: T.textSec, marginTop: "4px" }}>Heuristic and ML-based candidate scoring</p>
      </div>

      {/* ── Heuristic Model ── */}
      <ScoringBlock id="heuristic" icon={Brain} title="Heuristic Model (Default)" badge="Active" badgeVariant="success">
        <p style={{ fontSize: "13px", color: T.textSec, lineHeight: 1.7, marginBottom: "16px", marginTop: 0 }}>
          Position-weighted composite scoring across 5 biophysical features. This is the default scoring model used by the COMPASS pipeline.
        </p>

        <div style={{ background: T.bgSub, borderRadius: "4px", overflow: "hidden" }}>
          {SCORING_FEATURES.map((f, i) => (
            <div key={f.key} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderBottom: i < SCORING_FEATURES.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
              <div style={{ width: 140, fontSize: "13px", fontWeight: 600, color: T.text }}>{f.name}</div>
              <div style={{ flex: 1, height: 8, background: T.bg, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${f.weight * 100}%`, height: "100%", background: T.primary, borderRadius: 4 }} />
              </div>
              <span style={{ fontSize: "13px", fontWeight: 600, color: T.primary, width: 50, textAlign: "right" }}>{(f.weight * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop: "16px", fontSize: "12px", color: T.textSec, lineHeight: 1.6 }}>
          <strong>Formula:</strong> composite = {"\u03a3"}(feature_score {"\u00d7"} weight) where each feature_score {"\u2208"} [0, 1]
        </div>
      </ScoringBlock>

      {/* ── Compass-ML ── */}
      <ScoringBlock id="narsilml" icon={Cpu} title="Compass-ML" badge="Recommended" badgeVariant="success">
        <p style={{ fontSize: "13px", color: T.textSec, lineHeight: 1.7, marginBottom: "16px", marginTop: 0 }}>
          Dual-branch neural network combining a target-DNA CNN with RNA Foundation Model (RNA-FM) embeddings for crRNA secondary structure.
          R-Loop Propagation Attention (RLPA) encodes the biophysics of Cas12a's directional R-loop formation into the architecture.
          Dual-branch CNN + RNA-FM with R-Loop Propagation Attention (RLPA) and 9-class PAM encoding. Trained on 15,000 cis-cleavage measurements from Kim et al. (2018) with flanking shuffle augmentation. Activity score serves as primary ranking; heuristic provides independent biophysical QC.
        </p>

        {/* Architecture branches */}
        <div style={{ fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.04em" }}>Branch Ablation (Kim 2018 cross-library)</div>
        <div style={{ background: T.bgSub, borderRadius: "4px", overflow: "hidden", marginBottom: "20px" }}>
          {[
            { name: "CNN only (baseline)", rho: "0.740", delta: null },
            { name: "+ PAM encoding (9-class)", rho: "0.741", delta: "+0.2%" },
            { name: "+ RNA-FM embeddings", rho: "0.744", delta: "+0.5%" },
            { name: "+ RLPA attention", rho: "0.745", delta: "+0.7%" },
            { name: "Best single seed (production)", rho: "0.750", delta: "+1.3%" },
          ].map((f, i, arr) => (
            <div key={f.name} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
              <div style={{ flex: 1, fontSize: "13px", fontWeight: 600, color: T.text }}>{f.name}</div>
              <span style={{ fontSize: "13px", fontWeight: 600, fontFamily: FONT, color: T.text, width: 55, textAlign: "right" }}>{f.rho}</span>
              {f.delta ? (
                <span style={{ fontSize: "11px", fontWeight: 600, fontFamily: FONT, color: T.success, width: 55, textAlign: "right" }}>{f.delta}</span>
              ) : (
                <span style={{ width: 55, textAlign: "right", fontSize: "11px", color: T.textTer }}>baseline</span>
              )}
            </div>
          ))}
        </div>

        {/* Architecture details */}
        <div style={{ fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.04em" }}>Architecture</div>
        <div style={{ background: T.bgSub, borderRadius: "4px", overflow: "hidden" }}>
          {[
            ["Architecture", "CNN (k=3,5,7) + RNA-FM (640\u219264) + RLPA + PAM (9-class)"],
            ["CNN input", "One-hot 34nt (PAM + spacer + context)"],
            ["RNA-FM input", "Pre-cached 640-dim embeddings (frozen, 23M sequences)"],
            ["PAM encoding", "9-class learned embedding (Kleinstiver 2019 enAsCas12a)"],
            ["Training data", "Kim 2018 HT1-1 (15K guides) + flanking shuffle augmentation"],
            ["Parameters", "235K (CNN ~65K, RNA-FM proj ~41K, RLPA ~35K, PAM ~1K, heads ~8K)"],
            ["Val \u03c1", "0.750 (best of 3 seeds, Kim 2018 HT1-2)"],
            ["Ablation", "CNN 0.740 \u2192 +PAM 0.741 \u2192 +RNA-FM 0.744 \u2192 +RLPA 0.745"],
            ["Augmentation", "Flanking shuffle (p=0.3) + label noise (\u03c3=0.02)"],
            ["Loss", "Huber + differentiable Spearman (annealed)"],
          ].map(([k, v], i, arr) => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 16px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none", fontSize: "12px" }}>
              <span style={{ color: T.textSec, fontWeight: 500 }}>{k}</span>
              <span style={{ fontWeight: 600, color: T.text, fontSize: "12px" }}>{v}</span>
            </div>
          ))}
        </div>
      </ScoringBlock>

      {/* ── Discrimination Prediction (moved above B-JEPA) ── */}
      <ScoringBlock id="discrimination" icon={TrendingUp} title="Discrimination Prediction" badge="Trained" badgeVariant="success">
        <p style={{ fontSize: "13px", color: T.textSec, lineHeight: 1.7, margin: "0 0 16px" }}>
          Gradient-boosted model (XGBoost) trained on 6,136 paired MUT/WT trans-cleavage measurements from the EasyDesign dataset (Huang et al. 2024, LbCas12a).
          Predicts the discrimination ratio ({"\u0394"}log-k between perfect-match and single-mismatch targets) from 18 thermodynamic features encoding mismatch position, chemistry, R-loop energetics, and sequence context.
        </p>
        <div style={{ fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.04em" }}>Performance (3-fold stratified CV, guide-level split)</div>
        <div style={{ background: T.bgSub, borderRadius: "4px", overflow: "hidden", marginBottom: "16px" }}>
          {[
            { name: "Heuristic baseline", rmse: "0.641", corr: "0.298", delta: null },
            { name: "XGBoost 15 features (v1)", rmse: "0.540", corr: "0.459", delta: "\u221215% RMSE" },
            { name: "XGBoost 18 features (v2, production)", rmse: "0.520", corr: "0.565", delta: "\u221219% RMSE" },
          ].map((f, i, arr) => (
            <div key={f.name} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
              <div style={{ flex: 1, fontSize: "13px", fontWeight: 600, color: T.text }}>{f.name}</div>
              <span style={{ fontSize: "12px", fontFamily: FONT, color: T.textSec, width: 80, textAlign: "right" }}>RMSE {f.rmse}</span>
              <span style={{ fontSize: "12px", fontFamily: FONT, color: T.text, fontWeight: 600, width: 55, textAlign: "right" }}>r={f.corr}</span>
              {f.delta ? (
                <span style={{ fontSize: "11px", fontWeight: 600, fontFamily: FONT, color: T.success, width: 75, textAlign: "right" }}>{f.delta}</span>
              ) : (
                <span style={{ width: 75, textAlign: "right", fontSize: "11px", color: T.textTer }}>baseline</span>
              )}
            </div>
          ))}
        </div>
        <div style={{ fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.04em" }}>Top Features (by importance)</div>
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr 1fr" : "repeat(5, 1fr)", gap: "8px", marginBottom: "12px" }}>
          {[
            { label: "Seed \u0394G", desc: "R-loop stability at seed" },
            { label: "Total hybrid \u0394G", desc: "Full RNA:DNA energy" },
            { label: "Cumulative \u0394G", desc: "Energy at mismatch pos" },
            { label: "Energy ratio", desc: "|cum. \u0394G| / \u0394\u0394G" },
            { label: "GC content", desc: "Spacer GC fraction" },
          ].map(f => (
            <div key={f.label} style={{ background: T.bgSub, borderRadius: "4px", padding: "10px", textAlign: "center" }}>
              <div style={{ fontSize: "11px", fontWeight: 600, color: T.primary }}>{f.label}</div>
              <div style={{ fontSize: "9px", color: T.textTer, marginTop: "2px" }}>{f.desc}</div>
            </div>
          ))}
        </div>
        <div style={{ fontSize: "11px", color: T.textTer, lineHeight: 1.6 }}>
          Training data: EasyDesign (Huang et al. 2024). Features: R-loop {"\u0394"}G profiles (Sugimoto 1995 NN params), mismatch {"\u0394\u0394"}G penalties (Sugimoto 2000), position sensitivity (Strohkendl 2018). Guide-level CV split prevents data leakage.
        </div>
      </ScoringBlock>

      {/* ── B-JEPA (teaser) ── */}
      <ScoringBlock id="bjepa" icon={Brain} title="B-JEPA" badge="In development" dashed dimmed>
        <p style={{ fontSize: "13px", color: T.textSec, lineHeight: 1.7, margin: "0 0 16px" }}>
          Self-supervised foundation model using Joint-Embedding Predictive Architecture with latent grounding — pretrained on 6,326 complete bacterial reference genomes (10M fragments {"\u00d7"} 2048bp).
          Dynamic JEPA{"\u2192"}MLM loss scheduling: JEPA shapes high-dimensional representation space (RankMe {">"}450), then masked language modeling drives token-level genomic feature learning.
          Per-dimension variance floor prevents representational collapse. Targets downstream Cas12a guide efficiency prediction and MDR-TB drug resistance classification.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr 1fr" : "repeat(3, 1fr)", gap: "10px", marginBottom: "16px" }}>
          {[
            { label: "Pretraining", value: "6,326 bacterial genomes (NCBI RefSeq)" },
            { label: "Fragments", value: "10M (2048bp window, 512bp stride)" },
            { label: "Architecture", value: "12L \u00d7 576D \u00d7 9H encoder, 6L \u00d7 384D predictor" },
            { label: "Parameters", value: "64.3M" },
            { label: "Training", value: "v7.0" },
            { label: "Target \u03c1", value: "> 0.60" },
          ].map(s => (
            <div key={s.label} style={{ background: T.bgSub, borderRadius: "4px", padding: "12px", textAlign: "center" }}>
              <div style={{ fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "4px" }}>{s.label}</div>
              <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{s.value}</div>
            </div>
          ))}
        </div>
      </ScoringBlock>

      {/* API models — hidden by default, developer-only */}
      {models.length > 0 && (
        <div style={{ marginTop: "8px" }}>
          <button onClick={() => setShowApiRef(!showApiRef)} style={{ background: "none", border: "none", cursor: "pointer", fontSize: "12px", color: T.textTer, padding: "6px 0", display: "flex", alignItems: "center", gap: "6px" }}>
            <ChevronDown size={14} style={{ transform: showApiRef ? "rotate(0deg)" : "rotate(-90deg)", transition: "transform 0.2s" }} />
            API Reference ({models.length} models)
          </button>
          {showApiRef && (
            <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "24px", marginTop: "8px" }}>
              <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "12px" }}>Available Models from API</div>
              {models.map((m) => (
                <div key={m.id || m.name} style={{ padding: "12px", borderRadius: "4px", border: `1px solid ${T.borderLight}`, marginBottom: "8px" }}>
                  <div style={{ fontWeight: 600, color: T.text }}>{m.name}</div>
                  {m.description && <div style={{ fontSize: "12px", color: T.textSec, marginTop: "4px" }}>{m.description}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   RESEARCH PAGE — Experimental sandbox for scoring R&D
   ═══════════════════════════════════════════════════════════════════ */
const ResearchPage = ({ connected }) => {
  const mobile = useIsMobile();
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState("");
  const [comparison, setComparison] = useState(null);
  const [modelA, setModelA] = useState("heuristic");
  const [modelB, setModelB] = useState("compass_ml");
  const [comparing, setComparing] = useState(false);
  const [thermoTarget, setThermoTarget] = useState("");
  const [thermoStandaloneSeq, setThermoStandaloneSeq] = useState("TCGGTCAACCCCGACAGC");
  const [thermoMode, setThermoMode] = useState("standalone"); // "standalone" | "panel"
  const [thermoData, setThermoData] = useState(null);
  const [thermoLoading, setThermoLoading] = useState(false);
  const [thermoShowWT, setThermoShowWT] = useState(true);
  const [ablation, setAblation] = useState([]);
  const [sciBgOpen, setSciBgOpen] = useState(false);
  const [nucleaseProfiles, setNucleaseProfiles] = useState(null);
  const [nucleaseCoverage, setNucleaseCoverage] = useState(null);
  const [nucleaseLoading, setNucleaseLoading] = useState(false);
  const [expandedVariant, setExpandedVariant] = useState(null);

  useEffect(() => {
    if (connected) {
      listJobs().then(({ data }) => {
        if (data) {
          const completed = (data.jobs || data || []).filter(j => j.status === "completed");
          setJobs(completed);
          if (completed.length > 0 && !selectedJob) setSelectedJob(completed[0].job_id);
        }
      });
      getAblation().then(({ data }) => { if (data) setAblation(data); });
      getNucleaseProfiles().then(({ data }) => { if (data?.profiles) setNucleaseProfiles(data.profiles); });
    }
  }, [connected]);

  const handleCompare = async () => {
    if (!selectedJob) return;
    setComparing(true);
    const { data } = await compareScorers(selectedJob, modelA, modelB);
    if (data) setComparison(data);
    setComparing(false);
  };

  const handleThermo = async (label) => {
    if (!selectedJob || !label) return;
    setThermoLoading(true);
    setThermoTarget(label);
    setThermoMode("panel");
    const { data } = await getThermoProfile(selectedJob, label);
    if (data) setThermoData(data);
    setThermoLoading(false);
  };

  const handleThermoStandalone = async () => {
    const seq = thermoStandaloneSeq.trim().toUpperCase();
    if (!seq || seq.length < 15) return;
    setThermoLoading(true);
    setThermoTarget(seq);
    setThermoMode("standalone");
    const { data } = await getThermoStandalone(seq);
    if (data) setThermoData(data);
    setThermoLoading(false);
  };

  // Research-specific styles
  const RS = {
    bg: T.bgSub, cardBg: T.bg, border: T.border, text: T.text,
    muted: T.textSec, accent: T.primary, positive: T.success, negative: T.danger,
    mutLine: T.text, wtLine: T.textTer, seedBg: `${T.primary}0A`,
    snpLine: T.danger, barrier: "#D97706",
  };
  const rTooltip = { background: "#fff", border: `1px solid ${T.border}`, borderRadius: "3px", fontSize: "11px", fontFamily: MONO, color: T.text, boxShadow: "0 1px 3px rgba(0,0,0,0.06)" };
  const selectStyle = { padding: "6px 10px", borderRadius: "4px", border: `1px solid ${RS.border}`, fontSize: "12px", fontFamily: FONT, background: RS.cardBg, color: RS.text };
  const btnStyle = { padding: "6px 14px", borderRadius: "4px", border: "none", background: RS.accent, color: "#fff", fontSize: "12px", fontWeight: 600, cursor: "pointer" };
  const thStyle = { padding: "8px 12px", fontWeight: 600, color: RS.muted, fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${RS.border}` };
  const tdStyle = { padding: "8px 12px", fontSize: "12px", fontFamily: FONT, color: RS.text };

  // Heuristic feature weights for waterfall chart
  const HEURISTIC_WEIGHTS = [
    { name: "Seed position", key: "seed", weight: 0.30 },
    { name: "GC content", key: "gc", weight: 0.20 },
    { name: "Secondary struct.", key: "ss", weight: 0.15 },
    { name: "Off-target", key: "ot", weight: 0.20 },
    { name: "Thermo. stability", key: "thermo", weight: 0.15 },
  ];

  // Position importance (approximate RLPA/heuristic seed weights)
  const POS_WEIGHTS = Array.from({ length: 20 }, (_, i) => {
    const pos = i + 1;
    if (pos <= 4) return 1.0;
    if (pos <= 8) return 0.7 - (pos - 5) * 0.08;
    if (pos <= 12) return 0.35 - (pos - 9) * 0.04;
    return 0.18 - (pos - 13) * 0.015;
  });

  return (
    <div style={{ padding: mobile ? "24px 16px" : "32px 40px", background: RS.bg, minHeight: "100%" }}>
      {/* Header */}
      <div style={{ marginBottom: "28px" }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: RS.accent, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Research</div>
        <h2 style={{ fontSize: "20px", fontWeight: 600, color: RS.text, margin: 0, letterSpacing: "-0.02em", fontFamily: HEADING }}>Scoring R&D Sandbox</h2>
        <p style={{ fontSize: "13px", color: RS.muted, marginTop: "8px", lineHeight: 1.7, maxWidth: "720px" }}>
          Experimental workspace for scoring model development. Results here are exploratory — they inform model selection and feature engineering but do not affect production panel design. All thermodynamic calculations use nearest-neighbor parameters (Sugimoto et al. 1995 for RNA:DNA; SantaLucia 1998 for DNA:DNA) and are approximations of the true molecular energetics.
        </p>
      </div>

      {/* Job selector */}
      <div style={{ marginBottom: "24px", display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
        <span style={{ fontSize: "12px", fontWeight: 600, color: RS.muted }}>Panel run:</span>
        <select value={selectedJob} onChange={(e) => { setSelectedJob(e.target.value); setComparison(null); setThermoData(null); }} style={selectStyle}>
          {jobs.length === 0 && <option value="">No completed jobs</option>}
          {jobs.map(j => <option key={j.job_id} value={j.job_id}>{j.name || j.job_id}</option>)}
        </select>
      </div>

      {/* ═══ Section 1: Scorer Comparison Lab ═══ */}
      <CollapsibleSection title="Scorer Comparison Lab" defaultOpen={true}>
        <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap", marginBottom: "16px" }}>
          <select value={modelA} onChange={(e) => setModelA(e.target.value)} style={selectStyle}>
            <option value="heuristic">Heuristic</option>
            <option value="compass_ml">Compass-ML</option>
            <option value="compass_ml_diagnostic">Compass-ML Diagnostic</option>
          </select>
          <span style={{ fontSize: "12px", color: RS.muted, fontWeight: 600 }}>vs</span>
          <select value={modelB} onChange={(e) => setModelB(e.target.value)} style={selectStyle}>
            <option value="compass_ml">Compass-ML</option>
            <option value="heuristic">Heuristic</option>
            <option value="compass_ml_diagnostic">Compass-ML Diagnostic</option>
          </select>
          <button onClick={handleCompare} disabled={comparing || !selectedJob} style={{ ...btnStyle, opacity: comparing || !selectedJob ? 0.5 : 1 }}>
            {comparing ? "Comparing..." : "Compare"}
          </button>
        </div>

        {comparison && (() => {
          const { targets, summary } = comparison;
          const scoresA = targets.map(t => t.model_a.score || 0);
          const scoresB = targets.map(t => t.model_b.score || 0);
          const kdeA = gaussianKDE(scoresA, 0.06, 80);
          const kdeB = gaussianKDE(scoresB, 0.06, 80);
          const kdeOverlay = kdeA.map((p, i) => ({ x: p.x, a: p.density, b: kdeB[i]?.density || 0 }));
          return (
            <div>
              {/* Summary metrics */}
              <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "repeat(3, 1fr)", gap: "12px", marginBottom: "20px" }}>
                <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "12px 16px" }}>
                  <div style={{ fontSize: "11px", color: RS.muted, fontWeight: 600, marginBottom: "4px" }}>KENDALL TAU</div>
                  <div style={{ fontSize: "18px", fontWeight: 600, color: RS.text, fontFamily: FONT }}>{summary.kendall_tau?.toFixed(3) ?? "—"}</div>
                  <div style={{ fontSize: "10px", color: RS.muted, marginTop: "2px" }}>1.0 = identical ranking, 0 = unrelated</div>
                </div>
                <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "12px 16px" }}>
                  <div style={{ fontSize: "11px", color: RS.muted, fontWeight: 600, marginBottom: "4px" }}>MEAN SCORE DELTA</div>
                  <div style={{ fontSize: "18px", fontWeight: 600, color: summary.mean_score_delta > 0 ? RS.positive : summary.mean_score_delta < 0 ? RS.negative : RS.text, fontFamily: FONT }}>
                    {summary.mean_score_delta > 0 ? "+" : ""}{summary.mean_score_delta?.toFixed(4) || "0"}
                  </div>
                  <div style={{ fontSize: "10px", color: RS.muted, marginTop: "2px" }}>Average score change (B - A)</div>
                </div>
                <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "12px 16px" }}>
                  <div style={{ fontSize: "11px", color: RS.muted, fontWeight: 600, marginBottom: "4px" }}>DIAGNOSTIC IMPACT</div>
                  <div style={{ fontSize: "13px", fontWeight: 600, color: RS.text, lineHeight: 1.5 }}>
                    {summary.dropped.length === 0 && summary.gained.length === 0
                      ? `Both models: ${summary.above_threshold_a}/${summary.total_targets} above 0.4. No panel change.`
                      : <>
                          {summary.dropped.length > 0 && <div style={{ color: RS.negative }}>Dropped: {summary.dropped.join(", ")}</div>}
                          {summary.gained.length > 0 && <div style={{ color: RS.positive }}>Gained: {summary.gained.join(", ")}</div>}
                        </>
                    }
                  </div>
                </div>
              </div>

              {/* Comparison table */}
              <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", overflow: "hidden", marginBottom: "20px" }}>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <th style={{ ...thStyle, textAlign: "left" }}>Target</th>
                        <th style={{ ...thStyle, textAlign: "left" }}>Drug</th>
                        <th style={{ ...thStyle, textAlign: "left" }}>Strategy</th>
                        <th style={{ ...thStyle, textAlign: "right" }}>Score A</th>
                        <th style={{ ...thStyle, textAlign: "right" }}>Score B</th>
                        <th style={{ ...thStyle, textAlign: "right" }}>Delta</th>
                        <th style={{ ...thStyle, textAlign: "center" }}>Disc</th>
                        <th style={{ ...thStyle, textAlign: "center" }}>Rank A</th>
                        <th style={{ ...thStyle, textAlign: "center" }}>Rank B</th>
                        <th style={{ ...thStyle, textAlign: "center" }}>Rank Delta</th>
                        <th style={{ ...thStyle, textAlign: "center" }}>Thermo</th>
                      </tr>
                    </thead>
                    <tbody>
                      {targets.map((t, i) => {
                        const bigShift = Math.abs(t.rank_delta || 0) >= 3;
                        return (
                          <tr key={i} style={{ borderBottom: `1px solid ${RS.border}`, fontWeight: bigShift ? 700 : 400 }}>
                            <td style={{ ...tdStyle, fontWeight: 600 }}>{t.label}</td>
                            <td style={{ ...tdStyle, color: RS.muted }}>{t.drug || "—"}</td>
                            <td style={{ ...tdStyle, color: RS.muted, fontSize: "11px" }}>{t.strategy || "—"}</td>
                            <td style={{ ...tdStyle, textAlign: "right" }}>{t.model_a.score?.toFixed(3) ?? "—"}</td>
                            <td style={{ ...tdStyle, textAlign: "right" }}>{t.model_b.score?.toFixed(3) ?? "—"}</td>
                            <td style={{ ...tdStyle, textAlign: "right", color: t.score_delta > 0 ? RS.positive : t.score_delta < 0 ? RS.negative : RS.muted }}>
                              {t.score_delta != null ? `${t.score_delta > 0 ? "+" : ""}${t.score_delta.toFixed(3)}` : "—"}
                            </td>
                            <td style={{ ...tdStyle, textAlign: "center", color: RS.muted }}>{t.model_a.disc != null ? `${t.model_a.disc}x` : "—"}</td>
                            <td style={{ ...tdStyle, textAlign: "center" }}>#{t.model_a.rank ?? "—"}</td>
                            <td style={{ ...tdStyle, textAlign: "center" }}>#{t.model_b.rank ?? "—"}</td>
                            <td style={{ ...tdStyle, textAlign: "center", color: t.rank_delta > 0 ? RS.positive : t.rank_delta < 0 ? RS.negative : RS.muted }}>
                              {t.rank_delta != null ? (t.rank_delta > 0 ? `▲${t.rank_delta}` : t.rank_delta < 0 ? `▼${Math.abs(t.rank_delta)}` : "—") : "—"}
                            </td>
                            <td style={{ ...tdStyle, textAlign: "center" }}>
                              <button onClick={() => handleThermo(t.label)} style={{ background: "none", border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "2px 8px", cursor: "pointer", fontSize: "10px", color: RS.accent, fontWeight: 600 }}>View</button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Score distribution KDE overlay */}
              {!mobile && (
                <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "20px 24px" }}>
                  <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text, marginBottom: "4px" }}>Score Distribution Comparison</div>
                  <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "14px" }}>Overlaid KDE curves showing how each model distributes scores across candidates.</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={kdeOverlay} margin={{ top: 5, right: 15, bottom: 20, left: 15 }}>
                      <defs>
                        <linearGradient id="kdeCompA" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={RS.accent} stopOpacity={0.2} />
                          <stop offset="100%" stopColor={RS.accent} stopOpacity={0.02} />
                        </linearGradient>
                        <linearGradient id="kdeCompB" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={RS.barrier} stopOpacity={0.2} />
                          <stop offset="100%" stopColor={RS.barrier} stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="x" type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: RS.muted, fontFamily: MONO }} tickCount={11} axisLine={{ stroke: RS.border }} tickLine={false} />
                      <YAxis hide domain={[0, "auto"]} />
                      <Tooltip contentStyle={rTooltip} formatter={(v) => [v.toFixed(4), "Density"]} labelFormatter={(l) => `Score: ${l}`} />
                      <Area type="monotone" dataKey="a" stroke={RS.accent} strokeWidth={2} fill="url(#kdeCompA)" name={comparison.model_a} isAnimationActive={false} />
                      <Area type="monotone" dataKey="b" stroke={RS.barrier} strokeWidth={2} fill="url(#kdeCompB)" name={comparison.model_b} isAnimationActive={false} strokeDasharray="6 3" />
                      <Legend wrapperStyle={{ fontSize: "11px", fontFamily: MONO }} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          );
        })()}
      </CollapsibleSection>

      {/* ═══ Section 2: R-Loop Thermodynamic Explorer ═══ */}
      <CollapsibleSection title="R-Loop Thermodynamic Explorer" defaultOpen={false}>
        {/* Scientific background toggle */}
        <div style={{ marginBottom: "16px" }}>
          <button onClick={() => setSciBgOpen(!sciBgOpen)} style={{ background: "none", border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "6px 12px", cursor: "pointer", fontSize: "11px", color: RS.accent, fontWeight: 600, display: "flex", alignItems: "center", gap: "6px" }}>
            {sciBgOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            Scientific Background
          </button>
          {sciBgOpen && (
            <div style={{ marginTop: "10px", padding: "16px 20px", background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", fontSize: "12px", color: RS.muted, lineHeight: 1.8 }}>
              <p style={{ margin: "0 0 10px 0" }}>R-loop formation is the rate-limiting step of CRISPR-Cas12a target recognition (Strohkendl et al., Molecular Cell 2018; 2024). The crRNA spacer hybridises to the target strand of dsDNA, displacing the non-target strand, in a sequential PAM-proximal to PAM-distal process. Each dinucleotide step contributes a free energy increment that depends on the base-pair identity (nearest-neighbor model).</p>
              <p style={{ margin: "0 0 10px 0" }}>Zhang et al. (Nucleic Acids Research 2024, DOI: 10.1093/nar/gkae1124) demonstrated a linear correlation between Cas12a trans-cleavage kinetics and the free energy change required to unwind the crRNA spacer and DNA target from their self-folded states to a hybridisation-competent conformation. This "unwinding cost" is the dominant predictor of trans-cleavage rate.</p>
              <p style={{ margin: "0 0 10px 0" }}>CRISPRzip (Offerhaus et al., bioRxiv 2025) formalises R-loop formation as movement through a sequence-dependent free-energy landscape, combining nearest-neighbor RNA:DNA hybrid energetics with protein-mediated contributions inferred from high-throughput kinetics.</p>
              <p style={{ margin: "0 0 10px 0" }}>Aris et al. (Nature Communications 2025, DOI: 10.1038/s41467-025-57703-y) established a four-state kinetic model for Cas12a R-loop dynamics using single-molecule measurements, showing that R-loop formation is dynamic and reversible, with supercoiling-dependent interrogation.</p>
              <p style={{ margin: 0, fontStyle: "italic", fontSize: "11px", color: "#a3a3a3" }}>The profiles shown here use the Sugimoto et al. (1995) nearest-neighbor parameters for RNA:DNA hybrid thermodynamics and the SantaLucia (1998) unified parameters for DNA:DNA duplex stability. These are approximations — the true free-energy landscape includes protein-mediated contributions, supercoiling effects, and PAM-proximal protein contacts that stabilise early R-loop intermediates beyond what nucleic acid thermodynamics alone predict.</p>
            </div>
          )}
        </div>

        {/* Target input — dual mode */}
        {!thermoData && !thermoLoading && (
          <div style={{ fontSize: "12px", color: RS.muted }}>
            {/* Mode tabs */}
            <div style={{ display: "flex", gap: "0", marginBottom: "12px" }}>
              {[{ key: "standalone", label: "Standalone (enter sequence)" }, { key: "panel", label: "Panel-linked (select target)" }].map(m => (
                <button key={m.key} onClick={() => setThermoMode(m.key)} style={{ background: thermoMode === m.key ? RS.accent : "transparent", color: thermoMode === m.key ? "#fff" : RS.muted, border: `1px solid ${thermoMode === m.key ? RS.accent : RS.border}`, padding: "5px 12px", fontSize: "11px", fontWeight: 600, cursor: "pointer", borderRadius: m.key === "standalone" ? "6px 0 0 6px" : "0 6px 6px 0" }}>{m.label}</button>
              ))}
            </div>

            {thermoMode === "standalone" ? (
              <div>
                <div style={{ marginBottom: "6px" }}>Enter a DNA spacer sequence (15–30 nt) to compute the R-loop free energy profile:</div>
                <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                  <input value={thermoStandaloneSeq} onChange={(e) => setThermoStandaloneSeq(e.target.value.toUpperCase().replace(/[^ATCG]/g, ""))} placeholder="e.g. TCGGTCAACCCCGACAGC" style={{ ...selectStyle, flex: 1, maxWidth: "320px", fontFamily: MONO, letterSpacing: "0.04em" }} />
                  <button onClick={handleThermoStandalone} disabled={thermoStandaloneSeq.trim().length < 15} style={{ ...btnStyle, opacity: thermoStandaloneSeq.trim().length < 15 ? 0.5 : 1 }}>Compute</button>
                </div>
                <div style={{ fontSize: "10px", color: "#a3a3a3", marginTop: "4px" }}>Pre-filled: rpoB_S531L spacer (18 nt). No panel run needed.</div>
              </div>
            ) : (
              <div>
                <div style={{ marginBottom: "6px" }}>Select a target from a completed panel run, or enter a target label:</div>
                <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                  <input value={thermoTarget} onChange={(e) => setThermoTarget(e.target.value)} placeholder="e.g. rpoB_S531L" style={{ ...selectStyle, flex: 1, maxWidth: "240px" }} />
                  <button onClick={() => handleThermo(thermoTarget)} disabled={!thermoTarget || !selectedJob} style={{ ...btnStyle, opacity: !thermoTarget || !selectedJob ? 0.5 : 1 }}>Load</button>
                </div>
                {!selectedJob && <div style={{ fontSize: "10px", color: "#dc2626", marginTop: "4px" }}>No completed panel run selected. Switch to Standalone mode or run a panel first.</div>}
              </div>
            )}
          </div>
        )}
        {thermoLoading && <div style={{ fontSize: "12px", color: RS.muted }}>Computing thermodynamic profile...</div>}

        {thermoData && (() => {
          const mp = thermoData.mutant_profile;
          const wp = thermoData.wildtype_profile;
          const ppDg = thermoData.per_position_dg || [];
          const sc = thermoData.scalars || {};
          const eb = thermoData.energy_budget || {};
          const snpPos = thermoData.snp_position;

          // Build chart data for cumulative profile
          const cumData = (mp?.cumulative_dg || mp?.positions || []).map((val, i) => {
            const pos = mp?.positions ? mp.positions[i] : i + 1;
            const mutDg = mp?.cumulative_dg ? mp.cumulative_dg[i] : val;
            const wtDg = wp?.cumulative_dg ? wp.cumulative_dg[i] : null;
            return { pos, mutant: mutDg, wildtype: thermoShowWT ? wtDg : null };
          });

          // Per-position bar data
          const barData = ppDg.map((dg, i) => ({
            pos: i + 1,
            dg,
            isSeed: i + 1 <= 8,
            isSnp: i + 1 === snpPos,
          }));

          // Energy budget for horizontal bars
          const budgetItems = [
            { label: "crRNA spacer unfolding", value: eb.spacer_unfolding_cost || 0, color: RS.barrier, type: "cost" },
            { label: "dsDNA target unwinding", value: eb.target_unwinding_cost || 0, color: RS.negative, type: "cost" },
            { label: "R-loop hybrid formation", value: eb.hybrid_formation_dg || 0, color: RS.positive, type: "gain" },
          ];
          const netDg = eb.net_dg || 0;

          return (
            <div>
              {/* Header with target info and clear button */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                <div>
                  <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text }}>{thermoTarget}</div>
                  <div style={{ fontSize: "11px", color: RS.muted, fontFamily: MONO }}>{thermoData.crrna_spacer || thermoData.spacer_dna} | PAM: {thermoData.pam_seq}{snpPos ? ` | SNP pos: ${snpPos}` : ""}</div>
                </div>
                <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                  {wp && (
                    <label style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", color: RS.muted, cursor: "pointer" }}>
                      <input type="checkbox" checked={thermoShowWT} onChange={(e) => setThermoShowWT(e.target.checked)} />
                      MUT vs WT
                    </label>
                  )}
                  <button onClick={() => { setThermoData(null); setThermoTarget(""); }} style={{ background: "none", border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "3px 10px", cursor: "pointer", fontSize: "11px", color: RS.muted }}>Clear</button>
                </div>
              </div>

              {/* Metrics row */}
              <div style={{ display: "grid", gridTemplateColumns: mobile ? "repeat(3, 1fr)" : "repeat(7, 1fr)", gap: "8px", marginBottom: "20px" }}>
                {[
                  { label: "Hybrid dG", value: `${(eb.hybrid_formation_dg || 0).toFixed(2)} kcal/mol`, tip: "RNA:DNA hybrid formation energy (what the cumulative profile shows)" },
                  { label: "Net dG (nucleic acid)", value: `${(eb.net_dg || 0).toFixed(2)} kcal/mol`, tip: "hybrid + unwinding + unfolding; excludes protein stabilisation" },
                  { label: "Seed dG (1-8)", value: `${(sc.seed_dg || 0).toFixed(2)} kcal/mol`, tip: "Free energy of seed region hybridization" },
                  { label: "Tm (hybrid)", value: `${(sc.melting_tm || 0).toFixed(1)}\u00B0C`, tip: "Melting temperature of RNA:DNA hybrid" },
                  { label: "Unwinding cost", value: `+${((sc.target_unwinding || 0) + (sc.spacer_unfolding || 0)).toFixed(2)} kcal/mol`, tip: "Total cost: spacer unfolding + target unwinding" },
                  { label: "GC content", value: `${sc.gc_content || 0}%`, tip: "GC percentage of spacer" },
                  { label: "SNP barrier", value: sc.snp_barrier != null ? `+${Number(sc.snp_barrier).toFixed(2)} kcal/mol` : "N/A", tip: "Energy penalty at mismatch position (discrimination basis)" },
                ].map(m => (
                  <div key={m.label} style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "10px 12px" }} title={m.tip}>
                    <div style={{ fontSize: "9px", color: RS.muted, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em" }}>{m.label}</div>
                    <div style={{ fontSize: "13px", fontWeight: 600, color: RS.text, fontFamily: FONT, marginTop: "2px" }}>{m.value}</div>
                  </div>
                ))}
              </div>

              {/* Chart A: Cumulative R-Loop Free Energy Profile */}
              <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "20px 24px", marginBottom: "16px" }}>
                <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text, marginBottom: "4px" }}>Cumulative R-Loop Free Energy Profile</div>
                <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "14px" }}>
                  Cumulative dG along the R-loop from PAM-proximal (position 1) to PAM-distal. Steeper descent = stronger binding. Nearest-neighbor approximations (+-0.5 kcal/mol per step).
                </div>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={cumData} margin={{ top: 10, right: 20, bottom: 25, left: 20 }}>
                    <defs>
                      <linearGradient id="thermoMutFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={RS.mutLine} stopOpacity={0.05} />
                        <stop offset="100%" stopColor={RS.mutLine} stopOpacity={0.01} />
                      </linearGradient>
                      {/* Seed region highlight */}
                      <linearGradient id="seedHighlight" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor={RS.accent} stopOpacity={0.06} />
                        <stop offset="40%" stopColor={RS.accent} stopOpacity={0.06} />
                        <stop offset="40%" stopColor={RS.accent} stopOpacity={0} />
                        <stop offset="100%" stopColor={RS.accent} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="pos" tick={{ fontSize: 11, fill: RS.muted, fontFamily: MONO }} axisLine={{ stroke: RS.border }} tickLine={false} label={{ value: "Position (PAM-proximal \u2192 PAM-distal)", position: "insideBottom", offset: -12, fontSize: 11, fill: RS.muted }} />
                    <YAxis tick={{ fontSize: 11, fill: RS.muted, fontFamily: MONO }} axisLine={false} tickLine={false} label={{ value: "Cumulative dG (kcal/mol)", angle: -90, position: "insideLeft", offset: 5, fontSize: 11, fill: RS.muted }} />
                    <Tooltip contentStyle={rTooltip} formatter={(v, name) => [`${v?.toFixed(2)} kcal/mol`, name === "mutant" ? "Mutant" : "Wildtype"]} labelFormatter={(l) => `Position ${l}`} />
                    {snpPos && <ReferenceLine x={snpPos} stroke={RS.snpLine} strokeDasharray="4 3" strokeWidth={1.5} label={{ value: `SNP (pos ${snpPos})`, position: "insideTopRight", fontSize: 10, fill: RS.snpLine, fontWeight: 600 }} />}
                    <ReferenceLine x={8.5} stroke={RS.accent} strokeDasharray="2 4" strokeWidth={0.5} strokeOpacity={0.4} />
                    <Area type="monotone" dataKey="mutant" stroke={RS.mutLine} strokeWidth={2} fill="url(#thermoMutFill)" name="mutant" isAnimationActive={false} dot={false} />
                    {thermoShowWT && wp && <Line type="monotone" dataKey="wildtype" stroke={RS.wtLine} strokeWidth={1.5} strokeDasharray="6 3" dot={false} name="wildtype" isAnimationActive={false} />}
                  </AreaChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", gap: "16px", marginTop: "6px", fontSize: "10px", color: RS.muted }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                    <div style={{ width: "16px", height: "2px", background: RS.mutLine }} />
                    <span>Mutant (perfect match)</span>
                  </div>
                  {wp && thermoShowWT && <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                    <div style={{ width: "16px", height: "2px", background: RS.wtLine, borderTop: "1px dashed" }} />
                    <span>Wildtype (mismatch at SNP)</span>
                  </div>}
                  <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                    <div style={{ width: "8px", height: "8px", background: RS.seedBg, border: `1px solid ${RS.accent}33`, borderRadius: "2px" }} />
                    <span>Seed region (1-8)</span>
                  </div>
                </div>
                {/* Discrimination annotation */}
                {wp && thermoShowWT && sc.snp_barrier != null && (
                  <div style={{ marginTop: "12px", padding: "10px 14px", background: RS.seedBg, borderRadius: "4px", fontSize: "11px", color: RS.text, lineHeight: 1.6 }}>
                    <strong>Thermodynamic discrimination:</strong> The mismatch at position {snpPos} creates a +{Number(sc.snp_barrier).toFixed(2)} kcal/mol barrier in the wildtype R-loop.
                    {snpPos <= 4 ? " At this seed position, the barrier occurs early in R-loop propagation, likely causing complete R-loop collapse (Strohkendl et al., 2018)." : snpPos <= 8 ? " Within the seed region, this barrier significantly impedes R-loop extension." : " At this PAM-distal position, the barrier occurs after substantial R-loop formation and may be partially tolerated."}
                  </div>
                )}
              </div>

              {/* Chart B: Per-Position Energy Contribution */}
              <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "20px 24px", marginBottom: "16px" }}>
                <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text, marginBottom: "4px" }}>Per-Position Energy Contribution</div>
                <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "14px" }}>
                  dG contribution per dinucleotide step. GC-rich positions contribute more negative dG (taller bars downward). The mismatch position shows a positive bar (destabilising).
                </div>
                <ResponsiveContainer width="100%" height={150}>
                  <BarChart data={barData} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
                    <XAxis dataKey="pos" tick={{ fontSize: 11, fill: RS.muted, fontFamily: MONO }} axisLine={{ stroke: RS.border }} tickLine={false} label={{ value: "Position", position: "insideBottom", offset: -10, fontSize: 11, fill: RS.muted }} />
                    <YAxis tick={{ fontSize: 11, fill: RS.muted, fontFamily: MONO }} axisLine={false} tickLine={false} label={{ value: "dG (kcal/mol)", angle: -90, position: "insideLeft", offset: 5, fontSize: 10, fill: RS.muted }} />
                    <Tooltip contentStyle={rTooltip} formatter={(v) => [`${v?.toFixed(2)} kcal/mol`, "dG step"]} labelFormatter={(l) => `Position ${l}`} />
                    <ReferenceLine y={0} stroke={RS.border} strokeWidth={1} />
                    <Bar dataKey="dg" isAnimationActive={false} radius={[2, 2, 0, 0]}>
                      {barData.map((entry, i) => (
                        <Cell key={i} fill={entry.isSnp ? RS.snpLine : entry.isSeed ? RS.accent : RS.muted} fillOpacity={entry.isSnp ? 0.9 : entry.isSeed ? 0.7 : 0.4} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Chart C: Unwinding Cost Decomposition */}
              <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "20px 24px" }}>
                <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text, marginBottom: "4px" }}>Unwinding Cost Decomposition</div>
                <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "14px" }}>
                  Energy budget following Zhang et al. (NAR 2024). Net dG correlates linearly with trans-cleavage rate.
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                  {budgetItems.map(item => {
                    const maxAbs = Math.max(...budgetItems.map(b => Math.abs(b.value)), 1);
                    const pct = Math.min(Math.abs(item.value) / maxAbs * 100, 100);
                    return (
                      <div key={item.label} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                        <div style={{ width: "180px", fontSize: "12px", color: RS.muted, textAlign: "right", flexShrink: 0 }}>{item.label}</div>
                        <div style={{ flex: 1, height: "24px", background: "#f5f5f5", borderRadius: "4px", position: "relative", overflow: "hidden" }}>
                          <div style={{
                            position: "absolute", top: 0, left: item.type === "gain" ? 0 : undefined, right: item.type === "cost" ? 0 : undefined,
                            width: `${pct}%`, height: "100%", background: item.color, opacity: 0.2, borderRadius: "4px",
                          }} />
                          <div style={{ position: "absolute", top: 0, left: item.type === "gain" ? 0 : undefined, right: item.type === "cost" ? 0 : undefined, width: `${pct}%`, height: "100%", display: "flex", alignItems: "center", justifyContent: item.type === "gain" ? "flex-end" : "flex-start", padding: "0 8px" }}>
                            <span style={{ fontSize: "12px", fontWeight: 600, fontFamily: FONT, color: item.color }}>{item.value > 0 ? "+" : ""}{item.value.toFixed(2)}</span>
                          </div>
                        </div>
                        <div style={{ width: "60px", fontSize: "10px", color: RS.muted, flexShrink: 0 }}>{item.type === "cost" ? "cost" : "gain"}</div>
                      </div>
                    );
                  })}
                  {/* Net line */}
                  <div style={{ display: "flex", alignItems: "center", gap: "12px", borderTop: `1px solid ${RS.border}`, paddingTop: "8px", marginTop: "4px" }}>
                    <div style={{ width: "180px", fontSize: "12px", fontWeight: 600, color: RS.text, textAlign: "right" }}>Net dG (nucleic acid)</div>
                    <div style={{ flex: 1 }}>
                      <span style={{ fontSize: "14px", fontWeight: 600, fontFamily: FONT, color: netDg < 0 ? RS.positive : RS.negative }}>{netDg.toFixed(2)} kcal/mol</span>
                      <span style={{ fontSize: "11px", color: RS.muted, marginLeft: "8px" }}>{netDg < -15 ? "strongly favourable" : netDg < -5 ? "moderately favourable" : netDg < 0 ? "weakly favourable" : "unfavourable without protein"}</span>
                    </div>
                    <div style={{ width: "60px" }} />
                  </div>
                </div>
                {/* Protein stabilisation note */}
                {netDg >= 0 && (
                  <div style={{ marginTop: "12px", padding: "10px 14px", background: RS.seedBg, borderRadius: "4px", fontSize: "11px", color: RS.text, lineHeight: 1.6 }}>
                    <strong>Note:</strong> The positive net dG indicates that nucleic acid thermodynamics alone do not favour R-loop formation at this target. Cas12a protein provides 10{"\u2013"}30 kcal/mol of additional stabilisation through PAM recognition, REC domain contacts, and conformational coupling (Strohkendl et al. 2024; CRISPRzip, Offerhaus et al. 2025). The hybrid dG ({(eb.hybrid_formation_dg || 0).toFixed(2)} kcal/mol) remains the best available predictor of relative guide performance across candidates, as the protein contribution is approximately constant.
                  </div>
                )}
                {/* References */}
                <div style={{ marginTop: "12px", fontSize: "10px", color: "#a3a3a3", fontStyle: "italic" }}>
                  {(thermoData.references || []).join(" | ")}
                </div>
              </div>
            </div>
          );
        })()}
      </CollapsibleSection>

      {/* ═══ Section 3: Ablation Tracker ═══ */}
      <CollapsibleSection title="Ablation Tracker" defaultOpen={false}>
        {ablation.length === 0 ? (
          <div style={{ fontSize: "12px", color: RS.muted }}>No ablation data available.</div>
        ) : (() => {
          // Build scatter data — only rows with both kim and ed rho
          const scatterPts = ablation.filter(r => r.kim_rho != null && r.ed_rho != null);
          const allPts = ablation.map(r => ({ ...r, ed_rho: r.ed_rho ?? 0 }));
          const productionRow = ablation.find(r => r.notes && r.notes.toLowerCase().includes("production"));

          // Simple Pareto frontier (non-dominated points)
          const pareto = [];
          for (const p of scatterPts) {
            const dominated = scatterPts.some(q => q.kim_rho >= p.kim_rho && q.ed_rho >= p.ed_rho && (q.kim_rho > p.kim_rho || q.ed_rho > p.ed_rho));
            if (!dominated) pareto.push(p);
          }
          pareto.sort((a, b) => a.kim_rho - b.kim_rho);

          const DOT_COLORS = ["#2563eb", "#7c3aed", "#d97706", "#dc2626", "#16a34a", "#0891b2", "#e11d48"];

          return (
            <div>
              {/* Scatter plot */}
              {!mobile && scatterPts.length > 1 && (
                <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "20px 24px", marginBottom: "16px" }}>
                  <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text, marginBottom: "4px" }}>Cis vs Trans Cleavage Correlation</div>
                  <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "14px" }}>
                    Each model plotted by Kim 2018 rho (cis-cleavage) vs EasyDesign rho (trans-cleavage). Top-right = best all-rounder. Star = production checkpoint.
                  </div>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
                      <XAxis type="number" dataKey="kim_rho" name="Kim rho" domain={[0.38, 0.55]} tick={{ fontSize: 11, fontFamily: MONO, fill: RS.muted }} axisLine={{ stroke: RS.border }} tickLine={false} label={{ value: "Kim rho (cis)", position: "insideBottom", offset: -16, fontSize: 11, fill: RS.muted }} />
                      <YAxis type="number" dataKey="ed_rho" name="ED rho" domain={[-0.05, 0.65]} tick={{ fontSize: 11, fontFamily: MONO, fill: RS.muted }} axisLine={false} tickLine={false} label={{ value: "ED rho (trans)", angle: -90, position: "insideLeft", offset: 10, fontSize: 11, fill: RS.muted }} />
                      <Tooltip contentStyle={rTooltip} content={({ payload }) => {
                        if (!payload?.length) return null;
                        const d = payload[0]?.payload;
                        if (!d) return null;
                        return (
                          <div style={{ ...rTooltip, padding: "10px 14px" }}>
                            <div style={{ fontWeight: 600, fontSize: "12px", marginBottom: "4px" }}>{d.label}</div>
                            <div>Kim rho: {d.kim_rho?.toFixed(3)}</div>
                            <div>ED rho: {d.ed_rho?.toFixed(3)}</div>
                            <div style={{ fontSize: "10px", color: "#a3a3a3", marginTop: "4px" }}>{d.notes}</div>
                          </div>
                        );
                      }} />
                      {/* Pareto frontier line */}
                      {pareto.length > 1 && (
                        <Line data={pareto} dataKey="ed_rho" stroke={RS.accent} strokeWidth={1} strokeDasharray="6 4" dot={false} isAnimationActive={false} type="monotone" />
                      )}
                      <Scatter data={scatterPts} isAnimationActive={false}>
                        {scatterPts.map((entry, i) => {
                          const isProd = productionRow && entry.label === productionRow.label;
                          return <Cell key={i} fill={DOT_COLORS[i % DOT_COLORS.length]} r={isProd ? 8 : 5} stroke={isProd ? RS.accent : "#fff"} strokeWidth={isProd ? 2.5 : 1.5} />;
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                  {/* Labels for each point */}
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", marginTop: "8px" }}>
                    {scatterPts.map((p, i) => {
                      const isProd = productionRow && p.label === productionRow.label;
                      return (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                          <div style={{ width: isProd ? 10 : 8, height: isProd ? 10 : 8, borderRadius: "50%", background: DOT_COLORS[i % DOT_COLORS.length], border: isProd ? `2px solid ${RS.accent}` : "none" }} />
                          <span style={{ fontSize: "10px", color: RS.muted, fontWeight: isProd ? 700 : 400 }}>{p.label}{isProd ? " *" : ""}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Insight box */}
              <div style={{ padding: "12px 16px", background: RS.seedBg, borderRadius: "4px", fontSize: "12px", color: RS.text, lineHeight: 1.7, marginBottom: "16px" }}>
                <strong>Key finding:</strong> Models optimised for cis-cleavage gene editing (Kim 2018 benchmark) show near-zero predictive value for diagnostic trans-cleavage (rho = 0.04). The production checkpoint (multi-dataset, no domain adversarial) achieves rho = 0.55 on trans-cleavage while retaining rho = 0.49 on cis-cleavage — the best all-rounder across both benchmarks. Domain-adversarial training (Ganin et al., JMLR 2016) is counter-productive: forcing domain invariance destroys trans-cleavage-specific signal.
              </div>

              {/* Table */}
              <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", overflow: "hidden" }}>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <th style={{ ...thStyle, textAlign: "left" }}>Model</th>
                        <th style={{ ...thStyle, textAlign: "left" }}>Features</th>
                        <th style={{ ...thStyle, textAlign: "right" }}>Kim rho</th>
                        <th style={{ ...thStyle, textAlign: "right" }}>ED rho</th>
                        <th style={{ ...thStyle, textAlign: "left" }}>Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ablation.map((row, i) => {
                        const isProd = row.notes && row.notes.toLowerCase().includes("production");
                        return (
                          <tr key={i} style={{ borderBottom: `1px solid ${RS.border}`, background: isProd ? RS.seedBg : "transparent" }}>
                            <td style={{ ...tdStyle, fontWeight: 600 }}>
                              {row.label}
                              {isProd && <span style={{ marginLeft: "6px", fontSize: "9px", fontWeight: 600, color: RS.accent, background: `${RS.accent}15`, padding: "2px 6px", borderRadius: "3px", fontFamily: FONT }}>PRODUCTION</span>}
                            </td>
                            <td style={{ ...tdStyle, color: RS.muted, fontFamily: FONT, fontSize: "11px" }}>{row.features}</td>
                            <td style={{ ...tdStyle, textAlign: "right" }}>{row.kim_rho?.toFixed(3) ?? "—"}</td>
                            <td style={{ ...tdStyle, textAlign: "right", color: row.ed_rho ? RS.text : "#d4d4d4" }}>{row.ed_rho?.toFixed(3) ?? "—"}</td>
                            <td style={{ ...tdStyle, color: RS.muted, fontFamily: FONT, fontSize: "11px" }}>{row.notes}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          );
        })()}
      </CollapsibleSection>

      {/* ═══ Section 4: Feature Importance Analysis ═══ */}
      <CollapsibleSection title="Feature Importance Analysis" defaultOpen={false}>
        <div style={{ fontSize: "12px", color: RS.muted, marginBottom: "16px", lineHeight: 1.6 }}>
          Approximate feature contributions to the scoring model. Position importance reflects seed-region weighting (positions 1-4 most critical for R-loop nucleation). The waterfall shows additive contributions from the heuristic scorer components.
        </div>

        {/* 4A: Position importance heatmap */}
        <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "20px 24px", marginBottom: "16px" }}>
          <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text, marginBottom: "4px" }}>Position Importance</div>
          <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "14px" }}>
            Learned importance weight per spacer position. Dark = high importance. Seed region (1-8) drives R-loop nucleation.
          </div>
          <div style={{ display: "flex", gap: "2px", alignItems: "flex-end" }}>
            {POS_WEIGHTS.map((w, i) => {
              const h = Math.max(w * 50, 4);
              const opacity = 0.15 + w * 0.85;
              return (
                <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", flex: 1 }} title={`Position ${i + 1}: weight ${w.toFixed(2)}`}>
                  <div style={{ width: "100%", height: `${h}px`, background: RS.accent, opacity, borderRadius: "2px 2px 0 0", minWidth: "12px" }} />
                  <div style={{ fontSize: "9px", color: i < 8 ? RS.accent : RS.muted, fontFamily: MONO, marginTop: "3px", fontWeight: i < 4 ? 700 : 400 }}>{i + 1}</div>
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: "12px", marginTop: "10px", fontSize: "10px", color: RS.muted }}>
            <span style={{ fontWeight: 600, color: RS.accent }}>Seed (1-8)</span>
            <span>Mid (9-14)</span>
            <span>PAM-distal (15-20)</span>
          </div>
        </div>

        {/* 4B: Feature contribution waterfall */}
        <div style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "20px 24px" }}>
          <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text, marginBottom: "4px" }}>Heuristic Feature Breakdown</div>
          <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "14px" }}>
            Additive contribution of each heuristic scoring component. Weights represent relative importance in the composite score.
          </div>
          {(() => {
            // Use thermo data if available, otherwise show generic weights
            const baseScore = 0.50;
            const features = HEURISTIC_WEIGHTS.map(f => {
              let contrib = 0;
              if (thermoData && thermoData.scalars) {
                const sc = thermoData.scalars;
                if (f.key === "seed") contrib = sc.seed_dg ? Math.min(Math.abs(sc.seed_dg) / 30, 0.2) : 0.08;
                else if (f.key === "gc") contrib = sc.gc_content ? (sc.gc_content > 40 && sc.gc_content < 70 ? 0.06 : -0.02) : 0.04;
                else if (f.key === "ss") contrib = sc.spacer_unfolding ? -(sc.spacer_unfolding / 20) : -0.02;
                else if (f.key === "ot") contrib = 0.08;
                else if (f.key === "thermo") contrib = sc.net_dg ? Math.min(Math.abs(sc.net_dg) / 100, 0.1) : 0.03;
              } else {
                contrib = f.weight * 0.3 * (f.key === "ss" ? -1 : 1);
              }
              return { ...f, contrib: +contrib.toFixed(3) };
            });
            let running = baseScore;
            const waterfall = [{ name: "Base score", value: baseScore, running: baseScore, isBase: true }];
            for (const f of features) {
              running += f.contrib;
              waterfall.push({ name: f.name, value: f.contrib, running: +running.toFixed(3), isBase: false });
            }
            waterfall.push({ name: "Final score", value: running, running: +running.toFixed(3), isBase: true });

            return (
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                {waterfall.map((item, i) => {
                  const maxVal = Math.max(...waterfall.map(w => Math.abs(w.value)));
                  const pct = Math.min(Math.abs(item.value) / 1.0 * 100, 100);
                  const isPositive = item.value >= 0;
                  return (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                      <div style={{ width: "140px", fontSize: "12px", color: item.isBase ? RS.text : RS.muted, fontWeight: item.isBase ? 700 : 400, textAlign: "right", flexShrink: 0 }}>{item.name}</div>
                      <div style={{ flex: 1, height: "22px", position: "relative" }}>
                        {item.isBase ? (
                          <div style={{ position: "absolute", left: 0, top: 0, width: `${item.value * 100}%`, height: "100%", background: RS.accent, opacity: 0.15, borderRadius: "3px" }} />
                        ) : (
                          <div style={{
                            position: "absolute",
                            left: isPositive ? `${(item.running - item.value) * 100}%` : `${item.running * 100}%`,
                            top: 0,
                            width: `${Math.abs(item.value) * 100}%`,
                            height: "100%",
                            background: isPositive ? RS.positive : RS.negative,
                            opacity: 0.25,
                            borderRadius: "3px",
                          }} />
                        )}
                      </div>
                      <div style={{ width: "80px", fontSize: "12px", fontFamily: FONT, fontWeight: item.isBase ? 800 : 600, color: item.isBase ? RS.text : (isPositive ? RS.positive : RS.negative), textAlign: "right", flexShrink: 0 }}>
                        {item.isBase ? item.value.toFixed(3) : `${isPositive ? "+" : ""}${item.value.toFixed(3)}`}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </div>
      </CollapsibleSection>

      {/* ═══ Section 5: Platform Roadmap ═══ */}
      <CollapsibleSection title="Platform Roadmap" defaultOpen={false}>
        <p style={{ fontSize: "12px", color: RS.muted, marginBottom: "16px", lineHeight: 1.7 }}>
          Planned capabilities aligned with development milestones. Each module builds on experimental data collected during validation studies.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "14px" }}>
          {[
            {
              icon: <RefreshCw size={18} />,
              title: "Active Learning Loop",
              desc: "Feed experimental measurements back into Compass-ML. Fluorescence or electrochemical data recalibrates activity predictions, closing the gap between in silico and platform-specific signal.",
              milestone: "Phase 1 — Experimental validation",
              ready: true,
            },
            {
              icon: <Zap size={18} />,
              title: "Electrochemical Transfer Function",
              desc: "Model the relationship between solution-phase trans-cleavage and surface-tethered MB reporter degradation on LIG electrodes.",
              milestone: "Phase 2 — Electrode characterisation",
              ready: false,
            },
            {
              icon: <Grid3x3 size={18} />,
              title: "Spatial Multiplexing Optimiser",
              desc: "Assign targets to electrode pads on the spatially addressed array, minimising electrochemical crosstalk. Replaces solution-phase M8 when using in-situ complexation.",
              milestone: "Phase 2–3 — Array fabrication",
              ready: false,
            },
            {
              icon: <FlaskConical size={18} />,
              title: "Nuclease Adaptation Engine",
              desc: "Swap Cas12a variants via transfer learning. Freeze sequence encoders, fine-tune RLPA attention and output heads on variant-specific data.",
              milestone: "Phase 3 — Variant screening",
              ready: false,
            },
          ].map((card, i) => (
            <div key={i} style={{ background: RS.cardBg, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "20px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "10px" }}>
                <div style={{ color: card.ready ? RS.positive : RS.muted }}>{card.icon}</div>
                <div style={{ fontSize: "14px", fontWeight: 600, color: RS.text, fontFamily: HEADING }}>{card.title}</div>
              </div>
              <p style={{ fontSize: "12px", color: RS.muted, lineHeight: 1.7, margin: "0 0 12px" }}>{card.desc}</p>
              <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "8px" }}>
                <strong>Milestone:</strong> {card.milestone}
              </div>
              <span style={{
                display: "inline-block", fontSize: "10px", fontWeight: 600, padding: "3px 10px", borderRadius: "4px",
                background: card.ready ? `${RS.positive}18` : `${RS.muted}18`,
                color: card.ready ? RS.positive : RS.muted,
              }}>
                {card.ready ? "Ready to start" : "Planned"}
              </span>
            </div>
          ))}
        </div>
      </CollapsibleSection>

      {/* ═══ Section 6: Nuclease Comparison ═══ */}
      <CollapsibleSection title="Nuclease Variant Comparison" defaultOpen={false}>
        <p style={{ fontSize: "12px", color: RS.muted, marginBottom: "16px", lineHeight: 1.7, maxWidth: "800px" }}>
          Compare Cas12a variants on the current 14-target MDR-TB panel. PAM coverage is computed by running COMPASS's M2 PAM scanner against the
          H37Rv genome with each variant's published PAM set. Scoring and discrimination columns show whether Compass-ML has been trained on data
          for that variant — "Retraining required" indicates the scoring model needs variant-specific experimental data before predictions are valid.
        </p>

        {/* Load coverage button */}
        {!nucleaseCoverage && (
          <button
            onClick={() => {
              setNucleaseLoading(true);
              getNucleaseComparison().then(({ data }) => {
                if (data) {
                  setNucleaseProfiles(Object.values(data.profiles));
                  setNucleaseCoverage(data.coverage);
                }
                setNucleaseLoading(false);
              });
            }}
            disabled={nucleaseLoading}
            style={{ ...btnStyle, marginBottom: "16px", opacity: nucleaseLoading ? 0.6 : 1 }}
          >
            {nucleaseLoading ? "Scanning PAMs..." : "Run PAM Coverage Comparison"}
          </button>
        )}

        {/* Profile summary table (always shown if profiles loaded) */}
        {nucleaseProfiles && (
          <div style={{ overflowX: "auto", marginBottom: "20px" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", minWidth: 900 }}>
              <thead>
                <tr>
                  {["Variant", "Mutations", "PAM Set", "Targets", "PAM Deserts", "Scoring", "Disc."].map(h => (
                    <th key={h} style={thStyle}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {nucleaseProfiles.map(p => {
                  const cov = nucleaseCoverage?.[p.id];
                  const isExpanded = expandedVariant === p.id;
                  return (
                    <React.Fragment key={p.id}>
                      <tr
                        onClick={() => setExpandedVariant(isExpanded ? null : p.id)}
                        style={{ cursor: "pointer", borderBottom: `1px solid ${RS.border}`, background: isExpanded ? RS.seedBg : "transparent" }}
                      >
                        <td style={{ ...tdStyle, fontWeight: 600 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                            {p.display_name}
                          </div>
                        </td>
                        <td style={{ ...tdStyle, fontSize: "11px" }}>
                          {p.mutations ? p.mutations.join("/") : "\u2014"}
                        </td>
                        <td style={{ ...tdStyle, fontSize: "11px" }}>
                          {p.pam_canonical.join(", ")}
                          {p.pam_total_count > p.pam_canonical.length && (
                            <span style={{ color: RS.accent, marginLeft: "4px" }}>+{p.pam_total_count - p.pam_canonical.length}</span>
                          )}
                        </td>
                        <td style={tdStyle}>
                          {cov && !cov.error ? (
                            <span style={{ fontWeight: 600, color: cov.targets_with_pam === cov.targets_total ? RS.positive : RS.barrier }}>
                              {cov.targets_with_pam}/{cov.targets_total}
                            </span>
                          ) : cov?.error ? (
                            <span style={{ color: RS.muted, fontSize: "10px" }}>{cov.error.split(" — ")[0]}</span>
                          ) : (
                            <span style={{ color: RS.muted }}>—</span>
                          )}
                        </td>
                        <td style={{ ...tdStyle, fontSize: "10px" }}>
                          {cov && !cov.error ? (
                            cov.pam_desert_targets?.length > 0 ? (
                              <span style={{ color: RS.negative }}>{cov.pam_desert_targets.join(", ")}</span>
                            ) : (
                              <span style={{ color: RS.positive }}>None</span>
                            )
                          ) : "—"}
                        </td>
                        <td style={tdStyle}>
                          <span style={{
                            display: "inline-block", fontSize: "10px", fontWeight: 600, padding: "2px 8px", borderRadius: "4px",
                            background: p.scoring_trained ? `${RS.positive}18` : `${RS.barrier}18`,
                            color: p.scoring_trained ? RS.positive : RS.barrier,
                          }}>
                            {p.scoring_trained ? "Trained" : "Retraining required"}
                          </span>
                        </td>
                        <td style={tdStyle}>
                          <span style={{
                            display: "inline-block", fontSize: "10px", fontWeight: 600, padding: "2px 8px", borderRadius: "4px",
                            background: p.scoring_trained ? `${RS.positive}18` : `${RS.barrier}18`,
                            color: p.scoring_trained ? RS.positive : RS.barrier,
                          }}>
                            {p.scoring_trained ? "Trained" : "Retraining required"}
                          </span>
                        </td>
                      </tr>

                      {/* Expanded variant detail card */}
                      {isExpanded && (
                        <tr>
                          <td colSpan={7} style={{ padding: 0 }}>
                            <div style={{ padding: "16px 20px", background: RS.seedBg, borderBottom: `1px solid ${RS.border}` }}>
                              <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: "16px" }}>
                                {/* Left: Profile details */}
                                <div>
                                  <div style={{ fontSize: "13px", fontWeight: 600, color: RS.text, marginBottom: "8px", fontFamily: HEADING }}>
                                    {p.display_name}
                                  </div>
                                  <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "6px" }}>
                                    <strong>Organism:</strong> <em>{p.organism}</em>
                                  </div>
                                  {p.mutations && (
                                    <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "6px" }}>
                                      <strong>Mutations:</strong> {p.mutations.join(", ")}
                                    </div>
                                  )}
                                  <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "6px" }}>
                                    <strong>PAM:</strong> {p.pam_note}
                                  </div>
                                  <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "6px" }}>
                                    <strong>Seed:</strong> {p.seed_note}
                                  </div>
                                  <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "6px" }}>
                                    <strong>SNV discrimination:</strong> {p.snv_discrimination}
                                  </div>
                                  <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "6px" }}>
                                    <strong>Divalent cation:</strong> {p.divalent_cation}
                                  </div>
                                  <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "6px" }}>
                                    <strong>Temperature:</strong> {p.optimal_temp}\u00b0C optimal
                                    {p.temperature?.active_range_C && ` (active ${p.temperature.active_range_C[0]}\u2013${p.temperature.active_range_C[1]}\u00b0C)`}
                                  </div>
                                  {p.kinetics?.note && (
                                    <div style={{ fontSize: "11px", color: RS.muted, marginBottom: "6px" }}>
                                      <strong>Kinetics:</strong> {p.kinetics.note}
                                    </div>
                                  )}
                                </div>

                                {/* Right: Seed region heatmap */}
                                <div>
                                  <div style={{ fontSize: "11px", fontWeight: 600, color: RS.muted, marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                                    Seed Region Profile (positions 1\u201320)
                                  </div>
                                  <div style={{ display: "flex", gap: "2px", flexWrap: "wrap" }}>
                                    {Array.from({ length: 20 }, (_, i) => {
                                      const pos = i + 1;
                                      const isCritical = p.seed_positions.includes(pos);
                                      const isTolerant = p.tolerant_positions.includes(pos);
                                      const intensity = isCritical ? 1.0 : isTolerant ? 0.15 : 0.45;
                                      return (
                                        <div key={pos} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "2px" }}>
                                          <div style={{
                                            width: "24px", height: "24px", borderRadius: "3px",
                                            background: `rgba(37, 99, 235, ${intensity})`,
                                            display: "flex", alignItems: "center", justifyContent: "center",
                                            fontSize: "9px", fontFamily: MONO, fontWeight: 600,
                                            color: intensity > 0.5 ? "#fff" : RS.text,
                                          }}>
                                            {pos}
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                  <div style={{ display: "flex", gap: "12px", marginTop: "8px", fontSize: "10px", color: RS.muted }}>
                                    <span><span style={{ display: "inline-block", width: "10px", height: "10px", borderRadius: "2px", background: "rgba(37,99,235,1.0)", verticalAlign: "middle", marginRight: "4px" }} />Critical</span>
                                    <span><span style={{ display: "inline-block", width: "10px", height: "10px", borderRadius: "2px", background: "rgba(37,99,235,0.45)", verticalAlign: "middle", marginRight: "4px" }} />Intermediate</span>
                                    <span><span style={{ display: "inline-block", width: "10px", height: "10px", borderRadius: "2px", background: "rgba(37,99,235,0.15)", verticalAlign: "middle", marginRight: "4px" }} />Tolerant</span>
                                  </div>
                                </div>
                              </div>

                              {/* References */}
                              <div style={{ marginTop: "12px", borderTop: `1px solid ${RS.border}`, paddingTop: "10px" }}>
                                <div style={{ fontSize: "10px", fontWeight: 600, color: RS.muted, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "6px" }}>References</div>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                                  {p.references.map((ref, ri) => (
                                    <a
                                      key={ri}
                                      href={ref.doi ? `https://doi.org/${ref.doi}` : "#"}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      style={{
                                        fontSize: "10px", padding: "3px 8px", borderRadius: "4px",
                                        background: `${RS.accent}10`, color: RS.accent, textDecoration: "none",
                                        border: `1px solid ${RS.accent}30`,
                                      }}
                                      title={ref.data}
                                    >
                                      {ref.short} <ExternalLink size={8} style={{ verticalAlign: "middle" }} />
                                    </a>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Per-target coverage detail (shown after comparison run) */}
        {nucleaseCoverage && (() => {
          const variants = Object.values(nucleaseCoverage).filter(v => !v.error && v.per_target);
          if (variants.length === 0) return null;
          const allTargets = variants[0].per_target.map(t => t.target);
          return (
            <div style={{ overflowX: "auto", marginBottom: "20px" }}>
              <div style={{ fontSize: "12px", fontWeight: 600, color: RS.text, marginBottom: "8px", fontFamily: HEADING }}>Per-Target PAM Availability</div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px", minWidth: 700 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Target</th>
                    <th style={thStyle}>Drug</th>
                    {variants.map(v => <th key={v.variant_id} style={thStyle}>{v.display_name}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {allTargets.map((targetName, ti) => (
                    <tr key={targetName} style={{ borderBottom: `1px solid ${RS.border}` }}>
                      <td style={{ ...tdStyle, fontWeight: 600, fontSize: "11px" }}>{targetName}</td>
                      <td style={{ ...tdStyle, fontSize: "10px" }}>{variants[0].per_target[ti]?.drug || ""}</td>
                      {variants.map(v => {
                        const t = v.per_target[ti];
                        return (
                          <td key={v.variant_id} style={tdStyle}>
                            {t ? (
                              <span style={{
                                fontWeight: 600,
                                color: t.n_direct > 0 ? RS.positive : t.n_total > 0 ? RS.barrier : RS.negative,
                              }}>
                                {t.n_direct > 0 ? `${t.n_direct} direct` : t.n_total > 0 ? "proximity" : "none"}
                              </span>
                            ) : "\u2014"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })()}

        {/* Scientific insight box */}
        <div style={{ background: `${RS.barrier}10`, border: `1px solid ${RS.barrier}30`, borderRadius: "4px", padding: "12px 16px", marginTop: "8px" }}>
          <div style={{ display: "flex", gap: "10px", alignItems: "flex-start" }}>
            <AlertTriangle size={16} color={RS.barrier} style={{ flexShrink: 0, marginTop: 2 }} />
            <div>
              <div style={{ fontSize: "12px", fontWeight: 600, color: RS.text, marginBottom: "4px" }}>Mg{"\u00b2\u207a"} Concentration Inverts Seed Specificity</div>
              <p style={{ fontSize: "11px", color: RS.muted, lineHeight: 1.7, margin: 0 }}>
                Nguyen et al. (2024, <em>NAR</em> 52:9343) showed that at low Mg{"\u00b2\u207a"} ({"\u2264"}1 mM), seed mismatches become <em>more</em> tolerated
                while PAM-distal mismatches become <em>less</em> tolerated — partially inverting the canonical specificity pattern.
                The standard 10 mM MgCl{"\u2082"} used in most published assays may not reflect diagnostic buffer conditions.
                Buffer Mg{"\u00b2\u207a"} optimisation is a critical experimental variable for achieving reliable SNV discrimination.
              </p>
              <a
                href="https://doi.org/10.1093/nar/gkae613"
                target="_blank"
                rel="noopener noreferrer"
                style={{ fontSize: "10px", color: RS.accent, textDecoration: "none", marginTop: "6px", display: "inline-flex", alignItems: "center", gap: "4px" }}
              >
                DOI: 10.1093/nar/gkae613 <ExternalLink size={9} />
              </a>
            </div>
          </div>
        </div>

        {/* CasDx1 note */}
        <div style={{ background: `${RS.muted}08`, border: `1px solid ${RS.border}`, borderRadius: "4px", padding: "12px 16px", marginTop: "10px" }}>
          <div style={{ display: "flex", gap: "10px", alignItems: "flex-start" }}>
            <Lock size={14} color={RS.muted} style={{ flexShrink: 0, marginTop: 2 }} />
            <p style={{ fontSize: "11px", color: RS.muted, lineHeight: 1.7, margin: 0 }}>
              <strong>CasDx1</strong> (Mammoth Biosciences) showed superior SNV discrimination in SARS-CoV-2 detection
              (Fasching et al. 2022, <em>J Clin Microbiol</em>) but its PAM specificity and biochemical parameters are proprietary
              and cannot be configured. 97.3% SNP concordance on 261 clinical samples.
            </p>
          </div>
        </div>
      </CollapsibleSection>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   COMPASS PLATFORM — Root component
   ═══════════════════════════════════════════════════════════════════ */
const COMPASSPlatform = () => {
  const mobile = useIsMobile();
  const [page, setPage] = useState("home");
  const [connected, setConnected] = useState(false);
  const [pipelineJobId, setPipelineJobId] = useState(null);
  const [resultsJobId, setResultsJobId] = useState(null);
  const [resultsScorer, setResultsScorer] = useState("heuristic");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  /* Check API connectivity */
  useEffect(() => {
    healthCheck().then(({ data, error }) => {
      setConnected(!error && !!data);
    });
    const iv = setInterval(() => {
      healthCheck().then(({ data, error }) => {
        setConnected(!error && !!data);
      });
    }, 30000);
    return () => clearInterval(iv);
  }, []);

  const goTo = (pg, opts) => {
    if (opts?.jobId && pg === "pipeline") setPipelineJobId(opts.jobId);
    if (opts?.jobId && pg === "results") setResultsJobId(opts.jobId);
    if (opts?.scorer && pg === "results") setResultsScorer(opts.scorer);
    setPage(pg);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", fontFamily: FONT, color: T.text, background: T.bgSub }}>
      {/* Mobile top bar */}
      {mobile && (
        <header style={{
          display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px",
          background: T.sidebar, borderBottom: `1px solid ${T.border}`, flexShrink: 0,
        }}>
          <button onClick={() => setSidebarOpen(true)} style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", display: "flex" }}>
            <Menu size={22} color={T.text} />
          </button>
          <img src="/compass-logo.png" alt="COMPASS" style={{ height: "18px", objectFit: "contain" }} />
          {!connected && (
            <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "4px", fontSize: "10px", color: T.danger, fontWeight: 600 }}>
              <WifiOff size={10} /> API disconnected
            </div>
          )}
        </header>
      )}

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <Sidebar page={page} setPage={setPage} connected={connected} mobileOpen={sidebarOpen} setMobileOpen={setSidebarOpen} collapsed={sidebarCollapsed} setCollapsed={setSidebarCollapsed} />
        <main style={{ flex: 1, overflow: "auto" }}>
          <div key={page} style={{ animation: "pageIn 0.15s ease-out" }}>
            {page === "home" && <HomePage goTo={goTo} connected={connected} />}
            {page === "methods" && <MethodsPage />}
            {page === "pipeline" && <PipelinePage jobId={pipelineJobId} connected={connected} goTo={goTo} />}
            {page === "results" && <ResultsPage connected={connected} jobId={resultsJobId} scorer={resultsScorer} goTo={goTo} />}
            {page === "panels" && <PanelsPage connected={connected} />}
            {page === "mutations" && <MutationsPage />}
            {page === "scoring" && <ScoringPage connected={connected} />}
            {page === "research" && <ResearchPage connected={connected} />}
          </div>
        </main>
      </div>

      {/* Global styles */}
      <style>{`
        /* font loaded via index.html preload */
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes toastIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pageIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes pulseDot { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }
        @keyframes stepSlideIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes stepSwipeUp { from { opacity: 0; } to { opacity: 1; } }
        @keyframes substepSwipe { from { opacity: 0; } to { opacity: 1; } }
        @keyframes statReveal { from { opacity: 0; } to { opacity: 1; } }
        @keyframes subtlePulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        @keyframes indeterminateProgress { 0% { width: 0%; margin-left: 0%; } 50% { width: 60%; margin-left: 20%; } 100% { width: 0%; margin-left: 100%; } }
        @keyframes statFadeIn { from { opacity: 0; } to { opacity: 1; } }
        .spin { animation: spin 1s linear infinite; }
        *, *::before, *::after { box-sizing: border-box; }
        body { margin: 0; padding: 0; overflow: hidden; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: ${T.textTer}; }
        input, textarea, button, select { box-sizing: border-box; }
      `}</style>
    </div>
  );
};

const COMPASSApp = () => (
  <ToastProvider>
    <COMPASSPlatform />
  </ToastProvider>
);

export default COMPASSApp;
