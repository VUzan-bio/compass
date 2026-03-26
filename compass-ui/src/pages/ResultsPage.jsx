import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { createPortal } from "react-dom";
import {
  Activity, AlertTriangle, BarChart3, Check, CheckCircle, ChevronDown, ChevronRight,
  Copy, Crosshair, Cpu, Database, Download, ExternalLink, Eye, FileText, Filter,
  FlaskConical, Folder, GitBranch, Grid3x3, Info, Layers, List, Loader2, Lock,
  Map, Package, Play, RefreshCw, Search, Settings, Shield, Target, TrendingUp,
  X, Zap, Droplet,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, Legend, ComposedChart, ReferenceLine,
  LineChart, Line, Area, AreaChart,
} from "recharts";
import { T, FONT, HEADING, MONO, NUC } from "../tokens";
import { useIsMobile } from "../hooks/useIsMobile";
import { useToast } from "../components/Toast";
import { Badge, DrugBadge, Seq, Btn, tooltipStyle, gaussianKDE, stdDev } from "../components/ui/index.jsx";
import { CollapsibleSection, FigureSection } from "../components/CollapsibleSection";
import { AmpliconMap, MismatchProfile } from "../components/CandidateViewer";
import {
  MUTATIONS, RESULTS, CROSS_REACTIVITY_LABELS, CROSS_REACTIVITY_DRUG_GROUPS, MOCK_CROSS_REACTIVITY,
  SCORING_FEATURES, DRUG_LABELS, BIBLIOGRAPHY,
} from "../mockData";
import {
  getResults, exportResults, getFigureUrl, getTopK, getUmapData, getPoolData,
  getPresets, getDiagnostics, getWHOCompliance, runSweep, runPareto,
  compareScorers, getThermoProfile, getThermoStandalone, getAblation,
  getNucleaseProfiles, getNucleaseComparison, getEnzymes, listJobs,
} from "../api";
import { transformApiCandidate } from "../utils/api";
import ChipRender3D from "../ChipRender3D";

const RISK_COLORS = { green: T.riskGreen, amber: T.riskAmber, red: T.riskRed };
const RISK_BG = { green: T.riskGreenBg, amber: T.riskAmberBg, red: T.riskRedBg };
/* GreenBlue colormap; single-cell omics UMAP aesthetic.
   Stops: light gray → pale green → teal → blue → deep blue. */
const gradientColor = (t) => {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    { t: 0.00, r: 180, g: 185, b: 195 },  // #b4b9c3; soft gray (low)
    { t: 0.25, r: 171, g: 221, b: 164 },  // #abdda4; pale green
    { t: 0.50, r: 102, g: 194, b: 165 },  // #66c2a5; teal
    { t: 0.75, r: 50,  g: 136, b: 189 },  // #3288bd; blue
    { t: 1.00, r: 30,  g: 58,  b: 95 },   // #1E3A5F; deep navy (high)
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

/* Heatmap cell for Risk Assessment Matrix; 3 discrete pastel colors */
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
          <strong>Panel gap:</strong> {gaps.map(r => r.label).join(", ")}. No viable discrimination pathway. Requires alternative strategy or SM enhancement.
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

  // Pastel palette; matches UMAP embedding aesthetic
  const DRUG_LINE = { RIF: "#5B8BD4", INH: "#9B8EC4", EMB: "#66C2A5", PZA: "#8DA0CB", FQ: "#E78AC3", AG: "#A6D854", OTHER: "#B3B3B3", CTRL: "#B3B3B3" };

  // Full-width responsive SVG; use viewBox for scaling
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

      {/* Full-width SVG; viewBox scales to container */}
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

        {/* Candidate polylines; non-hovered first (dimmed), then hovered on top */}
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
            Panel-wide gap: <strong style={{ color: T.text }}>{AXIS_LABELS[weakest.axis]}</strong> axis averages {(weakest.avg * 100).toFixed(0)}%. Consider strengthening candidates on this dimension.
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

    // Clear; light background for UMAP
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
/* ─── Reusable collapsible in silico caveat banner ─── */
const InSilicoCaveat = () => {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginBottom: "20px" }}>
      <button onClick={() => setOpen(!open)} style={{ display: "flex", alignItems: "center", gap: "6px", background: "none", border: "none", cursor: "pointer", padding: "4px 0", marginBottom: open ? "2px" : 0, fontFamily: FONT, fontSize: "11px", fontWeight: 600, color: "#92400E", textTransform: "uppercase", letterSpacing: "0.04em" }}>
        <ChevronDown size={12} style={{ transform: open ? "rotate(0deg)" : "rotate(-90deg)", transition: "transform 0.15s", color: "#D97706" }} />
        <AlertTriangle size={12} color="#D97706" strokeWidth={2} />
        In silico prediction: experimental validation required
      </button>
      {open && (
        <div style={{ background: "#FFFBEB", border: "1px solid #F59E0B33", borderRadius: "4px", padding: "12px 16px", fontSize: "11px", color: "#92400E", lineHeight: 1.7 }}>
          <p style={{ margin: "0 0 6px" }}>
            Activity scores are predicted by Compass-ML (CNN + RNA-FM + RLPA) trained on human cell cis-cleavage data (Kim et al. 2018). The ranking between candidates is informative for synthesis prioritisation, but absolute values are not proportional to electrochemical signal on LIG-E (paper-based) electrodes.
          </p>
          <p style={{ margin: "0 0 6px" }}>
            Discrimination ratios (XGBoost, 18 thermodynamic features) are trained on 6,136 paired trans-cleavage measurements (Huang et al. 2024, LbCas12a). These are the most reliable in silico metric. Actual enAsCas12a discrimination on the electrochemical platform requires experimental confirmation.
          </p>
          <p style={{ margin: 0 }}>
            All predictions serve as a starting point for the wet-lab validation workflow on the deMello group's paper-based electrochemical platform (Bezinge et al. 2023).
          </p>
        </div>
      )}
    </div>
  );
};

const OverviewTab = ({ results, scorer, jobId, panelData }) => {
  const mobile = useIsMobile();
  const [glossaryOpen, setGlossaryOpen] = useState(false);

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
  // Separate direct (excl. IS6110) vs proximity discrimination
  const directNonIS = directResults.filter(r => r.gene !== "IS6110");
  const avgDiscDirect = directNonIS.length ? +(directNonIS.reduce((a, r) => a + r.disc, 0) / directNonIS.length).toFixed(1) : 0;
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

  const DRUG_PREVALENCE = {
    RIF: "~85%", INH: "~85%", EMB: "~45%", PZA: "~40%",
    FQ: "~40%", AG: "~90%", OTHER: "N/A"
  };

  // Model agreement; Spearman ρ between heuristic and Compass-ML (PAM-adjusted)
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
      <InSilicoCaveat />

      {/* Panel Interpretation; blue card with collapsible glossary */}
      <div style={{ background: "#EFF6FF", border: "1px solid #BFDBFE", borderRadius: "6px", padding: "16px 20px", marginBottom: "20px" }}>
        <div style={{ fontSize: "13px", fontWeight: 600, color: "#1D4ED8", marginBottom: "6px" }}>Panel Interpretation</div>
        <div style={{ fontSize: "12px", color: "#1E40AF", lineHeight: 1.6 }}>
          {assayReady}/{totalTargets} targets are assay-ready across {drugs.filter(d => d !== "OTHER").length} drug classes ({drugs.filter(d => d !== "OTHER").join(", ")}). {belowThreshold.length > 0 ? `${belowThreshold.length} below threshold - prioritised for first experimental round.` : "All targets meet readiness threshold."}
        </div>
        {/* Collapsible glossary */}
        <div style={{ marginTop: "10px" }}>
          <button onClick={() => setGlossaryOpen(!glossaryOpen)} style={{ background: "none", border: "none", padding: 0, cursor: "pointer", fontSize: "11px", fontWeight: 500, color: "#2563EB", display: "flex", alignItems: "center", gap: "4px" }}>
            {glossaryOpen ? <ChevronDown size={12} color="#2563EB" /> : <ChevronRight size={12} color="#2563EB" />} Metric definitions
          </button>
          {glossaryOpen && (
            <div style={{ marginTop: "8px", fontSize: "11px", color: "#1E3A5F", lineHeight: 1.7, paddingLeft: "12px", borderLeft: "2px solid #BFDBFE" }}>
              <div style={{ marginBottom: "4px" }}><strong>Direct avg ({avgDiscDirect}×):</strong> Mean Cas12a mismatch discrimination for direct-detection candidates (MUT/WT cleavage fold-difference).</div>
              <div style={{ marginBottom: "4px" }}><strong>Proximity (AS-RPA):</strong> Allele-specific RPA primer discrimination for candidates where the mutation falls outside the crRNA footprint. Estimated 11–100× selectivity.</div>
              <div style={{ marginBottom: "4px" }}><strong>Diagnostic-grade (≥3×):</strong> Number of direct candidates exceeding the 3× threshold for reliable clinical use on electrochemical or lateral-flow readout.</div>
              <div style={{ marginBottom: "4px" }}><strong>Avg. activity ({avgActivity}):</strong> Compass-ML predicted Cas12a on-target cis-cleavage efficiency (0–1). Higher = stronger trans-cleavage signal.</div>
              <div style={{ marginBottom: "4px" }}><strong>Avg. PAM-adjusted ({avgPamAdj || "N/A"}):</strong> Activity x PAM penalty. Predicts actual signal strength on the electrode after non-canonical PAM penalties (enAsCas12a expanded PAMs in GC-rich MTB).</div>
              <div><strong>Disc model:</strong> XGBoost on 18 thermodynamic features (trained on 6,136 EasyDesign crRNA-target pairs). Predicts MUT/WT discrimination from mismatch position, type, and flanking sequence context.</div>
            </div>
          )}
        </div>
      </div>

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

        {/* Three evidence columns; reordered: confidence → discrimination → readiness */}
        <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr 1fr", gap: mobile ? "16px" : "12px" }}>
          {/* Column 1: How confident are we? (discrimination; most important metric) */}
          <div style={{ background: T.bgSub, borderRadius: "6px", padding: "16px 20px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "14px" }}>
              <TrendingUp size={11} color={T.textTer} strokeWidth={2} />
              How confident are we?
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                  <TrendingUp size={12} color={T.textTer} strokeWidth={1.8} />
                  Direct avg
                </span>
                <span style={{ fontSize: "20px", fontWeight: 600, color: T.text, fontFamily: FONT }}>{avgDiscDirect}×</span>
              </div>
              {proximityCount > 0 && (
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: T.textSec }}>
                    <GitBranch size={12} color={T.purple} strokeWidth={1.8} />
                    Proximity
                  </span>
                  <span style={{ fontSize: "14px", fontWeight: 600, color: T.purple, fontFamily: FONT }}>AS-RPA (≥100× est.)</span>
                </div>
              )}
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
              <div style={{ fontSize: "10px", color: T.textTer, marginTop: "2px", paddingLeft: "16px" }}>
                Disc model: {discModel}
              </div>
            </div>
          </div>

          {/* Column 2: Can we detect resistance? */}
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

          {/* Old Column 2 removed; content moved to Column 1 */}

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

      {/* ── Drug Coverage table (promoted to position 2) ── */}
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
              {["Drug", "Candidates", "Avg Score", "Avg Disc (Direct)", "Primers", "Coverage"].map((h) => (
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
                  <td style={{ padding: "12px 24px", fontFamily: FONT, fontWeight: 600, color: T.textSec }}>{DRUG_PREVALENCE[d.drug] || "\u2014"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        </div>
        <div style={{ padding: "10px 24px", fontSize: "10px", color: T.textTer, lineHeight: 1.6, borderTop: `1px solid ${T.borderLight}` }}>
          Estimated resistance coverage from WHO mutation catalogue (2021). Actual coverage varies by geography and population.
        </div>
      </div>
      </FigureSection>

      {/* Risk Assessment Matrix */}
      {results.some(r => r.riskProfile != null) && (
        <FigureSection title="Risk Assessment Matrix" subtitle={`${results.length} targets scored across 5 biophysical axes; green = safe, amber = moderate risk, red = requires attention.`}>
          <RiskMatrix results={results} />
        </FigureSection>
      )}

      {/* ── Readiness Bar Chart (replaces parallel coordinates) ── */}
      {(() => {
        const readinessData = results.filter(r => r.readinessScore != null)
          .map(r => ({ label: r.label, drug: r.drug, score: r.readinessScore }))
          .sort((a, b) => b.score - a.score);
        if (readinessData.length === 0) return null;
        return (
          <FigureSection title="Diagnostic Readiness Score" subtitle="Candidates ranked by composite readiness score. Colored by drug class.">
            <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: mobile ? "16px" : "20px 24px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                {readinessData.map((r, i) => {
                  const pct = Math.round(r.score * 100);
                  const rs = r.score;
                  const barColor = rs >= 0.55 ? "#BBF7D0" : rs >= 0.4 ? "#FDE68A" : "#FECACA";
                  const scoreTextColor = rs >= 0.55 ? "#059669" : rs >= 0.4 ? "#D97706" : "#DC2626";
                  return (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <div style={{ width: mobile ? "80px" : "120px", fontSize: "11px", fontFamily: MONO, color: T.text, textAlign: "right", flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.label}</div>
                      <div style={{ flex: 1, height: "18px", background: T.bgSub, borderRadius: "3px", overflow: "hidden", position: "relative" }}>
                        <div style={{ width: `${pct}%`, height: "100%", background: barColor, borderRadius: "3px", transition: "width 300ms ease-out" }} />
                      </div>
                      <div style={{ width: "36px", fontSize: "11px", fontFamily: FONT, fontWeight: 600, color: scoreTextColor, textAlign: "right", flexShrink: 0 }}>{pct}%</div>
                    </div>
                  );
                })}
              </div>
              {/* Drug color legend */}
              <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "14px", flexWrap: "wrap" }}>
                {Object.entries(DRUG_CANVAS).filter(([d]) => d !== "OTHER" && readinessData.some(r => r.drug === d)).map(([drug, color]) => (
                  <div key={drug} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                    <div style={{ width: 10, height: 10, borderRadius: "2px", background: color, opacity: 0.85 }} />
                    <span style={{ fontSize: "10px", color: T.textTer, fontWeight: 600 }}>{drug}</span>
                  </div>
                ))}
              </div>
            </div>
          </FigureSection>
        );
      })()}

      {/* Score vs Discrimination Scatter; readiness-sized dots */}
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
                  {bottomRight.length > 0 ? ` ${bottomRight.length} Direct candidate${bottomRight.length > 1 ? "s have" : " has"} good scores but low Cas12a discrimination (${bottomRight.slice(0, 2).map(d => d.label).join(", ")}${bottomRight.length > 2 ? "…" : ""}); synthetic mismatch enhancement may improve these.` : ""}
                  {topLeft.length > 0 ? ` ${topLeft.length} candidate${topLeft.length > 1 ? "s" : ""} ${topLeft.length > 1 ? "have" : "has"} strong discrimination but weak scores; alternative spacers may help.` : ""}
                  {proximityCands.length > 0 ? ` ${proximityCands.length} Proximity candidate${proximityCands.length > 1 ? "s are" : " is"} not plotted. Their discrimination comes from AS-RPA primers, not crRNA mismatch. Of these, ${viableProx.length} show viable AS-RPA discrimination${nonViableProx > 0 ? ` and ${nonViableProx} ha${nonViableProx > 1 ? "ve" : "s"} no viable discrimination pathway (WC pair)` : ""}.` : ""}
                  {worstCandidate && worstCandidate !== bestCandidate ? ` Weakest Direct: ${worstCandidate.label} (${worstCandidate.score.toFixed(3)}, ${worstCandidate.disc.toFixed(1)}×).` : ""}
                </div>
              );
            })()}
          </div>
          </FigureSection>
        );
      })()}

      {/* ── Model Validation (collapsed: Model Card + UMAP + Scoring Comparison) ── */}
      <CollapsibleSection title="Model Validation" defaultOpen={false}>
        {/* Model Card */}
        {(() => {
          const cal = panelData?.calibration ?? null;
          return (
            <div style={{ background: T.bgSub, border: `1px solid ${T.border}`, borderRadius: "4px", padding: mobile ? "14px" : "16px 20px", marginBottom: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "10px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "10px" }}>
                <Cpu size={11} color={T.textTer} strokeWidth={2} />
                Model Card
              </div>
              <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr 1fr" : "1fr 1fr 1fr 1fr", gap: "8px" }}>
                {[
                  { l: "Validation \u03c1", v: cal?.val_rho != null ? (+cal.val_rho).toFixed(4) : "\u2014" },
                  { l: "Calibration T", v: cal?.temperature != null ? (+cal.temperature).toFixed(2) : "\u2014" },
                  { l: "Calibration \u03b1", v: cal?.alpha != null ? (+cal.alpha).toFixed(4) : "\u2014" },
                  { l: "Model params", v: cal?.model_params ?? "234K" },
                ].map(s => (
                  <div key={s.l} style={{ background: T.bg, border: `1px solid ${T.borderLight}`, borderRadius: "4px", padding: "8px 12px" }}>
                    <div style={{ fontSize: "9px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "3px" }}>{s.l}</div>
                    <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, fontFamily: MONO }}>{s.v}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {/* UMAP Embedding Space */}
        {jobId && (
          <div style={{ marginBottom: "16px" }}>
            <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, marginBottom: "8px" }}>Candidate Embedding Space</div>
            <UMAPPanel jobId={jobId} />
          </div>
        )}

        {/* Scoring Model Comparison */}
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
            <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "28px 32px" }}>
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
                    <strong style={{ color: T.primary }}>Interpretation:</strong> {agreePct}% of candidates are classified the same way by both models (above/below 0.5 threshold). {onLine}/{scatterData.length} score within \u00b10.05 of each other.
                    {aboveLine.length > 0 ? ` Compass-ML scores ${aboveLine.length} candidate${aboveLine.length > 1 ? "s" : ""} higher (${aboveLine.slice(0, 2).map(d => d.label).join(", ")}${aboveLine.length > 2 ? "\u2026" : ""}).` : ""}
                    {belowLine.length > 0 ? ` Heuristic scores ${belowLine.length} candidate${belowLine.length > 1 ? "s" : ""} higher (${belowLine.slice(0, 2).map(d => d.label).join(", ")}${belowLine.length > 2 ? "\u2026" : ""}).` : ""}
                    {modelAgreement != null ? ` Rank correlation \u03c1 = ${modelAgreement} (${modelAgreement >= 0.7 ? "strong agreement, QC corroborates activity predictions" : modelAgreement >= 0.4 ? "moderate agreement, QC catches biophysical edge cases activity model misses" : "weak agreement, models measure different things; QC serves as independent sanity check"}).` : ""}
                  </div>
                );
              })()}
            </div>
          );
        })()}
      </CollapsibleSection>
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
  const snpChange = snpNt ? `${wt[snpNt.pos - 1]}→${snpNt.base}` : "–";
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

      {/* SVG card; centered with generous padding */}
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

        {/* Legend + metadata row; below the SVG, centered */}
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

  // Overall assessment — factor in PAM-adjusted signal
  const pamAdj = r.pamAdjusted ?? (r.pamPenalty != null ? eff * r.pamPenalty : eff);
  const signalNote = (r.pamPenalty != null && r.pamPenalty < 0.8)
    ? ` However, the PAM-adjusted signal is ${pamAdj.toFixed(3)} (${r.pamPenalty}\u00d7 penalty), ${pamAdj <= 0.3 ? "at or below the 0.3 risk threshold. Longer incubation or higher RPA input may be needed for a resolvable SWV peak" : pamAdj < 0.4 ? "near the lower end of reliable detection" : "still within detectable range"}.`
    : "";
  if (eff >= 0.7) lines.push(`Strong candidate (activity score ${eff.toFixed(3)}).${signalNote || " High predicted Cas12a trans-cleavage rate, expected to produce a clear SWV signal decrease within 15-30 min on the electrochemical platform, well above the limit of detection."}`);
  else if (eff >= 0.5) lines.push(`Moderate candidate (activity score ${eff.toFixed(3)}).${signalNote || " Predicted trans-cleavage is sufficient for detection but not optimal. The SWV signal decrease may require 30-45 min to reach a confident positive call on the electrochemical platform."}`);
  else lines.push(`Weak candidate (activity score ${eff.toFixed(3)}).${signalNote || " Low predicted trans-cleavage rate. The electrochemical signal decrease may be near the detection limit, risking false negatives. Consider alternatives from the top-K list or synthetic mismatch optimisation."}`);

  // PAM quality
  const pam = (r.pam || "").toUpperCase();
  if (r.isCanonicalPam || pam.match(/^TTT[ACG]/)) {
    lines.push(`Canonical PAM (${r.pam}). Optimal Cas12a recognition, no activity penalty applied.`);
  } else {
    const penaltyStr = r.pamPenalty != null ? ` Activity penalty: ${r.pamPenalty}\u00d7 (Kleinstiver et al. 2019).` : "";
    lines.push(`Expanded PAM (${r.pam}${r.pamVariant ? `, ${r.pamVariant}` : ""}), recognized with reduced activity vs canonical TTTV.${penaltyStr} This is the best available PAM site in the GC-rich M. tuberculosis genomic context around this mutation.`);
  }

  // PAM disruption: binary discrimination override
  if (r.pamDisrupted) {
    const disruptionDetail = r.pamDisruptionType === "wt_pam_broken"
      ? "Binary discrimination: the resistance SNP disrupts the PAM consensus in the wildtype sequence. Cas12a cannot bind WT DNA at this locus, providing effectively infinite discrimination. This is the strongest possible discrimination mechanism: all-or-nothing PAM recognition gating."
      : "Binary discrimination: the resistance SNP disrupts the PAM consensus in the mutant sequence. Cas12a cannot bind MUT DNA at this locus. This inverts the expected detection logic, so signal absence indicates resistance.";
    lines.push(disruptionDetail);
  }

  // Discrimination
  const discModelName = r.discrimination?.model_name || "";
  const isNeuralDisc = r.discMethod === "neural";
  const isLearnedDisc = discModelName.includes("learned") || r.discMethod === "feature";
  const discSource = isNeuralDisc ? "neural discrimination head (Compass-ML multi-task, trained on 6,136 EasyDesign pairs)" : isLearnedDisc ? "learned model (XGBoost, 18 thermodynamic features)" : "heuristic model (position \u00D7 destabilisation)";
  if (r.pamDisrupted) {
    // Skip normal discrimination analysis; already covered above
  } else if (r.strategy === "Proximity") {
    lines.push(`Proximity detection: the resistance SNP falls outside the crRNA spacer${r.proximityDistance ? ` (${r.proximityDistance} bp away)` : ""}. Allele discrimination relies on AS-RPA primers (10\u2013100\u00D7 selectivity), not Cas12a mismatch intolerance. The Cas12a disc ratio (~${disc.toFixed(1)}\u00D7) is not relevant for this strategy.`);
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
      else lines.push(`SNP at seed position ${snpPos} (${snpChange}${mmChem}) gives limited discrimination (${disc.toFixed(1)}\u00D7, ${discSource}) despite being in the seed region (${rloopEffect}). The surrounding sequence context or mismatch chemistry may stabilise partial R-loop formation. Synthetic mismatch enhancement may improve this.`);
    } else {
      if (disc >= 3) lines.push(`SNP at PAM-distal position ${snpPos} (${snpChange}${mmChem}) provides ${disc.toFixed(1)}\u00D7 discrimination (${discSource}) with ${disc >= 10 ? "near-complete R-loop collapse" : disc >= 5 ? "substantial R-loop disruption" : "moderate R-loop destabilization"}. Although outside the seed, the mismatch is sufficient for diagnostic-grade allele differentiation.`);
      else lines.push(`SNP at PAM-distal position ${snpPos} (${snpChange}${mmChem}) gives limited discrimination (${disc.toFixed(1)}\u00D7, ${discSource}), ${disc >= 2 ? "partial R-loop disruption" : "minimal R-loop disruption"}. PAM-distal mismatches are better tolerated by Cas12a. Synthetic mismatch in the seed region could boost specificity.`);
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
    else lines.push(`Synthetic mismatch applied. An engineered base substitution creates a double-mismatch penalty on the wildtype template, boosting specificity${smImprovement}. Activity cost is position-dependent (Liang et al. 2023).`);
  }

  // GC content
  if (gc > 65) lines.push(`High GC content (${gc.toFixed(0)}%) increases R-loop thermodynamic stability but also raises the energetic cost of target strand unwinding. This is typical for M. tuberculosis (genome-wide GC ~65.6%).`);
  else if (gc < 40) lines.push(`Low GC content (${gc.toFixed(0)}%), unusual for M. tuberculosis. R-loop stability may be reduced, potentially lowering cleavage efficiency.`);

  // Off-targets
  if (r.ot > 0) lines.push(`${r.ot} potential off-target site${r.ot > 1 ? "s" : ""} detected in the H37Rv genome. Review cross-reactivity before synthesis. Off-targets within the same amplicon region could generate false positives.`);

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
          { l: "Activity QC", v: r.activityQc != null ? r.activityQc.toFixed(3) : (r.score ?? 0).toFixed(3), c: T.textTer },
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
          <div style={{ fontSize: "12px", fontWeight: 600, color: T.purple, fontFamily: HEADING, marginBottom: "4px" }}>Proximity Detection; PAM Desert</div>
          <div style={{ fontSize: "11px", color: "#2563EB", lineHeight: 1.5 }}>
            crRNA binds a conserved site {r.proximityDistance ? `${r.proximityDistance} bp` : "near"} the mutation. Discrimination via AS-RPA primers.
          </div>
        </div>
      )}

      {/* crRNA Spacer Architecture; full width, with Show Alternatives top-right */}
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
                { name: `${r.label}_crRNA`, seq: `AATTTCTACTCTTGTAGAT${displaySpacer}`, note: "Direct repeat + spacer", tm: null },
                ...(r.fwd ? [{ name: `${r.label}_FWD`, seq: r.fwd, note: r.strategy === "Direct" ? "Standard RPA forward" : "AS-RPA forward (allele-specific)", tm: r.fwdTm }] : []),
                ...(r.rev ? [{ name: `${r.label}_REV`, seq: r.rev, note: r.strategy === "Direct" ? "Standard RPA reverse" : "AS-RPA reverse (allele-specific)", tm: r.revTm }] : []),
              ].map((o, i, arr) => (
                <div key={o.name} style={{ padding: "10px 14px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                    <span style={{ fontSize: "10px", fontWeight: 600, fontFamily: MONO, color: T.text }}>{o.name}{o.tm != null && <span style={{ marginLeft: "8px", fontSize: "9px", fontWeight: 500, color: T.textSec, fontFamily: FONT }}>Tm {o.tm.toFixed(1)} °C</span>}</span>
                    <button onClick={(e) => { e.stopPropagation(); navigator.clipboard?.writeText(o.seq); toast(`${o.name} copied`); }} style={{ background: "none", border: `1px solid ${T.border}`, borderRadius: "5px", padding: "3px 8px", cursor: "pointer", fontSize: "9px", color: T.textSec, display: "flex", alignItems: "center", gap: "3px" }}><Copy size={9} /> Copy</button>
                  </div>
                  <div style={{ background: T.bg, borderRadius: "4px", padding: "8px 10px", border: `1px solid ${T.borderLight}`, marginBottom: "4px" }}>
                    <Seq s={o.seq} />
                  </div>
                  <div style={{ fontSize: "9px", color: T.textTer }}>{o.note}; {o.seq.length} nt</div>
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
                      ["CRyPTIC", ref.cryptic || "N/A"],
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
                    ["PAM Sequence", `${r.pam}${r.pamVariant ? ` (${r.pamVariant})` : ""}${r.pamPenalty != null && r.pamPenalty < 1.0 ? `; ${r.pamPenalty}× activity` : ""}`],
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

              {/* Score Breakdown; heuristic sub-scores */}
              {r.seedPositionScore != null && (
                <div style={{ gridColumn: mobile ? "1" : "1 / -1" }}>
                  <div style={{ fontSize: "11px", fontWeight: 600, color: T.textSec, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Score Breakdown</div>
                  <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", padding: "12px 14px" }}>
                    {[
                      { label: "Seed Position", value: r.seedPositionScore },
                      { label: "GC Content", value: r.gcPenalty },
                      { label: "Structure (MFE)", value: r.structurePenalty },
                      { label: "Homopolymer", value: r.homopolymerPenalty },
                      { label: "Off-target", value: r.offtargetPenalty },
                    ].filter(s => s.value != null).map(s => {
                      const barColor = s.value > 0.7 ? T.success : s.value > 0.4 ? T.warning : T.danger;
                      return (
                        <div key={s.label} style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
                          <span style={{ fontSize: "10px", color: T.textSec, width: "100px", flexShrink: 0 }}>{s.label}</span>
                          <div style={{ flex: 1, height: "8px", borderRadius: "4px", background: T.borderLight, overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${Math.max(s.value * 100, 2)}%`, borderRadius: "4px", background: barColor, transition: "width 0.3s" }} />
                          </div>
                          <span style={{ fontSize: "10px", fontWeight: 600, color: barColor, fontFamily: MONO, minWidth: "32px", textAlign: "right" }}>{s.value.toFixed(2)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Thermodynamics section */}
              {(r.mfe != null || r.thermoDdg != null || r.mismatchType != null) && (
                <div style={{ gridColumn: mobile ? "1" : "1 / -1" }}>
                  <div style={{ fontSize: "11px", fontWeight: 600, color: T.textSec, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Thermodynamics</div>
                  <div style={{ background: T.bgSub, border: `1px solid ${T.borderLight}`, borderRadius: "4px", overflow: "hidden" }}>
                    {[
                      ...(r.mfe != null ? [["MFE", `${r.mfe.toFixed(1)} kcal/mol`]] : []),
                      ...(r.thermoDdg != null ? [["\u0394\u0394G at mismatch", `${r.thermoDdg.toFixed(1)} kcal/mol`]] : []),
                      ...(r.mismatchType != null ? [["Mismatch type", r.mismatchType]] : []),
                    ].map(([k, v], i, arr) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "7px 14px", borderBottom: i < arr.length - 1 ? `1px solid ${T.borderLight}` : "none", fontSize: "11px" }}>
                        <span style={{ color: T.textSec }}>{k}</span>
                        <span style={{ fontWeight: 600, color: T.text, fontFamily: MONO }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
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
  const defaultSort = "readinessScore";
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
    const PREV_ORDER = { "rpoB_S531L": 45, "rpoB_H526Y": 15, "rpoB_D516V": 8, "katG_S315T": 60, "fabG1_C-15T": 25, "embB_M306V": 30, "embB_M306I": 15, "pncA_H57D": 5, "gyrA_D94G": 25, "gyrA_A90V": 15, "rrs_A1401G": 80, "eis_C-14T": 10, "IS6110_N0N": 0 };
    const getSortVal = (r) => {
      if (sortKey === "riskOverall") { const v = r.riskProfile?.overall; return v === "green" ? 2 : v === "amber" ? 1 : 0; }
      if (sortKey === "experimentalPriority") return r.experimentalPriority ?? 999;
      if (sortKey === "prevalence") return PREV_ORDER[r.label] || 0;
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

  // Mutation prevalence (% of drug-resistant isolates, WHO 2021 catalogue)
  const PREVALENCE = {
    "rpoB_S531L": { pct: "~45%", tip: "Most common RIF-R globally" },
    "rpoB_H526Y": { pct: "~15%", tip: "Second most common RIF-R" },
    "rpoB_D516V": { pct: "~8%", tip: "Third most common RIF-R" },
    "katG_S315T": { pct: "~60%", tip: "Dominant INH-R mechanism" },
    "fabG1_C-15T": { pct: "~25%", tip: "Second INH-R, inhA promoter" },
    "embB_M306V": { pct: "~30%", tip: "Most common EMB-R" },
    "embB_M306I": { pct: "~15%", tip: "Second EMB-R at codon 306" },
    "pncA_H57D": { pct: "~5%", tip: "Moderate PZA-R, variable by region" },
    "gyrA_D94G": { pct: "~25%", tip: "Most common FQ-R" },
    "gyrA_A90V": { pct: "~15%", tip: "Second most common FQ-R" },
    "rrs_A1401G": { pct: ">80%", tip: "Dominant AG-R mechanism" },
    "eis_C-14T": { pct: "~10%", tip: "KAN-R via eis promoter" },
    "IS6110_N0N": { pct: "N/A", tip: "Species ID control" },
  };

  // Risk flags per candidate
  const getRiskFlags = (r) => {
    const flags = [];
    const pamAdj = r.pamAdjusted ?? r.cnnCalibrated ?? r.score ?? 0;
    if (pamAdj < 0.3 && pamAdj > 0) flags.push({ icon: "\u2193", label: "Low signal", color: T.danger, tip: `PAM-adj ${pamAdj.toFixed(2)}` });
    if (r.gc && r.gc > 0.72) flags.push({ icon: "GC", label: "High GC", color: T.warning, tip: `${(r.gc * 100).toFixed(0)}% GC` });
    if (r.strategy === "Direct" && r.disc < 5 && r.disc > 0) flags.push({ icon: "D", label: "Low disc", color: T.warning, tip: `${r.disc.toFixed(1)}\u00d7` });
    if (r.ot > 0) flags.push({ icon: "OT", label: "Off-targets", color: T.danger, tip: `${r.ot} hits` });
    return flags;
  };

  // Columns: #, Target, Drug, Strategy, Signal, Disc, Prevalence, Risk, Readiness
  const cols = [
    ...(hasReadiness ? [{ key: "experimentalPriority", label: "#", w: 32 }] : []),
    { key: "label", label: "Target", w: 120 },
    { key: "drug", label: "Drug", w: 50 },
    { key: "strategy", label: "Strategy", w: 72 },
    { key: "pamAdjusted", label: "Signal", w: 90 },
    { key: "disc", label: "Disc", w: 70 },
    { key: "prevalence", label: "Prevalence", w: 70 },
    { key: "risk", label: "Risk", w: 70 },
    ...(hasReadiness ? [{ key: "readinessScore", label: "Readiness", w: 90 }] : []),
  ];

  // Hover state for spacer color reveal
  const [hoveredRow, setHoveredRow] = useState(null);

  return (
    <div>
      <InSilicoCaveat />

      {/* Explainer box; blue */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", padding: mobile ? "14px" : "16px 20px", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <List size={16} color={T.primary} strokeWidth={1.8} />
          <span style={{ fontSize: "13px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING }}>Candidate Scoring</span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", color: T.primaryDark, lineHeight: 1.6 }}>
          <div><span style={{ color: T.primary }}>Signal:</span> PAM-adjusted efficiency (Activity x PAM penalty). Predicts actual signal strength on the electrode. Below 0.3 = risk of weak SWV peak.</div>
          <div><span style={{ color: T.primary }}>Disc:</span> MUT/WT fold-difference. Direct: crRNA mismatch (XGBoost, 18 features). Proximity: AS-RPA primer selectivity. ≥3x diagnostic-grade, {"<"}2x insufficient.</div>
          <div><span style={{ color: T.primary }}>Prevalence:</span> % of drug-resistant isolates with this mutation (WHO catalogue 2023).</div>
          <div><span style={{ color: T.primary }}>Risk flags:</span> signal below 0.3 · GC primer GC {">"}70% · D discrimination below 5x</div>
          <div><span style={{ color: T.primary }}>Readiness:</span> Composite 0-100 across signal, discrimination, primers, off-target, GC. ≥40 = assay-ready. Ranks by technical feasibility, not clinical importance. Sort by Prevalence to prioritize by patient impact.</div>
          <div style={{ color: T.textSec }}>Expand any row for scored sequence, primers, Top-K alternatives, and predicted SWV curves.</div>
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

      {/* Candidates; cards on mobile, table on desktop */}
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden" }}>
       {mobile ? (
        /* ── Mobile card layout; monochrome ── */
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
                      {r.pamDisrupted && r.pamDisruptionType && (
                        <span style={{ display: "inline-block", marginLeft: "4px", padding: "1px 5px", borderRadius: "4px", fontSize: "8px", fontWeight: 600, background: "#7c3aed18", color: "#7c3aed", border: "1px solid #7c3aed33" }}>PAM Disrupted</span>
                      )}
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
                              {alt.has_primers ? <Badge variant="success">P</Badge> : <Badge variant="danger">–</Badge>}
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
        /* ── Desktop table layout; monochrome by default, color encodes meaning ── */
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
                    {/* Target; clean, no strategy subscript */}
                    <td style={{ padding: "10px 12px" }}>
                      <span style={{ fontWeight: 600, fontFamily: MONO, fontSize: "11px", color: T.text }}>{r.label}</span>
                    </td>
                    {/* Drug */}
                    <td style={{ padding: "10px 8px" }}><DrugBadge drug={r.drug} /></td>
                    {/* Strategy; prominent pill */}
                    <td style={{ padding: "10px 8px" }}>
                      <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600,
                        background: r.strategy === "Direct" ? "rgba(37,99,235,0.08)" : "rgba(124,58,237,0.08)",
                        color: r.strategy === "Direct" ? T.primary : "#7c3aed",
                        border: `1px solid ${r.strategy === "Direct" ? "rgba(37,99,235,0.2)" : "rgba(124,58,237,0.2)"}`,
                      }}>{r.strategy === "Direct" ? "Direct" : "Proximity"}</span>
                    </td>
                    {/* Signal; PAM-adjusted as primary, raw activity as secondary */}
                    <td style={{ padding: "10px 8px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <span style={{ fontFamily: FONT, fontWeight: 700, fontSize: "12px", color: pamAdjVal > 0.5 ? T.text : pamAdjVal > 0.3 ? T.warning : T.danger }}>{pamAdjVal.toFixed(3)}</span>
                        {r.pamPenalty != null && r.pamPenalty < 1.0 && (
                          <span style={{ fontSize: "9px", color: T.textTer }}>({activityVal.toFixed(2)}{"\u00d7"}{r.pamPenalty})</span>
                        )}
                      </div>
                      <div style={{ width: "100%", height: "3px", borderRadius: "2px", background: T.borderLight, marginTop: "3px", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${Math.min(pamAdjVal * 100, 100)}%`, borderRadius: "2px",
                          background: pamAdjVal > 0.5 ? T.primary : pamAdjVal > 0.3 ? T.warning : T.danger,
                        }} />
                      </div>
                    </td>
                    {/* Disc; with AS-RPA range for proximity */}
                    <td style={{ padding: "10px 8px", fontFamily: FONT, fontWeight: 600, fontSize: "11px", color: r.gene === "IS6110" ? T.textTer : r.strategy === "Proximity" ? "#7c3aed" : discColor }}>
                      {r.gene === "IS6110" ? <span style={{ fontSize: "10px", fontWeight: 400 }}>N/A</span>
                        : r.strategy === "Proximity" ? (() => {
                          const ad = r.asrpaDiscrimination;
                          if (ad && ad.disc_ratio) return <span>{ad.disc_ratio >= 100 ? "\u2265100" : `~${ad.disc_ratio.toFixed(0)}`}{"\u00d7"}</span>;
                          return <span style={{ fontSize: "10px" }}>AS-RPA</span>;
                        })()
                        : `${typeof r.disc === "number" ? r.disc.toFixed(1) : r.disc}\u00d7`}
                    </td>
                    {/* Prevalence; clinical importance */}
                    <td style={{ padding: "10px 8px" }}>
                      {(() => {
                        const prev = PREVALENCE[r.label];
                        if (!prev) return <span style={{ fontSize: "10px", color: T.textTer }}>{"\u2014"}</span>;
                        return <span style={{ fontSize: "11px", fontWeight: 600, color: T.text, fontFamily: FONT }} title={prev.tip}>{prev.pct}</span>;
                      })()}
                    </td>
                    {/* Risk; compact flag icons */}
                    <td style={{ padding: "10px 6px" }}>
                      {(() => {
                        const flags = getRiskFlags(r);
                        if (flags.length === 0) return <span style={{ fontSize: "10px", color: T.success, background: T.success + "20", border: `1px solid ${T.success}30`, padding: "2px 5px", borderRadius: "4px", fontWeight: 700 }}>{"\u2191"}</span>;
                        return (
                          <div style={{ display: "flex", gap: "3px", flexWrap: "wrap" }}>
                            {flags.map((f, i) => (
                              <span key={i} title={`${f.label}: ${f.tip}`} style={{ display: "inline-block", padding: "2px 5px", borderRadius: "4px", fontSize: "8px", fontWeight: 700, background: f.color + "20", border: `1px solid ${f.color}30`, color: f.color, fontFamily: MONO }}>{f.icon}</span>
                            ))}
                          </div>
                        );
                      })()}
                    </td>
                    {/* Readiness; gradient fill (keep strongest visual element) */}
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
                        ) : "N/A"}
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
                                    <td style={{ padding: "7px 12px", fontFamily: FONT, fontSize: "10px", color: alt.has_primers ? T.success : T.textTer }}>{alt.has_primers ? "Yes" : "–"}</td>
                                    <td style={{ padding: "7px 12px", fontSize: "10px", color: T.textSec }}>{alt.tradeoff || "N/A"}</td>
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
    return r ? ((r.cnnCalibrated ?? r.score) || 0).toFixed(2) : "N/A";
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
                <strong>{labels[hovCell.idx]}</strong>; on-target (S_eff = {getOnTargetScore(hovCell.idx)})
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
        <div style={{ fontSize: 12, color: T.text, fontFamily: FONT, lineHeight: 1.8 }}>
          <strong>{data.n_pairs} / {data.n_pairs}</strong> pairs tested<br />
          <strong>{data.none_count}</strong> pairs: no cross-reactivity (&lt; 1%)<br />
          <strong>{data.same_gene_pairs.length}</strong> pairs: low{"\u2013"}medium cross-reactivity (same-gene overlapping amplicons)
        </div>
        <div style={{ marginTop: 12, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 4, fontSize: 10, fontWeight: 600,
            background: data.panel_safe ? "#ECFDF5" : "#FEF2F2",
            color: data.panel_safe ? "#059669" : "#DC2626",
            fontFamily: MONO,
          }}>
            {data.panel_safe ? <><CheckCircle size={14} color="#059669" strokeWidth={2} style={{ display: "inline", verticalAlign: "middle" }} /> SAFE (in silico)</> : "\u26A0 REVIEW"} for spatially multiplexed electrode array
          </span>
        </div>
        {data.same_gene_pairs.length > 0 && (
          <div style={{ marginTop: 16, fontSize: 11, color: T.textSec, lineHeight: 1.8 }}>
            <strong style={{ display: "block", marginBottom: "8px" }}>Same-gene pairs with residual cross-reactivity:</strong>
            {data.same_gene_pairs.filter(p => p.sourceIdx < p.targetIdx || !data.same_gene_pairs.find(q => q.sourceIdx === p.targetIdx && q.targetIdx === p.sourceIdx && q.sourceIdx < q.targetIdx)).map(p => (
              <div key={`${p.source}-${p.target}`} style={{ marginLeft: 8, fontSize: 11, marginBottom: "6px", lineHeight: 1.7, display: "flex", gap: "6px" }}>
                <span style={{ color: T.textTer }}>{"·"}</span>
                <span><span style={{ fontFamily: MONO }}>{p.source} {"\u2194"} {p.target}</span><span style={{ fontFamily: FONT }}>: {(p.activity * 100).toFixed(1)}% ({p.risk.toUpperCase()}) - {p.note}</span></span>
              </div>
            ))}
            <div style={{ marginTop: 6, fontSize: 9, color: T.textTer }}>
              Managed by spatial separation: each crRNA contacts only its own amplicon within its physically isolated detection zone.
            </div>
          </div>
        )}
      </div>

      {/* Top Risk Pairs; sorted by off-target activity */}
      {(() => {
        const riskPairs = [...data.matrix]
          .filter(m => m.activity > 0.01)
          .sort((a, b) => b.activity - a.activity)
          .slice(0, 8);
        if (riskPairs.length === 0) return null;
        return (
          <div style={{ background: T.bg, border: `1px solid ${T.borderLight}`, borderRadius: 4, padding: "12px 16px", marginBottom: 16 }}>
            <div style={{ fontWeight: 600, color: T.text, marginBottom: 6, fontSize: 11, fontFamily: HEADING }}>Top Risk Pairs</div>
            <div style={{ fontSize: 10, color: T.textSec, marginBottom: 8, lineHeight: 1.5 }}>
              Highest-risk crRNA pairs ranked by predicted off-target activity. Only pairs with {">"}1% cross-reactivity shown.
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: FONT }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                  {["Source crRNA", "Off-target amplicon", "Activity", "Mismatches", "PAM", "Risk"].map(hd => (
                    <th key={hd} style={{ padding: "4px 8px", textAlign: "left", fontWeight: 600, color: T.textTer, fontSize: 9, textTransform: "uppercase", letterSpacing: "0.04em" }}>{hd}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {riskPairs.map((p, idx) => {
                  const riskColor = p.risk === "high" ? T.danger : p.risk === "medium" ? T.warning : "#D97706";
                  return (
                    <tr key={idx} style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                      <td style={{ padding: "4px 8px", fontFamily: MONO, fontWeight: 600, fontSize: 9 }}>{p.source}</td>
                      <td style={{ padding: "4px 8px", fontFamily: MONO, fontSize: 9, color: T.textSec }}>{p.target}</td>
                      <td style={{ padding: "4px 8px", fontWeight: 600, color: riskColor }}>{(p.activity * 100).toFixed(1)}%</td>
                      <td style={{ padding: "4px 8px", color: T.textSec }}>{p.mismatches} nt</td>
                      <td style={{ padding: "4px 8px" }}>
                        <span style={{ fontSize: 9, fontWeight: 500, color: p.pam_valid ? T.warning : T.success }}>{p.pam_valid ? "valid" : "absent"}</span>
                      </td>
                      <td style={{ padding: "4px 8px" }}>
                        <span style={{ display: "inline-block", padding: "1px 6px", borderRadius: 3, fontSize: 9, fontWeight: 600, background: riskColor + "15", color: riskColor, textTransform: "uppercase" }}>{p.risk}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {riskPairs.every(p => p.note != null) && (
              <div style={{ marginTop: 6, fontSize: 9, color: T.textTer }}>
                All high-activity pairs are same-gene overlapping amplicons, managed by spatial isolation on the electrode array.
              </div>
            )}
          </div>
        );
      })()}

      {/* Interpretation */}
      <div style={{ background: T.bg, border: `1px solid ${T.borderLight}`, borderRadius: 4, padding: "12px 16px", fontSize: 10, color: T.textSec, lineHeight: 1.7 }}>
        <div style={{ fontWeight: 600, color: T.text, marginBottom: 4 }}>Interpretation</div>
        Cross-reactivity is assessed by scoring each crRNA against all non-self amplicons in the 14-target panel.
        For the paper-based spatially multiplexed electrode array (Bezinge et al., Adv. Mater. 2023),
        each detection zone is isolated by wax-printed hydrophobic barriers, so each crRNA physically contacts only its own zone's amplicon, making inter-zone cross-reactivity impossible.
        <br /><br />
        This analysis validates that same-gene targets with overlapping amplicons (e.g., rpoB_S531L and rpoB_H526Y, which share the rpoB RRDR amplicon) do not produce false positives even in a hypothetical shared-solution format, and identifies any targets where crRNA redesign would improve panel orthogonality.
        <br /><br />
        <strong>PAM-level filtering:</strong> Cas12a requires a 5\u2032-TTTV PAM for activation. Off-target sites without a valid PAM are scored as zero regardless of spacer complementarity, as PAM recognition is an absolute prerequisite for R-loop initiation (Suea-Ngam et al., Chem. Sci. 2021, Fig. 4).
      </div>
    </div>
  );
};

const DiscriminationTab = ({ results }) => {
  const mobile = useIsMobile();
  const [expandedFeatures, setExpandedFeatures] = useState({});
  const nonControl = results.filter((r) => r.disc < 900);
  const directCands = nonControl.filter((r) => r.strategy === "Direct");
  const proximityCands = nonControl.filter((r) => r.strategy === "Proximity");
  const data = directCands.map((r) => ({ name: r.label, disc: +r.disc, score: r.score, drug: r.drug }));
  const excellent = directCands.filter((r) => r.disc >= 10).length;
  const good = directCands.filter((r) => r.disc >= 3 && r.disc < 10).length;
  const acceptable = directCands.filter((r) => r.disc >= 2 && r.disc < 3).length;
  const insufficient = directCands.filter((r) => r.disc < 2).length;

  // Helper: classify mismatch type (transition vs transversion)
  const TRANSITIONS_SET = new Set(["AG", "GA", "CT", "TC"]);
  const classifyMismatch = (mutBase, wtBase) => {
    if (!mutBase || !wtBase || mutBase === wtBase) return null;
    const pair = (mutBase + wtBase).toUpperCase();
    const isTransition = TRANSITIONS_SET.has(pair);
    return { change: `${wtBase}\u2192${mutBase}`, isTransition, label: isTransition ? "transition" : "transversion" };
  };
  const deriveMismatch = (r) => {
    if (r.mismatchType) return r.mismatchType;
    if (r.spacer && r.wtSpacer && r.spacer.length === r.wtSpacer.length) {
      for (let i = 0; i < r.spacer.length; i++) {
        if (r.spacer[i] !== r.wtSpacer[i]) {
          const info = classifyMismatch(r.spacer[i], r.wtSpacer[i]);
          if (info) return `${info.change} ${info.label}`;
        }
      }
    }
    return null;
  };
  const isMismatchTransversion = (r) => {
    if (r.mismatchType) return r.mismatchType.toLowerCase().includes("transversion");
    if (r.spacer && r.wtSpacer && r.spacer.length === r.wtSpacer.length) {
      for (let i = 0; i < r.spacer.length; i++) {
        if (r.spacer[i] !== r.wtSpacer[i]) {
          const info = classifyMismatch(r.spacer[i], r.wtSpacer[i]);
          if (info) return !info.isTransition;
        }
      }
    }
    return false;
  };

  return (
    <div>
      <InSilicoCaveat />

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
            is on resistant DNA versus normal DNA; for example, "5×" means the guide produces 5 times more signal on a resistant sample.
          </p>
          <p style={{ margin: 0 }}>
            A ratio ≥ 3× is considered diagnostic-grade; reliable enough for clinical use with electrochemical (SWV) or lateral-flow readout.
            ≥ 2× is the minimum for any detection method. Below 2× the guide cannot reliably distinguish resistant from susceptible bacteria
            and requires synthetic mismatch enhancement.
          </p>
        </div>
      </div>

      {/* Unified discrimination summary; all 13 targets */}
      <div style={{ background: T.bgSub, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "14px 18px", marginBottom: "16px" }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: T.textTer, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "10px" }}>Discrimination Summary, All Targets</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "4px" }}>
          {[...results].filter(r => r.gene !== "IS6110").sort((a, b) => {
            const aDisc = a.strategy === "Proximity" ? (a.asrpaDiscrimination?.disc_ratio || 100) : (a.disc || 0);
            const bDisc = b.strategy === "Proximity" ? (b.asrpaDiscrimination?.disc_ratio || 100) : (b.disc || 0);
            return bDisc - aDisc;
          }).map(r => {
            const disc = r.strategy === "Proximity" ? (r.asrpaDiscrimination?.disc_ratio || 100) : (r.disc || 0);
            const discLabel = r.strategy === "Proximity"
              ? (disc >= 100 ? "\u2265100\u00d7" : `~${disc.toFixed(0)}\u00d7`)
              : `${disc.toFixed(1)}\u00d7`;
            const source = r.strategy === "Proximity" ? "AS-RPA" : "crRNA";
            const maxBar = 100;
            const barW = Math.min(disc / maxBar * 100, 100);
            const barColor = r.strategy === "Proximity" ? "#7c3aed" : (disc >= 10 ? "#16a34a" : disc >= 3 ? "#2563EB" : "#d97706");
            return (
              <div key={r.label} style={{ display: "flex", alignItems: "center", gap: "8px", padding: "3px 0" }}>
                <span style={{ fontFamily: MONO, fontSize: "10px", fontWeight: 600, color: T.text, width: "110px", flexShrink: 0 }}>{r.label}</span>
                <div style={{ flex: 1, height: "6px", borderRadius: "3px", background: T.borderLight, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${barW}%`, borderRadius: "3px", background: barColor }} />
                </div>
                <span style={{ fontFamily: FONT, fontSize: "10px", fontWeight: 600, color: barColor, width: "50px", textAlign: "right" }}>{discLabel}</span>
                <span style={{ fontSize: "9px", color: T.textTer, width: "40px" }}>{source}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Threshold cards; glass style */}
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

      {proximityCands.length > 0 && (
        <div style={{ background: "#F5F3FF", border: "1px solid #DDD6FE", borderRadius: "4px", padding: "12px 16px", marginBottom: "24px", display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "13px", fontWeight: 600, color: "#7c3aed" }}>{proximityCands.length} proximity targets</span>
          <span style={{ fontSize: "12px", color: "#6D28D9" }}>Discrimination via AS-RPA primers (11{"\u2013"}100{"\u00d7"} estimated selectivity), see table below</span>
        </div>
      )}

      {/* Discrimination chart; horizontal lollipop */}
      {(() => {
        const DRUG_DC = { RIF: "#1E3A5F", INH: "#4338CA", EMB: "#059669", FQ: "#DC2626", AG: "#3730A3", PZA: "#059669", OTHER: "#9CA3AF" };
        const sorted = [...directCands].sort((a, b) => b.disc - a.disc);
        const discChart = sorted.map((r) => ({ name: r.label, disc: +r.disc, score: r.score, drug: r.drug, discConfidence: r.discConfidence ?? null }));
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
              <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>Discrimination Ratio; Direct Detection</div>
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
            <div style={{ position: "relative", paddingLeft: "120px", paddingRight: "72px" }}>
              {/* Threshold vertical lines */}
              {thresholds.map(t => {
                const pct = (t.val / maxDisc) * 100;
                return (
                  <div key={t.val} style={{ position: "absolute", left: `calc(120px + (100% - 192px) * ${pct / 100})`, top: 0, bottom: 0, width: 0, borderLeft: `1.5px dashed ${t.color}33`, zIndex: 0, pointerEvents: "none" }}>
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
                    <div key={d.name} style={{ display: "flex", alignItems: "center", height: "30px", marginLeft: "-120px", marginRight: "-72px" }}>
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
                      {/* Value + CI */}
                      <div style={{ width: "72px", flexShrink: 0, textAlign: "right", fontSize: "11px", fontFamily: FONT, fontWeight: 600, color: statusColor }}>
                        {d.disc.toFixed(1)}×
                        {d.discConfidence != null && (() => {
                          const ciWidth = d.disc * (1 - d.discConfidence);
                          return <span style={{ fontSize: "9px", fontWeight: 400, color: T.textTer, marginLeft: "2px" }}>{"\u00B1"}{ciWidth.toFixed(1)}</span>;
                        })()}
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
                  {bestDisc ? ` Highest: ${bestDisc.name} at ${bestDisc.disc.toFixed(1)}x, likely a seed-region mismatch (positions 1-8).` : ""}
                  {worstDisc ? ` Lowest: ${worstDisc.name} at ${worstDisc.disc.toFixed(1)}x${worstDisc.disc < 2 ? ", insufficient for any detection method, SM enhancement required." : worstDisc.disc < 3 ? ", acceptable but not diagnostic-grade." : "."}` : ""}
                  {below2.length > 0 ? ` ${below2.length} candidate${below2.length > 1 ? "s" : ""} (${below2.map(d => d.name).slice(0, 3).join(", ")}${below2.length > 3 ? "…" : ""}) fall below the 2x minimum. These have PAM-distal mismatches and require synthetic mismatch engineering.` : " All candidates meet the 2x minimum detection threshold."}
                  {excellent > 0 ? ` ${excellent} candidate${excellent > 1 ? "s" : ""} ${excellent > 1 ? "achieve" : "achieves"} excellent (≥ 10×) discrimination, suitable for lateral-flow deployment.` : ""}
                </div>
              );
            })()}
          </div>
        );
      })()}

      {/* Ranking table; Direct only */}
      <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", overflow: "hidden", marginBottom: "24px" }}>
        <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING, padding: "16px 20px", borderBottom: `1px solid ${T.border}` }}>Discrimination Ranking; Direct Detection</div>
        <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", minWidth: 820 }}>
          <thead>
            <tr style={{ background: T.bgSub }}>
              {["Rank", "Target", "Drug", "Discrimination", "Mismatch", "MM Pos", "Activity", "Signal", "Status"].map(h => (
                <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 600, color: T.textSec, borderBottom: `1px solid ${T.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...directCands].sort((a, b) => b.disc - a.disc).map((r, i) => {
              const mmInfo = deriveMismatch(r);
              const isTV = isMismatchTransversion(r);
              return (
              <React.Fragment key={r.label}>
              <tr style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: T.textTer }}>{i + 1}</td>
                <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600, fontSize: "11px" }}>{r.label}</td>
                <td style={{ padding: "10px 14px" }}><DrugBadge drug={r.drug} /></td>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: r.pamDisrupted ? "#7c3aed" : r.disc >= 3 ? T.success : r.disc >= 2 ? T.warning : T.danger }}>
                  {r.pamDisrupted ? <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}><span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, background: "#7c3aed18", color: "#7c3aed", border: "1px solid #7c3aed33" }}>PAM {"\u221E"}</span></span> : <>{typeof r.disc === "number" ? r.disc.toFixed(1) : r.disc}×{r.discConfidence != null && <span style={{ fontSize: "9px", fontWeight: 400, color: T.textTer, marginLeft: "3px" }}>{"\u00B1"}{(r.disc * (1 - r.discConfidence)).toFixed(1)}</span>}</>}
                </td>
                {/* Mismatch type classification (Feature 2) */}
                <td style={{ padding: "10px 14px", fontSize: "10px" }}>
                  {mmInfo ? (
                    <span style={{ fontFamily: MONO, fontWeight: 600, color: isTV ? T.success : T.warning }}>
                      {mmInfo}
                    </span>
                  ) : "\u2014"}
                </td>
                <td style={{ padding: "10px 14px", fontSize: "10px", fontFamily: MONO }}>
                  {(() => {
                    if (!r.spacer || !r.wtSpacer) return "\u2014";
                    for (let i = 0; i < Math.min(r.spacer.length, r.wtSpacer.length); i++) {
                      if (r.spacer[i] !== r.wtSpacer[i]) {
                        const pos = i + 1;
                        const region = pos <= 8 ? "seed" : "distal";
                        return <span><strong>{pos}</strong> <span style={{ fontSize: "9px", color: T.textTer }}>({region})</span></span>;
                      }
                    }
                    return "\u2014";
                  })()}
                </td>
                <td style={{ padding: "10px 14px", fontFamily: FONT }}>{(r.cnnCalibrated ?? r.score).toFixed(3)}</td>
                <td style={{ padding: "10px 12px", fontFamily: FONT, fontSize: "11px", color: (r.pamAdjusted ?? r.cnnCalibrated ?? r.score ?? 0) < 0.3 ? T.danger : T.textSec }}>
                  {(r.pamAdjusted ?? r.cnnCalibrated ?? r.score ?? 0).toFixed(3)}
                </td>
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
              {/* PAM disruption highlight row (Feature 3) */}
              {r.pamDisrupted && (
                <tr style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                  <td colSpan={9} style={{ padding: "6px 14px 6px 42px", background: "#7c3aed08" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <Shield size={12} color="#7c3aed" strokeWidth={2} />
                      <span style={{ fontSize: "10px", fontWeight: 600, color: "#7c3aed" }}>Binary discrimination: SNP disrupts PAM</span>
                      <span style={{ fontSize: "10px", color: T.textTer }}>
                        {r.pamDisruptionType ? `(${r.pamDisruptionType}) ` : ""}The SNP converts a valid TTTV PAM to a non-functional sequence. Cas12a cannot initiate R-loop formation on WT, providing the strongest possible selectivity mechanism.
                      </span>
                    </div>
                  </td>
                </tr>
              )}
              {/* XGBoost feature vector display (Feature 4) */}
              {r.discFeatureVector && (
                <tr style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                  <td colSpan={9} style={{ padding: "0 14px 0 42px" }}>
                    <button
                      onClick={() => setExpandedFeatures(prev => ({ ...prev, [r.label]: !prev[r.label] }))}
                      style={{ background: "none", border: "none", cursor: "pointer", padding: "6px 0", display: "flex", alignItems: "center", gap: "4px", fontSize: "10px", color: T.textTer }}
                    >
                      <ChevronRight size={10} style={{ transform: expandedFeatures[r.label] ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 150ms ease" }} />
                      <span style={{ fontWeight: 500 }}>Feature vector ({Object.keys(r.discFeatureVector).length} features)</span>
                    </button>
                    {expandedFeatures[r.label] && (
                      <div style={{ padding: "0 0 10px 14px", display: "flex", flexWrap: "wrap", gap: "4px 12px" }}>
                        {Object.entries(r.discFeatureVector)
                          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                          .map(([feat, val]) => (
                            <div key={feat} style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "10px", fontFamily: MONO }}>
                              <span style={{ color: T.textSec, fontWeight: 500 }}>{feat}:</span>
                              <span style={{ fontWeight: 600, color: Math.abs(val) > 0.5 ? T.primary : T.textTer }}>{typeof val === "number" ? val.toFixed(3) : val}</span>
                            </div>
                          ))}
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
      </div>

      {/* Proximity / AS-RPA section */}

      {proximityCands.length > 0 && (
        <div style={{ background: T.bg, border: `1px solid ${T.purple}33`, borderRadius: "4px", overflow: "hidden" }}>
          <div style={{ fontSize: "14px", fontWeight: 600, color: T.purple, fontFamily: HEADING, padding: "16px 20px", borderBottom: `1px solid ${T.purple}33` }}>
            AS-RPA Discrimination: Proximity Detection
            <span style={{ fontSize: "11px", fontWeight: 400, color: T.textTer, marginLeft: "10px" }}>{proximityCands.length} candidates (primer-based discrimination)</span>
          </div>
          <div style={{ padding: "16px 20px 8px", fontSize: "12px", color: T.textSec, lineHeight: 1.6 }}>
            These candidates use <strong>allele-specific RPA primers</strong> for discrimination. The crRNA binds outside the mutation site.
            Discrimination is provided by preferential primer extension on the mutant template.
            {proximityCands.some(r => r.asrpaDiscrimination) && (
              <span> Thermodynamic estimates below are based on 3′ terminal mismatch identity and penultimate mismatch design.</span>
            )}
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
            <thead>
              <tr style={{ background: T.bgSub }}>
                {["Target", "Drug", "Distance", "Activity", "3′ Mismatch", "Penult. MM", "Disc. Ratio", "Block", "Selectivity", "Primers"].map(h => (
                  <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 600, color: T.textSec, borderBottom: `1px solid ${T.borderLight}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {proximityCands.map((r) => {
                const d = r.asrpaDiscrimination;
                const blockColor = d?.block_class === "strong" ? T.success : d?.block_class === "moderate" ? T.warning : T.danger;
                return (
                  <React.Fragment key={r.label}>
                  <tr style={{ borderBottom: d ? "none" : `1px solid ${T.borderLight}` }}>
                    <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600, fontSize: "11px" }}>{r.label}</td>
                    <td style={{ padding: "10px 14px" }}><DrugBadge drug={r.drug} /></td>
                    <td style={{ padding: "10px 14px", fontFamily: FONT, color: T.purple }}>{r.proximityDistance ? `${r.proximityDistance} bp` : "\u2014"}</td>
                    <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: (r.cnnCalibrated ?? r.score) >= 0.7 ? T.success : (r.cnnCalibrated ?? r.score) >= 0.4 ? T.warning : T.danger }}>{(r.cnnCalibrated ?? r.score).toFixed(3)}</td>
                    <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600 }}>{d?.terminal_mismatch || "\u2014"}</td>
                    <td style={{ padding: "10px 14px", fontSize: "11px" }}>{d ? (d.has_penultimate_mm ? <span style={{ color: T.success, fontWeight: 600 }}>Yes</span> : <span style={{ color: T.textTer }}>No</span>) : "\u2014"}</td>
                    <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: d ? (d.block_class === "none" ? T.danger : d.disc_ratio >= 50 ? T.success : d.disc_ratio >= 10 ? T.warning : T.danger) : T.textTer }}>
                      {d ? (d.block_class === "none" ? "1\u00D7 (WC)" : d.disc_ratio >= 100 ? "\u2265100\u00D7" : `${d.disc_ratio.toFixed(0)}\u00D7`) : "\u2014"}
                    </td>
                    <td style={{ padding: "10px 14px" }}>
                      {d ? (d.block_class === "none"
                        ? <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, background: "#FEF2F2", color: T.danger, textTransform: "uppercase" }}>NO DISC</span>
                        : <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, background: blockColor + "20", color: blockColor, textTransform: "uppercase" }}>{d.block_class}</span>
                      ) : "\u2014"}
                    </td>
                    <td style={{ padding: "10px 12px", fontFamily: FONT, fontWeight: 600, fontSize: "11px", color: (d?.estimated_specificity ?? 0) >= 0.98 ? "#16a34a" : (d?.estimated_specificity ?? 0) >= 0.90 ? "#d97706" : T.danger }}>
                      {d?.estimated_specificity != null ? `${(d.estimated_specificity * 100).toFixed(0)}%` : "\u2014"}
                    </td>
                    <td style={{ padding: "10px 14px" }}>
                      {d?.block_class === "none"
                        ? <Badge variant="danger">Not viable</Badge>
                        : <Badge variant={r.hasPrimers ? "success" : "danger"}>{r.hasPrimers ? "AS-RPA" : "No primers"}</Badge>}
                    </td>
                  </tr>
                  {/* AS-RPA thermodynamic details (Feature 8) */}
                  {d && (
                    <tr style={{ borderBottom: `1px solid ${T.borderLight}` }}>
                      <td colSpan={10} style={{ padding: "4px 14px 10px 14px", background: `${T.purple}06` }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                          <Zap size={11} color="#7c3aed" strokeWidth={2} />
                          <span style={{ fontSize: "10px", fontWeight: 600, color: "#7c3aed" }}>Discrimination mechanism:</span>
                        </div>
                        <div style={{ fontSize: "10px", color: T.text, lineHeight: 1.6 }}>
                          <span style={{ fontWeight: 600 }}>3{"\u2032"} mismatch:</span>{" "}
                          <span style={{ fontFamily: MONO, fontWeight: 600 }}>{d.terminal_mismatch || "N/A"}</span>
                          {d.terminal_mismatch && (
                            <span style={{ color: T.textSec, marginLeft: "4px" }}>
                              ({d.block_class === "none" ? "Watson-Crick pair, no blocking"
                                : (() => {
                                  const mm = (d.terminal_mismatch || "").split(":");
                                  const purines = new Set(["A", "G"]);
                                  const b0 = mm[0] || "", b1 = mm[1] || "";
                                  const bothPurine = purines.has(b0) && purines.has(b1);
                                  const isWobble = (b0 === "G" && b1 === "T") || (b0 === "T" && b1 === "G") || (b0 === "U" && b1 === "G") || (b0 === "G" && b1 === "U");
                                  const label = bothPurine ? "purine\u00B7purine clash" : isWobble ? "wobble pair" : b0 === b1 ? "homo-mismatch" : "transversion";
                                  const strength = d.block_class === "strong" ? "strong steric block" : d.block_class === "moderate" ? "moderate block" : "weak block";
                                  return `${label}, ${strength}`;
                                })()})
                            </span>
                          )}
                          <span style={{ margin: "0 8px", color: T.borderLight }}>{"\u2502"}</span>
                          <span style={{ fontWeight: 600 }}>Penultimate MM:</span>{" "}
                          <span style={{ fontWeight: 600 }}>{d.has_penultimate_mm ? "Engineered" : "None"}</span>
                          {d.has_penultimate_mm && <span style={{ color: T.textSec, marginLeft: "4px" }}>(destabilises WT extension)</span>}
                          <span style={{ margin: "0 8px", color: T.borderLight }}>{"\u2502"}</span>
                          <span style={{ fontWeight: 600 }}>Selectivity:</span>{" "}
                          <span style={{ fontWeight: 600 }}>
                            {d.estimated_specificity != null ? `${(d.estimated_specificity * 100).toFixed(0)}%` : d.disc_ratio >= 100 ? "\u226599%" : `~${Math.round((1 - 1/d.disc_ratio) * 100)}%`}
                          </span>
                          <span style={{ color: T.textSec, marginLeft: "4px" }}>(allele-specific amplification selectivity)</span>
                          {d.delta_g != null && (
                            <>
                              <span style={{ margin: "0 8px", color: T.borderLight }}>{"\u2502"}</span>
                              <span style={{ fontWeight: 600 }}>{"\u0394\u0394"}G:</span>{" "}
                              <span style={{ fontFamily: MONO, fontWeight: 600 }}>{typeof d.delta_g === "number" ? d.delta_g.toFixed(1) : d.delta_g} kcal/mol</span>
                            </>
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
          {proximityCands.some(r => r.asrpaDiscrimination?.block_class === "none") && (
            <div style={{ padding: "12px 20px", fontSize: "11px", color: T.danger, background: "#FEF2F2", borderTop: `1px solid #FECACA` }}>
              <strong>Panel gap:</strong> {proximityCands.filter(r => r.asrpaDiscrimination?.block_class === "none").map(r => r.label).join(", ")}. Primer 3' base forms a Watson-Crick pair with the WT template (no mismatch = no discrimination).
              These targets require primer strand reversal, alternative SNP base selection, or a different discrimination strategy.
            </div>
          )}
          {proximityCands.some(r => r.asrpaDiscrimination) && (
            <div style={{ padding: "12px 20px", fontSize: "10px", color: T.textTer, borderTop: `1px solid ${T.purple}15` }}>
              Thermodynamic estimates, not experimentally validated. Ratios from Boltzmann conversion exp(ΔΔG/RT) at 37 °C, capped at 100× (empirical AS-RPA discrimination typically 10–100×; Ye et al. 2019).
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
      <InSilicoCaveat />
      {/* RPA Explanation */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}33`, borderRadius: "4px", padding: mobile ? "16px" : "20px 24px", marginBottom: "24px", display: "flex", gap: "14px", alignItems: "flex-start" }}>
        <Crosshair size={20} color={T.primaryDark} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <div style={{ fontSize: "14px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING, marginBottom: "4px" }}>Recombinase Polymerase Amplification (RPA)</div>
          <p style={{ fontSize: "13px", color: T.primaryDark, lineHeight: 1.6, margin: 0, opacity: 0.85 }}>
            RPA is an isothermal amplification method (37°C) that replaces PCR thermocycling. Each crRNA target needs a pair of
            25–35 nt primers flanking an 80–120 bp amplicon containing the crRNA binding site. The amplified product is then
            detected by Cas12a trans-cleavage of MB-ssDNA reporters on the electrochemical platform (SWV signal decrease on LIG-E (cellulose-derived) electrodes).
          </p>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "8px", marginTop: "10px", padding: "8px 12px", background: "rgba(255,255,255,0.5)", borderRadius: "4px", border: `1px solid ${T.primary}22` }}>
            <Droplet size={14} color={T.primaryDark} strokeWidth={2} style={{ flexShrink: 0, marginTop: 1 }} />
            <span style={{ fontSize: "11px", color: T.primaryDark, lineHeight: 1.5, opacity: 0.85 }}>
              Capped at 120 bp; cfDNA fragments in blood are ~100–160 bp (median ~140 bp). Shorter amplicons maximise template capture from fragmented circulating DNA.
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
            so allele discrimination comes from Cas12a mismatch intolerance; not from primers. Primers simply amplify the region
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
              <strong style={{ color: T.purple }}>PAM desert.</strong> These targets lack a TTTV PAM within the spacer window overlapping the SNP; common in M. tuberculosis (65.6% GC).
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

      {/* Locus grouping summary */}
      {(() => {
        const locusGroups = {};
        withPrimers.forEach(r => {
          if (r.locusGroup) {
            if (!locusGroups[r.locusGroup]) locusGroups[r.locusGroup] = [];
            locusGroups[r.locusGroup].push(r.label);
          }
        });
        const sharedLoci = Object.entries(locusGroups).filter(([, targets]) => targets.length > 1);
        const uniquePairs = withPrimers.length - sharedLoci.reduce((sum, [, t]) => sum + t.length - 1, 0);
        if (sharedLoci.length > 0) return (
          <div style={{ background: "#F0F9FF", border: "1px solid #BAE6FD", borderRadius: "4px", padding: "16px 20px", marginBottom: "16px", display: "flex", gap: "12px", alignItems: "flex-start" }}>
            <Layers size={16} color="#0284C7" style={{ flexShrink: 0, marginTop: 2 }} />
            <div>
              <div style={{ fontSize: "13px", fontWeight: 600, color: "#0284C7", fontFamily: HEADING, marginBottom: "4px" }}>
                Shared Amplicon Architecture: {withPrimers.length} targets, {uniquePairs} primer pairs
              </div>
              <p style={{ fontSize: "12px", color: T.textSec, lineHeight: 1.6, margin: 0 }}>
                Targets at the same genomic locus share a single RPA amplicon. Each amplicon is distributed to separate detection wells,
                each pre-loaded with a target-specific crRNA. This reduces the RPA multiplex from {withPrimers.length}-plex to {uniquePairs}-plex,
                within validated range for isothermal amplification.
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginTop: "8px" }}>
                {sharedLoci.map(([locus, targets]) => (
                  <div key={locus} style={{ display: "flex", alignItems: "center", gap: "4px", padding: "4px 8px", background: "rgba(255,255,255,0.7)", borderRadius: "4px", border: "1px solid #BAE6FD" }}>
                    <span style={{ fontSize: "11px", fontWeight: 600, color: "#0284C7" }}>{locus.replace("_locus", "")}:</span>
                    <span style={{ fontSize: "11px", color: T.textSec }}>{targets.length} targets, 1 primer pair</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
        return null;
      })()}

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
                : r.disc > 0 ? `${r.disc.toFixed(1)}×${r.hasSM ? " (post-SM)" : ""}` : "–";
              return (
              <tr key={r.label} style={{ borderBottom: `1px solid ${T.borderLight}`, transition: "background 0.15s", background: isHov ? `${T.primary}08` : "transparent" }}
                onMouseEnter={() => setHoveredRow(r.label)} onMouseLeave={() => setHoveredRow(null)}>
                <td style={{ padding: "10px 14px", fontFamily: MONO, fontWeight: 600, fontSize: "11px" }}>
                  {r.label}
                  {r.sharedAmpliconTargets && r.sharedAmpliconTargets.length > 0 && (
                    <span title={"Shared amplicon with " + r.sharedAmpliconTargets.join(", ")} style={{ marginLeft: "4px", display: "inline-flex", alignItems: "center", gap: "2px", padding: "1px 4px", background: "#E0F2FE", borderRadius: "3px", fontSize: "9px", color: "#0284C7", fontWeight: 600, verticalAlign: "middle" }}>
                      <Layers size={8} /> {r.sharedAmpliconTargets.length + 1}
                    </span>
                  )}
                </td>
                <td style={{ padding: "10px 14px" }}>
                  <Badge variant={r.strategy === "Direct" ? "success" : "purple"}>
                    {r.strategy === "Direct" ? "Standard" : "AS-RPA"}
                  </Badge>
                  {r.sharedLocusWithAsrpa && (
                    <div style={{ fontSize: "9px", color: "#0284C7", marginTop: "2px" }}>+ shared std</div>
                  )}
                </td>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, fontSize: "11px", color: r.gene === "IS6110" ? T.textTer : r.strategy === "Proximity" ? T.purple : r.disc >= 3 ? T.success : r.disc >= 2 ? T.warning : T.danger }}>
                  {discVal}
                </td>
                <td style={{ padding: "10px 14px" }}>
                  {(() => {
                    const fwdSeq = r.sharedLocusWithAsrpa && r.asrpaFwd ? r.asrpaFwd : r.fwd;
                    return isHov ? <Seq s={fwdSeq} /> : <span style={{ fontFamily: MONO, fontSize: "11.5px", letterSpacing: "1.2px", color: r.sharedLocusWithAsrpa ? T.purple : T.textTer }}>{fwdSeq}</span>;
                  })()}
                  {r.sharedLocusWithAsrpa && r.asrpaFwd && (
                    <div style={{ fontSize: "9px", color: "#0284C7", marginTop: "1px" }}>AS primer (allele-specific)</div>
                  )}
                </td>
                <td style={{ padding: "10px 14px" }}>
                  {(() => {
                    const revSeq = r.sharedLocusWithAsrpa && r.asrpaRev ? r.asrpaRev : r.rev;
                    return isHov ? <Seq s={revSeq} /> : <span style={{ fontFamily: MONO, fontSize: "11.5px", letterSpacing: "1.2px", color: r.sharedLocusWithAsrpa ? T.purple : T.textTer }}>{revSeq}</span>;
                  })()}
                </td>
                <td style={{ padding: "10px 14px", fontFamily: FONT, fontWeight: 600, color: (r.sharedLocusWithAsrpa ? r.asrpaAmpliconLength : r.amplicon) <= 100 ? T.success : (r.sharedLocusWithAsrpa ? r.asrpaAmpliconLength : r.amplicon) <= 120 ? T.warning : T.danger }}>
                  {r.sharedLocusWithAsrpa && r.asrpaAmpliconLength ? r.asrpaAmpliconLength : r.amplicon} bp
                </td>
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

  // ═══════════ PREDICTED ELECTROCHEMICAL READOUT; State ═══════════
  const [echemCandidate, setEchemCandidate] = useState("rpoB_S531L");
  const [echemTechnique, setEchemTechnique] = useState("SWV");
  const [echemTime, setEchemTime] = useState(30);       // minutes
  const [echemBloodTiter, setEchemBloodTiter] = useState(100); // cp/mL
  const [echemKtrans, setEchemKtrans] = useState(0.02); // s⁻¹ (mid-range estimate for LIG-E surface trans-cleavage on cellulose, 0.01–0.1 s⁻¹)
  const [echemAdvanced, setEchemAdvanced] = useState(false);
  const [echemGamma0, setEchemGamma0] = useState(1.5e11); // molecules/cm²
  const [echemPorosity, setEchemPorosity] = useState(3);
  const [echemIscale, setEchemIscale] = useState(3.0);    // μA
  const [echemShowFwdRev, setEchemShowFwdRev] = useState(false); // SWV i_fwd/i_rev toggle
  const [echemArch, setEchemArch] = useState("C"); // Reporter architecture: A=pAP/ALP, B=Silver, C=MB

  const kinetics = poolData?.kinetics || {
    phases: [
      { phase: "RPA amplification", solution_bound: "15\u201320 min", on_electrode: "15\u201320 min", description: "Low-plex shared RPA in upstream chambers (2\u20133 chambers, 3\u20134 primer pairs each). Amplicons distributed to detection wells by capillary flow.", is_bottleneck: false },
      { phase: "crRNA rehydration", solution_bound: "N/A", on_electrode: "2\u20135 min", description: "Amplicon reconstitutes dried crRNA + Cas12a in each detection well", is_bottleneck: false },
      { phase: "RNP formation", solution_bound: "0.5\u20131 min", on_electrode: "2\u20135 min", description: "Cas12a + crRNA form active RNP (in situ complexation). Amplicons already present in excess (>10\u2079 copies post-RPA).", is_bottleneck: false },
      { phase: "Target recognition + cis-cleavage", solution_bound: "~10 sec", on_electrode: "1\u20133 min", description: "RNP scans amplicons for PAM, R-loop forms, cis-cleavage activates trans-cleavage", is_bottleneck: false },
      { phase: "Surface trans-cleavage", solution_bound: "~5 min", on_electrode: "10\u201320 min", description: "Activated Cas12a cleaves pyrene-NHS-tethered ssDNA-MB reporters on LIG-E WE", is_bottleneck: true },
    ],
    totals: {
      detection_solution: "~6\u20138 min", detection_electrode: "15\u201330 min",
      rpa_time: "15\u201320 min", total_solution: "~23\u201328 min", total_electrode: "30\u201350 min",
      who_tpp_target: "< 120 min", who_tpp_pass: true,
    },
    parameters: [
      { param: "k_form (RNP association)", value: "1.75 \u00d7 10\u2075 M\u207b\u00b9s\u207b\u00b9", source: "Lesinski et al. 2024", note: "SPR measurement; analogous to on-pad scenario but different surface than LIG" },
      { param: "k_off (RNP dissociation)", value: "1.87 \u00d7 10\u207b\u2074 s\u207b\u00b9", source: "Lesinski et al. 2024", note: null },
      { param: "k_cis (cis-cleavage)", value: "0.03 s\u207b\u00b9", source: "Lesinski et al. 2024", note: null },
      { param: "k_trans (solution, free ssDNA)", value: "~2.0 s\u207b\u00b9", source: "Nalefski et al. 2021", note: "Free ssDNA in solution. NOT applicable to surface-tethered reporters." },
      { param: "k_trans (surface, estimated)", value: "0.01\u20130.1 s\u207b\u00b9", source: "Estimated", note: "Key experimental unknown." },
      { param: "[Cas12a]", value: "50 nM", source: "Design parameter", note: null },
      { param: "[crRNA] on pad", value: "~200 nM equivalent", source: "Design parameter", note: "Effective concentration after rehydration unknown." },
      { param: "MB-ssDNA probe density", value: "~10\u00b9\u2070\u201310\u00b9\u00b9 molecules/cm\u00b2", source: "Estimated for LIG", note: "Geometric density (estimated). Effective density is higher due to LIG-E porosity (3\u201310\u00d7 surface area). Directly affects signal magnitude and time-to-detection." },
    ],
    insights: [
      { title: "Rate-limiting step", text: "Surface trans-cleavage of pyrene-NHS-tethered ssDNA-MB reporters on LIG-E dominates detection time, not RNP formation or target recognition." },
      { title: "Shared amplicon architecture", text: "Targets at the same gene locus (e.g. rpoB S531L/H526Y/D516V) share one RPA amplicon from a common primer pair. The amplicon is distributed to separate detection wells, each with a target-specific crRNA. This reduces the RPA multiplex while maintaining detection plex." },
      { title: "In situ complexation", text: "Lesinski et al. 2024: dried Cas12a + crRNA reconstitute when amplicon solution arrives from upstream RPA chamber. RNP forms over ~15 min while amplicons are already in massive excess (>10\u2079 copies). No kinetic race between amplification and detection." },
      { title: "Reporter tethering", text: "ssDNA-MB reporters anchored to LIG-E via pyrene-NHS (PBASE): pyrene \u03c0-\u03c0 stacks on graphene surface (~74 kJ/mol binding), NHS ester covalently couples to amine-modified ssDNA. Stable under SWV cycling." },
      { title: "Experimental unknowns", text: "k_trans on LIG-tethered MB-ssDNA, crRNA rehydration kinetics, and capillary distribution uniformity across 14 wells have not been measured. These are key characterisation priorities." },
      { title: "Capacitive background", text: "SWV simulation models Faradaic current only. Real LIG-E electrodes have capacitive (non-Faradaic) baseline from double-layer charging on high-surface-area graphene foam. Signal-to-noise ratio in practice depends on the Faradaic-to-capacitive current ratio." },
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

  // Electrode layout; 7×2 grid (14-plex: 12 resistance mutations + IS6110 + RNaseP)
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

  // ═══════════ PREDICTED ELECTROCHEMICAL READOUT; Physics Engine ═══════════
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
      label: "MB", species: "surface-confined",
      reference: "surface-confined model",
      E0: -0.22, n: 2, // MB: 2e⁻, 2H⁺ reduction
      E_start: -0.05, E_end: -0.40,
      E_sw: 0.025, E_pulse: 0.050, frequency: 50, step: 0.004,
      scan_rate: 0.05, signal_direction: "off",
      peak_label: "MB reduction",
      peak_shape: "sech2", // surface-confined
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

  // Γ(t); exact integral with in situ RNP formation
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
  // Arch C (MB): sech² for surface-confined species
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

  // Unified SWV compute; architecture-aware
  const computeSWV = useCallback((E_array, Gamma) => {
    const { n, F, R, Temp, E0, E_sw } = ECHEM;
    const nFRT = n * F / (R * Temp);
    const scale = echemIscale * echemAeff * archCfg.I_scale_base;
    const ratio = Gamma / echemGamma0_mol;
    if (archCfg.peak_shape === "asymmetric") {
      // Arch A: pAP oxidation; asymmetric irreversible peak
      return E_array.map(E => scale * ratio * peakShape_asymmetric(E, E0, archCfg.alpha, 1, F, R, Temp));
    }
    if (archCfg.peak_shape === "stripping") {
      // Arch B: Ag stripping; asymmetric Gaussian
      return E_array.map(E => scale * ratio * peakShape_stripping(E, E0, archCfg.sigma_onset, archCfg.sigma_tail));
    }
    // Arch C: MB surface-confined sech² (positive peaks for SWV net current)
    return E_array.map(E => scale * ratio * peakShape_sech2(E, E0, nFRT, E_sw));
  }, [echemIscale, echemAeff, echemGamma0_mol, echemArch]);

  // SWV forward/reverse components for toggle display (Arch C only; other architectures use net only)
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

  // Unified DPV compute; architecture-aware
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
    // Arch C: MB; DPV uses difference of Nernst equilibria, same sech² family
    return E_array.map(E => scale * ratio * peakShape_sech2(E, E0, nFRT, E_pulse / 2));
  }, [echemAeff, echemGamma0_mol, echemArch]);

  // CV voltammogram; architecture-aware, scale-based (positive peaks for consistent peak detection)
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

  // CV duck-shape: forward + reverse scan; architecture-aware with visible Faradaic peaks
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
      // Arch C (MB): surface-confined; symmetric cathodic/anodic sech² peaks
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
      // Arch A (pAP): irreversible oxidation; forward peak, no reverse peak
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

  // Potential array; architecture-dependent range
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
      <InSilicoCaveat />

      {/* ── Explainer box ── */}
      <div style={{ background: T.primaryLight, border: `1px solid ${T.primary}25`, borderRadius: "4px", padding: mobile ? "14px" : "16px 20px", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <Grid3x3 size={16} color={T.primary} strokeWidth={1.8} />
          <span style={{ fontSize: "13px", fontWeight: 600, color: T.primaryDark, fontFamily: HEADING }}>Multiplex Engineering</span>
        </div>
        <div style={{ fontSize: "13px", color: T.primaryDark, lineHeight: 1.6 }}>
          {(() => {
            const locusGroups = {};
            results.forEach(r => { if (r.locusGroup) { if (!locusGroups[r.locusGroup]) locusGroups[r.locusGroup] = []; locusGroups[r.locusGroup].push(r.label); } });
            const sharedLoci = Object.entries(locusGroups).filter(([, t]) => t.length > 1);
            const uniquePairs = results.filter(r => r.hasPrimers).length - sharedLoci.reduce((s, [, t]) => s + t.length - 1, 0);
            const nChambers = Math.ceil(uniquePairs / 4);
            return `Low-plex shared RPA (${uniquePairs} primer pairs in ${nChambers} chambers) feeding ${results.length} spatially-isolated detection wells, each pre-loaded with target-specific crRNA + Cas12a. Amplicon distribution via capillary channels on cellulose. Each well contains a LIG working electrode with pyrene-NHS-tethered ssDNA-MB reporters. Shared CE + RE. SWV readout via multiplexed potentiostat. Detection of ${drugs.length} drug resistance classes from a single blood sample.`;
          })()}
        </div>
      </div>

      {/* ── Status banner ── */}
      <div style={{ background: "#FFFBEB", border: "1px solid #F59E0B44", borderRadius: "4px", padding: "10px 16px", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
        <AlertTriangle size={14} color="#D97706" strokeWidth={2} />
        <span style={{ fontSize: "12px", fontWeight: 500, color: "#92400E", fontFamily: FONT }}>Electrochemical predictions below are exploratory. Absolute signal values depend on reporter chemistry, surface functionalization, and electrode-specific parameters that require experimental characterization. Curves will be calibrated once the reporter architecture is selected and validated on the LIG-E platform.</span>
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

      {/* ═══════════ SECTION 4b: SA Optimization Convergence ═══════════ */}
      {panelData?.optimizer_score_trace && panelData.optimizer_score_trace.length > 0 && (() => {
        const trace = panelData.optimizer_score_trace;
        const trN = trace.length;
        const iterStep = 100;
        const totalIter = trN * iterStep;
        const trYMin = Math.min(...trace);
        const trYMax = Math.max(...trace);
        const trYPad = (trYMax - trYMin) * 0.1 || 0.01;
        const trMg = { top: 20, right: 16, bottom: 34, left: 52 };
        const trW = 520, trH = 200, trPW = trW - trMg.left - trMg.right, trPH = trH - trMg.top - trMg.bottom;
        const trXS = i => trMg.left + (i / Math.max(trN - 1, 1)) * trPW;
        const trYS = v => trMg.top + ((trYMax + trYPad - v) / (trYMax - trYMin + 2 * trYPad)) * trPH;
        const trPolyline = trace.map((v, i) => `${trXS(i).toFixed(1)},${trYS(v).toFixed(1)}`).join(" ");
        const trYTicks = Array.from({ length: 5 }, (_, i) => trYMin + (trYMax - trYMin) * (i / 4));
        return (
          <CollapsibleSection title="Optimization Convergence" defaultOpen={false} badge={{ text: `${totalIter.toLocaleString()} iter`, bg: T.primaryLight, color: T.primary }}>
            <div style={{ padding: "0", marginBottom: "16px" }}>
              <p style={{ fontSize: "12px", color: T.textSec, marginBottom: "12px", lineHeight: 1.6 }}>
                Simulated annealing convergence ({totalIter.toLocaleString()} iterations). The panel score trace is sampled every {iterStep} iterations.
                Higher scores indicate better multiplex panel configurations.
              </p>
              <div style={{ width: "100%", maxWidth: 560 }}>
                <svg viewBox={`0 0 ${trW} ${trH}`} width="100%" height="100%" style={{ fontFamily: MONO }}>
                  <rect width={trW} height={trH} fill="#FAFAFA" rx="6" />
                  {[0.25, 0.5, 0.75].map(f => <line key={`gx${f}`} x1={trMg.left + trPW * f} y1={trMg.top} x2={trMg.left + trPW * f} y2={trMg.top + trPH} stroke="#E8E8E8" strokeWidth="0.5" />)}
                  {trYTicks.map((v, i) => <line key={`gy${i}`} x1={trMg.left} y1={trYS(v)} x2={trMg.left + trPW} y2={trYS(v)} stroke="#E8E8E8" strokeWidth="0.5" />)}
                  <line x1={trMg.left} y1={trMg.top} x2={trMg.left} y2={trMg.top + trPH} stroke="#444" strokeWidth="1.5" />
                  <line x1={trMg.left} y1={trMg.top + trPH} x2={trMg.left + trPW} y2={trMg.top + trPH} stroke="#444" strokeWidth="1.5" />
                  <polyline points={trPolyline} fill="none" stroke={T.primary} strokeWidth="2" strokeLinejoin="round" />
                  <circle cx={trXS(trN - 1)} cy={trYS(trace[trN - 1])} r="3.5" fill={T.primary} stroke="#fff" strokeWidth="1.5" />
                  {[0, 0.25, 0.5, 0.75, 1].map(f => {
                    const iter = Math.round(f * totalIter);
                    return <text key={f} x={trMg.left + f * trPW} y={trMg.top + trPH + 14} textAnchor="middle" fill="#666" fontSize="8">{iter >= 1000 ? `${(iter / 1000).toFixed(iter % 1000 === 0 ? 0 : 1)}k` : iter}</text>;
                  })}
                  {trYTicks.map((v, i) => <text key={i} x={trMg.left - 5} y={trYS(v) + 3} textAnchor="end" fill="#666" fontSize="8">{v.toFixed(3)}</text>)}
                  <text x={trMg.left + trPW / 2} y={trH - 3} textAnchor="middle" fill="#444" fontSize="10">Iteration</text>
                  <text x={12} y={trMg.top + trPH / 2} textAnchor="middle" fill="#444" fontSize="10" transform={`rotate(-90, 12, ${trMg.top + trPH / 2})`}>Panel Score</text>
                  <text x={trXS(trN - 1) - 6} y={trYS(trace[trN - 1]) - 8} textAnchor="end" fill={T.primary} fontSize="9" fontWeight="600">{trace[trN - 1].toFixed(4)}</text>
                </svg>
              </div>
              <div style={{ marginTop: "8px", fontSize: "10px", color: T.textTer, lineHeight: 1.5 }}>
                Initial score: {trace[0].toFixed(4)} {"\u2192"} Final score: {trace[trN - 1].toFixed(4)} ({((trace[trN - 1] - trace[0]) / Math.abs(trace[0]) * 100).toFixed(1)}% improvement)
              </div>
            </div>
          </CollapsibleSection>
        );
      })()}

      {/* ═══════════ SECTION 6: In Situ RNP Formation Kinetics ═══════════ */}
      <CollapsibleSection title="In Situ RNP Formation Kinetics" defaultOpen={false} badge={{ text: kinetics.totals?.total_electrode || "30\u201350 min", bg: "#22c55e20", color: "#22c55e" }}>
        <div style={{ padding: "0", marginBottom: "24px" }}>
          <p style={{ fontSize: "12px", color: T.textSec, marginBottom: "16px", lineHeight: 1.6 }}>
            In situ RNP formation is integral to the per-pad one-pot architecture. Cas12a protein arrives in the sample buffer
            and encounters pad-specific lyophilized crRNA upon rehydration. Gradual RNP formation (Lesinski et al. 2024, Anal. Chem.)
            limits early cis-cleavage competition with RPA; critical at the low template concentrations expected from blood cfDNA.
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
            WHO TPP target: {kinetics.totals?.who_tpp_target || "< 120 min"}. Estimated total: {kinetics.totals?.total_electrode || "30\u201350 min"}; <strong style={{ color: T.success }}>within target</strong> with 2-4× margin.
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
          {/* Header description; architecture-dependent */}
          <p style={{ fontSize: "12px", color: T.textSec, marginBottom: "20px", lineHeight: 1.6 }}>
            <strong>SWV, DPV, and CV curves computed from COMPASS pipeline predictions and analytical electrochemistry for {
              echemArch === "A" ? "enzymatic pAP generation (diffusion-controlled, Bezinge 2023)"
              : echemArch === "B" ? "silver anodic stripping voltammetry (Suea-Ngam 2021)"
              : "surface-confined MB"
            }.</strong> {echemArch === "C"
              ? "Peak shapes follow surface-confined redox theory for adsorbed redox couples. Reporters tethered via pyrene-NHS (PBASE) to LIG-E graphene surface."
              : echemArch === "A"
              ? "Peak shapes follow Nicholson-Shain theory for irreversible diffusion-controlled oxidation."
              : "Peak shapes follow anodic stripping voltammetry dissolution kinetics."
            } Relative peak heights between candidates and between MUT/WT alleles are determined by Compass-ML efficiency and discrimination scores (trained on 15K real measurements). Absolute peak currents and detection times depend on electrode-specific parameters (surface trans-cleavage rate, reporter density) provided as adjustable sliders, to be locked to experimental values after the first electrode characterisation.
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
                      background: echemTechnique === tech ? T.bgHover : T.bg,
                      color: echemTechnique === tech ? T.text : T.textSec,
                      border: `1px solid ${T.border}`,
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
                      background: echemArch === arch.key ? T.bgHover : T.bg,
                      color: echemArch === arch.key ? T.text : T.textSec,
                      border: `1px solid ${T.border}`,
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

            {/* ═══ PANEL A: Voltammogram; negative (cathodic) MB peaks ═══ */}
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

                  // SWV / DPV; architecture-aware positive peaks
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
                      {/* SWV forward/reverse components; blue (fwd), orange (rev) */}
                      {echemShowFwdRev && echemTechnique === "SWV" && data[0].base_fwd != null && (
                        <>
                          <path d={pathD("base_fwd")} fill="none" stroke={EC.blue} strokeWidth="1.5" opacity="0.6" />
                          <path d={pathD("base_rev")} fill="none" stroke={EC.orange} strokeWidth="1.5" opacity="0.6" />
                          <path d={pathD("after_fwd")} fill="none" stroke={EC.blue} strokeWidth="1.2" opacity="0.35" strokeDasharray="4,2" />
                          <path d={pathD("after_rev")} fill="none" stroke={EC.orange} strokeWidth="1.2" opacity="0.35" strokeDasharray="4,2" />
                        </>
                      )}
                      {/* Main curves; i_net (purple) */}
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
              {/* Interpretation block; Panel A */}
              <div style={{ marginTop: "8px", padding: "10px 14px", borderRadius: "4px", background: "#EFF6FF", border: "1px solid #BFDBFE", fontSize: "11px", color: "#1D4ED8", lineHeight: 1.6 }}>
                <strong style={{ color: "#1D4ED8" }}>Interpretation:</strong>{" "}
                {echemMeta.deltaI > 50
                  ? `Strong signal reduction (${"\u0394"}I = ${echemMeta.deltaI}%). Trans-cleavage of MB reporters produces a clearly resolvable voltammetric shift at ${echemTime} min. The ${echemCandidateData.label} crRNA yields sufficient on-target activity for unambiguous electrochemical detection.`
                  : echemMeta.deltaI > 15
                  ? `Moderate signal reduction (${"\u0394"}I = ${echemMeta.deltaI}%). The ${echemCandidateData.label} crRNA generates a detectable but sub-optimal peak shift at ${echemTime} min. Extending incubation time or increasing k_trans via electrode surface optimization would improve signal-to-noise.`
                  : `Weak signal reduction (${"\u0394"}I = ${echemMeta.deltaI}%). At ${echemTime} min, the trans-cleavage signal for ${echemCandidateData.label} is near the detection limit. Consider increasing reporter density, incubation time, or optimizing surface chemistry to enhance k_trans.`
                }
                {echemCandidateData.isProximity
                  ? ` This is a Proximity candidate: allelic discrimination comes from AS-RPA primers (crRNA D = ${echemCandidateData.discrimination.toFixed(1)}\u00d7 is not the operative discrimination mechanism).`
                  : echemCandidateData.discrimination <= 2.0 && echemCandidateData.discrimination < 900
                  ? ` Note: D = ${echemCandidateData.discrimination.toFixed(1)}\u00d7 indicates poor crRNA-level discrimination. Allelic specificity depends on AS-RPA primer selectivity.`
                  : ""
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
                      {/* Positive zone: >30%; green tint */}
                      <rect x={mg.left} y={yS(100)} width={pw} height={yS(30) - yS(100)} fill={EC.green} opacity="0.04" />
                      {/* Indeterminate zone: 5-30%; yellow tint */}
                      <rect x={mg.left} y={yS(30)} width={pw} height={yS(5) - yS(30)} fill="#f59e0b" opacity="0.05" />
                      {/* Negative zone: <5%; red tint */}
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
              {/* Interpretation block; Panel B */}
              <div style={{ marginTop: "8px", padding: "10px 14px", borderRadius: "4px", background: "#EFF6FF", border: "1px solid #BFDBFE", fontSize: "11px", color: "#1D4ED8", lineHeight: 1.6 }}>
                <strong style={{ color: "#1D4ED8" }}>Interpretation:</strong>{" "}
                {echemTimeCourse.timeMut != null && echemTimeCourse.timeMut <= 20
                  ? `Rapid detection: MUT signal crosses the 3\u03c3 threshold at ~${echemTimeCourse.timeMut} min, well within the WHO TPP target of <120 min. `
                  : echemTimeCourse.timeMut != null
                  ? `Detection at ~${echemTimeCourse.timeMut} min. Signal accumulation is slower than ideal. Increasing k_trans or extending RPA amplification could accelerate time-to-result. `
                  : `MUT signal does not reach the 3\u03c3 threshold within 60 min at current k_trans. Surface optimization required. `
                }
                {echemTimeCourse.timeWt != null && echemTimeCourse.timeMut != null && echemTimeCourse.timeWt > echemTimeCourse.timeMut
                  ? `A ${echemTimeCourse.timeWt - echemTimeCourse.timeMut}-min discrimination window separates MUT from WT detection, enabling time-gated allelic discrimination.`
                  : echemTimeCourse.timeWt == null
                  ? `WT remains below threshold throughout, providing clean allelic discrimination at all time points.`
                  : `WT crosses threshold near MUT. Allelic discrimination relies on AS-RPA primer specificity rather than crRNA kinetics.`
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
                  {echemCandidateData.isProximity
                    ? <><strong style={{ color: EC.purple }}>AS-RPA</strong><span style={{ color: T.textTer }}> (crRNA: {echemDiscOverlay.narsilDisc}\u00d7, not used)</span></>
                    : <><strong style={{ color: echemDiscOverlay.measuredDisc < 1 ? "#ef4444" : echemDiscOverlay.measuredDisc < 2 ? "#f59e0b" : EC.purple }}>{echemDiscOverlay.measuredDisc === Infinity ? "\u221e" : `${echemDiscOverlay.measuredDisc}\u00d7`}</strong><span style={{ color: T.textTer }}> (COMPASS: {echemDiscOverlay.narsilDisc >= 900 ? "\u221e" : `${echemDiscOverlay.narsilDisc}\u00d7`})</span></>
                  }
                </div>
                {!echemCandidateData.isProximity && echemDiscOverlay.narsilDisc < 1 && <div style={{ color: "#ef4444", fontWeight: 600 }}>{"\u26a0"} D {"<"} 1: WT activates more than MUT</div>}
                {!echemCandidateData.isProximity && echemDiscOverlay.measuredDisc < 2 && echemDiscOverlay.measuredDisc !== Infinity && echemDiscOverlay.narsilDisc >= 1 && <div style={{ color: "#f59e0b", fontWeight: 600 }}>{"\u26a0"} poor discrimination</div>}
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

                  // SWV / DPV discrimination overlay; architecture-aware
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
              {/* Interpretation block; Panel C */}
              <div style={{ marginTop: "8px", padding: "10px 14px", borderRadius: "4px", background: "#EFF6FF", border: "1px solid #BFDBFE", fontSize: "11px", color: "#1D4ED8", lineHeight: 1.6 }}>
                <strong style={{ color: "#1D4ED8" }}>Interpretation:</strong>{" "}
                {echemCandidateData.isProximity
                  ? `Proximity candidate: allelic discrimination is provided by AS-RPA primers, not crRNA mismatch intolerance. The WT allele is not amplified (blocked at the primer level), so no WT signal reaches the electrode. crRNA-level discrimination (D = ${echemDiscOverlay.narsilDisc}\u00d7) is not relevant for this detection strategy. The voltammogram shows baseline vs MUT-only cleavage.`
                  : echemDiscOverlay.measuredDisc >= 3
                  ? `Diagnostic-grade allelic discrimination (D = ${echemDiscOverlay.measuredDisc === Infinity ? "\u221e" : echemDiscOverlay.measuredDisc + "\u00d7"}). The voltammetric \u0394I% difference between MUT and WT alleles is clearly resolvable, enabling reliable genotyping from electrochemical signal alone.`
                  : echemDiscOverlay.measuredDisc >= 1.5
                  ? `Moderate allelic discrimination (D = ${echemDiscOverlay.measuredDisc}\u00d7). The MUT/WT peak height difference is detectable but marginal. AS-RPA primer specificity provides additional discrimination at the amplification stage.`
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
                        <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: "4px", fontSize: "10px", fontWeight: 600, background: d.block_class === "strong" ? "#ECFDF5" : d.block_class === "moderate" ? "#FFFBEB" : "#FEF2F2", color: d.block_class === "strong" ? T.success : d.block_class === "moderate" ? "#D97706" : T.danger, textTransform: "lowercase" }}>{d.block_class}</span>
                      </td>
                      <td style={{ padding: "8px 12px", textAlign: "center", fontFamily: FONT, fontSize: "11px" }}>{Math.round(d.estimated_specificity * 100)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: "10px", color: T.textTer, marginTop: "8px" }}>
            Discrimination ratios computed via Boltzmann conversion: exp({"\u0394\u0394"}G / RT) at 37 {"\u00b0"}C. Ratios &gt; 100{"\u00d7"} capped; kinetic effects dominate at high {"\u0394\u0394"}G.
          </div>
        </div>
      </CollapsibleSection>
      )}


    </div>
  );
};


/* ═══════════════════════════════════════════════════════════════════
   DIAGNOSTICS TAB; Block 3 Sensitivity-Specificity Optimization
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
  const [mcResult, setMcResult] = useState(null);
  const [mcRunning, setMcRunning] = useState(false);

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
  // Normalize drug codes consistently; handles full names, lowercase, and short codes
  const normDrug = useCallback((d) => {
    if (!d) return "";
    const u = d.toUpperCase().trim();
    const MAP = { RIFAMPICIN: "RIF", ISONIAZID: "INH", FLUOROQUINOLONE: "FQ", FLUOROQUINOLONES: "FQ", ETHAMBUTOL: "EMB", PYRAZINAMIDE: "PZA", AMINOGLYCOSIDE: "AG", AMIKACIN: "AG", KANAMYCIN: "AG", SPECIES_CONTROL: "OTHER" };
    return MAP[u] || u;
  }, []);

  // Infer drug from gene name as fallback
  const inferDrug = useCallback((label) => {
    if (!label) return "";
    if (label.startsWith("rpoB")) return "RIF";
    if (label.startsWith("katG") || label.startsWith("fabG1")) return "INH";
    if (label.startsWith("embB")) return "EMB";
    if (label.startsWith("pncA")) return "PZA";
    if (label.startsWith("gyrA")) return "FQ";
    if (label.startsWith("rrs") || label.startsWith("eis")) return "AG";
    if (label.startsWith("IS6110")) return "OTHER";
    return "";
  }, []);

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
        const isControl = r.gene === "IS6110" || (normDrug(r.drug) || inferDrug(r.label)) === "OTHER";
        const ready = isControl ? (r.hasPrimers && eff >= effT) : (r.hasPrimers && eff >= effT && asrpaViable && (r.strategy === "Proximity" || disc >= discT));
        const drug = normDrug(r.drug) || inferDrug(r.label);
        return { target_label: r.label || "unknown", drug, efficiency: eff, discrimination: disc, is_assay_ready: ready, has_primers: !!r.hasPrimers, strategy: r.strategy || "Direct", asrpaViable };
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

  // Load presets on mount; try API first, fall back to hardcoded
  useEffect(() => {
    const fallbackPresets = [
      { name: "balanced", description: "WHO TPP-aligned; clinical diagnostic deployment.", efficiency_threshold: 0.4, discrimination_threshold: 3.0 },
      { name: "high_sensitivity", description: "Field screening; maximise coverage, tolerate lower discrimination.", efficiency_threshold: 0.2, discrimination_threshold: 1.5 },
      { name: "high_specificity", description: "Confirmatory; minimise false calls, reference lab use.", efficiency_threshold: 0.6, discrimination_threshold: 5.0 },
    ];
    if (!connected) { setPresets(fallbackPresets); return; }
    getPresets().then(({ data }) => { setPresets(data && data.length ? data : fallbackPresets); });
  }, [connected]);

  // Clear sweep/pareto when preset changes so stale charts don't persist
  useEffect(() => {
    setSweepData(null);
    setParetoData(null);
  }, [activePreset]);

  // Load diagnostics + WHO compliance; try API, fall back to client-side computation
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

  // Run sweep; try API, fall back to client-side
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

  // Run Pareto; try API, fall back to client-side
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
      <InSilicoCaveat />
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
              { label: "Specificity", value: `${Math.round(diagnostics.specificity * 100)}%`, color: diagnostics.specificity >= 0.98 ? T.success : diagnostics.specificity >= 0.85 ? "#d97706" : T.danger, icon: Shield },
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
                      {separation >= 0.15 ? " Good separation. The panel reliably distinguishes resistant from susceptible samples at the aggregate level." : separation >= 0.08 ? " Moderate separation. Borderline samples may produce ambiguous calls; consider tightening the panel to high-discrimination targets only." : " Poor separation. The panel cannot reliably distinguish MUT from WT; review target selection and consider dropping low-discrimination candidates."}
                      {` Overlap zone: ${overlapPct}%. This is the aggregate overlap; individual targets with high discrimination (e.g., disc >=10x) have near-zero overlap. In practice each target is read independently, so per-target separation matters more than panel-level aggregate.`}
                      {` Strongest MUT signal: ${bestMutLabel} (${mutSorted[0].toFixed(3)}). Weakest: ${worstMutLabel} (${mutSorted[mutSorted.length - 1].toFixed(3)}).`}
                      {plotResults.length === results.filter(r => r.gene !== "IS6110").length && activePreset !== "balanced" && ` Note: all candidates exceed the ${PRESET_LABELS[activePreset] || activePreset} thresholds. This profile produces identical results to a less stringent profile for the current panel.`}
                    </div>
                  );
                })()}
              </div>
            );
          } catch (e) { console.error("MUT vs WT chart error:", e); return null; } })()}

          {/* B2: Understanding Discrimination Scores; collapsible explainer */}
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
                M. tuberculosis has 65.8% GC content. GC-rich sequences around a mismatch stabilise the R-loop through additional hydrogen bonds, partially compensating for the mismatch. This is why some targets (EMB, PZA) show low predicted discrimination: their mutations sit in GC-rich regions at PAM-distal positions.
              </div>
              <div style={{ marginBottom: "14px" }}>
                <div style={{ fontSize: "13px", fontWeight: 600, color: T.text, marginBottom: "4px" }}>Prediction model</div>
                {results.some(r => r.discMethod === "neural")
                  ? "Discrimination ratios are predicted by Compass-ML's neural discrimination head, a multi-task extension (235K params) trained end-to-end on efficiency and discrimination simultaneously. The disc head takes paired encoder representations [mut, wt, mut\u2212wt, mut\u00D7wt] from the shared CNN+RNA-FM+RLPA backbone and outputs a predicted MUT/WT ratio via Softplus. Trained on 6,136 paired trans-cleavage measurements from EasyDesign (Huang et al. 2024, LbCas12a). 3-fold CV: r = 0.440."
                  : results.some(r => (r.discrimination?.model_name || "").includes("learned") || r.discMethod === "feature")
                  ? "Discrimination ratios are predicted by a gradient-boosted model (XGBoost) trained on 6,136 paired MUT/WT trans-cleavage measurements from the EasyDesign dataset (Huang et al. 2024, LbCas12a). The model uses 18 thermodynamic features including R-loop cumulative \u0394G, mismatch \u0394\u0394G penalties, and position sensitivity. Val: RMSE = 0.520, r = 0.565. Top feature: pam_to_mm_distance (0.148 importance)."
                  : "Discrimination ratios are predicted by a heuristic model using position sensitivity \u00D7 mismatch destabilisation scores. A trained model (XGBoost on 18 thermodynamic features) is available but was not loaded for this run."
                }
              </div>
              <div style={{ fontSize: "11px", color: T.textTer, borderTop: `1px solid ${T.borderLight}`, paddingTop: "10px" }}>
                These are in silico predictions. Experimental validation on the electrochemical platform will provide measured discrimination ratios through the active learning loop.
              </div>
            </div>
          </CollapsibleSection>

          {/* C: WHO Compliance Table */}
          {whoCompliance && whoCompliance.who_compliance && (() => {
            // Filter out species_control/UNKNOWN from WHO table; it's not a resistance drug class
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
                WHO Target Product Profile (TPP) 2024 defines minimum sensitivity and specificity thresholds per drug class for diagnostic deployment. Sensitivity = fraction of resistance-conferring mutations detected (pass/fail per drug class). Specificity = approximate in silico estimate: Direct targets use 1−1/disc (assumes perfectly separated signal distributions; actual specificity depends on signal variance and threshold selection). Proximity targets use thermodynamic AS-RPA mismatch penalty. ≥98% required; marked "Pending" when below threshold as experimental validation is needed. {results.some(r => (r.discrimination?.model_name || "").includes("learned")) ? "Discrimination ratios used here are from the learned model (XGBoost, 18 thermodynamic features)." : "Discrimination ratios used here are from the heuristic model."}
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
                        const tDrug = normDrug(t.drug) || inferDrug(t.target_label);
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
                        <td style={{ padding: "10px 14px", fontFamily: FONT, fontSize: "11px", color: avgDisc >= 3 ? T.success : avgDisc >= 2 ? T.warning : T.textTer }}>{avgDisc > 0 ? `${avgDisc.toFixed(1)}×` : "–"}</td>
                        <td style={{ padding: "10px 14px" }}>
                          {data.specificity != null ? (
                            <div>
                              <span style={{ fontFamily: FONT, fontWeight: 600, fontSize: "12px", color: data.specificity >= 0.98 ? T.success : data.specificity >= 0.90 ? T.warning : T.textTer }}>{Math.round(data.specificity * 100)}%</span>
                              {data.n_excluded_specificity > 0 && <div style={{ fontSize: "9px", color: T.textTer, marginTop: "2px" }}>{data.n_excluded_specificity} excluded</div>}
                              {/* Specificity decomposition */}
                              {(() => {
                                const dts = drugTargets.filter(t => t.strategy === "Direct" && t.discrimination > 0);
                                const pts = drugTargets.filter(t => t.strategy === "Proximity");
                                if (dts.length === 0 && pts.length === 0) return null;
                                return (
                                  <div style={{ fontSize: "8px", color: T.textTer, marginTop: "3px", lineHeight: 1.5, fontFamily: MONO }}>
                                    {dts.map(dt => (
                                      <div key={dt.target_label}>{dt.target_label.split("_").pop()}: crRNA {dt.discrimination.toFixed(1)}x {"\u2192"} {((1 - 1/Math.max(dt.discrimination, 1.01)) * 100).toFixed(0)}%</div>
                                    ))}
                                    {pts.map(pt => {
                                      const orig = results.find(r => r.label === pt.target_label);
                                      const asSpec = orig?.asrpaDiscrimination?.estimated_specificity;
                                      return <div key={pt.target_label}>{pt.target_label.split("_").pop()}: AS-RPA {"\u2192"} {asSpec != null ? `${(asSpec * 100).toFixed(0)}%` : "~95%"}</div>;
                                    })}
                                  </div>
                                );
                              })()}
                            </div>
                          ) : <span style={{ color: T.textTer }}>–</span>}
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
                    {" "}<strong>Specificity:</strong> {specPassing}/{whoEntries.length} classes meet the {"\u2265"}98% threshold (approximate in silico proxy; actual specificity requires experimental determination with clinical samples).
                    {specFailing.length > 0 && ` ${specFailing.length} class${specFailing.length > 1 ? "es" : ""} pending. Specificity estimates require experimental validation on the electrochemical platform.`}
                    {specPassing === whoEntries.length && " All classes pass specificity."}
                    {" "}Note: coverage denominators reflect panel targets only, not the full WHO mutation catalogue. Clinical sensitivity for a drug class depends on the epidemiological frequency of included mutations (e.g., INH: katG S315T covers ~60% of INH-resistant isolates; adding fabG1 C-15T raises coverage to ~85%).
                  </div>
                );
              })()}
            </div>
            );
          })()}

          {/* D2: Monte Carlo Robustness Analysis */}
          {diagnostics.per_target && diagnostics.per_target.length > 0 && (() => {
            const runMC = () => {
              setMcRunning(true);
              setTimeout(() => {
                const p = presets.find(x => x.name === activePreset) || { efficiency_threshold: 0.4, discrimination_threshold: 3.0 };
                const effT = p.efficiency_threshold || 0.4;
                const discT = p.discrimination_threshold || 3.0;
                const targets = diagnostics.per_target.filter(t => t.drug !== "OTHER");
                const N = 500;
                const sensResults = [];
                const specResults = [];
                for (let i = 0; i < N; i++) {
                  let ready = 0;
                  const specContribs = [];
                  for (const t of targets) {
                    const effPerturb = t.efficiency * (0.8 + Math.random() * 0.4);
                    const discPerturb = t.discrimination * (0.8 + Math.random() * 0.4);
                    const asrpaOk = t.strategy !== "Proximity" || true;
                    if (effPerturb >= effT && asrpaOk && (t.strategy === "Proximity" || discPerturb >= discT)) ready++;
                    // Specificity contribution
                    if (t.strategy === "Proximity") {
                      specContribs.push(0.95);
                    } else if (t.discrimination > 0) {
                      specContribs.push(1 - 1 / Math.max(discPerturb, 1.01));
                    }
                  }
                  sensResults.push(ready / targets.length * 100);
                  specResults.push(specContribs.length ? specContribs.reduce((a, v) => a + v, 0) / specContribs.length * 100 : 0);
                }
                sensResults.sort((a, b) => a - b);
                specResults.sort((a, b) => a - b);
                const sensMean = sensResults.reduce((a, v) => a + v, 0) / N;
                const sensCi95Low = sensResults[Math.floor(N * 0.025)];
                const sensCi95High = sensResults[Math.floor(N * 0.975)];
                const specMean = specResults.reduce((a, v) => a + v, 0) / N;
                const specCi95Low = specResults[Math.floor(N * 0.025)];
                const specCi95High = specResults[Math.floor(N * 0.975)];
                // Sensitivity histogram (10 bins)
                const sensMinV = Math.floor(sensResults[0] / 10) * 10;
                const sensMaxV = Math.ceil(sensResults[N - 1] / 10) * 10 || 100;
                const sensBinW = Math.max((sensMaxV - sensMinV) / 10, 1);
                const sensBins = Array.from({ length: 10 }, (_, i) => ({ lo: sensMinV + i * sensBinW, hi: sensMinV + (i + 1) * sensBinW, count: 0 }));
                for (const v of sensResults) {
                  const idx = Math.min(Math.floor((v - sensMinV) / sensBinW), 9);
                  sensBins[idx].count++;
                }
                const sensMaxCount = Math.max(...sensBins.map(b => b.count));
                // Specificity histogram (10 bins)
                const specMinV = Math.floor(specResults[0] / 10) * 10;
                const specMaxV = Math.ceil(specResults[N - 1] / 10) * 10 || 100;
                const specBinW = Math.max((specMaxV - specMinV) / 10, 1);
                const specBins = Array.from({ length: 10 }, (_, i) => ({ lo: specMinV + i * specBinW, hi: specMinV + (i + 1) * specBinW, count: 0 }));
                for (const v of specResults) {
                  const idx = Math.min(Math.floor((v - specMinV) / specBinW), 9);
                  specBins[idx].count++;
                }
                const specMaxCount = Math.max(...specBins.map(b => b.count));
                setMcResult({ sensMean, sensCi95Low, sensCi95High, sensBins, sensMaxCount, specMean, specCi95Low, specCi95High, specBins, specMaxCount, N });
                setMcRunning(false);
              }, 50);
            };
            return (
              <CollapsibleSection title="Robustness Analysis" defaultOpen={false}>
                <div style={{ padding: "14px 18px" }}>
                  <div style={{ fontSize: "11px", color: T.textSec, lineHeight: 1.6, marginBottom: 12 }}>
                    Monte Carlo robustness analysis: perturb each candidate{"'"}s activity and discrimination by {"\u00B1"}20% (uniform random, 500 iterations) and re-evaluate panel sensitivity and specificity under the current threshold profile.
                  </div>
                  {!mcResult ? (
                    <button onClick={runMC} disabled={mcRunning} style={{ padding: "8px 16px", borderRadius: 4, border: `1px solid ${T.primary}`, background: mcRunning ? T.bgSub : T.primary, color: mcRunning ? T.textSec : "#fff", fontFamily: FONT, fontSize: 11, fontWeight: 600, cursor: mcRunning ? "wait" : "pointer" }}>
                      {mcRunning ? "Running..." : "Run robustness analysis"}
                    </button>
                  ) : (
                    <div>
                      <div style={{ display: "flex", gap: 24, marginBottom: 16, flexWrap: "wrap" }}>
                        <div style={{ textAlign: "center" }}>
                          <div style={{ fontSize: 20, fontWeight: 700, color: T.text, fontFamily: MONO }}>{mcResult.sensMean.toFixed(1)}%</div>
                          <div style={{ fontSize: 9, color: T.textTer }}>Mean sensitivity</div>
                        </div>
                        <div style={{ textAlign: "center" }}>
                          <div style={{ fontSize: 14, fontWeight: 600, color: T.textSec, fontFamily: MONO }}>{mcResult.sensCi95Low.toFixed(1)}% {"\u2013"} {mcResult.sensCi95High.toFixed(1)}%</div>
                          <div style={{ fontSize: 9, color: T.textTer }}>Sensitivity 95% CI</div>
                        </div>
                        <div style={{ width: 1, background: T.borderLight, alignSelf: "stretch" }} />
                        <div style={{ textAlign: "center" }}>
                          <div style={{ fontSize: 20, fontWeight: 700, color: T.text, fontFamily: MONO }}>{mcResult.specMean.toFixed(1)}%</div>
                          <div style={{ fontSize: 9, color: T.textTer }}>Mean specificity</div>
                        </div>
                        <div style={{ textAlign: "center" }}>
                          <div style={{ fontSize: 14, fontWeight: 600, color: T.textSec, fontFamily: MONO }}>{mcResult.specCi95Low.toFixed(1)}% {"\u2013"} {mcResult.specCi95High.toFixed(1)}%</div>
                          <div style={{ fontSize: 9, color: T.textTer }}>Specificity 95% CI</div>
                        </div>
                      </div>
                      {/* Histograms; 2 columns */}
                      <div style={{ display: "grid", gridTemplateColumns: mobile ? "1fr" : "1fr 1fr", gap: 16 }}>
                        <div>
                          <div style={{ fontSize: 10, fontWeight: 600, color: T.textSec, marginBottom: 4 }}>Sensitivity distribution</div>
                          <svg viewBox="0 0 300 80" style={{ width: "100%", maxWidth: 400, display: "block" }}>
                            {mcResult.sensBins.map((b, i) => {
                              const barH = mcResult.sensMaxCount > 0 ? (b.count / mcResult.sensMaxCount) * 60 : 0;
                              return (
                                <g key={i}>
                                  <rect x={i * 30} y={65 - barH} width={28} height={barH} fill={T.primary} opacity={0.7} rx={2} />
                                  <text x={i * 30 + 14} y={78} fontSize="7" fill={T.textTer} textAnchor="middle" fontFamily={MONO}>{b.lo.toFixed(0)}%</text>
                                </g>
                              );
                            })}
                          </svg>
                        </div>
                        <div>
                          <div style={{ fontSize: 10, fontWeight: 600, color: T.textSec, marginBottom: 4 }}>Specificity distribution</div>
                          <svg viewBox="0 0 300 80" style={{ width: "100%", maxWidth: 400, display: "block" }}>
                            {mcResult.specBins.map((b, i) => {
                              const barH = mcResult.specMaxCount > 0 ? (b.count / mcResult.specMaxCount) * 60 : 0;
                              return (
                                <g key={i}>
                                  <rect x={i * 30} y={65 - barH} width={28} height={barH} fill="#d97706" opacity={0.7} rx={2} />
                                  <text x={i * 30 + 14} y={78} fontSize="7" fill={T.textTer} textAnchor="middle" fontFamily={MONO}>{b.lo.toFixed(0)}%</text>
                                </g>
                              );
                            })}
                          </svg>
                        </div>
                      </div>
                      <div style={{ fontSize: 9, color: T.textTer, marginTop: 8 }}>
                        Distribution of panel sensitivity and specificity under {"\u00B1"}20% perturbation. Narrow distribution = robust panel design.
                      </div>
                    </div>
                  )}
                </div>
              </CollapsibleSection>
            );
          })()}

          {/* D: Per-Target Breakdown with Top-K */}
          {diagnostics.per_target && diagnostics.per_target.length > 0 && (
            <CollapsibleSection title={`Per-Target Breakdown (${diagnostics.per_target.length} targets)`} defaultOpen={false}>
              <div style={{ padding: "10px 14px", marginBottom: "12px", background: T.primaryLight, borderRadius: "4px", fontSize: "11px", color: T.primaryDark, lineHeight: 1.6 }}>
                <strong>Per-target assay readiness assessment.</strong> Each row shows the selected candidate's predicted efficiency and discrimination ratio against the active profile thresholds.
                Click any row to expand the <strong>Top-K alternative candidates</strong>; ranked alternatives with tradeoff annotations for experimental fallback planning.
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
                      const drugDisplay = normDrug(t.drug) || inferDrug(t.target_label) || "CTRL";
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
                                  return <span style={{ fontSize: "10px", fontWeight: 600, color: c }} title={`AS-RPA ${ad.terminal_mismatch}; ${ad.block_class}`}>{ad.disc_ratio >= 100 ? "≥100" : ad.disc_ratio.toFixed(0)}× <span style={{ fontWeight: 500, color: T.purple }}>AS-RPA</span></span>;
                                }
                                return <span style={{ fontSize: "10px", color: T.purple, fontWeight: 600 }}>AS-RPA</span>;
                              })() : (
                                <span style={{ fontFamily: FONT, fontWeight: 600, fontSize: "12px", color: discColor }}>{disc > 0 ? `${disc.toFixed(1)}×` : "–"}</span>
                              )}
                            </td>
                            <td style={{ padding: "10px 12px" }}>{t.has_primers ? <CheckCircle size={14} color={T.success} /> : <span style={{ color: T.textTer }}>–</span>}</td>
                            <td style={{ padding: "10px 12px" }}>
                              {(() => {
                                const isControl = t.drug === "OTHER" || t.target_label === "IS6110_N0N";
                                if (isControl) return (
                                  <span style={{ display: "inline-flex", alignItems: "center", gap: "3px", fontSize: "10px", fontWeight: 600, padding: "3px 10px", borderRadius: "3px", background: "rgba(16,185,129,0.1)", color: T.success }}>ID Control</span>
                                );
                                if (t.is_assay_ready) return (
                                  <span style={{ display: "inline-flex", alignItems: "center", gap: "3px", fontSize: "10px", fontWeight: 600, padding: "3px 10px", borderRadius: "3px", background: "rgba(16,185,129,0.1)", color: T.success }}>Ready</span>
                                );
                                return (
                                  <span style={{ display: "inline-flex", alignItems: "center", gap: "3px", fontSize: "10px", fontWeight: 600, padding: "3px 10px", borderRadius: "3px", background: T.bgSub, color: T.textTer }}>Not ready</span>
                                );
                              })()}
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
                                    <div style={{ fontSize: "11px", color: T.textTer, padding: "8px 0" }}>No alternative candidates available for this target.</div>
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
                                                <td style={{ padding: "7px 10px", fontFamily: FONT, color: T.textSec }}>{isProximity ? <span style={{ fontSize: "10px", color: T.purple }}>AS-RPA</span> : aDisc > 0 ? `${aDisc.toFixed(1)}×` : "–"}</td>
                                                <td style={{ padding: "7px 10px", fontFamily: FONT, color: alt.offtarget_count === 0 ? T.success : alt.offtarget_count != null ? T.warning : T.textTer }}>{alt.offtarget_count ?? "–"}</td>
                                                <td style={{ padding: "7px 10px", fontFamily: MONO, fontSize: "10px", color: T.textTer, letterSpacing: "0.3px" }}>{spacer ? `${spacer.slice(0, 10)} ${spacer.slice(10, 20)}` : "–"}</td>
                                                <td style={{ padding: "7px 10px", fontSize: "10px", color: T.textTer }}>
                                                  {isSelected ? <span style={{ fontWeight: 600, color: T.primary }}>Selected candidate</span> : (notes || "comparable")}
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
   RESULTS PAGE; Tab container with accordion candidates
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
          calibration: data?.calibration || null,
          optimizer_score_trace: data?.optimizer_score_trace || null,
        });
        setLoading(false);
      });
    } else if (activeJob.startsWith("mock-")) {
      /* Mock mode; adapt mock data to scorer + panel encoded in job ID */
      const isHeuristic = activeJob.includes("-heuristic-");
      // Extract selected mutation indices from job ID (format: mock-scorer-0,1,2,...-timestamp)
      const parts = activeJob.split("-");
      const indicesStr = parts.length >= 4 ? parts.slice(2, -1).join("-") : "";
      const selectedIndices = indicesStr ? indicesStr.split(",").map(Number).filter(n => !isNaN(n)) : null;
      // Filter RESULTS to only selected mutations (+ IS6110 control always included)
      let filtered = RESULTS;
      if (selectedIndices && selectedIndices.length > 0 && selectedIndices.length < RESULTS.length) {
        const selectedLabels = new Set(selectedIndices.map(i => {
          const m = MUTATIONS[i];
          if (!m) return null;
          return m.category === "gene_presence" ? m.gene : `${m.gene}_${m.ref}${m.pos}${m.alt}`;
        }).filter(Boolean));
        filtered = RESULTS.filter(r => selectedLabels.has(r.label) || r.gene === "IS6110" || r.category === "gene_presence");
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
          {tab === "overview" && <OverviewTab results={results} scorer={scorerProp} jobId={activeJob} panelData={panelData} />}
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


export {
  RISK_COLORS, RISK_BG, gradientColor, gradientCSS,
  ResultsPage,
};
