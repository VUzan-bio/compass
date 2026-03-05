import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";

/* ═══════════════════════════════════════════════════════════════
   CONSTANTS & HELPERS
   ═══════════════════════════════════════════════════════════════ */
const DRUG_HEX = { RIF: 0x4338CA, INH: 0xD97706, EMB: 0x059669, PZA: 0x0891B2, FQ: 0xDC2626, AG: 0x7C3AED, CTRL: 0x9ca3af, OTHER: 0x888888 };
const DRUG_CSS = { RIF: "#4338CA", INH: "#D97706", EMB: "#059669", PZA: "#0891B2", FQ: "#DC2626", AG: "#7C3AED", CTRL: "#9ca3af", OTHER: "#888888" };
const addAt = (parent, mesh, x, y, z) => { mesh.position.set(x, y, z); parent.add(mesh); return mesh; };

/* ── Architecture-dependent SWV curve generators ── */

// Architecture A: pAP/ALP — peak at +0.15 V, signal-ON
const miniSWV_pAP = (Gamma, G0) => {
  const nFRT = (2 * 96485) / (8.314 * 310.15);
  const pts = [];
  for (let i = 0; i <= 80; i++) {
    const E = -0.2 + i * 0.005625;
    const xp = nFRT * (E + 0.0125 - 0.15);
    const xm = nFRT * (E - 0.0125 - 0.15);
    const base = (1 / (1 + Math.exp(xp)) - 1 / (1 + Math.exp(xm))) * 2.5;
    pts.push({ E, I: (1.0 - Gamma / G0 * 0.6 + 0.4) * base });
  }
  return pts;
};

// Architecture B: Silver metallization — peak at +0.16 V, signal-OFF
const miniSWV_Ag = (Gamma, G0) => {
  const nFRT = (1 * 96485) / (8.314 * 310.15);
  const pts = [];
  for (let i = 0; i <= 80; i++) {
    const E = -0.3 + i * 0.01;
    const xp = nFRT * (E + 0.03 - 0.16);
    const xm = nFRT * (E - 0.03 - 0.16);
    pts.push({ E, I: (Gamma / G0) * (1 / (1 + Math.exp(xp)) - 1 / (1 + Math.exp(xm))) * 3.0 });
  }
  return pts;
};

// Architecture C: Direct MB-ssDNA — peak at -0.22 V, signal-OFF
const miniSWV_MB = (Gamma, G0) => {
  const nFRT = (2 * 96485) / (8.314 * 310.15);
  const pts = [];
  for (let i = 0; i <= 80; i++) {
    const E = -0.5 + i * 0.00625;
    const xp = nFRT * (E + 0.0125 - (-0.22));
    const xm = nFRT * (E - 0.0125 - (-0.22));
    pts.push({ E, I: (Gamma / G0) * (1 / (1 + Math.exp(xp)) - 1 / (1 + Math.exp(xm))) * 3.0 });
  }
  return pts;
};

const miniEIS = (Gamma, G0) => {
  const Rs = 50, Rct0 = 2000, Cdl = 20e-6;
  const Rct = Rct0 * (Gamma / G0) + 100;
  const pts = [];
  for (let i = 0; i <= 60; i++) {
    const omega = 2 * Math.PI * Math.pow(10, i * 0.1);
    const d = 1 + (omega * Rct * Cdl) ** 2;
    pts.push({ Zr: Rs + Rct / d, Zi: omega * Rct * Rct * Cdl / d });
  }
  return pts;
};

const ARCH_META = {
  A: { name: "Vertical-flow ALP", short: "ALP", peak: "+0.15 V", label: "pAP oxidation", freq: "2.5 Hz", amp: "25 mV", step: "5 mV", eRange: "−0.2 to +0.25 V", sig: "ON", lod: "1 cp/µL", ref: "Bezinge 2023", swv: miniSWV_pAP },
  B: { name: "Silver metallization", short: "Ag", peak: "+0.16 V", label: "Ag⁰ → Ag⁺", freq: "200 Hz", amp: "60 mV", step: "10 mV", eRange: "−0.3 to +0.5 V", sig: "OFF", lod: "3.5 fM", ref: "Suea-Ngam 2021", swv: miniSWV_Ag },
  C: { name: "Direct MB-ssDNA", short: "MB", peak: "−0.22 V", label: "MB reduction", freq: "50 Hz", amp: "25 mV", step: "4 mV", eRange: "−0.5 to 0.0 V", sig: "OFF", lod: "~pM", ref: "Conceptual", swv: miniSWV_MB },
};

// Sprite text label helper
const makeSprite = (text, color, size, bold) => {
  const canvas = document.createElement("canvas");
  const dpr = 2;
  canvas.width = 512 * dpr; canvas.height = 64 * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.font = `${bold ? "bold " : ""}${Math.round(size || 18)}px monospace`;
  ctx.fillStyle = color || "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 256, 32);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(size ? size * 0.28 : 5.0, size ? size * 0.07 : 1.2, 1);
  return sprite;
};

// 14-target electrode layout: 7 columns × 2 rows
const ELECTRODE_TARGETS = [
  { row: 0, col: 0, target: "IS6110", drug: "CTRL", color: DRUG_HEX.CTRL, label: "IS6110\nTB ID" },
  { row: 0, col: 1, target: "rpoB_S531L", drug: "RIF", color: DRUG_HEX.RIF, label: "rpoB S531L\nRIF" },
  { row: 0, col: 2, target: "rpoB_H526Y", drug: "RIF", color: DRUG_HEX.RIF, label: "rpoB H526Y\nRIF" },
  { row: 0, col: 3, target: "rpoB_D516V", drug: "RIF", color: DRUG_HEX.RIF, label: "rpoB D516V\nRIF" },
  { row: 0, col: 4, target: "katG_S315T", drug: "INH", color: DRUG_HEX.INH, label: "katG S315T\nINH" },
  { row: 0, col: 5, target: "fabG1_C-15T", drug: "INH", color: DRUG_HEX.INH, label: "fabG1 C-15T\nINH" },
  { row: 0, col: 6, target: "embB_M306V", drug: "EMB", color: DRUG_HEX.EMB, label: "embB M306V\nEMB" },
  { row: 1, col: 0, target: "embB_M306I", drug: "EMB", color: DRUG_HEX.EMB, label: "embB M306I\nEMB" },
  { row: 1, col: 1, target: "pncA_H57D", drug: "PZA", color: DRUG_HEX.PZA, label: "pncA H57D\nPZA" },
  { row: 1, col: 2, target: "gyrA_D94G", drug: "FQ", color: DRUG_HEX.FQ, label: "gyrA D94G\nFQ" },
  { row: 1, col: 3, target: "gyrA_A90V", drug: "FQ", color: DRUG_HEX.FQ, label: "gyrA A90V\nFQ" },
  { row: 1, col: 4, target: "rrs_A1401G", drug: "AG", color: DRUG_HEX.AG, label: "rrs A1401G\nKAN/AMK" },
  { row: 1, col: 5, target: "eis_C-14T", drug: "AG", color: DRUG_HEX.AG, label: "eis C-14T\nKAN" },
  { row: 1, col: 6, target: "RNaseP", drug: "CTRL", color: DRUG_HEX.CTRL, label: "RNaseP\nHuman Ctrl" },
];

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function ChipRender3D({ electrodeLayout, targetDrug, targetStrategy, getEfficiency, results, computeGamma, echemTime, echemKtrans, echemGamma0_mol, HEADING, MONO }) {
  const mountRef = useRef(null);
  const stateRef = useRef(null);
  const [mode, setMode] = useState(1);
  const [arch, setArch] = useState("B");
  const [selectedPad, setSelectedPad] = useState(null);
  const [cas12aActive, setCas12aActive] = useState(false);
  const [tooltipInfo, setTooltipInfo] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [incubationMin, setIncubationMin] = useState(echemTime);
  const [curveMode, setCurveMode] = useState("SWV");
  const [showFluidics, setShowFluidics] = useState(true);
  const [showWaxBarriers, setShowWaxBarriers] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showCapillaryFlow, setShowCapillaryFlow] = useState(false);
  const [expandedCurve, setExpandedCurve] = useState(false);
  const [expandedETransfer, setExpandedETransfer] = useState(false);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;
    const W = container.clientWidth;
    const H = Math.round(W * 9 / 16);
    container.style.height = H + "px";

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.setClearColor(0xF8F9FA);
    container.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 500);
    const scene = new THREE.Scene();

    // ── Lighting ──
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const key = new THREE.DirectionalLight(0xFFF5E6, 0.75);
    key.position.set(-30, 50, 40); key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 1; key.shadow.camera.far = 200;
    key.shadow.camera.left = -50; key.shadow.camera.right = 50;
    key.shadow.camera.top = 50; key.shadow.camera.bottom = -50;
    key.shadow.bias = -0.001;
    scene.add(key);
    scene.add(new THREE.DirectionalLight(0xE6F0FF, 0.3).translateX(40).translateY(30).translateZ(-20));
    scene.add(new THREE.DirectionalLight(0xFFFFFF, 0.2).translateY(20).translateZ(-50));
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.ShadowMaterial({ opacity: 0.15 }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -1; ground.receiveShadow = true;
    scene.add(ground);

    // ══════════════════════════════════════════════════════════
    // MATERIALS — Cellulose paper platform
    // ══════════════════════════════════════════════════════════
    const celluloseMat = new THREE.MeshStandardMaterial({ color: 0xF5F0E8, roughness: 0.85, metalness: 0.0 });
    const ligMat = new THREE.MeshStandardMaterial({ color: 0x374151, roughness: 0.9, metalness: 0.1 });
    const waxMat = new THREE.MeshStandardMaterial({ color: 0x3D2B1F, roughness: 0.7, metalness: 0.0 });
    const agMat = new THREE.MeshStandardMaterial({ color: 0xC0C0C0, metalness: 0.7, roughness: 0.2 });
    const channelMat = new THREE.MeshStandardMaterial({ color: 0x88BBDD, transparent: true, opacity: 0.25, roughness: 0.4 });
    const bloodMat = new THREE.MeshStandardMaterial({ color: 0xCC3333, transparent: true, opacity: 0.45, roughness: 0.6 });
    const plasmaMat = new THREE.MeshStandardMaterial({ color: 0xF5D0A0, transparent: true, opacity: 0.35, roughness: 0.5 });

    // ══════════════════════════════════════════════════════════
    // MODE 1: CHIP OVERVIEW — ~40 × 25 mm cellulose paper
    // ══════════════════════════════════════════════════════════
    const chipGroup = new THREE.Group();
    scene.add(chipGroup);

    const chipW = 40, chipD = 25, chipH = 3.9; // 390 µm → 3.9 scaled

    // Cellulose paper substrate (white, fibrous)
    const body = new THREE.Mesh(new THREE.BoxGeometry(chipW, chipH, chipD), celluloseMat);
    body.castShadow = true; body.receiveShadow = true;
    chipGroup.add(body);

    // Fiber texture on top surface
    const fiberMat = new THREE.MeshStandardMaterial({ color: 0xEDE8DD, roughness: 0.95 });
    for (let i = 0; i < 60; i++) {
      const fx = (Math.random() - 0.5) * (chipW - 2);
      const fz = (Math.random() - 0.5) * (chipD - 2);
      const fLen = 1.0 + Math.random() * 2.5;
      const fib = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, fLen, 3), fiberMat);
      fib.position.set(fx, chipH / 2 + 0.01, fz);
      fib.rotation.x = Math.PI / 2 + (Math.random() - 0.5) * 0.3;
      fib.rotation.y = Math.random() * Math.PI;
      chipGroup.add(fib);
    }

    // ── LIG-E region (embedded IN cellulose, dark) ──
    const gridOriginX = -10;
    const gridOriginZ = -3;
    const colSpacing = 3.5;
    const rowSpacing = 6;
    const weRadius = 1.2;

    const ligPadX = gridOriginX + 3 * colSpacing;
    // LIG-E is embedded — show as darker region flush with surface
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(28, 0.08, 16), ligMat), ligPadX, chipH / 2 - 0.02, gridOriginZ + rowSpacing / 2);

    // ── Wax-printed hydrophobic barriers ──
    const waxGroup = new THREE.Group();
    chipGroup.add(waxGroup);

    // Outer barrier rectangle around detection zone
    const wBW = 30, wBD = 18;
    const waxH = 0.15;
    [[-wBW / 2, 0, wBD / 2], [wBW / 2, 0, wBD / 2], [-wBW / 2, 0, -wBD / 2], [wBW / 2, 0, -wBD / 2]].forEach(() => {});
    // Top/bottom horizontal barriers
    addAt(waxGroup, new THREE.Mesh(new THREE.BoxGeometry(wBW, waxH, 0.2), waxMat), ligPadX, chipH / 2 + 0.05, gridOriginZ - 1.5);
    addAt(waxGroup, new THREE.Mesh(new THREE.BoxGeometry(wBW, waxH, 0.2), waxMat), ligPadX, chipH / 2 + 0.05, gridOriginZ + rowSpacing + 1.5);
    // Left/right vertical barriers
    addAt(waxGroup, new THREE.Mesh(new THREE.BoxGeometry(0.2, waxH, rowSpacing + 3), waxMat), gridOriginX - 2, chipH / 2 + 0.05, gridOriginZ + rowSpacing / 2);
    addAt(waxGroup, new THREE.Mesh(new THREE.BoxGeometry(0.2, waxH, rowSpacing + 3), waxMat), gridOriginX + 6 * colSpacing + 2, chipH / 2 + 0.05, gridOriginZ + rowSpacing / 2);
    // Inter-column barriers (between each electrode)
    for (let c = 0; c < 6; c++) {
      const bx = gridOriginX + (c + 0.5) * colSpacing;
      addAt(waxGroup, new THREE.Mesh(new THREE.BoxGeometry(0.12, waxH, rowSpacing + 2.5), waxMat), bx, chipH / 2 + 0.05, gridOriginZ + rowSpacing / 2);
    }
    // Inter-row barrier
    addAt(waxGroup, new THREE.Mesh(new THREE.BoxGeometry(wBW, waxH, 0.12), waxMat), ligPadX, chipH / 2 + 0.05, gridOriginZ + rowSpacing / 2);

    // ── Working Electrodes (14 pads) ──
    const padPositions = [];
    const padMeshes = [];
    const labelSprites = [];
    ELECTRODE_TARGETS.forEach((e, idx) => {
      const px = gridOriginX + e.col * colSpacing;
      const pz = gridOriginZ + e.row * rowSpacing;
      padPositions.push({ x: px, z: pz, ...e, idx });

      // LIG-E floor (dark, porous, embedded)
      addAt(chipGroup, new THREE.Mesh(new THREE.CylinderGeometry(weRadius, weRadius, 0.08, 24), ligMat), px, chipH / 2, pz);

      // Pore texture dots
      for (let p = 0; p < 6; p++) {
        const a2 = Math.random() * Math.PI * 2, rd2 = Math.random() * (weRadius - 0.2);
        addAt(chipGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.09, 4),
          new THREE.MeshStandardMaterial({ color: 0x1a1a1a })),
          px + Math.cos(a2) * rd2, chipH / 2 + 0.06, pz + Math.sin(a2) * rd2);
      }

      // Drug class color ring
      const ringMat = new THREE.MeshStandardMaterial({ color: e.color, emissive: e.color, emissiveIntensity: 0.35, transparent: true, opacity: 0.85, side: THREE.DoubleSide });
      const ring = new THREE.Mesh(new THREE.RingGeometry(weRadius - 0.2, weRadius, 24), ringMat);
      ring.rotation.x = -Math.PI / 2;
      addAt(chipGroup, ring, px, chipH / 2 + 0.07, pz);

      // Target label sprite
      const lbl = makeSprite(e.target, "#555555", 7, true);
      lbl.position.set(px, chipH / 2 + 1.5, pz);
      chipGroup.add(lbl); labelSprites.push(lbl);

      // Drug class sub-label
      const drugLbl = makeSprite(e.drug, DRUG_CSS[e.drug] || "#888", 6);
      drugLbl.position.set(px, chipH / 2 + 1.1, pz);
      chipGroup.add(drugLbl); labelSprites.push(drugLbl);

      // Raycast hit mesh
      const pm = new THREE.Mesh(new THREE.CylinderGeometry(weRadius, weRadius, 1.0, 16), new THREE.MeshBasicMaterial({ visible: false }));
      pm.position.set(px, chipH / 2 + 0.5, pz);
      pm.userData = { target: e.target, drug: e.drug, idx, padColor: e.color };
      chipGroup.add(pm); padMeshes.push(pm);
    });

    // ── Counter Electrode — LIG-E arc ──
    const ceX = gridOriginX - 5;
    const ceZ = gridOriginZ + rowSpacing / 2;
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(3, 0.1, 8), ligMat), ceX, chipH / 2, ceZ);
    addAt(chipGroup, makeSprite("CE", "#888888", 9, true), ceX, chipH / 2 + 1.6, ceZ);

    // ── Ag/AgCl Reference Electrode ──
    const reX = gridOriginX + 6 * colSpacing + 4;
    const reZ = gridOriginZ + rowSpacing / 2;
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.1, 8), ligMat), reX, chipH / 2, reZ);
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.02, 8),
      new THREE.MeshStandardMaterial({ color: 0xE8E8F0, transparent: true, opacity: 0.6, metalness: 0.5 })), reX, chipH / 2 + 0.06, reZ);
    addAt(chipGroup, makeSprite("Ag/AgCl RE", "#8888aa", 7, true), reX, chipH / 2 + 1.6, reZ);

    // ── Contact Pads along bottom edge ──
    const padEdgeZ = chipD / 2 - 1.2;
    const numPads = 16;
    const padStartX = -chipW / 2 + 4;
    const padSpacing = (chipW - 8) / (numPads - 1);
    const contactMat = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.6, metalness: 0.2 });
    for (let i = 0; i < numPads; i++) {
      const px = padStartX + i * padSpacing;
      addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.1, 1.0), contactMat), px, chipH / 2 + 0.05, padEdgeZ);
    }

    // LIG traces from WE to contact pads
    ELECTRODE_TARGETS.forEach((e, idx) => {
      const px = gridOriginX + e.col * colSpacing;
      const pz = gridOriginZ + e.row * rowSpacing;
      const padX = padStartX + idx * padSpacing;
      const traceW = 0.1;
      const vLen = padEdgeZ - pz - weRadius;
      if (vLen > 0.5) {
        addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(traceW, 0.03, vLen), ligMat),
          px, chipH / 2 + 0.02, pz + weRadius + vLen / 2);
      }
      const dx = padX - px;
      if (Math.abs(dx) > 0.2) {
        addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(Math.abs(dx), 0.03, traceW), ligMat),
          (px + padX) / 2, chipH / 2 + 0.02, padEdgeZ - 0.8);
      }
      addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(traceW, 0.03, 1.0), ligMat),
        padX, chipH / 2 + 0.02, padEdgeZ - 0.3);
    });

    // ── Microfluidic overlay (blood cfDNA workflow) ──
    const fluidicsGroup = new THREE.Group();
    chipGroup.add(fluidicsGroup);

    // Blood inlet absorption pad (top center, red-tinted)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(6, 0.2, 3), bloodMat), 0, chipH / 2 + 0.1, -chipD / 2 + 2);
    addAt(fluidicsGroup, makeSprite("Blood inlet", "#CC3333", 8, true), 0, chipH / 2 + 1.8, -chipD / 2 + 2);

    // Plasma separation zone (cellulose filtration — gradient from red to yellow)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(6, 0.15, 2), plasmaMat), 0, chipH / 2 + 0.1, -chipD / 2 + 4.5);
    addAt(fluidicsGroup, makeSprite("Plasma sep.", "#B8860B", 7), 0, chipH / 2 + 1.6, -chipD / 2 + 4.5);

    // Channel: plasma → RPA zone
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.06, 2), channelMat), 0, chipH / 2 + 0.08, -chipD / 2 + 6.5);

    // ── RPA amplification zone (prominent, labeled) ──
    const rpaZoneZ = -chipD / 2 + 8;
    const rpaMat = new THREE.MeshStandardMaterial({ color: 0xCC8844, transparent: true, opacity: 0.55, roughness: 0.7 });
    const rpaBorderMat = new THREE.MeshStandardMaterial({ color: 0x996633, roughness: 0.6 });

    // Shared RPA zone background (large visible zone)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(18, 0.08, 3.5),
      new THREE.MeshStandardMaterial({ color: 0xE8D5B8, transparent: true, opacity: 0.3 })), 0, chipH / 2 + 0.06, rpaZoneZ);
    // Zone label
    addAt(fluidicsGroup, makeSprite("RPA Amplification Zone", "#7A5530", 9, true), 0, chipH / 2 + 2.2, rpaZoneZ);

    // 3 sub-multiplex chambers
    const rpaLabels = ["RPA-A (5-plex)", "RPA-B (5-plex)", "RPA-C (4-plex)"];
    for (let i = 0; i < 3; i++) {
      const rx = -6 + i * 6;
      // Chamber body
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(4.5, 0.18, 2.5), rpaMat), rx, chipH / 2 + 0.12, rpaZoneZ);
      // Chamber border
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(4.7, 0.04, 0.12), rpaBorderMat), rx, chipH / 2 + 0.2, rpaZoneZ - 1.3);
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(4.7, 0.04, 0.12), rpaBorderMat), rx, chipH / 2 + 0.2, rpaZoneZ + 1.3);
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.04, 2.7), rpaBorderMat), rx - 2.35, chipH / 2 + 0.2, rpaZoneZ);
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.04, 2.7), rpaBorderMat), rx + 2.35, chipH / 2 + 0.2, rpaZoneZ);
      // Lyophilized pellet dots (show dried reagents)
      for (let d = 0; d < 4; d++) {
        const dx = rx - 1.2 + d * 0.8;
        addAt(fluidicsGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.06, 8),
          new THREE.MeshStandardMaterial({ color: 0xAA7733, roughness: 0.9 })), dx, chipH / 2 + 0.22, rpaZoneZ);
      }
      addAt(fluidicsGroup, makeSprite(rpaLabels[i], "#7A5530", 7), rx, chipH / 2 + 1.6, rpaZoneZ);
    }

    // Distribution channels from RPA → detection grid (wider, more visible)
    const trunkZ = gridOriginZ - 1.5;
    const distChannelMat = new THREE.MeshStandardMaterial({ color: 0x88BBDD, transparent: true, opacity: 0.35, roughness: 0.4 });
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(6 * colSpacing + 3, 0.06, 0.4), distChannelMat),
      gridOriginX + 3 * colSpacing, chipH / 2 + 0.08, trunkZ);
    // Feeder channels from RPA to trunk
    for (let i = 0; i < 3; i++) {
      const rx = -6 + i * 6;
      const feedLen = Math.abs(rpaZoneZ + 1.3 - trunkZ);
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.06, feedLen), distChannelMat),
        rx, chipH / 2 + 0.08, (rpaZoneZ + 1.3 + trunkZ) / 2);
    }
    for (let c = 0; c < 7; c++) {
      const cx = gridOriginX + c * colSpacing;
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.06, Math.abs(gridOriginZ - trunkZ)), distChannelMat),
        cx, chipH / 2 + 0.08, (gridOriginZ + trunkZ) / 2);
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.06, rowSpacing), distChannelMat),
        cx, chipH / 2 + 0.08, gridOriginZ + rowSpacing / 2);
    }

    // Waste absorption pad (bottom)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(chipW - 10, 0.12, 2),
      new THREE.MeshStandardMaterial({ color: 0x886666, transparent: true, opacity: 0.25 })),
      0, chipH / 2 + 0.1, chipD / 2 - 2);
    addAt(fluidicsGroup, makeSprite("Waste pad", "#884444", 7), 0, chipH / 2 + 1.5, chipD / 2 - 2);

    // Insertion guide mark
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(chipW - 3, 0.08, 0.2),
      new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.5 })), 0, chipH / 2 + 0.04, chipD / 2 - 0.6);

    // ── Capillary flow particles (toggled) ──
    const capillaryParticles = [];
    const capMat = new THREE.MeshStandardMaterial({ color: 0x3B82F6, transparent: true, opacity: 0.7 });
    for (let i = 0; i < 30; i++) {
      const cp = new THREE.Mesh(new THREE.SphereGeometry(0.08, 4, 4), capMat);
      cp.position.set(
        gridOriginX + Math.random() * 6 * colSpacing,
        chipH / 2 - 0.5 - Math.random() * 2,
        gridOriginZ + Math.random() * rowSpacing
      );
      cp.visible = false;
      chipGroup.add(cp);
      capillaryParticles.push({ mesh: cp, speed: 0.02 + Math.random() * 0.04, baseX: cp.position.x });
    }

    // ══════════════════════════════════════════════════════════
    // MODE 2: CROSS-SECTION — Architecture-dependent layer stack
    // ══════════════════════════════════════════════════════════
    const crossGroup = new THREE.Group();
    crossGroup.visible = false;
    scene.add(crossGroup);

    const secR = 3.5;

    // ── Common base: Cellulose paper (390 µm → 4.0 units) ──
    addAt(crossGroup, new THREE.Mesh(new THREE.CylinderGeometry(secR, secR, 4.0, 32),
      new THREE.MeshStandardMaterial({ color: 0xF5F0E8, roughness: 0.85 })), 0, 2.0, 0).castShadow = true;

    // Cellulose fiber texture in cross-section
    const fibCrossMat = new THREE.MeshStandardMaterial({ color: 0xE8E3D8, roughness: 0.95 });
    for (let i = 0; i < 40; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.3);
      const fy = 0.3 + Math.random() * 3.4;
      const fib = new THREE.Mesh(new THREE.CylinderGeometry(0.04 + Math.random() * 0.03, 0.04, 0.3 + Math.random() * 0.8, 4), fibCrossMat);
      fib.position.set(Math.cos(a) * rr, fy, Math.sin(a) * rr);
      fib.rotation.z = Math.PI / 2 + (Math.random() - 0.5) * 0.4;
      fib.rotation.y = Math.random() * Math.PI;
      crossGroup.add(fib);
    }

    // ── LIG-E layer (embedded in top surface, 0.8 units) ──
    addAt(crossGroup, new THREE.Mesh(new THREE.CylinderGeometry(secR, secR, 0.8, 32),
      new THREE.MeshStandardMaterial({ color: 0x2D2D2D, roughness: 0.9 })), 0, 4.4, 0).castShadow = true;

    // LIG-E pore texture
    const poreMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 1 });
    for (let i = 0; i < 80; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.15);
      const pSize = 0.02 + Math.random() * 0.08;
      const pDepth = 0.04 + Math.random() * 0.3;
      const p = new THREE.Mesh(new THREE.CylinderGeometry(pSize, pSize * 0.6, pDepth, 5), poreMat);
      p.position.set(Math.cos(a) * rr, 4.8 + Math.random() * 0.05, Math.sin(a) * rr);
      p.rotation.x = (Math.random() - 0.5) * 0.5;
      crossGroup.add(p);
    }
    // Side pores
    for (let i = 0; i < 30; i++) {
      const a = Math.random() * Math.PI * 2;
      const py = 4.05 + Math.random() * 0.7;
      const p = new THREE.Mesh(new THREE.SphereGeometry(0.03 + Math.random() * 0.06, 4, 4), poreMat);
      p.position.set(Math.cos(a) * (secR - 0.01), py, Math.sin(a) * (secR - 0.01));
      crossGroup.add(p);
    }

    const baseY = 4.85;

    // ────────────────────────────────────────
    // Architecture B: Silver metallization (default)
    // ────────────────────────────────────────
    const archBGroup = new THREE.Group();
    crossGroup.add(archBGroup);

    // Pyrene monolayer (thin amber)
    const pyreneMat = new THREE.MeshStandardMaterial({ color: 0xF59E0B, roughness: 0.3, metalness: 0.2 });
    for (let i = 0; i < 50; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.3);
      const px = Math.cos(a) * rr, pz = Math.sin(a) * rr;
      const pyMesh = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.02, 0.06), pyreneMat);
      pyMesh.position.set(px, baseY, pz);
      pyMesh.rotation.y = Math.random() * Math.PI;
      archBGroup.add(pyMesh);
    }

    // ssDNA probes with silver deposits
    const strandMat = new THREE.MeshStandardMaterial({ color: 0x60A5FA });
    const silverMat = new THREE.MeshStandardMaterial({ color: 0xC0C0C0, metalness: 0.8, roughness: 0.15 });
    const cutMat = new THREE.MeshStandardMaterial({ color: 0x3d7a63 });
    const reporters = [];

    for (let i = 0; i < 40; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.4);
      const x = Math.cos(a) * rr, z = Math.sin(a) * rr;
      const h = 1.2 + Math.random() * 0.6;
      const rnd = Math.random();
      const bendMag = rnd < 0.3 ? 0.12 : rnd > 0.7 ? 0.03 : 0.07;
      const bendX = (Math.random() - 0.5) * 0.4;
      const bendZ = (Math.random() - 0.5) * 0.4;

      // Pyrene anchor (amber dot at base)
      addAt(archBGroup, new THREE.Mesh(new THREE.SphereGeometry(0.04, 6, 6), pyreneMat), x, baseY, z);

      // ssDNA segments
      const segs = [];
      const nSeg = 5, segH = h / nSeg;
      let cx = x, cz = z, cy = baseY + 0.04;
      for (let s = 0; s < nSeg; s++) {
        const t_param = s / nSeg;
        cx += Math.sin(i * 2.3 + s * 1.7) * bendMag + bendX * (t_param * t_param);
        cz += Math.cos(i * 1.9 + s * 1.3) * bendMag + bendZ * (t_param * t_param);
        const segMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, segH, 4), strandMat.clone());
        segMesh.position.set(cx, cy + segH / 2, cz);
        segMesh.rotation.x = bendX * 0.3 + Math.sin(s * 1.4 + i) * 0.15;
        segMesh.rotation.z = bendZ * 0.3 + Math.cos(s * 1.1 + i * 0.7) * 0.15;
        segMesh.userData._rx0 = segMesh.rotation.x;
        segMesh.userData._rz0 = segMesh.rotation.z;
        archBGroup.add(segMesh); segs.push(segMesh);
        cy += segH;
      }

      // Silver nanoparticles along the ssDNA
      const nAg = 2 + Math.floor(Math.random() * 3);
      const agSpheres = [];
      for (let ai = 0; ai < nAg; ai++) {
        const agY = baseY + 0.3 + (ai / nAg) * h * 0.7;
        const agX = cx + (Math.random() - 0.5) * 0.15;
        const agZ = cz + (Math.random() - 0.5) * 0.15;
        const agSize = 0.04 + Math.random() * 0.05;
        const agSphere = new THREE.Mesh(new THREE.SphereGeometry(agSize, 6, 6), silverMat.clone());
        agSphere.position.set(agX, agY, agZ);
        archBGroup.add(agSphere);
        agSpheres.push(agSphere);
      }

      // Cut stub and detached silver
      const cleavageTime = 2 + Math.random() * 26;
      const cutH = h * (0.15 + Math.random() * 0.15);
      const st = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, cutH, 4), cutMat);
      st.position.set(x, baseY + cutH / 2, z);
      st.visible = false; archBGroup.add(st);

      const detachedAg = new THREE.Mesh(new THREE.SphereGeometry(0.06, 6, 6),
        new THREE.MeshStandardMaterial({ color: 0xC0C0C0, transparent: true, opacity: 0.5 }));
      detachedAg.position.set(cx + (Math.random() - 0.5) * 0.5, baseY + h + 1.5 + Math.random() * 2, cz + (Math.random() - 0.5) * 0.5);
      detachedAg.visible = false; archBGroup.add(detachedAg);

      reporters.push({ segs, agSpheres, stub: st, detachedMB: detachedAg, cleavageTime, baseX: x, baseZ: z, archGroup: "B" });
    }

    // ────────────────────────────────────────
    // Architecture A: Vertical-flow ALP
    // ────────────────────────────────────────
    const archAGroup = new THREE.Group();
    archAGroup.visible = false;
    crossGroup.add(archAGroup);

    // Nitrocellulose membrane layer above LIG-E
    addAt(archAGroup, new THREE.Mesh(new THREE.CylinderGeometry(secR, secR, 0.6, 32),
      new THREE.MeshStandardMaterial({ color: 0xF0EDE5, roughness: 0.8, transparent: true, opacity: 0.7 })), 0, baseY + 1.5, 0);
    addAt(archAGroup, makeSprite("Nitrocellulose", "#888", 7), secR + 1.0, baseY + 1.5, 0);

    // Anti-DIG Y-shaped antibodies on nitrocellulose
    const abMat = new THREE.MeshStandardMaterial({ color: 0x8B5CF6, roughness: 0.5 });
    for (let i = 0; i < 15; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.5);
      const abX = Math.cos(a) * rr, abZ = Math.sin(a) * rr;
      // Y stem
      addAt(archAGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.25, 3), abMat), abX, baseY + 1.95, abZ);
      // Y arms
      addAt(archAGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.15, 3), abMat), abX - 0.06, baseY + 2.15, abZ).rotation.z = 0.5;
      addAt(archAGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.15, 3), abMat), abX + 0.06, baseY + 2.15, abZ).rotation.z = -0.5;
    }

    // pAP molecules (green dots in solution)
    const papMat = new THREE.MeshStandardMaterial({ color: 0x34D399, emissive: 0x115533, emissiveIntensity: 0.3, transparent: true, opacity: 0.7 });
    for (let i = 0; i < 20; i++) {
      const px = (Math.random() - 0.5) * (secR * 1.4);
      const pz = (Math.random() - 0.5) * (secR * 1.4);
      addAt(archAGroup, new THREE.Mesh(new THREE.SphereGeometry(0.06, 6, 6), papMat),
        px, baseY + 2.8 + Math.random() * 2, pz);
    }

    // ── Vertical flow pore channels (Arch A key feature) ──
    const poreChannelMat = new THREE.MeshStandardMaterial({ color: 0x88BBDD, transparent: true, opacity: 0.15, roughness: 0.4 });
    const poreChannelDarkMat = new THREE.MeshStandardMaterial({ color: 0x556677, transparent: true, opacity: 0.2, roughness: 0.5 });
    // Vertical channels through cellulose (y=0 to y=4.0)
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2 + Math.random() * 0.3;
      const rr = 0.5 + Math.random() * (secR - 1.2);
      const cx = Math.cos(a) * rr, cz = Math.sin(a) * rr;
      const chRadius = 0.06 + Math.random() * 0.04;
      // Channel through cellulose
      addAt(archAGroup, new THREE.Mesh(new THREE.CylinderGeometry(chRadius, chRadius, 4.0, 6), poreChannelMat), cx, 2.0, cz);
      // Channel through LIG-E
      addAt(archAGroup, new THREE.Mesh(new THREE.CylinderGeometry(chRadius * 0.8, chRadius * 0.8, 0.8, 6), poreChannelDarkMat), cx, 4.4, cz);
      // Channel through nitrocellulose
      addAt(archAGroup, new THREE.Mesh(new THREE.CylinderGeometry(chRadius * 0.7, chRadius * 0.7, 0.6, 6), poreChannelMat), cx, baseY + 1.5, cz);
      // Small pore openings at top surface of LIG-E
      addAt(archAGroup, new THREE.Mesh(new THREE.CylinderGeometry(chRadius * 1.2, chRadius * 0.5, 0.1, 6),
        new THREE.MeshStandardMaterial({ color: 0x3388AA, transparent: true, opacity: 0.3 })), cx, baseY + 0.05, cz);
    }

    // ── Animated fluid wicking particles (Arch A) ──
    const wickParticleMat = new THREE.MeshStandardMaterial({ color: 0x3B82F6, emissive: 0x1a4480, emissiveIntensity: 0.4, transparent: true, opacity: 0.8 });
    const wickParticles = [];
    for (let i = 0; i < 24; i++) {
      const a = (i % 12) / 12 * Math.PI * 2 + Math.random() * 0.3;
      const rr = 0.5 + Math.random() * (secR - 1.2);
      const wp = new THREE.Mesh(new THREE.SphereGeometry(0.05, 6, 6), wickParticleMat);
      wp.position.set(Math.cos(a) * rr, -0.5 + Math.random() * 7, Math.sin(a) * rr);
      wp.visible = false;
      archAGroup.add(wp);
      wickParticles.push({ mesh: wp, speed: 0.015 + Math.random() * 0.025, baseA: a, baseR: rr });
    }

    // ────────────────────────────────────────
    // Architecture C: Direct MB-ssDNA
    // ────────────────────────────────────────
    const archCGroup = new THREE.Group();
    archCGroup.visible = false;
    crossGroup.add(archCGroup);

    // Pyrene monolayer (same as B)
    for (let i = 0; i < 50; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.3);
      const pym = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.02, 0.06), pyreneMat);
      pym.position.set(Math.cos(a) * rr, baseY, Math.sin(a) * rr);
      pym.rotation.y = Math.random() * Math.PI;
      archCGroup.add(pym);
    }

    // ssDNA-MB probes (like old MB-ssDNA)
    const mbMat = new THREE.MeshStandardMaterial({ color: 0x2563eb, emissive: 0x1a3a6b, emissiveIntensity: 0.3 });
    const reportersC = [];
    for (let i = 0; i < 40; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.4);
      const x = Math.cos(a) * rr, z = Math.sin(a) * rr;
      const h = 1.3 + Math.random() * 0.7;
      const rnd = Math.random();
      const bentToward = rnd < 0.3;
      const bendMag = bentToward ? 0.14 : 0.05;
      const bendX = (Math.random() - 0.5) * (bentToward ? 1.0 : 0.3);
      const bendZ = (Math.random() - 0.5) * (bentToward ? 1.0 : 0.3);

      addAt(archCGroup, new THREE.Mesh(new THREE.SphereGeometry(0.04, 6, 6), pyreneMat), x, baseY, z);

      const segs = [];
      const nSeg = 6, segH = h / nSeg;
      let cx = x, cz = z, cy = baseY + 0.04;
      for (let s = 0; s < nSeg; s++) {
        const t_param = s / nSeg;
        cx += Math.sin(i * 2.1 + s * 1.6) * bendMag + bendX * (t_param * t_param);
        cz += Math.cos(i * 1.7 + s * 1.2) * bendMag + bendZ * (t_param * t_param);
        const segMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, segH, 4), strandMat.clone());
        segMesh.position.set(cx, cy + segH / 2, cz);
        segMesh.rotation.x = bendX * 0.3 + Math.sin(s * 1.4 + i) * 0.15;
        segMesh.rotation.z = bendZ * 0.3 + Math.cos(s * 1.1 + i) * 0.15;
        segMesh.userData._rx0 = segMesh.rotation.x;
        segMesh.userData._rz0 = segMesh.rotation.z;
        archCGroup.add(segMesh); segs.push(segMesh);
        cy += segH;
      }

      // MB sphere at tip
      const mbY = bentToward ? baseY + h * 0.35 : baseY + h + 0.06;
      const mb = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 10), mbMat.clone());
      mb.position.set(cx, mbY, cz);
      archCGroup.add(mb);

      const cleavageTime = 2 + Math.random() * 26;
      const cutH = h * (0.15 + Math.random() * 0.2);
      const st = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, cutH, 4), cutMat);
      st.position.set(x, baseY + cutH / 2, z);
      st.visible = false; archCGroup.add(st);

      const detMB = new THREE.Mesh(new THREE.SphereGeometry(0.08, 8, 8),
        new THREE.MeshStandardMaterial({ color: 0x2563eb, transparent: true, opacity: 0.6 }));
      detMB.position.set(cx + (Math.random() - 0.5) * 0.5, baseY + h + 1.5 + Math.random() * 2, cz + (Math.random() - 0.5) * 0.5);
      detMB.visible = false; archCGroup.add(detMB);

      reportersC.push({ segs, mb, stub: st, detachedMB: detMB, cleavageTime, bentToward, baseX: x, baseZ: z, archGroup: "C" });
    }

    // ── Solution-phase elements (shared) ──
    const rnpMat = new THREE.MeshStandardMaterial({ color: 0x33AA55, emissive: 0x115522, emissiveIntensity: 0.2, transparent: true, opacity: 0.75 });
    const rnps = [];
    for (let i = 0; i < 8; i++) {
      const rx = (Math.random() - 0.5) * (secR * 1.4);
      const rz = (Math.random() - 0.5) * (secR * 1.4);
      const ry = baseY + 3.0 + Math.random() * 3.0;
      const rnp = new THREE.Mesh(new THREE.IcosahedronGeometry(0.18, 0), rnpMat);
      rnp.position.set(rx, ry, rz);
      rnp.visible = false; crossGroup.add(rnp);
      rnps.push(rnp);
    }

    // RPA amplicon dsDNA
    const ampliconGroup = new THREE.Group();
    ampliconGroup.visible = false;
    const helixMat1 = new THREE.MeshStandardMaterial({ color: 0x5577CC, transparent: true, opacity: 0.6 });
    const helixMat2 = new THREE.MeshStandardMaterial({ color: 0xCC5577, transparent: true, opacity: 0.6 });
    for (let t = 0; t < 20; t++) {
      const angle = t * 0.5;
      const y = baseY + 5.5 + t * 0.07;
      addAt(ampliconGroup, new THREE.Mesh(new THREE.SphereGeometry(0.04, 4, 4), helixMat1),
        Math.cos(angle) * 0.12, y, Math.sin(angle) * 0.12);
      addAt(ampliconGroup, new THREE.Mesh(new THREE.SphereGeometry(0.04, 4, 4), helixMat2),
        Math.cos(angle + Math.PI) * 0.12, y, Math.sin(angle + Math.PI) * 0.12);
    }
    crossGroup.add(ampliconGroup);

    // ── Scale bars ──
    const sbMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.03, 4.0, 0.03), sbMat), -secR - 0.8, 2.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.03, 0.03), sbMat), -secR - 0.8, 0.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.03, 0.03), sbMat), -secR - 0.8, 4.0, 0);
    addAt(crossGroup, makeSprite("390 µm", "#777", 9), -secR - 1.7, 2.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.8, 0.03), sbMat), -secR - 0.5, 4.4, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.03, 0.03), sbMat), -secR - 0.5, 4.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.03, 0.03), sbMat), -secR - 0.5, 4.8, 0);
    addAt(crossGroup, makeSprite("LIG-E", "#777", 8), -secR - 1.4, 4.4, 0);

    // ══════════════════════════════════════════════════════════
    // MODE 3: SIDE PROFILE — Cellulose + LIG-E
    // ══════════════════════════════════════════════════════════
    const sideGroup = new THREE.Group();
    sideGroup.visible = false;
    scene.add(sideGroup);

    const sW = 40, sCelH = 3.0, sLigH = 0.4, sWaxH = 0.15;
    // Cellulose base
    addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(sW, sCelH, 8),
      new THREE.MeshStandardMaterial({ color: 0xF5F0E8, roughness: 0.85 })), 0, sCelH / 2, 0);
    // LIG-E patches (embedded in surface)
    for (let i = 0; i < 7; i++) {
      const lx = -12 + i * 3.5;
      addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(2.5, sLigH, 8),
        new THREE.MeshStandardMaterial({ color: 0x2D2D2D, roughness: 0.85 })), lx, sCelH - sLigH / 2, 0);
    }
    // Wax barriers
    for (let i = 0; i < 6; i++) {
      const bx = -10.5 + i * 3.5;
      addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(0.15, sWaxH, 8.1), waxMat), bx, sCelH + sWaxH / 2, 0);
    }
    // Contact pads
    for (let i = 0; i < 6; i++) {
      addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.1, 1.2), contactMat), -12 + i * 5, sCelH + 0.05, 5);
    }
    // Labels
    addAt(sideGroup, makeSprite("Cellulose CF3, 390 µm", "#8B7355", 8), -sW / 2 - 4, sCelH / 2, 0);
    addAt(sideGroup, makeSprite("LIG-E, 23 Ω/sq", "#555", 7), -sW / 2 - 4, sCelH - sLigH / 2, 0);
    addAt(sideGroup, makeSprite("Wax barriers", "#3D2B1F", 7), -sW / 2 - 4, sCelH + sWaxH, 0);
    addAt(sideGroup, makeSprite("Contact pads →", "#666", 7), 12, sCelH + 0.5, 5);

    // ══════════════════════════════════════════════════════════
    // CAMERA ORBIT
    // ══════════════════════════════════════════════════════════
    let orbit = { theta: 0.3, phi: -0.45, dist: 60, target: new THREE.Vector3(0, 0, 0) };
    let tgtOrbit = { theta: 0.3, phi: -0.45, dist: 60, target: new THREE.Vector3(0, 0, 0) };
    let isDragging = false, prevMouse = { x: 0, y: 0 };
    const canvas = renderer.domElement;
    const raycaster = new THREE.Raycaster();

    const onDown = (e) => { isDragging = true; const p = e.touches ? e.touches[0] : e; prevMouse = { x: p.clientX, y: p.clientY }; };
    const onUp = () => { isDragging = false; };
    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const p = e.touches ? e.touches[0] : e;
      if (!isDragging) {
        const mx = ((p.clientX - rect.left) / rect.width) * 2 - 1;
        const my = -((p.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(new THREE.Vector2(mx, my), camera);
        const hits = raycaster.intersectObjects(padMeshes);
        if (hits.length > 0) {
          const ud = hits[0].object.userData;
          canvas.style.cursor = "pointer";
          stateRef.current._hovIdx = ud.idx;
          setTooltipInfo({ target: ud.target, drug: ud.drug });
          setTooltipPos({ x: p.clientX - rect.left, y: p.clientY - rect.top });
        } else {
          canvas.style.cursor = "grab";
          stateRef.current._hovIdx = -1;
          setTooltipInfo(null);
        }
      }
      if (!isDragging) return;
      const dx = p.clientX - prevMouse.x, dy = p.clientY - prevMouse.y;
      tgtOrbit.theta += dx * 0.005;
      tgtOrbit.phi = Math.max(-1.3, Math.min(-0.08, tgtOrbit.phi + dy * 0.005));
      prevMouse = { x: p.clientX, y: p.clientY };
    };
    const onWheel = (e) => { e.preventDefault(); tgtOrbit.dist = Math.max(12, Math.min(120, tgtOrbit.dist + e.deltaY * 0.06)); };
    const onClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const my = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(new THREE.Vector2(mx, my), camera);
      const hits = raycaster.intersectObjects(padMeshes);
      if (hits.length > 0) stateRef.current._selectPad(hits[0].object.userData.idx, hits[0].object.userData.target);
    };

    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("touchstart", onDown, { passive: true });
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("touchmove", onMove, { passive: true });
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("click", onClick);

    // ── Animation loop ──
    let frameId, time = 0;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      time += 0.016;
      orbit.theta += (tgtOrbit.theta - orbit.theta) * 0.07;
      orbit.phi += (tgtOrbit.phi - orbit.phi) * 0.07;
      orbit.dist += (tgtOrbit.dist - orbit.dist) * 0.07;
      orbit.target.lerp(tgtOrbit.target instanceof THREE.Vector3 ? tgtOrbit.target : new THREE.Vector3(tgtOrbit.target.x, tgtOrbit.target.y, tgtOrbit.target.z), 0.07);
      camera.position.x = orbit.target.x + orbit.dist * Math.sin(orbit.theta) * Math.cos(orbit.phi);
      camera.position.y = orbit.target.y + orbit.dist * Math.sin(-orbit.phi);
      camera.position.z = orbit.target.z + orbit.dist * Math.cos(orbit.theta) * Math.cos(orbit.phi);
      camera.lookAt(orbit.target);

      // Animate reporters (Architecture B — silver)
      reporters.forEach((r, i) => {
        r.segs.forEach((seg, si) => {
          if (seg.visible) {
            seg.rotation.x = seg.userData._rx0 + Math.sin(time * 1.9 + i * 0.7 + si * 0.5) * 0.05;
            seg.rotation.z = seg.userData._rz0 + Math.cos(time * 1.5 + i * 1.1 + si * 0.3) * 0.05;
          }
        });
        if (r.detachedMB.visible) {
          r.detachedMB.position.y += Math.sin(time * 0.5 + i) * 0.002;
          r.detachedMB.position.x += Math.sin(time * 0.3 + i * 1.3) * 0.001;
        }
      });

      // Animate reporters (Architecture C — MB)
      reportersC.forEach((r, i) => {
        r.segs.forEach((seg, si) => {
          if (seg.visible) {
            seg.rotation.x = seg.userData._rx0 + Math.sin(time * 1.9 + i * 0.7 + si * 0.5) * 0.06;
            seg.rotation.z = seg.userData._rz0 + Math.cos(time * 1.5 + i * 1.1 + si * 0.3) * 0.06;
          }
        });
        if (r.mb && r.mb.visible) r.mb.material.emissiveIntensity = 0.2 + 0.15 * Math.sin(time * 3.14 + i);
        if (r.detachedMB.visible) {
          r.detachedMB.position.y += Math.sin(time * 0.5 + i) * 0.002;
          r.detachedMB.position.x += Math.sin(time * 0.3 + i * 1.3) * 0.001;
        }
      });

      // Animate RNP complexes
      rnps.forEach((rnp, i) => {
        if (rnp.visible) {
          rnp.rotation.y += 0.01;
          rnp.position.y += Math.sin(time * 0.7 + i * 2) * 0.003;
          rnp.position.x += Math.cos(time * 0.4 + i * 1.5) * 0.002;
        }
      });

      // Vertical wicking particles (Arch A)
      wickParticles.forEach((wp, i) => {
        if (wp.mesh.visible) {
          wp.mesh.position.y += wp.speed;
          wp.mesh.position.x += Math.sin(time * 1.5 + i * 2.1) * 0.002;
          wp.mesh.position.z += Math.cos(time * 1.2 + i * 1.7) * 0.002;
          // Reset to bottom when reaching top
          if (wp.mesh.position.y > baseY + 2.2) {
            wp.mesh.position.y = -0.2 + Math.random() * 0.5;
            const a = wp.baseA + (Math.random() - 0.5) * 0.4;
            const rr = wp.baseR + (Math.random() - 0.5) * 0.3;
            wp.mesh.position.x = Math.cos(a) * rr;
            wp.mesh.position.z = Math.sin(a) * rr;
          }
        }
      });

      // Capillary flow particles
      capillaryParticles.forEach((cp, i) => {
        if (cp.mesh.visible) {
          cp.mesh.position.x += cp.speed;
          cp.mesh.position.y += Math.sin(time * 2 + i) * 0.005;
          if (cp.mesh.position.x > gridOriginX + 6 * colSpacing + 3) {
            cp.mesh.position.x = gridOriginX - 2;
          }
        }
      });

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const w = container.clientWidth, h2 = Math.round(w * 9 / 16);
      container.style.height = h2 + "px";
      renderer.setSize(w, h2);
      camera.aspect = w / h2; camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);

    stateRef.current = {
      chipGroup, crossGroup, sideGroup, fluidicsGroup, waxGroup, labelSprites,
      reporters, reportersC, padMeshes, rnps, ampliconGroup, orbit, tgtOrbit,
      archAGroup, archBGroup, archCGroup, capillaryParticles, wickParticles,
      _hovIdx: -1,
      _selectPad: (idx, target) => { setSelectedPad({ idx, target }); setMode(2); },
      _toMode1: () => { setMode(1); setSelectedPad(null); setCas12aActive(false); },
    };

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchend", onUp);
      window.removeEventListener("resize", onResize);
      scene.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) { if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose()); else obj.material.dispose(); }
      });
      renderer.dispose();
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
    };
  }, []);

  // ── Mode transitions ──
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.chipGroup.visible = mode === 1;
    s.crossGroup.visible = mode === 2;
    s.sideGroup.visible = mode === 3;
    if (mode === 1) {
      s.tgtOrbit.dist = 60; s.tgtOrbit.theta = 0.3; s.tgtOrbit.phi = -0.45;
      s.tgtOrbit.target = new THREE.Vector3(0, 0, 0);
      setExpandedCurve(false); setExpandedETransfer(false);
    } else if (mode === 2) {
      s.tgtOrbit.dist = 18; s.tgtOrbit.theta = 0.4; s.tgtOrbit.phi = -0.3;
      s.tgtOrbit.target = new THREE.Vector3(0, 4.5, 0);
    } else if (mode === 3) {
      s.tgtOrbit.dist = 25; s.tgtOrbit.theta = 0.0; s.tgtOrbit.phi = -0.1;
      s.tgtOrbit.target = new THREE.Vector3(0, 2, 0);
    }
  }, [mode, selectedPad]);

  // ── Toggle visibility ──
  useEffect(() => { const s = stateRef.current; if (s?.fluidicsGroup) s.fluidicsGroup.visible = showFluidics; }, [showFluidics]);
  useEffect(() => { const s = stateRef.current; if (s?.waxGroup) s.waxGroup.visible = showWaxBarriers; }, [showWaxBarriers]);
  useEffect(() => { const s = stateRef.current; if (s?.labelSprites) s.labelSprites.forEach(l => { l.visible = showLabels; }); }, [showLabels]);
  useEffect(() => {
    const s = stateRef.current;
    if (s?.capillaryParticles) s.capillaryParticles.forEach(cp => { cp.mesh.visible = showCapillaryFlow; });
  }, [showCapillaryFlow]);

  // ── Architecture switching ──
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.archAGroup.visible = arch === "A";
    s.archBGroup.visible = arch === "B";
    s.archCGroup.visible = arch === "C";
  }, [arch]);

  // ── Progressive Cas12a cleavage ──
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;

    // Architecture B reporters (silver)
    if (arch === "B") {
      s.reporters.forEach(r => {
        if (cas12aActive) {
          const cleaved = incubationMin >= r.cleavageTime;
          r.segs.forEach(seg => { seg.visible = !cleaved; });
          r.agSpheres.forEach(ag => { ag.visible = !cleaved; });
          r.stub.visible = cleaved;
          r.detachedMB.visible = cleaved;
        } else {
          r.segs.forEach(seg => { seg.visible = true; });
          r.agSpheres.forEach(ag => { ag.visible = true; });
          r.stub.visible = false;
          r.detachedMB.visible = false;
        }
      });
    }

    // Architecture C reporters (MB)
    if (arch === "C") {
      s.reportersC.forEach(r => {
        if (cas12aActive) {
          const cleaved = incubationMin >= r.cleavageTime;
          r.segs.forEach(seg => { seg.visible = !cleaved; });
          r.mb.visible = !cleaved;
          r.stub.visible = cleaved;
          r.detachedMB.visible = cleaved;
        } else {
          r.segs.forEach(seg => { seg.visible = true; });
          r.mb.visible = true;
          r.stub.visible = false;
          r.detachedMB.visible = false;
        }
      });
    }

    s.rnps.forEach(rnp => { rnp.visible = cas12aActive; });
    s.ampliconGroup.visible = cas12aActive;

    // Vertical wicking particles (Arch A) — flow on activation
    if (s.wickParticles) {
      s.wickParticles.forEach(wp => { wp.mesh.visible = cas12aActive && arch === "A"; });
    }
  }, [cas12aActive, incubationMin, arch]);

  // ── Computed data for selected pad ──
  const selTarget = selectedPad ? ELECTRODE_TARGETS[selectedPad.idx] : null;
  const selDrug = selTarget?.drug || null;
  const selR = selectedPad ? results.find(x => x.label === selectedPad.target) : null;
  const selEff = selectedPad ? getEfficiency(selectedPad.target) : null;
  const selStrat = selectedPad ? targetStrategy(selectedPad.target) : null;
  const selDisc = selR?.disc && selR.disc < 900 ? selR.disc : null;
  const selScore = selR?.ensembleScore || selR?.score || null;

  const deltaI = selectedPad && computeGamma && echemGamma0_mol ? (() => {
    const G = computeGamma(incubationMin * 60, getEfficiency(selectedPad.target), echemKtrans);
    return ((1 - G / echemGamma0_mol) * 100).toFixed(1);
  })() : null;

  // ── Architecture-dependent curve data ──
  const am = ARCH_META[arch];
  const curveData = selectedPad ? (() => {
    const G_after = computeGamma(incubationMin * 60, getEfficiency(selectedPad.target), echemKtrans);
    if (curveMode === "SWV") return { before: am.swv(echemGamma0_mol, echemGamma0_mol), after: am.swv(G_after, echemGamma0_mol) };
    if (curveMode === "EIS") return { before: miniEIS(echemGamma0_mol, echemGamma0_mol), after: miniEIS(G_after, echemGamma0_mol) };
    return null;
  })() : null;

  const svgPathVolt = (pts, w, h) => {
    if (!pts?.length) return "";
    const maxI = Math.max(...pts.map(p => Math.abs(p.I)), 0.001);
    return pts.map((p, i) => {
      const x = (i / (pts.length - 1)) * w;
      const y = h - (Math.abs(p.I) / maxI) * h * 0.82 - h * 0.06;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  };

  const svgPathEIS = (pts, w, h) => {
    if (!pts?.length) return "";
    const maxZr = Math.max(...pts.map(p => p.Zr), 1);
    const maxZi = Math.max(...pts.map(p => p.Zi), 1);
    return pts.map((p, i) => {
      const x = ((p.Zr - 40) / (maxZr - 40 + 1)) * w * 0.88 + w * 0.06;
      const y = h - (p.Zi / (maxZi + 1)) * h * 0.82 - h * 0.06;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  };

  const _reporters = stateRef.current?.reporters || [];
  const _reportersC = stateRef.current?.reportersC || [];
  const allReporters = arch === "B" ? _reporters : arch === "C" ? _reportersC : [];
  const cleavedCount = cas12aActive && allReporters.length > 0 ? Math.round(allReporters.length * Math.min(1, incubationMin / 28)) : 0;

  // Architecture-dependent layer labels
  const LAYER_LABELS = {
    A: [
      { label: "Streptavidin-ALP enzyme", color: "#8B5CF6", dim: "" },
      { label: "Anti-DIG antibodies (nitrocellulose)", color: "#8B5CF6", dim: "" },
      { label: "Biotin-DIG reporters (solution)", color: "#60A5FA", dim: "" },
      { label: "LIG-E (23 ± 2 Ω/sq, porous, hydrophilic)", color: "#2D2D2D", dim: "" },
      { label: "Cellulose CF3 ((NH₄)SO₃NH₂ pre-treated)", color: "#F5F0E8", dim: "390 µm" },
    ],
    B: [
      { label: "Ag deposits (Ag⁰ → Ag⁺ + e⁻ at +0.16 V)", color: "#C0C0C0", dim: "" },
      { label: "ssDNA probe (zone-specific, amine-terminated)", color: "#60A5FA", dim: "" },
      { label: "Pyrene-NHS (PBASE, π-π on graphene, ~74 kJ/mol)", color: "#F59E0B", dim: "" },
      { label: "LIG-E (23 Ω/sq, cellulose-derived, porous)", color: "#2D2D2D", dim: "" },
      { label: "Cellulose paper (CF3, (NH₄)SO₃NH₂ pre-treated)", color: "#F5F0E8", dim: "390 µm" },
    ],
    C: [
      { label: "MB (E° = −0.22 V, n = 2e⁻)", color: "#2563eb", dim: "" },
      { label: "ssDNA-MB probe (pyrene-tethered)", color: "#60A5FA", dim: "" },
      { label: "Pyrene-NHS (PBASE, π-π stacking)", color: "#F59E0B", dim: "" },
      { label: "LIG-E (23 Ω/sq, hydrophilic)", color: "#2D2D2D", dim: "" },
      { label: "Cellulose paper (CF3, 390 µm)", color: "#F5F0E8", dim: "390 µm" },
    ],
  };

  // Architecture-dependent electron transfer SVG
  const eTransferSVG = {
    A: (
      <svg width={100} height={40} viewBox="0 0 100 40">
        <rect x={0} y={32} width={100} height={8} fill="#374151" rx={1} opacity={0.6} />
        <text x={50} y={38} fontSize="4" fill="#999" textAnchor="middle">LIG-E surface</text>
        <circle cx={30} cy={15} r={4} fill="#34D399" />
        <text x={30} y={16.5} fontSize="3.5" fill="#fff" textAnchor="middle">pAP</text>
        <path d="M30,19 L30,32" stroke="#34D399" strokeWidth="1" strokeDasharray="2,1" />
        <text x={42} y={24} fontSize="3.5" fill="#16A34A">e⁻ →</text>
        <circle cx={70} cy={8} r={3} fill="#8B5CF6" />
        <text x={70} y={9.5} fontSize="3" fill="#fff" textAnchor="middle">ALP</text>
        <path d="M67,10 L50,18 L33,15" stroke="#8B5CF6" strokeWidth="0.5" strokeDasharray="1,1" fill="none" />
        <text x={80} y={8} fontSize="3.5" fill="#8B5CF6">p-APP→pAP</text>
      </svg>
    ),
    B: (
      <svg width={100} height={40} viewBox="0 0 100 40">
        <rect x={0} y={32} width={100} height={8} fill="#374151" rx={1} opacity={0.6} />
        <text x={50} y={38} fontSize="4" fill="#999" textAnchor="middle">LIG-E (graphene)</text>
        <line x1={25} y1={32} x2={25} y2={10} stroke="#60A5FA" strokeWidth="1" />
        <circle cx={22} cy={15} r={3} fill="#C0C0C0" />
        <circle cx={28} cy={20} r={2.5} fill="#C0C0C0" />
        <text x={12} y={22} fontSize="3.5" fill="#C0C0C0">Ag⁰</text>
        <path d="M28,22 L40,30" stroke="#16A34A" strokeWidth="1" strokeDasharray="2,1" />
        <text x={42} y={28} fontSize="3.5" fill="#16A34A">Ag⁰→Ag⁺+e⁻</text>
        <text x={5} y={8} fontSize="3.5" fill="#60A5FA">ssDNA scaffold</text>
        <rect x={23} y={31} width={4} height={2} fill="#F59E0B" rx={0.5} />
        <text x={32} y={33} fontSize="3" fill="#F59E0B">pyrene π-π</text>
      </svg>
    ),
    C: (
      <svg width={100} height={40} viewBox="0 0 100 40">
        <rect x={0} y={32} width={100} height={8} fill="#374151" rx={1} opacity={0.6} />
        <text x={50} y={38} fontSize="4" fill="#999" textAnchor="middle">LIG-E (graphene)</text>
        <path d="M20,32 Q18,20 22,10" fill="none" stroke="#60A5FA" strokeWidth="1" />
        <circle cx={22} cy={10} r={3} fill="#2563eb" />
        <text x={22} y={11.5} fontSize="3" fill="#fff" textAnchor="middle">MB</text>
        <text x={10} y={20} fontSize="3.5" fill="#16A34A">⚡ e⁻</text>
        <text x={5} y={6} fontSize="3.5" fill="#16A34A">&lt;2 nm</text>
        <path d="M65,32 Q63,22 68,4" fill="none" stroke="#60A5FA" strokeWidth="1" />
        <circle cx={68} cy={4} r={3} fill="#2563eb" />
        <text x={68} y={5.5} fontSize="3" fill="#fff" textAnchor="middle">MB</text>
        <text x={74} y={10} fontSize="3.5" fill="#ef4444">✗ too far</text>
        <rect x={18} y={31} width={4} height={2} fill="#F59E0B" rx={0.5} />
        <text x={28} y={33} fontSize="3" fill="#F59E0B">pyrene</text>
      </svg>
    ),
  };

  // ══════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════
  return (
    <div style={{ position: "relative", borderRadius: "10px", overflow: "hidden", background: "#F8F9FA" }}>
      <div ref={mountRef} style={{ width: "100%", minHeight: 280 }} />

      {/* ═══ MODE 1: CHIP OVERVIEW ═══ */}
      {mode === 1 && (
        <>
          <div style={{ position: "absolute", top: 12, left: 16, fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: HEADING, textShadow: "0 1px 3px rgba(255,255,255,0.9)" }}>
            COMPASS 14-Plex MDR-TB Chip
          </div>
          <div style={{ position: "absolute", top: 12, right: 16, display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
            {[
              { label: "Fluidics", active: showFluidics, toggle: () => setShowFluidics(!showFluidics) },
              { label: "Wax Barriers", active: showWaxBarriers, toggle: () => setShowWaxBarriers(!showWaxBarriers) },
              { label: "Labels", active: showLabels, toggle: () => setShowLabels(!showLabels) },
              { label: "Capillary Flow", active: showCapillaryFlow, toggle: () => setShowCapillaryFlow(!showCapillaryFlow) },
            ].map(b => (
              <button key={b.label} onClick={b.toggle} style={{
                fontSize: 8, fontWeight: 600, padding: "2px 8px", borderRadius: 4,
                border: "1px solid #D1D5DB", cursor: "pointer", fontFamily: MONO,
                background: b.active ? "rgba(51,136,170,0.15)" : "rgba(255,255,255,0.88)",
                color: b.active ? "#1a6680" : "#6B7280",
              }}>
                {b.active ? "Hide" : "Show"} {b.label}
              </button>
            ))}
            <button onClick={() => setMode(3)} style={{
              fontSize: 8, fontWeight: 600, padding: "2px 8px", borderRadius: 4,
              border: "1px solid #D1D5DB", cursor: "pointer", fontFamily: MONO,
              background: "rgba(255,255,255,0.88)", color: "#6B7280",
            }}>
              Side Profile
            </button>
          </div>

          {/* Architecture toggle */}
          <div style={{ position: "absolute", top: 40, right: 16, display: "flex", gap: 2 }}>
            {["A", "B", "C"].map(a => (
              <button key={a} onClick={() => setArch(a)} style={{
                fontSize: 8, fontWeight: arch === a ? 700 : 500, padding: "2px 8px", borderRadius: 3,
                border: arch === a ? "1px solid #374151" : "1px solid #D1D5DB", cursor: "pointer", fontFamily: MONO,
                background: arch === a ? "#374151" : "rgba(255,255,255,0.9)",
                color: arch === a ? "#fff" : "#6B7280",
              }}>
                {a}: {ARCH_META[a].short}
              </button>
            ))}
          </div>

          {/* Drug class legend */}
          <div style={{ position: "absolute", bottom: 12, left: 16, display: "flex", gap: 4, flexWrap: "wrap" }}>
            {[
              { d: "TB ID", c: "#22d3ee" }, { d: "RIF", c: "#ef4444" }, { d: "INH", c: "#f97316" },
              { d: "EMB", c: "#eab308" }, { d: "PZA", c: "#eab308" }, { d: "FQ", c: "#a855f7" },
              { d: "KAN/AMK", c: "#ec4899" }, { d: "Human Ctrl", c: "#22d3ee" },
            ].map(l => (
              <span key={l.d} style={{ fontSize: 7, fontWeight: 700, fontFamily: MONO, display: "flex", alignItems: "center", gap: 2, background: "rgba(255,255,255,0.9)", padding: "1px 5px", borderRadius: 3 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: l.c, display: "inline-block" }} />{l.d}
              </span>
            ))}
          </div>

          {/* Chip specs footer */}
          <div style={{ position: "absolute", bottom: 12, right: 16 }}>
            <span style={{ fontSize: 7, color: "#6B7280", background: "rgba(255,255,255,0.9)", padding: "2px 6px", borderRadius: 3, fontFamily: MONO }}>
              ~40 × 25 mm · Cellulose CF3 + LIG-E · 14 WE + CE + RE · Pyrene-ssDNA · SWV readout
            </span>
          </div>
        </>
      )}

      {/* ═══ MODE 2: CROSS-SECTION ═══ */}
      {mode === 2 && selectedPad && (
        <>
          {/* Backdrop for expanded panels */}
          {(expandedCurve || expandedETransfer) && (
            <div onClick={() => { setExpandedCurve(false); setExpandedETransfer(false); }} style={{
              position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.3)",
              zIndex: 50, backdropFilter: "blur(2px)",
            }} />
          )}
          {/* Info panel */}
          <div style={{ position: "absolute", top: 12, left: 16, background: "rgba(255,255,255,0.95)", padding: "10px 14px", borderRadius: 8, border: "1px solid #E3E8EF", maxWidth: 280, backdropFilter: "blur(8px)" }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#111827", fontFamily: HEADING }}>
              {selectedPad.target} · <span style={{ color: DRUG_CSS[selDrug] || "#888" }}>{selDrug}</span>
            </div>
            <div style={{ fontSize: 9, color: "#6B7280", fontFamily: MONO, marginTop: 3, lineHeight: 1.7 }}>
              S_eff = {selEff?.toFixed(3)}{selScore != null && ` · Score = ${selScore.toFixed(2)}`}{selDisc ? ` · D = ${selDisc.toFixed(1)}×` : ""}<br />
              {selStrat} detection
              {selR?.hasPrimers && <span style={{ color: "#16A34A" }}> · RPA primers</span>}
            </div>
            {selR && (
              <div style={{ fontSize: 8, color: "#9CA3AF", fontFamily: MONO, marginTop: 3, lineHeight: 1.6, borderTop: "1px solid #E3E8EF", paddingTop: 3 }}>
                {selR.spacer && <div>crRNA: <span style={{ color: "#374151", letterSpacing: "0.5px" }}>{selR.spacer.slice(0, 20)}{selR.spacer.length > 20 ? "…" : ""}</span></div>}
                {selR.pam && <div>PAM: <span style={{ color: "#374151" }}>{selR.pam}</span>{selR.pamVariant && <span> ({selR.pamVariant})</span>}</div>}
                {selR.pamDisrupted != null && <div>PAM-overlap: <span style={{ color: selR.pamDisrupted ? "#7c3aed" : "#374151" }}>{selR.pamDisrupted ? "yes — binary disc" : "no"}</span></div>}
                {selR.amplicon && <div>Amplicon: <span style={{ color: selR.amplicon <= 120 ? "#16A34A" : "#ef4444" }}>{selR.amplicon} bp</span>{selR.amplicon <= 120 ? " (cfDNA ✓)" : " (⚠ >120)"}</div>}
                {deltaI != null && (
                  <div>
                    Expected ΔI%: <span style={{ color: "#16A34A", fontWeight: 700 }}>{deltaI}%</span> @ {incubationMin} min
                    {selDisc != null && selDisc <= 2.0 && (
                      <div style={{ marginTop: 2, padding: "2px 4px", background: "#FEF3C7", borderRadius: 2, fontSize: 7, color: "#92400E", lineHeight: 1.4 }}>
                        ⚠ D = {selDisc.toFixed(1)}× — WT allele ΔI% ≈ MUT ΔI% (S_eff_WT = {(selEff / selDisc).toFixed(3)}).
                        Discrimination relies on AS-RPA primer selectivity, not crRNA alone.
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          <button onClick={() => stateRef.current?._toMode1()} style={{ position: "absolute", top: 12, right: 16, fontSize: 10, fontWeight: 600, padding: "6px 14px", borderRadius: 6, border: "1px solid #E3E8EF", background: "#fff", cursor: "pointer", fontFamily: HEADING, color: "#374151" }}>
            ← Back to chip
          </button>

          {/* Architecture toggle */}
          <div style={{ position: "absolute", top: 42, right: 16, display: "flex", gap: 2 }}>
            {["A", "B", "C"].map(a => (
              <button key={a} onClick={() => setArch(a)} style={{
                fontSize: 8, fontWeight: arch === a ? 700 : 500, padding: "2px 8px", borderRadius: 3,
                border: arch === a ? "1px solid #374151" : "1px solid #D1D5DB", cursor: "pointer", fontFamily: MONO,
                background: arch === a ? "#374151" : "rgba(255,255,255,0.9)",
                color: arch === a ? "#fff" : "#6B7280",
              }}>
                {a}: {ARCH_META[a].short}
              </button>
            ))}
          </div>

          {/* Layer labels */}
          <div style={{ position: "absolute", left: 16, top: "36%", transform: "translateY(-50%)", display: "flex", flexDirection: "column", gap: 2 }}>
            {(LAYER_LABELS[arch] || []).map(l => (
              <div key={l.label} style={{ fontSize: 7, fontFamily: MONO, color: "#374151", background: "rgba(255,255,255,0.92)", padding: "2px 6px", borderRadius: 3, borderLeft: `3px solid ${l.color}`, lineHeight: 1.3 }}>
                ← {l.label}{l.dim && <span style={{ color: "#9CA3AF" }}> ({l.dim})</span>}
              </div>
            ))}
            <div style={{ fontSize: 6, color: "#B0B0B0", fontFamily: MONO, fontStyle: "italic", marginTop: 1, paddingLeft: 4 }}>
              ⚠ vertical scale exaggerated
            </div>
          </div>

          {/* ── Electrochemistry curves panel (expandable) ── */}
          {curveData && (
            <div onClick={() => setExpandedCurve(!expandedCurve)} style={{ position: expandedCurve ? "fixed" : "absolute", bottom: expandedCurve ? "50%" : 55, right: expandedCurve ? "50%" : 16, transform: expandedCurve ? "translate(50%, 50%)" : "none", background: "rgba(255,255,255,0.98)", padding: expandedCurve ? "16px 20px" : "8px 12px", borderRadius: 8, border: "1px solid #E3E8EF", backdropFilter: "blur(8px)", minWidth: expandedCurve ? 420 : 190, zIndex: expandedCurve ? 100 : 10, cursor: "pointer", boxShadow: expandedCurve ? "0 8px 40px rgba(0,0,0,0.2)" : "none", transition: "all 0.2s ease" }}>
              <div style={{ display: "flex", gap: 2, marginBottom: 4, alignItems: "center" }}>
                {["SWV", "EIS"].map(m => (
                  <button key={m} onClick={(e) => { e.stopPropagation(); setCurveMode(m); }} style={{
                    fontSize: expandedCurve ? 10 : 7, fontWeight: curveMode === m ? 700 : 500, fontFamily: MONO,
                    padding: expandedCurve ? "4px 12px" : "2px 8px", borderRadius: 3, border: "none", cursor: "pointer",
                    background: curveMode === m ? "#374151" : "#F3F4F6",
                    color: curveMode === m ? "#fff" : "#6B7280",
                  }}>{m}</button>
                ))}
                <span style={{ fontSize: expandedCurve ? 9 : 6, color: "#B0B0B0", marginLeft: 4, fontFamily: MONO }}>Signal-{am.sig}</span>
                <span style={{ fontSize: expandedCurve ? 8 : 5.5, color: "#C0C0C0", marginLeft: "auto", fontFamily: MONO }}>{expandedCurve ? "click to collapse" : "click to expand"}</span>
              </div>

              <svg width={expandedCurve ? 400 : 190} height={expandedCurve ? 200 : 90} viewBox="0 0 190 90" style={{ display: "block" }}>
                {curveMode === "SWV" ? (
                  <>
                    <path d={svgPathVolt(curveData.before, 190, 90)} fill="none" stroke="#93C5FD" strokeWidth={arch === "A" ? "2.5" : "1.5"} strokeDasharray="4,3" />
                    <path d={svgPathVolt(curveData.after, 190, 90)} fill="none" stroke="#ef4444" strokeWidth="2" />
                    <line x1="0" y1="86" x2="190" y2="86" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="2" y="85" fontSize="5" fill="#9CA3AF">{am.eRange.split(" to ")[0]}</text>
                    <text x="140" y="85" fontSize="5" fill="#9CA3AF">{am.eRange.split(" to ")[1]} V</text>
                    <line x1="95" y1="0" x2="95" y2="90" stroke="#E5E7EB" strokeWidth="0.5" strokeDasharray="2,2" />
                    <text x="97" y="8" fontSize="5" fill="#9CA3AF">{am.peak} ({am.label})</text>
                    {deltaI && <text x="130" y="20" fontSize="6.5" fill="#ef4444" fontWeight="bold">ΔI = {deltaI}%</text>}
                    {arch === "A" && cas12aActive && (
                      <>
                        {/* Signal-ON annotation arrow */}
                        <line x1="105" y1="55" x2="105" y2="30" stroke="#16A34A" strokeWidth="1.2" markerEnd="url(#arrowUp)" />
                        <defs><marker id="arrowUp" markerWidth="6" markerHeight="6" refX="3" refY="6" orient="auto"><path d="M1,6 L3,0 L5,6" fill="#16A34A" /></marker></defs>
                        <text x="110" y="46" fontSize="5.5" fill="#16A34A" fontWeight="bold">peak GROWS</text>
                        <text x="110" y="53" fontSize="4.5" fill="#16A34A">+target → +pAP</text>
                      </>
                    )}
                    {arch !== "A" && cas12aActive && (
                      <>
                        <line x1="105" y1="30" x2="105" y2="55" stroke="#ef4444" strokeWidth="1.2" markerEnd="url(#arrowDown)" />
                        <defs><marker id="arrowDown" markerWidth="6" markerHeight="6" refX="3" refY="0" orient="auto"><path d="M1,0 L3,6 L5,0" fill="#ef4444" /></marker></defs>
                        <text x="110" y="40" fontSize="5.5" fill="#ef4444" fontWeight="bold">peak SHRINKS</text>
                        <text x="110" y="47" fontSize="4.5" fill="#ef4444">signal-OFF</text>
                      </>
                    )}
                  </>
                ) : (
                  <>
                    <path d={svgPathEIS(curveData.before, 190, 90)} fill="none" stroke="#93C5FD" strokeWidth="2" strokeDasharray="4,3" />
                    <path d={svgPathEIS(curveData.after, 190, 90)} fill="none" stroke="#ef4444" strokeWidth="2" />
                    <line x1="0" y1="86" x2="190" y2="86" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="2" y="85" fontSize="5" fill="#9CA3AF">Z' (Ω)</text>
                    <line x1="2" y1="0" x2="2" y2="86" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="4" y="8" fontSize="5" fill="#9CA3AF">−Z'' (Ω)</text>
                  </>
                )}
              </svg>

              <div style={{ fontSize: 6, color: "#9CA3AF", fontFamily: MONO, display: "flex", gap: 8, marginTop: 2, alignItems: "center" }}>
                <span><span style={{ color: "#93C5FD", fontWeight: 700 }}>- - -</span> baseline (t=0)</span>
                <span><span style={{ color: "#ef4444", fontWeight: 700 }}>——</span> t={incubationMin}m</span>
                <span style={{ color: am.sig === "ON" ? "#16A34A" : "#ef4444", fontWeight: 700, fontSize: 7 }}>Signal-{am.sig}</span>
              </div>
              <div style={{ fontSize: 5.5, color: "#C0C0C0", fontFamily: MONO, marginTop: 1 }}>
                Freq: {am.freq} | Amp: {am.amp} | Step: {am.step}
              </div>
            </div>
          )}

          {/* ── Electron transfer inset (expandable) ── */}
          <div onClick={() => setExpandedETransfer(!expandedETransfer)} style={{
            position: expandedETransfer ? "fixed" : "absolute",
            bottom: expandedETransfer ? "50%" : 55,
            left: expandedETransfer ? "50%" : 16,
            transform: expandedETransfer ? "translate(-50%, 50%)" : "none",
            background: "rgba(255,255,255,0.96)", padding: expandedETransfer ? "16px 20px" : "5px 8px",
            borderRadius: expandedETransfer ? 10 : 6, border: "1px solid #E3E8EF", backdropFilter: "blur(8px)",
            zIndex: expandedETransfer ? 100 : 10, cursor: "pointer",
            boxShadow: expandedETransfer ? "0 8px 40px rgba(0,0,0,0.2)" : "none",
            transition: "all 0.2s ease",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: expandedETransfer ? 8 : 2 }}>
              <div style={{ fontSize: expandedETransfer ? 11 : 6.5, fontWeight: 700, color: "#374151", fontFamily: MONO }}>e⁻ Transfer — Arch {arch}: {ARCH_META[arch].name}</div>
              <span style={{ fontSize: expandedETransfer ? 8 : 5, color: "#C0C0C0", fontFamily: MONO, marginLeft: 8 }}>{expandedETransfer ? "click to collapse" : "click to expand"}</span>
            </div>
            <div style={{ transform: expandedETransfer ? "scale(2.8)" : "scale(1)", transformOrigin: "top left", width: expandedETransfer ? 280 : "auto", height: expandedETransfer ? 112 : "auto" }}>
              {eTransferSVG[arch]}
            </div>
            {expandedETransfer && (
              <div style={{ marginTop: 120, fontSize: 10, color: "#6B7280", fontFamily: MONO, lineHeight: 1.7, maxWidth: 380 }}>
                {arch === "A" && "ALP converts p-APP → pAP in solution. pAP diffuses to LIG-E surface and is oxidized (pAP → quinoneimine + 2H⁺ + 2e⁻). Electrons flow through LIG-E to potentiostat. Signal-ON: more target → more cleavage → more captured ALP → more pAP → higher peak."}
                {arch === "B" && "Ag⁰ nanoparticles deposited along ssDNA scaffold. Cas12a trans-cleavage releases Ag-ssDNA fragments into solution. Remaining surface-bound Ag⁰ is oxidized via ASV (Ag⁰ → Ag⁺ + e⁻). Signal-OFF: more cleavage → less surface Ag → smaller stripping peak."}
                {arch === "C" && "MB redox reporter tethered to ssDNA probe via pyrene π-π stacking on LIG-E. Intact probe: MB within electron-tunneling distance (<2 nm) → strong signal. Cas12a cleaves probe → MB diffuses away → signal loss. Signal-OFF."}
              </div>
            )}
          </div>

          {/* ── Cas12a controls ── */}
          <div style={{ position: "absolute", bottom: 10, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 10, alignItems: "center", background: "rgba(255,255,255,0.95)", padding: "6px 14px", borderRadius: 8, border: "1px solid #E3E8EF", flexWrap: "wrap", justifyContent: "center", backdropFilter: "blur(8px)" }}>
            <button onClick={() => setCas12aActive(!cas12aActive)} style={{
              fontSize: 10, fontWeight: 700, padding: "5px 14px", borderRadius: 6, cursor: "pointer", fontFamily: MONO,
              background: cas12aActive ? "#DC2626" : "#16A34A", color: "#fff", border: "none",
            }}>
              {cas12aActive ? "Reset" : "Activate Cas12a"}
            </button>
            {cas12aActive && (
              <div style={{ fontSize: 9, fontFamily: MONO, color: "#6B7280", display: "flex", gap: 6, alignItems: "center" }}>
                <span style={{ fontWeight: 700, color: "#16A34A" }}>ΔI% = {deltaI}%</span>
                <span style={{ color: "#33AA55" }}>{cleavedCount}/{allReporters.length} cleaved</span>
              </div>
            )}
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ fontSize: 8, color: "#6B7280", fontFamily: MONO }}>t =</span>
              <input type="range" min="0" max="30" step="1" value={incubationMin} onChange={e => setIncubationMin(+e.target.value)} style={{ width: 80, accentColor: "#374151" }} />
              <span style={{ fontSize: 9, fontWeight: 700, fontFamily: MONO, color: "#374151", minWidth: 32 }}>{incubationMin} min</span>
            </div>
          </div>

          {/* Solution-phase legend */}
          {cas12aActive && (
            <div style={{ position: "absolute", right: 16, top: "36%", display: "flex", flexDirection: "column", gap: 2 }}>
              {[
                { label: "Cas12a RNP (in situ, ~1250 t/hr)", color: "#33AA55" },
                { label: arch === "B" ? "Detached Ag (signal loss)" : "Cleaved MB (diffusing)", color: arch === "B" ? "#C0C0C0" : "#2563eb" },
                { label: "RPA amplicon (≤120 bp, cfDNA)", color: "#5577CC" },
              ].map(l => (
                <div key={l.label} style={{ fontSize: 6.5, fontFamily: MONO, color: "#374151", background: "rgba(255,255,255,0.9)", padding: "2px 5px", borderRadius: 3, borderLeft: `3px solid ${l.color}` }}>
                  {l.label}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ═══ MODE 3: SIDE PROFILE ═══ */}
      {mode === 3 && (
        <>
          <div style={{ position: "absolute", top: 12, left: 16, fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: HEADING, textShadow: "0 1px 3px rgba(255,255,255,0.9)" }}>
            Side Profile — Layer Assembly
          </div>
          <button onClick={() => setMode(1)} style={{ position: "absolute", top: 12, right: 16, fontSize: 10, fontWeight: 600, padding: "6px 14px", borderRadius: 6, border: "1px solid #E3E8EF", background: "#fff", cursor: "pointer", fontFamily: HEADING, color: "#374151" }}>
            ← Back to chip
          </button>
          <div style={{ position: "absolute", bottom: 12, left: 16, background: "rgba(255,255,255,0.94)", padding: "8px 12px", borderRadius: 8, border: "1px solid #E3E8EF" }}>
            <div style={{ fontSize: 7.5, fontFamily: MONO, color: "#374151", lineHeight: 1.8 }}>
              <div style={{ borderLeft: "3px solid #F5F0E8", paddingLeft: 6, marginBottom: 2 }}>Cellulose CF3 paper — 390 µm (Cytiva, ~11 µm pores)</div>
              <div style={{ borderLeft: "3px solid #888", paddingLeft: 6, marginBottom: 2 }}>Pre-treatment: 0.8 M ammonium sulfamate</div>
              <div style={{ borderLeft: "3px solid #2D2D2D", paddingLeft: 6, marginBottom: 2 }}>LIG-E — CO₂ 4.8 W, 16 cm/s, 10 mm defocus, N₂, 23 Ω/sq</div>
              <div style={{ borderLeft: "3px solid #3D2B1F", paddingLeft: 6, marginBottom: 2 }}>Wax barriers — hydrophobic zone isolation</div>
              <div style={{ borderLeft: "3px solid #555", paddingLeft: 6 }}>Contact pads — reader pogo pins connect here</div>
            </div>
          </div>
        </>
      )}

      {/* ═══ TOOLTIP ═══ */}
      {tooltipInfo && mode === 1 && (
        <div style={{ position: "absolute", left: tooltipPos.x + 14, top: tooltipPos.y - 12, pointerEvents: "none", background: "rgba(255,255,255,0.96)", border: "1px solid #E3E8EF", borderRadius: 6, padding: "6px 10px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)", zIndex: 10 }}>
          <div style={{ fontSize: 10, fontWeight: 700, fontFamily: MONO, color: DRUG_CSS[tooltipInfo.drug] || "#333" }}>{tooltipInfo.target}</div>
          <div style={{ fontSize: 9, color: "#6B7280", marginTop: 1 }}>
            {tooltipInfo.drug} · S_eff = {getEfficiency(tooltipInfo.target).toFixed(3)}
            {(() => { const r = results.find(x => x.label === tooltipInfo.target); return r?.disc && r.disc < 900 ? ` · D = ${r.disc.toFixed(1)}×` : ""; })()}
          </div>
          <div style={{ fontSize: 8, color: "#9CA3AF", marginTop: 1 }}>{targetStrategy(tooltipInfo.target)} · Click to inspect</div>
        </div>
      )}
    </div>
  );
}
