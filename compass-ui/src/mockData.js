import {
  Database, Search, Filter, Shield, BarChart3, GitBranch, Zap, TrendingUp,
  Grid3x3, Crosshair, Check, Package, Download,
} from "lucide-react";

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
  // E. coli ESBL / CRE
  "blaCTX-M-15_presence": { who: "WHO Priority", catalogue: "CARD (2024)", cryptic: null, freq: "Most prevalent ESBL globally" },
  "blaCTX-M-14_presence": { who: "WHO Priority", catalogue: "CARD (2024)", cryptic: null, freq: "2nd most common CTX-M worldwide" },
  "blaCTX-M-27_presence": { who: "WHO Priority", catalogue: "CARD (2024)", cryptic: null, freq: "Emerging in E. coli ST131 C2 clade" },
  "blaNDM-1_presence": { who: "WHO Critical", catalogue: "CARD (2024); WHO GLASS (2023)", cryptic: null, freq: "Dominant MBL in South Asia, spreading globally" },
  "blaNDM-5_presence": { who: "WHO Critical", catalogue: "CARD (2024)", cryptic: null, freq: "Enhanced activity NDM variant; community E. coli" },
  "blaKPC-2_presence": { who: "WHO Critical", catalogue: "CARD (2024); WHO GLASS (2023)", cryptic: null, freq: "Dominant serine carbapenemase in Americas" },
  "blaKPC-3_presence": { who: "WHO Critical", catalogue: "CARD (2024)", cryptic: null, freq: "KPC variant; K. pneumoniae → E. coli HGT" },
  "blaOXA-48_presence": { who: "WHO Critical", catalogue: "CARD (2024); WHO GLASS (2023)", cryptic: null, freq: "Weak carbapenem hydrolysis; hard to detect phenotypically" },
  "mcr-1_presence": { who: "WHO Critical", catalogue: "CARD (2024); Liu et al. Lancet ID (2016)", cryptic: null, freq: "Plasmid-mediated colistin-R; last-resort drug" },
  "ecoli_gyrA_S83L": { who: "Associated", catalogue: "CARD (2024); EUCAST (2023)", cryptic: null, freq: "Primary FQ-R in E. coli QRDR" },
  "ecoli_gyrA_D87N": { who: "Associated", catalogue: "CARD (2024); EUCAST (2023)", cryptic: null, freq: "2nd-step FQ-R; +S83L = high-level" },
  "ecoli_parC_S80I": { who: "Associated", catalogue: "CARD (2024)", cryptic: null, freq: "Topo IV; contributes to high-level FQ-R" },
  "ecoli_parC_E84V": { who: "Interim", catalogue: "CARD (2024)", cryptic: null, freq: "Auxiliary FQ-R determinant" },
  "ampC_P-42L": { who: "Associated", catalogue: "CARD (2024); Jacoby (2009)", cryptic: null, freq: "Promoter ↑ AmpC → cephalosporin-R" },
  // S. aureus MRSA / VRSA
  "mecA_presence": { who: "WHO Critical", catalogue: "EUCAST (2023); CLSI (2024)", cryptic: null, freq: "Defines MRSA; PBP2a on SCCmec" },
  "mecC_presence": { who: "WHO Critical", catalogue: "EUCAST (2023); Podkowik et al. (2025)", cryptic: null, freq: "mecA homolog SCCmec XI; missed by some PCRs" },
  "vanA_presence": { who: "WHO Critical", catalogue: "CLSI (2024); CDC VRSA (2023)", cryptic: null, freq: "Defines VRSA; transferred from Enterococcus" },
  "blaZ_presence": { who: "Associated", catalogue: "EUCAST (2023)", cryptic: null, freq: ">90% S. aureus carry blaZ" },
  "sa_gyrA_S84L": { who: "Associated", catalogue: "EUCAST (2023); CLSI (2024)", cryptic: null, freq: "Primary FQ-R in S. aureus QRDR" },
  "grlA_S80F": { who: "Associated", catalogue: "EUCAST (2023); Hooper & Jacoby (2016)", cryptic: null, freq: "ParC equivalent; primary FQ target in S. aureus" },
  "grlA_S80Y": { who: "Interim", catalogue: "EUCAST (2023)", cryptic: null, freq: "Alternative grlA QRDR mutation" },
  "sa_rpoB_H481N": { who: "Associated", catalogue: "EUCAST (2023); CLSI (2024)", cryptic: null, freq: "RIF-R RRDR cluster I (≡ E. coli H526N)" },
  "sa_rpoB_S464P": { who: "Associated", catalogue: "EUCAST (2023)", cryptic: null, freq: "High-level RIF-R; RRDR" },
  "fusA_L461K": { who: "Associated", catalogue: "EUCAST (2023); Castanheira et al. (2020)", cryptic: null, freq: "Fusidic acid-R; EF-G domain III" },
  "dfrB_F99Y": { who: "Associated", catalogue: "CARD (2024); EUCAST (2023)", cryptic: null, freq: "Trimethoprim-R; DHFR active site" },
  "ermA_presence": { who: "Associated", catalogue: "CLSI (2024); Leclercq (2002)", cryptic: null, freq: "MLSb resistance (macrolides/lincosamides/streptogramin B)" },
  "ermC_presence": { who: "Associated", catalogue: "CLSI (2024); Leclercq (2002)", cryptic: null, freq: "Plasmid-borne MLSb; inducible/constitutive" },
  "cfr_presence": { who: "Associated", catalogue: "CARD (2024); Long et al. (2006)", cryptic: null, freq: "23S rRNA methyltransferase; LZD + PhLOPSA-R" },
  "mprF_S295L": { who: "Associated", catalogue: "EUCAST (2023); Bayer et al. (2015)", cryptic: null, freq: "Daptomycin non-susceptibility; ↑ lysyl-PG" },
  // N. gonorrhoeae AMR
  "penA_A501V": { who: "WHO Urgent", catalogue: "WHO gonococcal AMR (2023); Unemo et al. (2019)", cryptic: null, freq: "Mosaic penA; reduced CRO susceptibility" },
  "penA_A501T": { who: "WHO Urgent", catalogue: "WHO gonococcal AMR (2023)", cryptic: null, freq: "Alternative mosaic mutation at 501" },
  "penA_G545S": { who: "WHO Urgent", catalogue: "WHO gonococcal AMR (2023); Grad et al. (2016)", cryptic: null, freq: "Mosaic penA; ESC-R in combination" },
  "penA_I312M": { who: "Interim", catalogue: "Unemo et al. (2019)", cryptic: null, freq: "Mosaic penA; intermediate CRO" },
  "penA_V316T": { who: "Interim", catalogue: "Unemo et al. (2019)", cryptic: null, freq: "Mosaic penA; combinatorial ESC-R" },
  "penA_T483S": { who: "Interim", catalogue: "Unemo et al. (2019)", cryptic: null, freq: "Mosaic penA; ESC-R contribution" },
  "ng_gyrA_S91F": { who: "Associated", catalogue: "WHO gonococcal AMR (2023); EUCAST (2023)", cryptic: null, freq: "Primary CIP-R; QRDR hot spot" },
  "ng_gyrA_D95A": { who: "Associated", catalogue: "WHO gonococcal AMR (2023)", cryptic: null, freq: "2nd-step FQ-R; +S91F = high-level" },
  "ng_gyrA_D95G": { who: "Associated", catalogue: "WHO gonococcal AMR (2023)", cryptic: null, freq: "Alternative D95 substitution" },
  "ng_parC_D86N": { who: "Interim", catalogue: "Unemo et al. (2019)", cryptic: null, freq: "Auxiliary FQ-R; Topo IV" },
  "ng_parC_S87R": { who: "Interim", catalogue: "Unemo et al. (2019)", cryptic: null, freq: "parC QRDR mutation; FQ-R" },
  "23S_C2611T": { who: "Associated", catalogue: "WHO gonococcal AMR (2023); Chisholm et al. (2010)", cryptic: null, freq: "High-level AZM-R; single 23S rRNA SNP" },
  "23S_A2059G": { who: "Associated", catalogue: "WHO gonococcal AMR (2023)", cryptic: null, freq: "AZM-R; macrolide target site" },
  "mtrR_A-35del": { who: "Associated", catalogue: "WHO gonococcal AMR (2023); Ohnishi et al. (2011)", cryptic: null, freq: "Promoter del → ↑ MtrCDE efflux" },
  "tetM_presence": { who: "Associated", catalogue: "CARD (2024); Roberts (2005)", cryptic: null, freq: "Ribosomal protection; conjugative TET-R" },
  "folP_R228S": { who: "Associated", catalogue: "Unemo et al. (2019)", cryptic: null, freq: "Sulfonamide-R; DHPS active site" },
};

const MUTATIONS = [
  { gene: "rpoB", ref: "S", pos: 531, alt: "L", drug: "RIF", drugFull: "Rifampicin", conf: "High", tier: 1, category: "aa_substitution" },
  { gene: "rpoB", ref: "H", pos: 526, alt: "Y", drug: "RIF", drugFull: "Rifampicin", conf: "High", tier: 1, category: "aa_substitution" },
  { gene: "rpoB", ref: "D", pos: 516, alt: "V", drug: "RIF", drugFull: "Rifampicin", conf: "High", tier: 1, category: "aa_substitution" },
  { gene: "katG", ref: "S", pos: 315, alt: "T", drug: "INH", drugFull: "Isoniazid", conf: "High", tier: 1, category: "aa_substitution" },
  { gene: "fabG1", ref: "C", pos: -15, alt: "T", drug: "INH", drugFull: "Isoniazid", conf: "High", tier: 1, category: "promoter" },
  { gene: "embB", ref: "M", pos: 306, alt: "V", drug: "EMB", drugFull: "Ethambutol", conf: "High", tier: 1, category: "aa_substitution" },
  { gene: "embB", ref: "M", pos: 306, alt: "I", drug: "EMB", drugFull: "Ethambutol", conf: "Moderate", tier: 2, category: "aa_substitution" },
  { gene: "pncA", ref: "H", pos: 57, alt: "D", drug: "PZA", drugFull: "Pyrazinamide", conf: "Moderate", tier: 2, category: "aa_substitution" },
  { gene: "gyrA", ref: "D", pos: 94, alt: "G", drug: "FQ", drugFull: "Fluoroquinolones", conf: "High", tier: 1, category: "aa_substitution" },
  { gene: "gyrA", ref: "A", pos: 90, alt: "V", drug: "FQ", drugFull: "Fluoroquinolones", conf: "High", tier: 1, category: "aa_substitution" },
  { gene: "rrs", ref: "A", pos: 1401, alt: "G", drug: "AG", drugFull: "Amikacin", conf: "High", tier: 1, category: "rrna" },
  { gene: "eis", ref: "C", pos: -14, alt: "T", drug: "AG", drugFull: "Amikacin", conf: "High", tier: 1, category: "promoter" },
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
  gene: "IS6110", ref: "N", pos: 0, alt: "N", drug: "OTHER", drugFull: "Other", conf: "N/A", tier: 0, category: "gene_presence",
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
    interpretation: "In silico estimate (same-gene pairs from sequence homology; inter-gene pairs are conservative upper bounds, not experimentally measured). In the spatially multiplexed paper electrode array, each detection zone is physically isolated by wax-printed hydrophobic barriers, so cross-reactivity between zones is impossible. These scores are relevant only for hypothetical solution-phase multiplex formats.",
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
  "Initializing": 0, "Target Resolution": 0, "PAM Scanning": 1,
  "Candidate Filtering": 2, "Off-Target Screening": 3, "Heuristic Scoring": 4,
  "Compass-ML: Encoding": 4, "Compass-ML: Encoding targets": 4,
  "Compass-ML: RNA-FM": 4, "Compass-ML: RNA-FM embeddings": 4,
  "Compass-ML: CNN inference": 4, "Compass-ML: Calibration": 4,
  "Mismatch Pairs": 5, "SM Enhancement": 6, "SM Enhancement complete": 6,
  "Discrimination Scoring": 7, "Discrimination complete": 7,
  "Multiplex Optimization": 8, "RPA Primer Design": 9, "Co-Selection Validation": 10,
  "Panel Assembly": 11, "Export": 12, "Complete": 12, "Serializing Results": 12,
};

// Progress-to-step mapping: uses numeric progress value as fallback
// when current_module string lookup fails
const PROGRESS_TO_STEP = [
  [0.95, 12], [0.90, 11], [0.85, 10], [0.78, 9], [0.70, 8],
  [0.65, 7], [0.55, 6], [0.45, 5], [0.40, 4], [0.20, 3],
  [0.15, 2], [0.10, 1], [0.05, 0], [0.02, 0],
];
function resolveStep(data) {
  // Try module name first
  if (data.current_module) {
    const idx = MODULE_NAME_MAP[data.current_module];
    if (idx !== undefined) return idx;
  }
  // Fallback: derive step from numeric progress
  if (typeof data.progress === "number") {
    for (const [threshold, step] of PROGRESS_TO_STEP) {
      if (data.progress >= threshold) return step;
    }
  }
  return 0;
}

/* Scoring feature weights; matches compass/core/constants.py HEURISTIC_WEIGHTS exactly */
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
  // RNA:DNA Thermodynamics
  { id: "sugimoto1995", authors: "Sugimoto N, Nakano S, Katoh M, et al.", year: 1995, title: "Thermodynamic parameters to predict stability of RNA/DNA hybrid duplexes", journal: "Biochemistry", doi: "10.1021/bi00035a029", pmid: "7545436", category: "R-Loop Thermodynamics" },
  { id: "sugimoto2000", authors: "Sugimoto N, Nakano M, Nakano S", year: 2000, title: "Thermodynamics-structure relationship of single mismatches in RNA/DNA duplexes", journal: "Biochemistry", doi: "10.1021/bi000819p", category: "R-Loop Thermodynamics" },
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
  { id: "bezinge2023", authors: "Bezinge L, Shih CJ, deMello AJ, et al.", year: 2023, title: "Paper-Based Laser-Pyrolyzed Electrofluidics: An Electrochemical Platform for Capillary-Driven Diagnostic Bioassays", journal: "Advanced Materials", doi: "10.1002/adma.202302893", category: "Electrochemical Platform" },
  { id: "sueangam2021", authors: "Suea-Ngam A, Howes PD, Stanley CE, deMello AJ", year: 2021, title: "An amplification-free ultra-sensitive electrochemical CRISPR/Cas biosensor for drug-resistant bacteria detection", journal: "Chemical Science", doi: "10.1039/D1SC02197D", pmid: "34703560", category: "Electrochemical Platform" },
  { id: "lesinski2024", authors: "Lesinski AM, Aksoy YA, Bhatt A, et al.", year: 2024, title: "Kinetic Characterization of the CRISPR-Cas12a Trans-Cleavage Reaction Using Surface Plasmon Resonance", journal: "Analytical Chemistry", doi: "10.1021/acs.analchem.4c03600", category: "Electrochemical Platform" },
  { id: "kohabir2024", authors: "Kohabir KAV, Noori N, Engstr\u00f6m JOA, et al.", year: 2024, title: "Synthetic mismatches enable specific CRISPR-Cas12a-based detection of genome-wide SNVs tracked by ARTEMIS", journal: "Cell Reports Methods", doi: "10.1016/j.crmeth.2024.100912", category: "CRISPR Diagnostics" },
  // Bioinformatics
  { id: "langmead2012", authors: "Langmead B, Salzberg SL", year: 2012, title: "Fast gapped-read alignment with Bowtie 2", journal: "Nature Methods", doi: "10.1038/nmeth.1923", pmid: "22388286", category: "Bioinformatics" },
  { id: "piepenburg2006", authors: "Piepenburg O, Williams CH, Stemple DL, Armes NA", year: 2006, title: "DNA detection using recombination proteins", journal: "PLoS Biology", doi: "10.1371/journal.pbio.0040204", pmid: "16756388", category: "Isothermal Amplification" },
];

// ── Organism presets for multi-species support ──
const ORGANISMS = [
  {
    id: "mtb", name: "M. tuberculosis", reference: "H37Rv",
    accession: "NC_000962.3", gc: 0.656, priority: "WHO Critical",
    description: "Multi-drug resistant tuberculosis (MDR/XDR-TB)",
    mutations: MUTATIONS, // reuse existing MTB mutations
  },
  {
    id: "ecoli", name: "E. coli", reference: "K-12 MG1655",
    accession: "NC_000913.3", gc: 0.508, priority: "WHO Critical",
    description: "ESBL-producing E. coli + Enterobacterales carbapenemase panel (NDM/KPC/OXA-48)",
    mutations: [
      // ESBLs — gene presence
      { gene: "blaCTX-M-15", ref: "-", pos: 0, alt: "presence", drug: "CTX", drugFull: "Cefotaxime", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "blaCTX-M-14", ref: "-", pos: 0, alt: "presence", drug: "CTX", drugFull: "Cefotaxime", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "blaCTX-M-27", ref: "-", pos: 0, alt: "presence", drug: "CTX", drugFull: "Cefotaxime", conf: "High", tier: 1, category: "gene_presence" },
      // Carbapenemases — gene presence
      { gene: "blaNDM-1", ref: "-", pos: 0, alt: "presence", drug: "CAR", drugFull: "Carbapenem", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "blaNDM-5", ref: "-", pos: 0, alt: "presence", drug: "CAR", drugFull: "Carbapenem", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "blaKPC-2", ref: "-", pos: 0, alt: "presence", drug: "CAR", drugFull: "Carbapenem", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "blaKPC-3", ref: "-", pos: 0, alt: "presence", drug: "CAR", drugFull: "Carbapenem", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "blaOXA-48", ref: "-", pos: 0, alt: "presence", drug: "CAR", drugFull: "Carbapenem", conf: "High", tier: 1, category: "gene_presence" },
      // Colistin — last resort
      { gene: "mcr-1", ref: "-", pos: 0, alt: "presence", drug: "CST", drugFull: "Colistin", conf: "High", tier: 1, category: "gene_presence" },
      // Fluoroquinolone QRDR mutations
      { gene: "gyrA", ref: "S", pos: 83, alt: "L", drug: "CIP", drugFull: "Ciprofloxacin", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "gyrA", ref: "D", pos: 87, alt: "N", drug: "CIP", drugFull: "Ciprofloxacin", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "parC", ref: "S", pos: 80, alt: "I", drug: "CIP", drugFull: "Ciprofloxacin", conf: "Moderate", tier: 2, category: "aa_substitution" },
      { gene: "parC", ref: "E", pos: 84, alt: "V", drug: "CIP", drugFull: "Ciprofloxacin", conf: "Moderate", tier: 2, category: "aa_substitution" },
      // AmpC promoter
      { gene: "ampC", ref: "P", pos: -42, alt: "L", drug: "AMP", drugFull: "Ampicillin", conf: "High", tier: 1, category: "promoter" },
    ],
  },
  {
    id: "saureus", name: "S. aureus", reference: "NCTC 8325",
    accession: "NC_007795.1", gc: 0.328, priority: "WHO Critical",
    description: "Methicillin-resistant S. aureus (MRSA) & vancomycin-resistant (VRSA)",
    mutations: [
      // Beta-lactam resistance — gene presence
      { gene: "mecA", ref: "-", pos: 0, alt: "presence", drug: "MET", drugFull: "Methicillin", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "mecC", ref: "-", pos: 0, alt: "presence", drug: "MET", drugFull: "Methicillin", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "blaZ", ref: "-", pos: 0, alt: "presence", drug: "PEN", drugFull: "Penicillin", conf: "High", tier: 1, category: "gene_presence" },
      // Glycopeptide resistance
      { gene: "vanA", ref: "-", pos: 0, alt: "presence", drug: "VAN", drugFull: "Vancomycin", conf: "High", tier: 1, category: "gene_presence" },
      // Fluoroquinolone QRDR
      { gene: "gyrA", ref: "S", pos: 84, alt: "L", drug: "CIP", drugFull: "Ciprofloxacin", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "grlA", ref: "S", pos: 80, alt: "F", drug: "CIP", drugFull: "Ciprofloxacin", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "grlA", ref: "S", pos: 80, alt: "Y", drug: "CIP", drugFull: "Ciprofloxacin", conf: "Moderate", tier: 2, category: "aa_substitution" },
      // Rifampicin RRDR
      { gene: "rpoB", ref: "H", pos: 481, alt: "N", drug: "RIF", drugFull: "Rifampicin", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "rpoB", ref: "S", pos: 464, alt: "P", drug: "RIF", drugFull: "Rifampicin", conf: "High", tier: 1, category: "aa_substitution" },
      // Other targets
      { gene: "fusA", ref: "L", pos: 461, alt: "K", drug: "FA", drugFull: "Fusidic acid", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "dfrB", ref: "F", pos: 99, alt: "Y", drug: "SXT", drugFull: "Trimethoprim", conf: "High", tier: 1, category: "aa_substitution" },
      // Macrolide/lincosamide — gene presence
      { gene: "ermA", ref: "-", pos: 0, alt: "presence", drug: "ERY", drugFull: "Erythromycin", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "ermC", ref: "-", pos: 0, alt: "presence", drug: "ERY", drugFull: "Erythromycin", conf: "High", tier: 1, category: "gene_presence" },
      // Oxazolidinone / lipopeptide
      { gene: "cfr", ref: "-", pos: 0, alt: "presence", drug: "LZD", drugFull: "Linezolid", conf: "High", tier: 1, category: "gene_presence" },
      { gene: "mprF", ref: "S", pos: 295, alt: "L", drug: "DAP", drugFull: "Daptomycin", conf: "Moderate", tier: 2, category: "aa_substitution" },
    ],
  },
  {
    id: "ngonorrhoeae", name: "N. gonorrhoeae", reference: "FA 1090",
    accession: "NC_002946.2", gc: 0.525, priority: "WHO High",
    description: "Extended-spectrum cephalosporin & azithromycin-resistant gonorrhoea",
    mutations: [
      // Mosaic penA — ESC resistance (WHO urgent threat)
      // Numbering: Ambler PBP2 convention per Unemo et al. 2019 / WHO gonococcal AMR surveillance
      // Resistance is COMBINATORIAL — individual SNPs have modest effect; high-level CRO-R requires >=3 mosaic mutations
      // Panel interpretation: flag "mosaic penA detected" when >=2 of these electrodes fire
      { gene: "penA", ref: "A", pos: 501, alt: "V", drug: "CRO", drugFull: "Ceftriaxone", conf: "High", tier: 1, category: "aa_substitution", combinatorial: "penA_mosaic" },
      { gene: "penA", ref: "A", pos: 501, alt: "T", drug: "CRO", drugFull: "Ceftriaxone", conf: "High", tier: 1, category: "aa_substitution", combinatorial: "penA_mosaic" },
      { gene: "penA", ref: "G", pos: 545, alt: "S", drug: "CRO", drugFull: "Ceftriaxone", conf: "High", tier: 1, category: "aa_substitution", combinatorial: "penA_mosaic" },
      { gene: "penA", ref: "I", pos: 312, alt: "M", drug: "CRO", drugFull: "Ceftriaxone", conf: "Moderate", tier: 2, category: "aa_substitution", combinatorial: "penA_mosaic" },
      { gene: "penA", ref: "V", pos: 316, alt: "T", drug: "CRO", drugFull: "Ceftriaxone", conf: "Moderate", tier: 2, category: "aa_substitution", combinatorial: "penA_mosaic" },
      { gene: "penA", ref: "T", pos: 483, alt: "S", drug: "CRO", drugFull: "Ceftriaxone", conf: "Moderate", tier: 2, category: "aa_substitution", combinatorial: "penA_mosaic" },
      // Fluoroquinolone QRDR
      { gene: "gyrA", ref: "S", pos: 91, alt: "F", drug: "CIP", drugFull: "Ciprofloxacin", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "gyrA", ref: "D", pos: 95, alt: "A", drug: "CIP", drugFull: "Ciprofloxacin", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "gyrA", ref: "D", pos: 95, alt: "G", drug: "CIP", drugFull: "Ciprofloxacin", conf: "High", tier: 1, category: "aa_substitution" },
      { gene: "parC", ref: "D", pos: 86, alt: "N", drug: "CIP", drugFull: "Ciprofloxacin", conf: "Moderate", tier: 2, category: "aa_substitution" },
      { gene: "parC", ref: "S", pos: 87, alt: "R", drug: "CIP", drugFull: "Ciprofloxacin", conf: "Moderate", tier: 2, category: "aa_substitution" },
      // Azithromycin — dual therapy failure driver
      { gene: "23S rRNA", ref: "C", pos: 2611, alt: "T", drug: "AZM", drugFull: "Azithromycin", conf: "High", tier: 1, category: "rrna" },
      { gene: "23S rRNA", ref: "A", pos: 2059, alt: "G", drug: "AZM", drugFull: "Azithromycin", conf: "High", tier: 1, category: "rrna" },
      // Efflux pump
      { gene: "mtrR", ref: "A", pos: -35, alt: "del", drug: "AZM", drugFull: "Azithromycin", conf: "High", tier: 1, category: "promoter" },
      // Tetracycline — gene presence
      { gene: "tetM", ref: "-", pos: 0, alt: "presence", drug: "TET", drugFull: "Tetracycline", conf: "High", tier: 1, category: "gene_presence" },
      // Sulfonamide
      { gene: "folP", ref: "R", pos: 228, alt: "S", drug: "SXT", drugFull: "Sulfonamide", conf: "High", tier: 1, category: "aa_substitution" },
    ],
  },
];

export {
  seq, WHO_REFS, MUTATIONS, RESULTS,
  CROSS_REACTIVITY_LABELS, CROSS_REACTIVITY_DRUG_GROUPS, MOCK_CROSS_REACTIVITY,
  MODULES, MODULE_NAME_MAP, PROGRESS_TO_STEP, resolveStep,
  SCORING_FEATURES, DRUG_LABELS, BIBLIOGRAPHY,
  ORGANISMS,
};
