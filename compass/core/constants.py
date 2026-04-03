"""Biological constants for COMPASS.

Centralised to avoid magic numbers scattered across modules.
Sources cited inline.
"""

# ---------------------------------------------------------------------------
# Reference genome
# ---------------------------------------------------------------------------
H37RV_ACCESSION = "NC_000962.3"
H37RV_LENGTH = 4_411_532
H37RV_GC_CONTENT = 0.656

# ---------------------------------------------------------------------------
# Cas12a parameters
# ---------------------------------------------------------------------------
ASCAS12A_PAM = "TTTV"
ENASCAS12A_PAMS = ("TTYN", "VTTV")
SPACER_LENGTH_MIN = 18
SPACER_LENGTH_MAX = 24
SPACER_LENGTH_DEFAULT = 20
SEED_REGION_END = 8

# Cas12a direct repeat (crRNA scaffold) — 19-nt mature DR after processing.
# Used for full crRNA MFE computation: DR + spacer folds as one molecule.
# The DR forms a conserved pseudoknot that Cas12a recognises; spacer nucleotides
# adjacent to the DR can form unintended base-pairs that compete with R-loop formation.
# Zetsche et al., Cell 2015; Fonfara et al., Nature 2016.
CAS12A_DR = "UAAUUUCUACUAAGUGUAGA"  # RNA sequence, 5'→3'

TRUNCATED_SPACER_LENGTH = 17
WOBBLE_POSITION = 14

# ---------------------------------------------------------------------------
# Hard filter thresholds (Module 2)
# Defaults are for generic organisms; organism-specific bounds are in
# candidates/filters.py OrganismPresets.
# ---------------------------------------------------------------------------
GC_MIN = 0.40
GC_MAX = 0.60

# TB-specific GC optimum — M.tb genome is 65.6% GC, so the scoring
# optimum shifts from 0.50 to 0.55 (Kim et al. 2018 shows Cas12a
# tolerates higher GC well).
GC_OPTIMAL_DEFAULT = 0.50
GC_OPTIMAL_MTB = 0.65  # computed from H37Rv TTTV-adjacent spacer median (was 0.55 hardcoded guess)
HOMOPOLYMER_MAX = 4
MFE_THRESHOLD = -2.0

# ---------------------------------------------------------------------------
# Off-target screening (Module 3)
# ---------------------------------------------------------------------------
OFFTARGET_MISMATCH_THRESHOLD = 3
BOWTIE2_SEED_LENGTH = 20
BOWTIE2_MAX_MISMATCHES = 3

OFFTARGET_SEED_WEIGHT = 10.0
OFFTARGET_TRUNK_WEIGHT = 2.0
OFFTARGET_TAIL_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# RPA primer constraints (Module 8)
# Widened for M.tb (65.6% GC genome): rpoB RRDR ~70%, gyrA QRDR ~72%.
# RPA uses recombinase at 37°C — much more Tm-tolerant than PCR.
# Published M.tb RPA assays use Tm 62–70°C (Ai et al. 2019, Cao et al. 2018).
#
# BRIDGE project: blood cfDNA input. Circulating free DNA fragments are
# ~100–160 bp (median ~140 bp for human cfDNA; Lo et al. 2010,
# Mouliere et al. 2018). M.tb cfDNA fragment sizes may differ
# (bacterial DNA lacks nucleosomal packaging).
# Hard cap at 120 bp ensures the full amplicon fits within a single cfDNA
# fragment — longer amplicons span fragment junctions and fail to amplify.
# ---------------------------------------------------------------------------
RPA_PRIMER_LENGTH_MIN = 25
RPA_PRIMER_LENGTH_MAX = 38
RPA_TM_MIN = 57.0
RPA_TM_MAX = 72.0
RPA_AMPLICON_MIN = 80
# Amplicon max is now sample_type-dependent in PrimerConfig.
# cfDNA: 120bp (blood fragments ~100-160bp), genomic: 250bp.
# This constant is kept as the cfDNA default for backward compatibility.
RPA_AMPLICON_MAX_CFDNA = 120
RPA_AMPLICON_MAX_GENOMIC = 250
RPA_AMPLICON_SOFT_PENALTY_START = 100  # bp — linear penalty above this threshold
PRIMER_DIMER_DG_THRESHOLD = -6.0

# Allele-specific RPA (Ye et al. 2019)
ASRPA_MISMATCH_POSITIONS = [-2, -3]
ASRPA_DELIBERATE_MISMATCH = {"A": "C", "T": "G", "G": "T", "C": "A"}

# One-pot (Bell et al. Sci Adv 2025)
ONEPOT_ASYMMETRIC_RATIO = 10
ONEPOT_TEMPERATURE = 37.0

# ---------------------------------------------------------------------------
# Heuristic scoring weights (Module 4, Level 1)
# ---------------------------------------------------------------------------
# Weights learned from 2,858 EasyDesign perfect-match guides via Ridge
# regression on measured trans-cleavage activity (log-k fluorescence).
# PAM quality is the dominant predictor (applied multiplicatively in scorer).
# GC has near-zero marginal signal after controlling for structure.
HEURISTIC_WEIGHTS = {
    "seed_position": 0.35,
    "gc": 0.05,
    "structure": 0.30,
    "homopolymer": 0.10,
    "offtarget": 0.20,
}

# TB-optimised weights: GC structure penalties increased for M.tb's
# 65.6% GC genome where GC-related biophysical issues are more prevalent.
# Derived via grid search on Kim 2018 HT1-2 validation set (TB-like GC subset).
HEURISTIC_WEIGHTS_MTB = {
    "seed_position": 0.30,
    "gc": 0.25,
    "structure": 0.22,
    "homopolymer": 0.08,
    "offtarget": 0.15,
}

# ---------------------------------------------------------------------------
# Flanking sequence extraction
# ---------------------------------------------------------------------------
FLANKING_WINDOW = 500

# ---------------------------------------------------------------------------
# IUPAC degenerate bases
# ---------------------------------------------------------------------------
IUPAC_EXPAND: dict[str, set[str]] = {
    "A": {"A"}, "T": {"T"}, "G": {"G"}, "C": {"C"},
    "V": {"A", "C", "G"}, "Y": {"C", "T"}, "N": {"A", "T", "G", "C"},
    "K": {"G", "T"}, "R": {"A", "G"}, "S": {"G", "C"},
    "W": {"A", "T"}, "M": {"A", "C"},
}


def pam_matches(seq: str, pattern: str) -> bool:
    """Check if a 4-nt sequence matches a degenerate PAM pattern."""
    if len(seq) != len(pattern):
        return False
    return all(nt.upper() in IUPAC_EXPAND.get(p, set()) for nt, p in zip(seq, pattern))


# ---------------------------------------------------------------------------
# M.tb gene name synonyms (Mycobrowser / TubercuList)
# KEY FIX: fabG1 → Rv1483 (NOT inhA), katG → Rv1908c, pncA → Rv2043c
# ---------------------------------------------------------------------------
MTB_GENE_SYNONYMS: dict[str, list[str]] = {
    "rpoB":   ["Rv0667"],
    "katG":   ["Rv1908c"],
    "inhA":   ["Rv1484"],
    "fabG1":  ["Rv1483", "mabA"],
    "embB":   ["Rv3795"],
    "embC":   ["Rv3793"],
    "embA":   ["Rv3794"],
    "pncA":   ["Rv2043c"],
    "gyrA":   ["Rv0006"],
    "gyrB":   ["Rv0005"],
    "rrs":    ["MTB000019", "Rvnr01"],
    "rrl":    ["MTB000020", "Rvnr02"],
    "eis":    ["Rv2416c"],
    "tlyA":   ["Rv1694"],
    "rpsL":   ["Rv0682"],
    "Rv0678": ["mmpR5"],
    "mmpR5":  ["Rv0678"],
    "rplC":   ["Rv0701"],
    "ddn":    ["Rv3547"],
    "fbiA":   ["Rv3261"],
    "fbiB":   ["Rv3262"],
    "fbiC":   ["Rv1173"],
    "fgd1":   ["Rv0407"],
    "ethA":   ["Rv3854c"],
    "folC":   ["Rv2447c"],
    "thyA":   ["Rv2764c"],
    "pepQ":   ["Rv2535c"],
    "ahpC":   ["Rv2428"],
    "rpoC":   ["Rv0668"],
}

MTB_SYSTEMATIC_TO_COMMON: dict[str, str] = {}
for _common, _sys_list in MTB_GENE_SYNONYMS.items():
    for _sys in _sys_list:
        MTB_SYSTEMATIC_TO_COMMON[_sys] = _common
        MTB_SYSTEMATIC_TO_COMMON[_sys.lower()] = _common

# ---------------------------------------------------------------------------
# IS6110 — MTB species identification control
# ---------------------------------------------------------------------------
IS6110_SPACER = "AATGTCGCCGCGATCGAGCG"
IS6110_PAM = "TTTG"
IS6110_COPIES_PER_GENOME = "6-16"

# Published IS6110 RPA primers (Ai et al., Emerging Microbes & Infections 2019)
IS6110_FWD_PRIMER = "GATCGTCTCGATCTGCGTAAAGGTGGACTACCAGGGTATCTGCGT"
IS6110_REV_PRIMER = "CCGCTTCCAGCCCAGTGACGAGCGTAAGCCTCAACTACCACAGT"
IS6110_AMPLICON_LENGTH = 143

# ---------------------------------------------------------------------------
# RNaseP — Human extraction/sample adequacy control
# Detects human RNaseP (RPPH1) gene in the sample as proof that:
#   1. DNA extraction succeeded
#   2. Sufficient human material is present (sputum / blood)
# This is the standard CDC extraction control used in TB diagnostics.
# Coordinates from GRCh38 RPPH1 locus (chr14:20,811,208-20,811,520).
# ---------------------------------------------------------------------------
RNASEP_SPACER = "GCGCGAGCGCATGCCTGCAG"   # 20-nt spacer targeting RPPH1 exon
RNASEP_PAM = "TTTG"
RNASEP_FWD_PRIMER = "AGATTTGGACCTGCGAGCGCTCGAGCAGGACG"
RNASEP_REV_PRIMER = "GAGCGGCTGTCTCCACAAGTCCGCCTCGACGA"
RNASEP_AMPLICON_LENGTH = 95
RNASEP_GENE = "RPPH1"
RNASEP_ORGANISM = "human"
