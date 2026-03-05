# Discrimination Biology: Why Cas12a Distinguishes Some Mutations Better Than Others

This document explains the biophysical basis for CRISPR-Cas12a discrimination ratios
in the COMPASS pipeline, why certain drug classes show low predicted specificity, and how
the pipeline addresses these challenges.

## Three Factors Governing Discrimination

### 1. Mismatch Position in the Spacer (Dominant Factor)

Cas12a initiates R-loop formation at the PAM-proximal (seed) end of the spacer and
propagates directionally toward the PAM-distal end. This directionality creates a
position-dependent sensitivity gradient:

| Seed Position | Region        | Typical Disc Ratio | Effect on R-loop               |
|---------------|---------------|--------------------|---------------------------------|
| 1–4           | PAM-proximal  | 10–50×             | Blocks R-loop initiation        |
| 5–8           | Mid-seed      | 3–10×              | Partial R-loop stalling         |
| 9–14          | Mid-spacer    | 1.5–3×             | Tolerated, mild destabilisation |
| 15–20         | PAM-distal    | 1–2×               | Well-tolerated                  |

**Reference:** Strohkendl I, Saifuddin FA, Rybarski JR, et al. Kinetic basis for DNA
target specificity of CRISPR-Cas12a. *Mol Cell* **71**, 816–824 (2018).
[DOI: 10.1016/j.molcel.2018.06.043](https://doi.org/10.1016/j.molcel.2018.06.043)

**Implementation:** `compass/scoring/discrimination.py` — position-dependent penalty
lookup; `compass-net/attention/rloop_attention.py` — RLPA causal attention encoding
directional R-loop propagation.

### 2. Mismatch Type (Transition vs Transversion)

Not all single-nucleotide changes are equal. The Watson-Crick geometry of the
mismatch determines how much it destabilises the RNA:DNA heteroduplex:

- **Transversions** (purine ↔ pyrimidine, e.g., A→C, G→T): Larger geometric
  distortion, stronger R-loop disruption, higher discrimination.
- **Transitions** (purine ↔ purine or pyrimidine ↔ pyrimidine, e.g., A→G, C→T):
  Smaller distortion, partially tolerated, lower discrimination.

The heuristic scorer (`compass/scoring/heuristic.py`) applies a fixed penalty per
position but does not model mismatch type. Compass-ML learns mismatch-type effects
implicitly from the one-hot encoded sequence context.

**Reference:** Kim D, Kim J, Hur JK, et al. Genome-wide analysis reveals
specificities of Cpf1 endonucleases in human cells. *Nat Biotechnol* **34**,
863–868 (2016). [DOI: 10.1038/nbt.3609](https://doi.org/10.1038/nbt.3609)

### 3. Sequence Context (GC Content)

*M. tuberculosis* has 65.8% GC content — among the highest of any human pathogen.
GC-rich sequences flanking a mismatch provide additional hydrogen bonds that
stabilise the R-loop even in the presence of the mismatch, partially compensating
for the destabilising effect.

This means:
- AT-rich regions amplify mismatch penalties → higher discrimination
- GC-rich regions rescue mismatches → lower discrimination

**Implementation:** `compass/scoring/heuristic.py` — GC content feature (weight 0.20);
`compass-net/features/thermodynamic.py` — local GC and melting temperature features.

## Why EMB and PZA Show Low Specificity

The ethambutol (EMB) and pyrazinamide (PZA) resistance mutations in the 14-plex
panel have two compounding disadvantages:

1. **PAM-distal SNP positions:** The embB M306V/I and pncA H57D mutations, when
   mapped to available Cas12a spacer designs, place the discriminating SNP at
   positions 12–18 of the spacer — well outside the seed region.

2. **High local GC content:** Both *embB* and *pncA* sit in GC-rich genomic
   regions (>68% local GC), which stabilises the wildtype R-loop and reduces
   the discrimination penalty.

The result: predicted discrimination ratios of 1.2–2.0×, insufficient for
diagnostic-grade specificity (≥3× required for WHO TPP compliance).

## How the Pipeline Addresses Low Discrimination

### Synthetic Mismatch Enhancement (Module 6)

For targets with low natural discrimination, Module 6 (`compass/candidates/synthetic_mismatch.py`)
introduces an engineered second mismatch in the seed region of the crRNA. This creates
a double-mismatch penalty on wildtype DNA that is disproportionately larger than the
single-mismatch penalty on mutant DNA:

- **Mutant target:** 0 mismatches (perfect match) → full Cas12a activity
- **Wildtype target:** 2 mismatches (natural + synthetic) → severely reduced activity
- **Net effect:** Discrimination ratio increases from ~1.5× to ~4–8×

The SM optimizer selects positions and substitutions that maximise the wildtype penalty
while preserving mutant activity above the efficiency threshold.

### Compass-ML vs Heuristic Scoring

The heuristic scorer computes discrimination as a position-dependent lookup table.
It captures the dominant effect (position) but misses mismatch type and sequence
context interactions.

Compass-ML (`compass/scoring/compass_ml_scorer.py`) improves discrimination prediction
through three mechanisms:

1. **CNN branch** — Learns nonlinear nucleotide interactions from 15K Cas12a
   measurements (Kim et al. 2018), capturing mismatch-type effects implicitly.

2. **RNA-FM branch** — Pre-trained RNA foundation model embeddings encode crRNA
   secondary structure and thermodynamic stability, capturing GC-context effects.

3. **RLPA attention** — Biophysically-informed causal attention that models
   directional R-loop propagation, encoding the position-dependent kinetics
   from Strohkendl et al. directly into the architecture.

4. **Multi-task discrimination head** — Phase 2 training adds a dedicated head
   predicting MUT/WT activity ratios (trained on self-distilled proxy labels).

**Expected impact on diagnostics:**
- Sensitivity: Similar to heuristic (~93%) — both agree on which targets have
  viable candidates.
- Specificity: Should improve — Compass-ML's discrimination predictions are better
  calibrated, moving from ~79% toward higher values.
- Ranking: More trustworthy — candidates Compass-ML says are discriminating are
  more likely to actually discriminate in the lab.

### Active Learning Loop (Future)

Computational predictions cannot cross the WHO TPP 98% specificity threshold alone.
The active learning module will:

1. Select the highest-information candidates for electrochemical testing
2. Measure real MUT/WT signal ratios on the platform
3. Feed measured discrimination back into Compass-ML fine-tuning
4. Iteratively close the gap between predicted and measured performance

This is the path from in silico 79% specificity to experimentally validated
98% specificity.

## Comparison: Heuristic vs Compass-ML Diagnostics

To generate a thesis-quality comparison figure:

```bash
# Run with heuristic scorer (default)
python scripts/run_full_pipeline.py --config config/default.yaml

# Run with Compass-ML scorer
python scripts/run_full_pipeline.py --config config/default.yaml \
  --override scoring.scorer=compass_ml \
  --override scoring.compass_ml_weights=compass/weights/compass_ml_best.pt
```

The Diagnostics tab will show `Scored by: Heuristic (Level 1)` vs
`Scored by: Compass-ML (Level 3)`, allowing direct comparison of sensitivity,
specificity, and per-drug-class WHO compliance.
