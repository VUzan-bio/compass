# COMPASS

**CRISPR-Optimized Multiplex Panel And Spacer Selection**

End-to-end computational pipeline for designing multiplexed CRISPR-Cas12a diagnostic panels targeting drug-resistant *Mycobacterium tuberculosis* from blood.

Live platform: [compass-design.app](https://compass-design.app/)

---

COMPASS takes WHO-catalogued drug-resistance mutations as input and produces a complete, optimised diagnostic panel: crRNA sequences, RPA primer pairs, discrimination predictions, and clinical compliance metrics — ready for experimental validation on electrochemical or fluorescence platforms. The pipeline handles PAM deserts in the GC-rich *M. tuberculosis* genome (65.6% GC) through automatic proximity detection with allele-specific RPA primer design. Guide scoring uses Compass-ML — a dual-branch CNN + RNA-FM architecture with physics-informed R-loop attention — calibrated via Spearman-optimised ensemble weighting against both cis-cleavage (Kim et al. 2018) and trans-cleavage (Huang et al. 2024) benchmarks. Discrimination prediction uses a gradient-boosted model trained on 6,136 paired MUT/WT measurements with 15 thermodynamic features. COMPASS produces a 15-channel MDR-TB panel covering rifampicin, isoniazid, ethambutol, pyrazinamide, fluoroquinolone, and aminoglycoside resistance, plus an IS6110 species identification control.

## Pipeline

<img width="936" height="470" alt="compass-archit" src="https://github.com/user-attachments/assets/800af375-aea3-4ae2-80d2-4e0f7912fcfe" />

<br />

Ten modules execute sequentially across three processing blocks:

**Block 1 — Candidate Generation (M1–M4)**

| Module | Function | Method |
|--------|----------|--------|
| M1 Target Resolution | WHO mutations → genomic coordinates on H37Rv | 5-strategy offset resolver against NC_000962.3 |
| M2 PAM Scanning | Identify Cas12a-compatible protospacer sites | Both strands: TTTV canonical + enAsCas12a relaxed (TTTN, TTCN, TCTV, CTTV); spacer 18–23 nt |
| M3 Candidate Filtering | Biophysical constraint enforcement | GC 30–70%, homopolymer < 5 nt, self-complementarity MFE > −3.0 kcal/mol |
| M4 Off-Target Screening | Genome-wide specificity check | Bowtie2 against H37Rv (4.41 Mb); ≤3 mismatches flagged |

**Block 2 — Scoring & Optimisation (M5–M9)**

| Module | Function | Method |
|--------|----------|--------|
| M5 Efficiency Scoring | Predict Cas12a cleavage activity | Compass-ML ensemble (see Architecture) |
| M5.5 Mismatch Pair Generation | Create MUT/WT spacer pairs | Complement substitution at SNP position |
| M6 Discrimination Scoring | Predict MUT/WT selectivity | Learned model: LightGBM, 15 thermodynamic features, 6,136 pairs (r = 0.46); fallback: position-dependent heuristic (r = 0.30) |
| M7 Synthetic Mismatch Enhancement | Boost discrimination for borderline candidates | Deliberate mismatches at seed positions 2–6; 2–6× → 10–100× |
| M8 Multiplex Optimisation | Select optimal panel combination | Simulated annealing, 10,000 iterations; objective: efficiency + discrimination − cross-reactivity |
| M9 RPA Primer Co-Design | Design amplification primers per target | Standard RPA (28–38 nt, Tm 57–72°C) + allele-specific primers for proximity candidates |

**Block 3 — Clinical Assessment (M10)**

| Module | Function | Method |
|--------|----------|--------|
| M10 Panel Assembly | Compile final panel with clinical metrics | Per-drug-class sensitivity/specificity against WHO TPP 2024; three operating presets; ranked backup alternatives |

For mutations in PAM-desert regions (e.g., *rpoB* RRDR at 70%+ GC with no T-rich PAM within 50 bp), the pipeline automatically falls back to **proximity detection**: the crRNA targets a nearby accessible site while an allele-specific RPA primer provides mutation-specific amplification.

## Compass-ML Architecture

Dual-branch architecture with physics-informed attention for diagnostic guide scoring. 235,000 trainable parameters.

```
Target DNA (4×34 one-hot) ──→ Multi-scale Conv1d (k=3,5,7; 32ch) → BN → (B, 34, 64) ─┐
                                                                                         ├→ concat (128-dim) → RLPA → pool → Efficiency Head → σ
crRNA spacer (20×640 RNA-FM) → Linear(640→64) + zero-pad to 34 ──→ (B, 34, 64) ────────┘                                  └→ Discrimination Head → Softplus
```

**CNN branch.** Multi-scale parallel Conv1d (kernel sizes 3, 5, 7; 32 channels each) with batch normalisation and dropout (0.3). Input: 34-nucleotide one-hot encoded target context (4 nt PAM + 20 nt protospacer + 10 nt flanking). Output: 64-dimensional features per position.

**RNA-FM branch.** Frozen RNA-FM embeddings (Chen et al. 2022; 23M training sequences, 640-dim per-nucleotide) projected to 64 dimensions via learned linear layer, zero-padded from 20 to 34 positions for alignment with the CNN branch. Captures guide RNA folding stability and accessibility properties.

**R-Loop Propagation Attention (RLPA).** Single-head causal self-attention with 32-dim Q/K/V projections and a learnable 34×34 positional bias matrix. The causal (lower-triangular) mask encodes the directional R-loop propagation of Cas12a (PAM-proximal → PAM-distal), motivated by the kinetic observation that R-loop formation is sequential and reversible (Strohkendl et al. 2018). RLPA improved cross-dataset generalisation by +6.7% on the Kim 2018 cross-library evaluation (test ρ: 0.496 → 0.534). ~25,000 parameters.

**Output heads.** Efficiency: 128 → 64 → 32 → 1 with GELU activation and sigmoid output. Discrimination: 1024 → 64 → 32 → 1 with Softplus output. Loss: L_Huber(efficiency) + 0.5 × (1 − ρ_soft_Spearman) + λ_disc × L_Huber(log D), where ρ_soft uses the differentiable ranking of Blondel et al. (2020).

### Training Data

| Dataset | Enzyme | Measurement | Guides | Source |
|---------|--------|-------------|--------|--------|
| Kim et al. 2018 | AsCas12a | Indel frequency (cis-cleavage) | ~15,000 | HT-PAMDA, three HEK293T libraries |
| Huang et al. 2024 (EasyDesign) | LbCas12a | FAM-quencher fluorescence (trans-cleavage) | ~10,000 | Pathogen-diverse diagnostic targets |

The production checkpoint (multi-dataset, no domain adversarial training) achieves:
- **Trans-cleavage ρ = 0.55** (EasyDesign benchmark — the diagnostic-relevant readout)
- **Cis-cleavage ρ = 0.49** (Kim 2018 benchmark)

Models trained only on cis-cleavage data show ρ = 0.04 on the trans-cleavage benchmark — the multi-dataset approach provides a 12× improvement in diagnostic prediction accuracy.

### Temperature Calibration & Ensemble

Each scorer applies quantile-matched temperature scaling: `calibrated = sigmoid(logit(raw) / T)`, then `ensemble = α × heuristic + (1 − α) × calibrated`. The calibration file (compass/weights/calibration.json) stores T and α per scorer, fitted on the validation set.

| Scorer | Parameters | T | α | Val ρ | Test ρ |
|--------|-----------|---|---|-------|--------|
| Heuristic | 5 weights | — | — | — | ~0.18 |
| SeqCNN | 110K | 7.53 | 0.007 | 0.74 | 0.53 |
| Compass-ML | 235K | 0.74 | 0.028 | 0.71 | 0.55 |

## Learned Discrimination Model

Discrimination prediction replaces the position-dependent heuristic (Strohkendl et al. 2018) with a gradient-boosted model trained on paired MUT/WT measurements.

**Training data.** 6,136 paired measurements extracted from the EasyDesign dataset: same crRNA tested on both perfect-match (0-mismatch) and single-mismatch targets, from 1,224 unique guides. The discrimination ratio for each pair is the ratio of trans-cleavage activity on the perfect match vs the mismatched target.

**Features (15).** Four categories:
- Position: spacer position, seed binary, normalised position, sensitivity region
- Mismatch chemistry: ΔΔG destabilisation penalty, wobble pair flag, purine-purine flag, transition/transversion class
- Thermodynamics: cumulative ΔG at mismatch position, seed ΔG (positions 1–8), total hybrid ΔG, energy ratio (cumulative/penalty)
- Sequence context: global GC content, local GC (±2 nt window)

Thermodynamic parameters from Sugimoto et al. (2000, Biochemistry) for RNA:DNA mismatch penalties and Sugimoto et al. (1995, Biochemistry) for RNA:DNA hybrid nearest-neighbour parameters.

**Results.** 3-fold cross-validation (guide-level stratified):

| Model | RMSE | Pearson r |
|-------|------|-----------|
| Position heuristic (Strohkendl 2018) | 0.641 | 0.298 |
| **Learned (LightGBM, 15 features)** | **0.540** | **0.459** |

Top features by importance: seed ΔG, total hybrid ΔG, cumulative ΔG at mismatch, energy ratio — thermodynamic features dominate over position alone.

## Post-Optimisation Analysis

### Primer Dimer Thermodynamics

All pairwise primer interactions (465 pairs for a 15-target panel) are evaluated using SantaLucia (2004) nearest-neighbour parameters. Two ΔG values per pair:
- **ΔG_full**: most stable dimer across all alignment positions
- **ΔG_3prime**: most stable dimer anchored at the 3' end of at least one primer (extensible — produces amplification artifacts)

Thresholds: ΔG_3prime < −6.0 kcal/mol = high risk; < −4.0 = moderate risk. Displayed as a 30×30 heatmap on the Multiplex tab. Currently post-optimisation analysis; integration into the simulated annealing cost function is planned.

### AS-RPA Thermodynamic Discrimination

For proximity candidates, the forward primer's 3' terminal nucleotide matches only the mutant allele. Discrimination is estimated from the ΔΔG between matched (MUT) and mismatched (WT) primer-template complexes at the 3' anchor region:
- Terminal C:C mismatch → ΔΔG ≈ 6.3 kcal/mol → strong block
- Terminal G:T wobble → ΔΔG ≈ 0.5 kcal/mol → weak block
- Boltzmann conversion: disc ≈ exp(ΔΔG / RT) at 37°C, capped at 100×

Mismatch penalty data from Allawi & SantaLucia (1997, 1998) and RPA-specific tolerance from systematic mismatch profiling (PMC12179515, 2025). Penultimate mismatch strategy per Ye et al. (2019).

## Diagnostics Presets

Three operating modes control candidate selection:

| Preset | Efficiency ≥ | Discrimination ≥ | Use case |
|--------|-------------|-------------------|----------|
| High Sensitivity | 0.30 | 2× | Field screening, maximum coverage |
| Balanced (WHO TPP) | 0.40 | 3× | Clinical diagnostic deployment |
| High Specificity | 0.60 | 5× | Confirmatory testing, reference labs |

WHO TPP compliance is evaluated per drug class: ≥95% sensitivity for RIF, ≥90% for INH and FQ, ≥80% for EMB, PZA, and AG. Specificity is approximated as 1 − 1/disc for Direct candidates and from thermodynamic AS-RPA estimates for Proximity candidates — all marked "Pending" as experimental validation is required.

## Platform

The web platform ([compass-design.app](https://compass-design.app)) provides six result tabs per panel run:

| Tab | Content |
|-----|---------|
| **Overview** | Score distribution, drug class coverage, score-vs-discrimination scatter plot |
| **Candidates** | Per-candidate detail: spacer architecture, interpretation, oligo sequences, evidence metadata. Expandable rows with Top-K alternatives |
| **Discrimination** | Direct detection ranking (learned model) + AS-RPA thermodynamic estimates for proximity candidates |
| **Primers** | Standard and allele-specific RPA primer pairs, amplicon sizes, SM status |
| **Multiplex** | Panel composition, primer dimer ΔG heatmap, AS-RPA discrimination table, crRNA cross-reactivity matrix |
| **Diagnostics** | WHO TPP compliance per drug class, MUT vs WT density plots (filtered per preset), per-target readiness breakdown, parameter sweep |

The **Research** page provides experimental tools: Scorer Comparison Lab, R-Loop Thermodynamic Explorer (per-position cumulative ΔG profiles with MUT/WT overlay), Ablation Tracker (cis vs trans benchmark scatter plot), and Feature Importance analysis.

## Repository Structure

```
compass/                   Core pipeline library (10 modules)
  core/                  Target resolution, PAM scanning, filtering, scoring
  primers/               Standard RPA + AS-RPA primer design
  multiplex/             Simulated annealing optimiser, primer dimer analysis
  research/              Thermodynamic profiling, scorer comparison
  nuclease/              NucleaseProfile configuration system
  scoring/               Heuristic, SeqCNN, Compass-ML scorers
  weights/               Model checkpoints + calibration files

compass-net/               Standalone ML model
  models/                Compass-ML architecture, discrimination model
  data/                  Data loaders, discrimination pair extraction
  features/              Thermodynamic feature computation
  scripts/               Training scripts (Phase 1–3, multi-dataset)

api/                     FastAPI REST + WebSocket backend (22 endpoints)
compass-ui/                React 19 + Vite SPA frontend
tests/                   97 tests across 6 files
```

## Training

### Compass-ML

```bash
# Phase 1: CNN + RNA-FM + RLPA (Kim 2018)
cd compass-net && python scripts/run_phase1_rlpa.py

# Phase 2: Multi-task (efficiency + discrimination heads)
python scripts/run_phase2_multitask.py

# Phase 3: Multi-dataset (Kim 2018 + EasyDesign, no domain adversarial)
python scripts/run_multidataset.py

# Temperature calibration
python -m compass.scoring.calibrate_compass_ml
```

### Discrimination Model

```bash
# Extract paired MUT/WT measurements from EasyDesign
python compass-net/scripts/train_discrimination.py \
    --data_dir compass-net/data/external/easydesign/ \
    --output compass/weights/disc_model.joblib
```

### SeqCNN (baseline)

```bash
python -m compass.scoring.train_cnn --data-dir compass/data/kim2018/ --epochs 200
python -m compass.scoring.calibrate
```

## Limitations

- **Domain shift.** Trained on wild-type AsCas12a (Kim 2018) and LbCas12a (EasyDesign); deployed on enAsCas12a (E174R/S542R/K548R). The engineered variant's altered PAM recognition and potentially different cleavage kinetics are not captured by the training data. Active learning from experimental validation is the intended calibration mechanism.
- **GC regime.** Training data median GC ≈ 50%; *M. tuberculosis* targets range 50–78% GC. The heuristic penalises high GC; Compass-ML, trained on diverse sequences, partially compensates. Experimental measurement on high-GC targets will determine whether scores underpredict or overpredict actual performance.
- **Discrimination model.** Pearson r = 0.46 explains ~21% of variance. The remaining 79% includes protein-mediated effects (conformational activation kinetics, NTS threading), mismatch-type-specific structural perturbations, and experimental noise. Position and thermodynamic features are necessary but not sufficient.
- **Multiplex modelling.** Cross-reactivity is sequence-based (Bowtie2); primer dimer stability is thermodynamic (SantaLucia NN, post-optimisation). Enzyme competition (15 crRNAs competing for Cas12a), RPA amplification bias, and reporter crosstalk are not modelled.
- **Specificity proxy.** The formula 1 − 1/disc assumes perfectly separated signal distributions. Actual specificity depends on signal variance and threshold selection. All specificity values are marked "Pending" pending experimental validation.
- **Single reference genome.** All designs target H37Rv. Lineage-specific SNPs near target sites could affect PAM availability or primer binding in non-H37Rv strains (e.g., lineage 2/Beijing, ~25% of global MDR-TB).

## References

### CRISPR-Cas12a Biology
1. Zetsche B, Gootenberg JS, Abudayyeh OO, et al. Cpf1 is a single RNA-guided endonuclease of a class 2 CRISPR-Cas system. *Cell* **163**, 759–771 (2015). [DOI](https://doi.org/10.1016/j.cell.2015.09.038)
2. Chen JS, Ma E, Harrington LB, et al. CRISPR-Cas12a target binding unleashes indiscriminate single-stranded DNase activity. *Science* **360**, 436–439 (2018). [DOI](https://doi.org/10.1126/science.aar6245)
3. Strohkendl I, Saifuddin FA, Rybarski JR, et al. Kinetic basis for DNA target specificity of CRISPR-Cas12a. *Molecular Cell* **71**, 816–824 (2018). [DOI](https://doi.org/10.1016/j.molcel.2018.06.043)
4. Kleinstiver BP, Sousa AA, Walton RT, et al. Engineered CRISPR-Cas12a variants with increased activities and improved targeting ranges for gene, epigenetic and base editing. *Nature Biotechnology* **37**, 276–282 (2019). [DOI](https://doi.org/10.1038/s41587-018-0011-0)
5. Strohkendl I, Saha A, Moy C, et al. Cas12a domain flexibility guides R-loop formation and forces RuvC resetting. *Molecular Cell* **84**, 2717–2731 (2024). [DOI](https://doi.org/10.1016/j.molcel.2024.05.032)
6. Swarts DC, van der Oost J, Jinek M. Structural basis for guide RNA processing and seed-dependent DNA targeting by CRISPR-Cas12a. *Molecular Cell* **66**, 221–233 (2017). [DOI](https://doi.org/10.1016/j.molcel.2017.03.016)

### R-Loop Thermodynamics
7. Sugimoto N, Nakano S, Katoh M, et al. Thermodynamic parameters to predict stability of RNA/DNA hybrid duplexes. *Biochemistry* **34**, 11211–11216 (1995). [DOI](https://doi.org/10.1021/bi00035a029)
8. SantaLucia J Jr. A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics. *PNAS* **95**, 1460–1465 (1998). [DOI](https://doi.org/10.1073/pnas.95.4.1460)
9. Zhang J, Guan X, Moon J, et al. Interpreting CRISPR-Cas12a enzyme kinetics through free energy change of nucleic acids. *Nucleic Acids Research* **52**, 14077–14092 (2024). [DOI](https://doi.org/10.1093/nar/gkae1124)
10. Aris KDP, Cofsky JC, Shi H, et al. Dynamic basis of supercoiling-dependent DNA interrogation by Cas12a via R-loop intermediates. *Nature Communications* **16**, 2939 (2025). [DOI](https://doi.org/10.1038/s41467-025-57703-y)

### Guide Activity Prediction
11. Kim HK, Min S, Song M, et al. Deep learning improves prediction of CRISPR-Cpf1 guide RNA activity. *Nature Biotechnology* **36**, 239–241 (2018). [DOI](https://doi.org/10.1038/nbt.4061)
12. Huang B, Mu K, Li G, et al. Deep learning enhancing guide RNA design for CRISPR/Cas12a-based diagnostics. *iMeta* **3**, e214 (2024). [DOI](https://doi.org/10.1002/imt2.214)
13. Chen J, Hu Z, Sun S, et al. Interpretable RNA Foundation Model from unannotated data for highly accurate RNA structure and function predictions. arXiv:2204.00300 (2022). [arXiv](https://arxiv.org/abs/2204.00300)
14. Blondel M, Teboul O, Berthet Q, Djolonga J. Fast differentiable sorting and ranking. *ICML* (2020). [Soft Spearman loss in Compass-ML training]
15. Yao Z, Li W, He K, et al. Facilitating crRNA design by integrating DNA interaction features of CRISPR-Cas12a system. *Advanced Science* **12**, e2501269 (2025). [DOI](https://doi.org/10.1002/advs.202501269)

### Mismatch Discrimination
16. Sugimoto N, Nakano M, Nakano S. Thermodynamics-structure relationship of single mismatches in RNA/DNA duplexes. *Biochemistry* **39**, 11270–11281 (2000). [DOI](https://doi.org/10.1021/bi000819p)
17. Allawi HT, SantaLucia J Jr. Thermodynamics of internal C·T mismatches in DNA. *Nucleic Acids Research* **26**, 2694–2701 (1998). [DOI](https://doi.org/10.1093/nar/26.11.2694)
18. Kohabir KAV, et al. Synthetic mismatches enable specific CRISPR-Cas12a-based detection of genome-wide SNVs tracked by ARTEMIS. *Cell Reports Methods* **4**, 100912 (2024). [DOI](https://doi.org/10.1016/j.crmeth.2024.100912)
19. Nguyen GT, et al. CRISPR-Cas12a exhibits metal-dependent specificity switching. *Nucleic Acids Research* **52**, 9343–9359 (2024). [DOI](https://doi.org/10.1093/nar/gkae613)

### CRISPR Diagnostic Design Tools
20. Low SJ, O'Neill M, Kerry WJ, et al. PathoGD: an integrative genomics approach to primer and guide RNA design for CRISPR-based diagnostics. *Communications Biology* **8**, 147 (2025). [DOI](https://doi.org/10.1038/s42003-025-07591-1)

### Clinical Standards
21. WHO. Target product profiles for tuberculosis diagnosis and detection of drug resistance. Geneva: World Health Organization (2024). ISBN: 978-92-4-009769-8.
22. WHO. Catalogue of mutations in *Mycobacterium tuberculosis* complex and their association with drug resistance, 2nd edition. Geneva: World Health Organization (2023).
23. CRyPTIC Consortium. A data compendium associating the genomes of 12,289 *Mycobacterium tuberculosis* isolates with quantitative resistance phenotypes to 13 antibiotics. *PLoS Biology* **20**, e3001721 (2022). [DOI](https://doi.org/10.1371/journal.pbio.3001721)

### CRISPR Diagnostics
24. Broughton JP, Deng X, Yu G, et al. CRISPR-Cas12-based detection of SARS-CoV-2. *Nature Biotechnology* **38**, 870–874 (2020). [DOI](https://doi.org/10.1038/s41587-020-0513-4)
25. Ai JW, Zhou X, Xu T, et al. CRISPR-based rapid and ultra-sensitive diagnostic test for *Mycobacterium tuberculosis*. *Emerging Microbes & Infections* **8**, 1361–1369 (2019). [DOI](https://doi.org/10.1080/22221751.2019.1664939)

### Bioinformatics
26. Langmead B, Salzberg SL. Fast gapped-read alignment with Bowtie 2. *Nature Methods* **9**, 357–359 (2012). [DOI](https://doi.org/10.1038/nmeth.1923)

## Citation

```bibtex
@software{compass2025,
  author = {Uzan, Valentin},
  title = {COMPASS: Guide RNA Automated Resistance Diagnostics},
  year = {2025},
  url = {https://github.com/VUzan-bio/compass},
  note = {Computational pipeline for multiplexed CRISPR-Cas12a MDR-TB diagnostics}
}
```

## Acknowledgements

Developed for the BRIDGE Discovery grant project on CRISPR-Cas12a electrochemical diagnostics for drug-resistant tuberculosis, in collaboration with the deMello Group (ETH Zurich) and CSEM.


## License

MIT License. See [LICENSE](LICENSE) for details.
