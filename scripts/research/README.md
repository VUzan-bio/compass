# COMPASS Multi-Species ML Research Scripts

## Execution Order

Scripts must be run in dependency order. Each phase gates the next.

### Phase 0: Infrastructure (Days 1-3)
```bash
# 1. Download CARD database and build target metadata
python scripts/research/setup_card_data.py
# Output: data/card/amr_target_metadata.csv (42 targets with prevalence)

# 2. Extract genomic contexts for EVO-2/B-JEPA
python scripts/research/extract_target_contexts.py
# Output: data/card/target_contexts.json (±250bp contexts)
# Requires: reference genomes (python scripts/download_references.py)
```

### Phase 1: Kim 2018 Full Benchmark (Weeks 1-2)
```bash
# 3. Train 4 ablation configs × 3 seeds on full 15K HT1-1
python scripts/research/benchmark_kim2018_full.py
# Output: results/research/kim2018_benchmark/
# Decision gate: test ρ ≥ 0.75 → proceed; < 0.65 → architecture review

# 4. Calibration evaluation (run after benchmark completes)
python scripts/research/calibration_evaluation.py
# Output: results/research/calibration/
```

### Phase 2: Foundation Model LLR (Weeks 3-4)
```bash
# 5. EVO-2 DNA-level LLR on 42 AMR targets (requires GPU)
python scripts/research/evo2_amr_targets.py
# Output: results/research/evo2_llr/
# Decision gate: |ρ| > 0.5 in ≥2 organisms → add as Compass-ML scalar

# 6. ESM-2 protein-level LLR on 28 AA substitution targets
python scripts/research/esm2_amr_targets.py
# Output: results/research/esm2_llr/
# Compares DNA vs protein LLR (multi-scale hypothesis)
```

### Phase 3: Cross-Species Transfer (Weeks 4-6)
```bash
# 7. Domain-adversarial training: Kim 2018 + EasyDesign
python scripts/research/cross_species_transfer.py
# Output: results/research/cross_species/
# Decision gate: EasyDesign ρ improves ≥0.05 without Kim degradation >0.02

# 8. Discrimination model ablation (R-loop physics prior)
python scripts/research/discrimination_ablation.py
# Output: results/research/discrimination_ablation/
```

### Phase 4: Validation (Weeks 6-10)
```bash
# 9. Literature concordance: published crRNAs vs COMPASS rankings
python scripts/research/literature_concordance.py
# Output: results/research/literature_concordance/
# Decision gate: ≥60% in top-10 → compelling validation

# 10. B-JEPA clustering analysis on 42 AMR target loci
python scripts/research/bjepa_clustering.py
# Output: results/research/bjepa_clustering/
# Decision gate: ARI(mechanism) > 0.3 → functional structure learned
```

## Dependencies

| Script | Requires |
|--------|----------|
| setup_card_data.py | Internet (CARD download) |
| extract_target_contexts.py | Reference genomes, setup_card_data.py |
| benchmark_kim2018_full.py | Kim 2018 Excel, RNA-FM cache |
| calibration_evaluation.py | benchmark_kim2018_full.py checkpoints |
| evo2_amr_targets.py | extract_target_contexts.py, EVO-2 7B (GPU) |
| esm2_amr_targets.py | setup_card_data.py, ESM-2 650M |
| cross_species_transfer.py | EasyDesign Table_S2, RNA-FM cache |
| discrimination_ablation.py | EasyDesign Table_S2 |
| literature_concordance.py | COMPASS pipeline results |
| bjepa_clustering.py | extract_target_contexts.py, B-JEPA checkpoint |

## Key Metrics per Experiment

| Experiment | Primary Metric | Threshold |
|------------|---------------|-----------|
| Kim 2018 benchmark | Spearman ρ (HT2+3) | ≥ 0.75 |
| EVO-2 LLR | ρ(|LLR|, prevalence) | > 0.5 in ≥2 orgs |
| ESM-2 LLR | ρ(|LLR|, prevalence) | Comparison with EVO-2 |
| Cross-species | EasyDesign ρ improvement | ≥ +0.05 |
| Discrimination | Spearman ρ (paired) | Blend > XGBoost-only |
| Calibration | ECE | < 0.05 |
| Literature | Top-10 concordance | ≥ 60% |
| B-JEPA | ARI (mechanism) | > 0.3 |
