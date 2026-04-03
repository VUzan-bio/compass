"""Pipeline configuration.

Loaded from YAML, validated with Pydantic. One config drives the entire run.
Extended for end-to-end pipeline: SM enhancement, discrimination scoring,
AS-RPA design, multiplex optimization, and panel assembly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ReferenceConfig(BaseModel):
    genome_fasta: Path
    genome_index: Optional[Path] = None
    human_index: Optional[Path] = None
    # Cross-reactivity screening indices (closely related species).
    # "ntm_indices" is the legacy name (NTM = non-tuberculous mycobacteria);
    # "cross_reactivity_indices" is the organism-agnostic alias.
    # Both fields feed into the same off-target screening databases.
    ntm_indices: list[Path] = Field(default_factory=list)
    cross_reactivity_indices: list[Path] = Field(default_factory=list)
    gff_annotation: Optional[Path] = None
    genbank_annotation: Optional[Path] = None

    @model_validator(mode="after")
    def _merge_cross_reactivity(self) -> ReferenceConfig:
        """Merge ntm_indices and cross_reactivity_indices into a single list."""
        if self.cross_reactivity_indices:
            all_indices = list(self.ntm_indices) + list(self.cross_reactivity_indices)
            # Deduplicate while preserving order
            seen = set()
            merged = []
            for p in all_indices:
                if str(p) not in seen:
                    seen.add(str(p))
                    merged.append(p)
            self.ntm_indices = merged
        return self


class CandidateConfig(BaseModel):
    spacer_lengths: list[int] = Field(default=[20, 21, 23])
    use_enascas12a: bool = True
    cas_variant: Optional[str] = None
    enzyme_id: Optional[str] = None  # maps to ENZYME_REGISTRY; overrides cas_variant/use_enascas12a
    require_seed_mutation: bool = True
    gc_min: float = 0.40
    gc_max: float = 0.60
    homopolymer_max: int = 4
    mfe_threshold: float = -2.0

    def resolve_enzyme_id(self) -> str:
        """Resolve the effective enzyme ID from config fields.

        Priority: enzyme_id > cas_variant > use_enascas12a flag.
        """
        if self.enzyme_id:
            return self.enzyme_id
        if self.cas_variant:
            return self.cas_variant
        return "enAsCas12a" if self.use_enascas12a else "AsCas12a"


class SyntheticMismatchConfig(BaseModel):
    """SM enhancement for improving crRNA discrimination."""
    enabled: bool = True
    cas_variant: str = "enAsCas12a"
    min_activity_vs_mut: float = 0.3
    min_discrimination_improvement: float = 1.5
    max_synthetic_mismatches: int = 2
    allow_double_sm: bool = True


class ScoringConfig(BaseModel):
    use_heuristic: bool = True
    use_ml: bool = False
    use_discrimination: bool = True
    ml_model_path: Optional[Path] = None
    ml_model_name: str = "heuristic"
    # Compass-ML integration (replaces SeqCNN when weights available)
    scorer: str = "compass_ml"  # "compass_ml" (default) or "seq_cnn" (legacy)
    compass_ml_weights: Optional[Path] = None
    rnafm_cache_dir: Optional[Path] = None
    compass_ml_use_rlpa: bool = True
    compass_ml_use_rnafm: bool = True
    jepa_encoder_path: Optional[Path] = None
    jepa_head_path: Optional[Path] = None
    jepa_mode: str = "efficiency"
    discrimination_min_ratio: float = 2.0
    discrimination_method: str = "auto"  # "auto", "learned", "heuristic"
    discrimination_model_path: Optional[Path] = None  # path to disc_xgb.pkl


class MultiplexConfig(BaseModel):
    max_plex: int = 14
    optimizer: str = "simulated_annealing"
    max_iterations: int = 10_000
    cross_reactivity_threshold: float = 0.3
    efficiency_weight: float = 0.5
    discrimination_weight: float = 0.2
    cross_reactivity_weight: float = 0.3
    include_species_control: bool = True
    # Backwards-compat alias: include_is6110 maps to include_species_control
    include_is6110: Optional[bool] = None

    @model_validator(mode="after")
    def _migrate_is6110_flag(self) -> MultiplexConfig:
        """Map legacy include_is6110 → include_species_control."""
        if self.include_is6110 is not None:
            self.include_species_control = self.include_is6110
        return self


class PrimerConfig(BaseModel):
    primer_length_min: int = 25
    primer_length_max: int = 38
    tm_min: float = 57.0
    tm_max: float = 72.0
    tm_opt: Optional[float] = None  # Organism-dependent; loaded from profile
    amplicon_min: int = 80
    amplicon_max: int = 250
    sample_type: str = "genomic"  # "cfDNA" (80-120bp) or "genomic" (80-250bp)
    enable_allele_specific: bool = True
    as_rpa_deliberate_mm_pos: list[int] = Field(default=[-2, -3])
    max_pairs_per_candidate: int = 10

    @model_validator(mode="after")
    def _set_amplicon_from_sample_type(self) -> PrimerConfig:
        """Set amplicon_max from sample_type if using default."""
        if self.sample_type == "cfDNA" and self.amplicon_max == 250:
            self.amplicon_max = 120
        return self


class PipelineConfig(BaseModel):
    """Top-level config — one object drives the full pipeline."""
    name: str = "compass_run"
    output_dir: Path = Path("results")
    organism: str = "mtb"
    reference: ReferenceConfig
    candidates: CandidateConfig = CandidateConfig()
    synthetic_mismatch: SyntheticMismatchConfig = SyntheticMismatchConfig()
    scoring: ScoringConfig = ScoringConfig()
    multiplex: MultiplexConfig = MultiplexConfig()
    primers: PrimerConfig = PrimerConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
