"""Pipeline configuration.

Loaded from YAML, validated with Pydantic. One config drives the entire run.
Extended for end-to-end pipeline: SM enhancement, discrimination scoring,
AS-RPA design, multiplex optimization, and panel assembly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ReferenceConfig(BaseModel):
    genome_fasta: Path
    genome_index: Optional[Path] = None
    human_index: Optional[Path] = None
    ntm_indices: list[Path] = Field(default_factory=list)
    gff_annotation: Optional[Path] = None
    genbank_annotation: Optional[Path] = None


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
    scorer: str = "seq_cnn"  # "seq_cnn" or "compass_ml"
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
    include_is6110: bool = True


class PrimerConfig(BaseModel):
    primer_length_min: int = 25
    primer_length_max: int = 38
    tm_min: float = 57.0
    tm_max: float = 72.0
    amplicon_min: int = 80
    amplicon_max: int = 250
    enable_allele_specific: bool = True
    as_rpa_deliberate_mm_pos: list[int] = Field(default=[-2, -3])
    max_pairs_per_candidate: int = 10


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
