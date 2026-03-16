"""API request/response models.

Decoupled from compass.core.types — the API layer serializes pipeline
outputs into these response models. No direct import of domain types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ======================================================================
# Enums
# ======================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineMode(str, Enum):
    BASIC = "basic"
    FULL = "full"


class ExportFormat(str, Enum):
    JSON = "json"
    TSV = "tsv"
    CSV = "csv"
    FASTA = "fasta"


# ======================================================================
# Request models
# ======================================================================

class MutationInput(BaseModel):
    gene: str
    ref_aa: str
    position: int
    alt_aa: str
    drug: str = "OTHER"


class PipelineRunRequest(BaseModel):
    name: str = "COMPASS Run"
    mode: PipelineMode = PipelineMode.FULL
    mutations: list[MutationInput]
    enzyme_id: Optional[str] = None  # "AsCas12a" or "enAsCas12a"; None = use default
    config_overrides: dict = Field(default_factory=dict)


# ======================================================================
# Response models
# ======================================================================

class JobResponse(BaseModel):
    job_id: str
    name: str
    status: JobStatus
    mode: PipelineMode
    n_mutations: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_module: Optional[str] = None
    error: Optional[str] = None


class CandidateSummary(BaseModel):
    candidate_id: str
    spacer_seq: str
    pam_seq: str
    pam_variant: str
    strand: str
    gc_content: float
    detection_strategy: str
    mutation_position_in_spacer: Optional[int] = None
    wt_spacer_seq: Optional[str] = None
    composite_score: float
    cnn_score: Optional[float] = None
    cnn_calibrated: Optional[float] = None
    ensemble_score: Optional[float] = None
    discrimination_ratio: Optional[float] = None
    discrimination: Optional[dict] = None
    disc_method: Optional[str] = None  # "neural", "neural_enhanced", or "feature"
    neural_disc: Optional[float] = None  # Neural disc head prediction (if available)
    feature_disc: Optional[float] = None  # Feature-based disc prediction (if available)
    thermo_ddg: Optional[float] = None  # RNA:DNA hybrid ddG at mismatch (kcal/mol)
    mm_position_pam: Optional[int] = None  # Mismatch position relative to PAM (1-20)
    pam_penalty: Optional[float] = None       # PAM activity penalty (1.0 = canonical, <1.0 = expanded)
    is_canonical_pam: Optional[bool] = None   # True if TTTV
    activity_qc: Optional[float] = None       # Biophysical quality sub-score (GC, structure, homopolymer, offtarget)
    discrimination_qc: Optional[float] = None # SNP discrimination sub-score (seed pos, mismatch type, flanking GC)
    mismatch_type_score: Optional[float] = None  # Transversion (1.0) vs transition (0.5) at mutation site
    flanking_gc_score: Optional[float] = None    # Local GC around mismatch (lower = better discrimination)
    enzyme_id: Optional[str] = None           # Cas12a variant used
    pam_disrupted: Optional[bool] = None       # SNP disrupts PAM consensus → binary discrimination
    pam_disruption_type: Optional[str] = None  # "wt_pam_broken" | "mut_pam_broken" | None
    ml_scores: list[dict] = Field(default_factory=list)
    rank: Optional[int] = None

    # Heuristic sub-scores (detailed breakdown)
    seed_position_score: Optional[float] = None
    gc_penalty: Optional[float] = None
    structure_penalty: Optional[float] = None
    homopolymer_penalty: Optional[float] = None
    offtarget_penalty: Optional[float] = None
    proximity_bonus: Optional[float] = None

    # Thermodynamic data
    mfe: Optional[float] = None               # minimum free energy kcal/mol
    cumulative_dg: Optional[float] = None
    local_dg: Optional[float] = None

    # Off-target details
    offtarget_hits: Optional[int] = None
    offtarget_is_clean: Optional[bool] = None

    # Discrimination model details
    disc_confidence: Optional[float] = None
    disc_feature_vector: Optional[dict] = None
    disc_model_name: Optional[str] = None

    # ML metadata
    ml_confidence: Optional[float] = None

    # Mismatch details
    mismatch_type: Optional[str] = None       # "transition" or "transversion"
    ref_base_at_mutation: Optional[str] = None


class TargetResult(BaseModel):
    label: str
    gene: str
    mutation: str
    drug: str
    detection_strategy: str
    n_candidates: int
    selected_candidate: Optional[CandidateSummary] = None
    has_primers: bool = False
    fwd_primer: Optional[str] = None
    rev_primer: Optional[str] = None
    amplicon_length: Optional[int] = None
    # Primer details
    fwd_tm: Optional[float] = None
    rev_tm: Optional[float] = None
    fwd_gc: Optional[float] = None
    rev_gc: Optional[float] = None
    primer_dimer_dg: Optional[float] = None
    amplicon_seq: Optional[str] = None
    proximity_distance: Optional[int] = None
    has_sm: bool = False
    sm_enhanced_spacer: Optional[str] = None
    sm_position: Optional[int] = None
    sm_discrimination_score: Optional[float] = None
    sm_improvement_factor: Optional[float] = None
    sm_original_base: Optional[str] = None
    sm_replacement_base: Optional[str] = None
    asrpa_discrimination: Optional[dict] = None
    # Readiness scoring (computed post-hoc over panel)
    readiness_score: Optional[float] = None
    readiness_components: Optional[dict] = None
    experimental_priority: Optional[int] = None
    risk_profile: Optional[dict] = None
    priority_reason: Optional[str] = None


class PanelSummary(BaseModel):
    plex: int
    complete_targets: int
    panel_score: Optional[float] = None
    mean_discrimination: Optional[float] = None
    direct_count: int = 0
    proximity_count: int = 0
    enzyme_id: Optional[str] = None
    canonical_pam_count: int = 0       # targets using TTTV
    expanded_pam_count: int = 0        # targets using non-canonical PAMs


class ModuleStats(BaseModel):
    module_id: str
    module_name: str
    duration_ms: int
    candidates_in: int
    candidates_out: int
    detail: str
    breakdown: dict = Field(default_factory=dict)


class PipelineResultResponse(BaseModel):
    job_id: str
    panel: PanelSummary
    targets: list[TargetResult]
    module_stats: list[ModuleStats] = []
    total_duration_ms: int = 0
    primer_dimer_matrix: Optional[list[list[float]]] = None
    primer_dimer_labels: Optional[list[str]] = None
    primer_dimer_report: Optional[dict] = None
    # Calibration metadata
    calibration: Optional[dict] = None        # {val_rho, temperature, alpha, model_params}
    # SA convergence
    optimizer_score_trace: Optional[list] = None


# ======================================================================
# Scoring models (Phase 4)
# ======================================================================

class ScoringModelInfo(BaseModel):
    name: str
    status: str  # "ready", "no_weights", "disabled"
    description: str = ""


class JEPAScoreRequest(BaseModel):
    spacer_sequences: list[str]
    encoder_path: Optional[str] = None


class JEPAScoreResponse(BaseModel):
    predictions: list[dict]


# ======================================================================
# Validation models (Phase 4)
# ======================================================================

class ValidationUpload(BaseModel):
    measurements: list[dict]


class UncertaintyCandidate(BaseModel):
    candidate_id: str
    target_label: str
    spacer_seq: str
    uncertainty: float
    predicted_efficiency: float


# ======================================================================
# Spatially-addressed electrode array models
# ======================================================================

class PoolStatsSchema(BaseModel):
    pool_id: str
    targets: list[str]
    n_targets: int
    n_primers: int
    high_risk_dimers: int
    worst_dg: float


class PoolingResultSchema(BaseModel):
    pools: dict[str, list[str]]
    pool_stats: dict[str, PoolStatsSchema]
    total_high_risk_single_tube: int
    total_high_risk_after_pooling: int
    reduction_pct: float
    electrode_layout: list[list[str]]
    target_to_pool: dict[str, str]


class KineticEstimateSchema(BaseModel):
    target: str
    t_rnp_formation: float
    t_target_recognition: float
    t_signal_generation: float
    t_total: float
    efficiency: float
    is_weak: bool


class KineticsResultSchema(BaseModel):
    estimates: list[KineticEstimateSchema]
    parameters: dict
    insight: str


class AmpliconPadSpecificitySchema(BaseModel):
    labels: list[str]
    matrix: list[list[float]]
    n_targets: int
    high_risk_pairs: list[dict]
    co_amplicon_note: str
    validation_note: str
