"""Thread-safe job management and background pipeline execution.

The pipeline is CPU-bound (sequence scanning, thermodynamic calculations).
We use ThreadPoolExecutor rather than asyncio because the pipeline is
synchronous Python. FastAPI endpoints are async but delegate to the pool.

max_workers=2 allows two concurrent pipeline runs. Each run needs ~1-2 GB
RAM and 4 PyTorch threads, so 2 is the practical limit on Railway 8 GB.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class _PipelineProgressHandler(logging.Handler):
    """Attaches to compass.pipeline.runner's logger to track pipeline stages.

    Uses the standard logging infrastructure instead of monkey-patching
    logger.info. Each job gets its own handler added/removed around the
    run_full() call, so concurrent jobs don't interfere.
    """

    def __init__(
        self,
        job: "PipelineJob",
        state: "AppState",
        stage_progress: dict[str, tuple[float, str]],
    ) -> None:
        super().__init__(level=logging.INFO)
        self._job = job
        self._state = state
        self._stage_progress = stage_progress

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            for key, (prog, mod) in self._stage_progress.items():
                if key in msg:
                    self._state._update_progress(self._job, prog, mod)
                    break
        except Exception:
            self.handleError(record)

from api.schemas import JobStatus, MutationInput, PipelineMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global model cache: load heavy objects once, reuse across pipeline runs.
# Each run gets its own scorer *instance* (with mutable state like
# _collected_embeddings) but shares the expensive torch model + RNA-FM cache.
# ---------------------------------------------------------------------------
_model_cache_lock = threading.Lock()
_cached_torch_model = None       # torch.nn.Module (read-only after load)
_cached_rnafm_cache = None       # EmbeddingCache (read-only lookups)
_cached_calibration = None       # dict with temperature, alpha, etc.
_cached_disc_model = None        # XGBoost/LightGBM model object
_model_cache_ready = False


def _warm_model_cache():
    """Load heavy model files into module-level globals (first run only)."""
    global _cached_torch_model, _cached_rnafm_cache, _cached_calibration
    global _cached_disc_model, _model_cache_ready

    with _model_cache_lock:
        if _model_cache_ready:
            return
        logger.info("Warming global model cache (first run)...")

        # Load Compass-ML into a temporary scorer, then steal its internals
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        tmp = CompassMlScorer(
            weights_path=None,
            rnafm_cache_dir=str(Path("compass/data/embeddings/rnafm")),
            use_rlpa=True,
            use_rnafm=True,
            collect_embeddings=False,
        )
        _cached_torch_model = tmp.model        # nn.Module in eval mode
        _cached_rnafm_cache = tmp._rnafm_cache  # EmbeddingCache
        _cached_calibration = {
            "temperature": tmp.temperature,
            "alpha": tmp.alpha,
            "calibrated": tmp.calibrated,
            "_val_rho": tmp._val_rho,
            "_calibration_meta": tmp._calibration_meta,
        }
        logger.info("Compass-ML model + RNA-FM cache loaded.")

        # Load discrimination model
        from compass.scoring.learned_discrimination import (
            LearnedDiscriminationScorer,
        )
        tmp_disc = LearnedDiscriminationScorer()
        _cached_disc_model = getattr(tmp_disc, "_model", None)
        logger.info("Discrimination model loaded.")

        _model_cache_ready = True


def _inject_cached_models(pipeline) -> None:
    """Inject pre-loaded model weights into a pipeline's scorers.

    Each pipeline keeps its own scorer instances (with per-run mutable state)
    but the heavy torch model, RNA-FM cache, and XGBoost model are shared.
    """
    if not _model_cache_ready:
        return

    ml = getattr(pipeline, "ml_scorer", None)
    if ml is not None and _cached_torch_model is not None:
        ml.model = _cached_torch_model
        ml._rnafm_cache = _cached_rnafm_cache
        if _cached_calibration:
            ml.temperature = _cached_calibration["temperature"]
            ml.alpha = _cached_calibration["alpha"]
            ml.calibrated = _cached_calibration["calibrated"]
            ml._val_rho = _cached_calibration["_val_rho"]
            ml._calibration_meta = _cached_calibration["_calibration_meta"]

    disc = getattr(pipeline, "disc_scorer", None)
    if disc is not None and _cached_disc_model is not None:
        disc._model = _cached_disc_model
        disc._model_loaded = True


class PipelineJob:
    """Tracks one pipeline run."""

    def __init__(
        self,
        name: str,
        mode: PipelineMode,
        mutations: list[MutationInput],
        config_overrides: dict,
    ) -> None:
        self.job_id: str = secrets.token_hex(16)
        self.name = name
        self.mode = mode
        self.mutations = mutations
        self.config_overrides = config_overrides
        self.status: JobStatus = JobStatus.PENDING
        self.progress: float = 0.0
        self.current_module: Optional[str] = None
        self.created_at: datetime = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.result: Optional[dict[str, Any]] = None

    def to_response(self) -> dict:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "mode": self.mode,
            "n_mutations": len(self.mutations),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "current_module": self.current_module,
            "error": self.error,
        }


class AppState:
    """Global application state with thread-safe job management."""

    def __init__(self, results_dir: str = "results/api") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, PipelineJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._rehydrate_from_disk()

    def submit_job(self, job: PipelineJob) -> str:
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(self._run_pipeline, job.job_id)
        return job.job_id

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[PipelineJob]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True,
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _rehydrate_from_disk(self) -> None:
        """Restore completed jobs from saved result files on startup."""
        for result_path in self.results_dir.glob("*.json"):
            job_id = result_path.stem
            if job_id in self._jobs:
                continue
            try:
                with open(result_path) as f:
                    result = json.load(f)
                # Build a stub PipelineJob for the completed run
                job = PipelineJob.__new__(PipelineJob)
                job.job_id = job_id
                job.name = result.get("name", job_id[:12])
                job.mode = PipelineMode.FULL if "members" in result else PipelineMode.BASIC
                job.mutations = []
                job.config_overrides = {}
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.current_module = "Complete"
                job.created_at = datetime.now(timezone.utc)
                job.started_at = None
                job.completed_at = None
                job.error = None
                job.result = result
                self._jobs[job_id] = job
                logger.info("Rehydrated job %s from disk", job_id)
            except Exception as e:
                logger.warning("Failed to rehydrate %s: %s", result_path.name, e)

    # ------------------------------------------------------------------
    # Pipeline execution (runs in thread pool)
    # ------------------------------------------------------------------

    def _update_progress(self, job: PipelineJob, progress: float, module: str) -> None:
        job.progress = progress
        job.current_module = module
        logger.info("Job %s: %.0f%% — %s", job.job_id, progress * 100, module)

    def _run_pipeline(self, job_id: str) -> None:
        job = self._jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)

        try:
            from compass.core.config import PipelineConfig, ReferenceConfig
            from compass.core.types import Drug, Mutation, MutationCategory
            from compass.pipeline.runner import COMPASSPipeline

            # Build config
            self._update_progress(job, 0.02, "Initializing")

            output_dir = self.results_dir / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Resolve organism from config_overrides (default: mtb)
            organism_id = job.config_overrides.get("organism", "mtb")

            # Per-organism reference file resolution via env vars with sensible defaults
            _REF_DEFAULTS = {
                "mtb": ("data/references/H37Rv.fasta", "data/references/H37Rv", "data/references/H37Rv.gff3"),
                "ecoli": ("data/references/ecoli_K12.fasta", "data/references/ecoli_K12", "data/references/ecoli_K12.gff3"),
                "saureus": ("data/references/saureus_NCTC8325.fasta", "data/references/saureus_NCTC8325", "data/references/saureus_NCTC8325.gff3"),
                "ngonorrhoeae": ("data/references/ngono_FA1090.fasta", "data/references/ngono_FA1090", "data/references/ngono_FA1090.gff3"),
            }
            fasta_default, idx_default, gff_default = _REF_DEFAULTS.get(organism_id, _REF_DEFAULTS["mtb"])

            # MTB uses dedicated env vars for backwards compat; others use organism-prefixed env vars
            if organism_id == "mtb":
                fasta_path = os.environ.get("H37RV_FASTA_PATH", fasta_default)
                index_path = os.environ.get("BOWTIE2_INDEX_PATH", idx_default)
                gff_path = os.environ.get("H37RV_GFF_PATH", gff_default)
            else:
                prefix = organism_id.upper()
                fasta_path = os.environ.get(f"{prefix}_FASTA_PATH", fasta_default)
                index_path = os.environ.get(f"{prefix}_INDEX_PATH", idx_default)
                gff_path = os.environ.get(f"{prefix}_GFF_PATH", gff_default)

            # Pre-flight check: ensure reference genome exists
            fasta_p = Path(fasta_path)
            if not fasta_p.exists():
                _ORG_LABELS = {
                    "mtb": "M. tuberculosis H37Rv",
                    "ecoli": "E. coli K-12 MG1655",
                    "saureus": "S. aureus NCTC 8325",
                    "ngonorrhoeae": "N. gonorrhoeae FA 1090",
                }
                label = _ORG_LABELS.get(organism_id, organism_id)
                raise FileNotFoundError(
                    f"Reference genome not found for {label}: {fasta_p}. "
                    f"Run 'python scripts/download_references.py' to download "
                    f"all required reference genomes."
                )

            ref_config = ReferenceConfig(
                genome_fasta=fasta_p,
                genome_index=Path(index_path),
                gff_annotation=Path(gff_path),
            )

            # Dot-path override mapping: frontend key → nested config path
            _OVERRIDE_MAP = {
                # Candidate generation
                "gc_min": ("candidates", "gc_min"),
                "gc_max": ("candidates", "gc_max"),
                "homopolymer_max": ("candidates", "homopolymer_max"),
                "mfe_threshold": ("candidates", "mfe_threshold"),
                "spacer_lengths": ("candidates", "spacer_lengths"),
                # Scoring
                "scorer": ("scoring", "scorer"),
                "compass_ml_use_rlpa": ("scoring", "compass_ml_use_rlpa"),
                "compass_ml_use_rnafm": ("scoring", "compass_ml_use_rnafm"),
                "discrimination_min_ratio": ("scoring", "discrimination_min_ratio"),
                # Multiplex
                "max_plex": ("multiplex", "max_plex"),
                "efficiency_weight": ("multiplex", "efficiency_weight"),
                "discrimination_weight": ("multiplex", "discrimination_weight"),
                # Primers
                "tm_min": ("primers", "tm_min"),
                "tm_max": ("primers", "tm_max"),
                "amplicon_min": ("primers", "amplicon_min"),
                "amplicon_max": ("primers", "amplicon_max"),
                "sample_type": ("primers", "sample_type"),
                # Synthetic mismatch
                "sm_enabled": ("synthetic_mismatch", "enabled"),
            }

            config_kwargs: dict[str, Any] = {
                "name": job.name,
                "output_dir": output_dir,
                "organism": organism_id,
                "reference": ref_config,
            }

            # Group overrides by sub-config section
            sub_config_kwargs: dict[str, dict[str, Any]] = {}
            for key, val in job.config_overrides.items():
                if key in _OVERRIDE_MAP:
                    section, field = _OVERRIDE_MAP[key]
                    sub_config_kwargs.setdefault(section, {})[field] = val
                else:
                    logger.warning("Unknown config override '%s' — ignored", key)

            # Build sub-configs from overrides
            from compass.core.config import (
                CandidateConfig, ScoringConfig, MultiplexConfig,
                PrimerConfig, SyntheticMismatchConfig,
            )
            _SUB_CONFIG_CLASSES = {
                "candidates": CandidateConfig,
                "scoring": ScoringConfig,
                "multiplex": MultiplexConfig,
                "primers": PrimerConfig,
                "synthetic_mismatch": SyntheticMismatchConfig,
            }
            for section, kwargs in sub_config_kwargs.items():
                cls = _SUB_CONFIG_CLASSES.get(section)
                if cls:
                    config_kwargs[section] = cls(**kwargs)

            # Auto-resolve RNA-FM cache for scoring
            scoring_section = sub_config_kwargs.get("scoring", {})
            if scoring_section.get("scorer", "compass_ml") == "compass_ml":
                if "scoring" not in config_kwargs:
                    config_kwargs["scoring"] = ScoringConfig()
                if config_kwargs["scoring"].rnafm_cache_dir is None:
                    config_kwargs["scoring"].rnafm_cache_dir = Path(
                        "compass/data/embeddings/rnafm"
                    )

            # Pre-warm model cache (blocks only the very first run)
            # Compass-ML is MTB-only; non-MTB organisms use heuristic/seq_cnn
            if organism_id == "mtb":
                _warm_model_cache()

            # When models are cached, override scoring config to skip
            # redundant disk loading inside COMPASSPipeline.__init__.
            # The scorer will initialise with model=None, then we inject
            # the cached torch model + RNA-FM cache afterwards.
            if _model_cache_ready and organism_id == "mtb":
                from compass.core.config import ScoringConfig
                skip_scoring = ScoringConfig(
                    scorer="compass_ml",
                    compass_ml_weights=Path("/dev/null/skip"),  # non-existent → skip load
                    rnafm_cache_dir=None,  # skip RNA-FM cache load
                    compass_ml_use_rlpa=True,
                    compass_ml_use_rnafm=True,
                )
                config_kwargs["scoring"] = skip_scoring

            config = PipelineConfig(**config_kwargs)
            pipeline = COMPASSPipeline(config)

            # Inject pre-loaded model weights into the pipeline's scorers.
            # Each run still has its own scorer instances with independent
            # mutable state, but shares the heavy model objects.
            if organism_id == "mtb":
                _inject_cached_models(pipeline)
            if _model_cache_ready and organism_id == "mtb":
                ml = getattr(pipeline, "ml_scorer", None)
                has_model = ml is not None and ml.model is not None
                has_cache = ml is not None and ml._rnafm_cache is not None
                cache_len = len(ml._rnafm_cache) if has_cache else 0
                disc = getattr(pipeline, "disc_scorer", None)
                has_disc = disc is not None and getattr(disc, "_model_loaded", False)
                logger.info(
                    "Model injection: CNN=%s, RNA-FM cache=%s (%d sequences), "
                    "calibrated=%s, disc=%s",
                    has_model, has_cache, cache_len,
                    getattr(ml, "calibrated", False) if ml else False,
                    has_disc,
                )

            # Convert mutations
            self._update_progress(job, 0.05, "Target Resolution")

            drug_map = {d.value: d for d in Drug}
            drug_map.update({d.name: d for d in Drug})

            mutations = []
            for m in job.mutations:
                drug = drug_map.get(m.drug.upper(), Drug.OTHER)
                cat = MutationCategory.LARGE_DELETION if m.mutation == "gene_presence" else None
                mutations.append(Mutation(
                    gene=m.gene,
                    ref_aa=m.ref_aa,
                    position=m.position,
                    alt_aa=m.alt_aa,
                    drug=drug,
                    category=cat,
                    notes="Gene presence detection" if m.mutation == "gene_presence" else None,
                ))

            # Execute pipeline
            if job.mode == PipelineMode.BASIC:
                self._update_progress(job, 0.15, "PAM Scanning")
                scored_by_target = pipeline.run(mutations)

                self._update_progress(job, 0.80, "Serializing Results")

                result = {
                    "mode": "basic",
                    "targets": {},
                }
                for label, scored_list in scored_by_target.items():
                    result["targets"][label] = [
                        sc.model_dump(mode="json") for sc in scored_list
                    ]

                job.result = result

            else:  # FULL mode
                # Hook into pipeline stages via progress updates
                self._update_progress(job, 0.10, "PAM Scanning")

                # Attach a progress handler to the pipeline runner's logger.
                # Each job gets its own handler; concurrent jobs don't interfere.
                import compass.pipeline.runner as runner_mod

                stage_progress: dict[str, tuple[float, str]] = {
                    "Module 3:": (0.10, "Candidate Filtering"),
                    "Module 4:": (0.15, "Off-Target Screening"),
                    "Module 5:": (0.20, "Heuristic Scoring"),
                    "Module 5.ML: Encoding": (0.25, "Compass-ML: Encoding"),
                    "Module 5.ML: Computing RNA-FM": (0.30, "Compass-ML: RNA-FM"),
                    "Module 5.ML: Running CNN": (0.35, "Compass-ML: CNN inference"),
                    "Module 5.ML: Inference complete": (0.40, "Compass-ML: Calibration"),
                    "Module 5.5": (0.45, "Mismatch Pairs"),
                    "Module 6:": (0.50, "SM Enhancement"),
                    "Enhancement batch:": (0.55, "SM Enhancement complete"),
                    "Module 6.5": (0.60, "Discrimination Scoring"),
                    "Discrimination scoring:": (0.65, "Discrimination complete"),
                    "Module 7:": (0.70, "Multiplex Optimization"),
                    "Module 8:": (0.78, "RPA Primer Design"),
                    "Module 8.5": (0.85, "Co-Selection Validation"),
                    "Module 9:": (0.90, "Panel Assembly"),
                    "PANEL COMPLETE": (0.95, "Export"),
                }

                _progress_handler = _PipelineProgressHandler(job, self, stage_progress)
                _runner_logger = logging.getLogger(runner_mod.logger.name)
                _runner_logger.addHandler(_progress_handler)

                try:
                    panel = pipeline.run_full(mutations)
                finally:
                    _runner_logger.removeHandler(_progress_handler)

                self._update_progress(job, 0.95, "Serializing Results")

                job.result = panel.model_dump(mode="json")

                # Persist calibration metadata from ML scorer
                if hasattr(pipeline, 'ml_scorer') and getattr(pipeline.ml_scorer, 'model', None) is not None:
                    job.result["calibration"] = {
                        "val_rho": getattr(pipeline.ml_scorer, '_val_rho', None),
                        "temperature": getattr(pipeline.ml_scorer, 'temperature', None) if hasattr(pipeline.ml_scorer, 'calibrated') else None,
                        "alpha": getattr(pipeline.ml_scorer, 'alpha', None) if hasattr(pipeline.ml_scorer, 'calibrated') else None,
                    }

                job.result["module_stats"] = pipeline.last_stats
                job.result["total_duration_ms"] = sum(
                    s.get("duration_ms", 0) for s in pipeline.last_stats
                )

                # Cache scored candidates for Block 3 endpoints (top-K, sweep, pareto)
                from api.routes.optimisation import cache_pipeline_result
                if hasattr(pipeline, "_scored_by_target"):
                    cache_pipeline_result(
                        job_id=job_id,
                        members=panel.members,
                        candidates_by_target=pipeline._scored_by_target,
                    )

            # Save result to disk
            result_path = self.results_dir / f"{job_id}.json"
            with open(result_path, "w") as f:
                json.dump(job.result, f, indent=2, default=str)

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            self._update_progress(job, 1.0, "Complete")

            logger.info("Job %s completed successfully", job_id)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error = f"Pipeline failed: {e}" if isinstance(e, FileNotFoundError) else f"Pipeline failed: {type(e).__name__}"
            logger.error("Job %s failed: %s\n%s", job_id, e, traceback.format_exc())
