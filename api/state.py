"""Thread-safe job management and background pipeline execution.

The pipeline is CPU-bound (sequence scanning, thermodynamic calculations).
We use ThreadPoolExecutor rather than asyncio because the pipeline is
synchronous Python. FastAPI endpoints are async but delegate to the pool.
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

from api.schemas import JobStatus, MutationInput, PipelineMode

logger = logging.getLogger(__name__)


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
        self._executor = ThreadPoolExecutor(max_workers=1)
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
            from compass.core.types import Drug, Mutation
            from compass.pipeline.runner import COMPASSPipeline

            # Build config
            self._update_progress(job, 0.02, "Initializing")

            output_dir = self.results_dir / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            ref_config = ReferenceConfig(
                genome_fasta=Path(os.environ.get(
                    "H37RV_FASTA_PATH", "data/references/H37Rv.fasta")),
                genome_index=Path(os.environ.get(
                    "BOWTIE2_INDEX_PATH", "data/references/H37Rv")),
                gff_annotation=Path(os.environ.get(
                    "H37RV_GFF_PATH", "data/references/H37Rv.gff3")),
            )

            # Apply whitelisted overrides only
            ALLOWED_OVERRIDES = {
                "cas_type", "pam_length", "spacer_length",
                "gc_min", "gc_max", "off_target_threshold",
                "multiplex_size", "primer_opt_tm", "primer_min_tm",
                "primer_max_tm", "amplicon_size_range",
            }
            # Scoring overrides applied to nested ScoringConfig
            SCORING_OVERRIDES = {"scorer", "compass_ml_use_rlpa", "compass_ml_use_rnafm"}
            config_kwargs: dict[str, Any] = {
                "name": job.name,
                "output_dir": output_dir,
                "reference": ref_config,
            }
            scoring_kwargs: dict[str, Any] = {}
            for key, val in job.config_overrides.items():
                if key in ALLOWED_OVERRIDES:
                    config_kwargs[key] = val
                elif key in SCORING_OVERRIDES:
                    scoring_kwargs[key] = val

            if scoring_kwargs:
                from compass.core.config import ScoringConfig
                # Auto-resolve Compass-ML weights path
                if scoring_kwargs.get("scorer", "compass_ml") == "compass_ml":
                    # Don't set compass_ml_weights — let CompassMlScorer auto-detect
                    # the best available checkpoint (phase1_v2 > diagnostic > best)
                    scoring_kwargs.setdefault(
                        "rnafm_cache_dir",
                        Path("compass/data/embeddings/rnafm"),
                    )
                config_kwargs["scoring"] = ScoringConfig(**scoring_kwargs)

            config = PipelineConfig(**config_kwargs)
            pipeline = COMPASSPipeline(config)

            # Convert mutations
            self._update_progress(job, 0.05, "Target Resolution")

            drug_map = {d.value: d for d in Drug}
            drug_map.update({d.name: d for d in Drug})

            mutations = []
            for m in job.mutations:
                drug = drug_map.get(m.drug.upper(), Drug.OTHER)
                mutations.append(Mutation(
                    gene=m.gene,
                    ref_aa=m.ref_aa,
                    position=m.position,
                    alt_aa=m.alt_aa,
                    drug=drug,
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

                # Patch logger to track progress
                import compass.pipeline.runner as runner_mod
                original_info = runner_mod.logger.info

                stage_progress = {
                    "Module 3:": (0.15, "Candidate Filtering"),
                    "Module 4:": (0.20, "Off-Target Screening"),
                    "Module 5.5": (0.30, "Mismatch Pairs"),
                    "Module 5:": (0.25, "Heuristic Scoring"),
                    "Module 6.5": (0.50, "Discrimination Scoring"),
                    "Module 6:": (0.40, "SM Enhancement"),
                    "Module 7:": (0.60, "Multiplex Optimization"),
                    "Module 8.5": (0.75, "Co-Selection Validation"),
                    "Module 8:": (0.70, "RPA Primer Design"),
                    "Module 9:": (0.85, "Panel Assembly"),
                    "PANEL COMPLETE": (0.95, "Export"),
                }

                def _tracking_info(msg: str, *args: Any, **kwargs: Any) -> None:
                    try:
                        formatted = msg % args if args else msg
                        for key, (prog, mod) in stage_progress.items():
                            if key in formatted:
                                self._update_progress(job, prog, mod)
                                break
                    except Exception:
                        pass
                    original_info(msg, *args, **kwargs)

                runner_mod.logger.info = _tracking_info  # type: ignore[assignment]

                try:
                    panel = pipeline.run_full(mutations)
                finally:
                    runner_mod.logger.info = original_info  # type: ignore[assignment]

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
            job.error = f"Pipeline failed: {type(e).__name__}"
            logger.error("Job %s failed: %s\n%s", job_id, e, traceback.format_exc())
