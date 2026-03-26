"""Organism registry — loads per-organism configuration from JSON data files.

Each organism has a JSON file in data/organisms/ containing:
  - gene synonyms (systematic ↔ common name mapping)
  - codon numbering offsets for known catalogues
  - species-specific identification control (crRNA + primers)
  - heuristic scoring weights calibrated to genome GC
  - reference genome metadata

This replaces hardcoded MTB_GENE_SYNONYMS, mtb_offsets, IS6110 constants,
and organism-specific scoring weights with a single data-driven registry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "organisms"


@dataclass(frozen=True)
class SpeciesControl:
    """Pre-validated crRNA + primers for organism-specific identification."""
    name: str                         # e.g. "IS6110", "nuc", "uidA"
    description: str
    reference: str                    # publication citation
    spacer: str                       # 20-nt crRNA spacer
    pam: str                          # 4-nt PAM
    genomic_start: int
    genomic_mid: int
    fwd_primer: Optional[str] = None  # None = design at runtime
    rev_primer: Optional[str] = None
    amplicon_length: Optional[int] = None
    channel_name: str = ""


@dataclass(frozen=True)
class OrganismProfile:
    """Complete organism configuration loaded from JSON."""
    organism_id: str                                  # e.g. "mtb", "ecoli"
    name: str                                         # e.g. "Escherichia coli"
    reference_accession: str                          # e.g. "NC_000913.3"
    reference_name: str                               # e.g. "K-12 MG1655"
    genome_length: int
    genome_gc: float
    gene_synonyms: dict[str, list[str]]               # common → [systematic IDs]
    codon_offsets: dict[str, list[int]]                # gene → [offset candidates]
    species_control: Optional[SpeciesControl] = None
    heuristic_weights: Optional[dict[str, float]] = None
    gc_optimal: float = 0.50

    @property
    def systematic_to_common(self) -> dict[str, str]:
        """Reverse mapping: systematic ID → common gene name."""
        rev: dict[str, str] = {}
        for common, sys_list in self.gene_synonyms.items():
            for sys_id in sys_list:
                rev[sys_id] = common
                rev[sys_id.lower()] = common
        return rev


# ======================================================================
# Registry cache
# ======================================================================

_REGISTRY: dict[str, OrganismProfile] = {}


def _parse_species_control(data: dict) -> Optional[SpeciesControl]:
    """Parse species_control block from JSON, returning None if absent."""
    sc = data.get("species_control")
    if sc is None:
        return None
    # Skip if spacer is null/empty (organism has no validated control yet)
    if not sc.get("spacer"):
        return None
    return SpeciesControl(
        name=sc["name"],
        description=sc.get("description", ""),
        reference=sc.get("reference", ""),
        spacer=sc["spacer"],
        pam=sc["pam"],
        genomic_start=sc.get("genomic_start", 0),
        genomic_mid=sc.get("genomic_mid", 0),
        fwd_primer=sc.get("fwd_primer"),
        rev_primer=sc.get("rev_primer"),
        amplicon_length=sc.get("amplicon_length"),
        channel_name=sc.get("channel_name", f"{sc['name']}_ID"),
    )


def load_organism(organism_id: str, data_dir: Path | None = None) -> OrganismProfile:
    """Load organism profile from JSON, with caching.

    Args:
        organism_id: Short ID matching a JSON filename (e.g. "mtb", "ecoli").
        data_dir: Override for the data/organisms/ directory.

    Returns:
        OrganismProfile with all organism-specific configuration.

    Raises:
        FileNotFoundError: If no JSON file exists for organism_id.
    """
    if organism_id in _REGISTRY:
        return _REGISTRY[organism_id]

    search_dir = data_dir or _DATA_DIR
    json_path = search_dir / f"{organism_id}.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"No organism profile for '{organism_id}'. "
            f"Expected: {json_path}\n"
            f"Available: {[p.stem for p in search_dir.glob('*.json')]}"
        )

    with open(json_path) as f:
        data = json.load(f)

    profile = OrganismProfile(
        organism_id=data["organism_id"],
        name=data["name"],
        reference_accession=data["reference_accession"],
        reference_name=data.get("reference_name", ""),
        genome_length=data.get("genome_length", 0),
        genome_gc=data.get("genome_gc", 0.50),
        gene_synonyms=data.get("gene_synonyms", {}),
        codon_offsets={k: v for k, v in data.get("codon_offsets", {}).items()},
        species_control=_parse_species_control(data),
        heuristic_weights=data.get("heuristic_weights"),
        gc_optimal=data.get("gc_optimal", 0.50),
    )

    _REGISTRY[organism_id] = profile
    logger.info("Loaded organism profile: %s (%s)", profile.name, profile.organism_id)
    return profile


def list_organisms(data_dir: Path | None = None) -> list[str]:
    """Return available organism IDs from the data directory."""
    search_dir = data_dir or _DATA_DIR
    return sorted(p.stem for p in search_dir.glob("*.json"))


def get_gene_synonyms(organism_id: str) -> dict[str, list[str]]:
    """Convenience: load organism and return gene synonym dict."""
    return load_organism(organism_id).gene_synonyms


def get_codon_offsets(organism_id: str) -> dict[str, list[int]]:
    """Convenience: load organism and return codon offset dict."""
    return load_organism(organism_id).codon_offsets


def get_species_control(organism_id: str) -> Optional[SpeciesControl]:
    """Convenience: load organism and return species control (or None)."""
    return load_organism(organism_id).species_control
