"""Centralized enzyme configuration registry.

Defines Cas12a variants with PAM recognition, activity penalties, and
metadata. PAM activity penalties for enAsCas12a are from Kleinstiver
et al., Nature Biotechnology 2019 (Fig. 2d, indel quantification on
endogenous human sites).

WT AsCas12a: strict TTTV-only recognition.
enAsCas12a (E174R/S542R/K548R): expanded PAM tolerance — ~5× more
candidate sites in GC-rich genomes like M. tuberculosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── PAM activity penalties (Kleinstiver et al. 2019, Fig. 2d) ────────
# Values represent relative indel activity vs canonical TTTV = 1.0.
# These are applied as multiplicative penalties to composite scores.

KLEINSTIVER_2019_PENALTIES: Dict[str, float] = {
    "TTTV": 1.00,   # canonical — no penalty
    "TTTT": 0.75,   # recognized by enAsCas12a but not WT; not in IUPAC V
    "TTCV": 0.65,   # moderate activity
    "TATV": 0.55,   # moderate activity
    "CTTV": 0.45,   # reduced
    "TCTV": 0.40,   # reduced
    "TGTV": 0.35,   # low
    "ATTV": 0.30,   # low
    "GTTV": 0.25,   # low
}


@dataclass(frozen=True)
class PAMSpec:
    """A single PAM pattern with IUPAC notation and activity weight."""
    pattern: str       # IUPAC pattern, e.g. "TTTV"
    activity: float    # relative activity (0-1), Kleinstiver 2019
    label: str         # display label

    @property
    def is_canonical(self) -> bool:
        return self.pattern == "TTTV"


@dataclass(frozen=True)
class EnzymeConfig:
    """Configuration for a Cas12a variant."""
    enzyme_id: str              # unique identifier
    display_name: str           # human-readable name
    description: str            # short description
    pam_specs: Tuple[PAMSpec, ...]  # PAM patterns with activities
    seed_start: int = 1         # seed region start (1-indexed from PAM)
    seed_end: int = 8           # seed region end
    spacer_lengths: Tuple[int, ...] = (18, 19, 20, 21, 22, 23)
    scanner_key: str = ""       # maps to scanner.CONFIGS key
    source: str = ""            # literature reference
    mutations: str = ""         # protein mutations (empty for WT)

    @property
    def pam_patterns(self) -> List[str]:
        """All IUPAC PAM patterns this enzyme recognizes."""
        return [p.pattern for p in self.pam_specs]

    @property
    def n_pam_variants(self) -> int:
        return len(self.pam_specs)

    def get_activity(self, pam_variant: str) -> float:
        """Look up activity penalty for a PAM variant label.

        Falls back to lowest activity (0.2) for unknown PAMs.
        """
        for spec in self.pam_specs:
            if spec.label == pam_variant or spec.pattern == pam_variant:
                return spec.activity
        return 0.2  # unknown PAM — conservative penalty

    def is_canonical_pam(self, pam_variant: str) -> bool:
        """Check if a PAM variant is canonical (TTTV)."""
        return pam_variant == "TTTV"

    def to_dict(self) -> dict:
        return {
            "enzyme_id": self.enzyme_id,
            "display_name": self.display_name,
            "description": self.description,
            "pam_specs": [
                {"pattern": p.pattern, "activity": p.activity, "label": p.label}
                for p in self.pam_specs
            ],
            "seed_region": f"{self.seed_start}-{self.seed_end}",
            "spacer_lengths": list(self.spacer_lengths),
            "n_pam_variants": self.n_pam_variants,
            "source": self.source,
            "mutations": self.mutations,
        }


# ── Enzyme registry ──────────────────────────────────────────────────

WT_CAS12A = EnzymeConfig(
    enzyme_id="AsCas12a",
    display_name="WT AsCas12a",
    description="Wild-type Acidaminococcus sp. Cas12a. Strict TTTV PAM.",
    pam_specs=(
        PAMSpec("TTTV", 1.0, "TTTV"),
    ),
    seed_start=1,
    seed_end=8,
    scanner_key="AsCas12a",
    source="Zetsche et al., Cell 2015",
    mutations="",
)

EN_AS_CAS12A = EnzymeConfig(
    enzyme_id="enAsCas12a",
    display_name="enAsCas12a",
    description=(
        "Engineered AsCas12a (E174R/S542R/K548R) with expanded PAM "
        "recognition. ~5× more candidate sites in GC-rich genomes."
    ),
    pam_specs=(
        PAMSpec("TTTV", 1.00, "TTTV"),
        PAMSpec("TTTT", 0.75, "TTTT"),
        PAMSpec("TTCV", 0.65, "TTCV"),
        PAMSpec("TATV", 0.55, "TATV"),
        PAMSpec("CTTV", 0.45, "CTTV"),
        PAMSpec("TCTV", 0.40, "TCTV"),
        PAMSpec("TGTV", 0.35, "TGTV"),
        PAMSpec("ATTV", 0.30, "ATTV"),
        PAMSpec("GTTV", 0.25, "GTTV"),
    ),
    seed_start=1,
    seed_end=8,
    scanner_key="enAsCas12a",
    source="Kleinstiver et al., Nature Biotechnology 2019",
    mutations="E174R/S542R/K548R",
)

# Registry: enzyme_id → EnzymeConfig
ENZYME_REGISTRY: Dict[str, EnzymeConfig] = {
    WT_CAS12A.enzyme_id: WT_CAS12A,
    EN_AS_CAS12A.enzyme_id: EN_AS_CAS12A,
}

# Default enzyme for the platform
DEFAULT_ENZYME_ID = "enAsCas12a"


def get_enzyme(enzyme_id: str) -> EnzymeConfig:
    """Get enzyme config by ID. Raises KeyError if not found."""
    if enzyme_id not in ENZYME_REGISTRY:
        raise KeyError(
            f"Unknown enzyme '{enzyme_id}'. "
            f"Available: {list(ENZYME_REGISTRY.keys())}"
        )
    return ENZYME_REGISTRY[enzyme_id]


def list_enzymes() -> List[EnzymeConfig]:
    """Return all registered enzymes."""
    return list(ENZYME_REGISTRY.values())


def get_pam_penalty(enzyme_id: str, pam_variant: str) -> float:
    """Get multiplicative PAM activity penalty for scoring.

    Returns 1.0 for canonical TTTV (no penalty).
    Returns enzyme-specific activity for expanded PAMs.
    """
    enzyme = get_enzyme(enzyme_id)
    return enzyme.get_activity(pam_variant)
