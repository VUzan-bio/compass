"""
NucleaseProfile loader.

Loads variant-specific parameters from JSON config files.
Pipeline modules (M2, M3) read PAM sets and seed lengths from the
active profile instead of hardcoded values.

Usage:
    profile = NucleaseProfile.load("enAsCas12a")
    pams = profile.get_all_pams()  # canonical + expanded
    seed = profile.seed_positions()  # [1,2,3,4,5,6,7]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

PROFILES_DIR = Path(__file__).parent / "profiles"


class NucleaseProfile:
    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(cls, variant_id: str) -> "NucleaseProfile":
        path = PROFILES_DIR / f"{variant_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No profile for {variant_id}. "
                f"Available: {cls.available()}"
            )
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def available(cls) -> List[str]:
        return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))

    @classmethod
    def load_all(cls) -> Dict[str, "NucleaseProfile"]:
        return {vid: cls.load(vid) for vid in cls.available()}

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def display_name(self) -> str:
        return self._data["display_name"]

    @property
    def organism(self) -> str:
        return self._data.get("organism", "")

    @property
    def mutations(self) -> Optional[List[str]]:
        return self._data.get("mutations")

    @property
    def scanner_variant(self) -> Optional[str]:
        """Maps to compass.candidates.scanner.CONFIGS key, if available."""
        return self._data.get("scanner_variant")

    @property
    def scoring_trained(self) -> bool:
        return self._data.get("scoring_trained", False)

    def get_canonical_pams(self) -> List[str]:
        return self._data["pam"]["canonical"]

    def get_all_pams(self) -> List[str]:
        """All PAMs the variant can use (canonical + expanded)."""
        pams = list(self._data["pam"]["canonical"])
        for key in ["tier1_expanded", "tier2_expanded", "non_canonical"]:
            pams.extend(self._data["pam"].get(key, []))
        return pams

    @property
    def pam_note(self) -> str:
        return self._data["pam"].get("note", "")

    def seed_positions(self) -> List[int]:
        return self._data["seed"]["critical_positions"]

    def tolerant_positions(self) -> List[int]:
        return self._data["seed"]["tolerant_positions"]

    @property
    def seed_note(self) -> str:
        return self._data["seed"].get("note", "")

    @property
    def kinetics(self) -> dict:
        return self._data.get("kinetics", {})

    @property
    def optimal_temperature(self) -> int:
        return self._data["temperature"]["optimal_C"]

    @property
    def temperature(self) -> dict:
        return self._data.get("temperature", {})

    @property
    def divalent_cation(self) -> str:
        return self._data["divalent_cation"]

    @property
    def snv_discrimination(self) -> str:
        return str(self._data.get("snv_discrimination", "unknown"))

    @property
    def spacer_length(self) -> dict:
        return self._data.get("spacer_length", {})

    @property
    def references(self) -> List[dict]:
        return self._data.get("references", [])

    def to_summary(self) -> dict:
        """Summary for API/frontend display."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "organism": self.organism,
            "mutations": self.mutations,
            "pam_canonical": self.get_canonical_pams(),
            "pam_all": self.get_all_pams(),
            "pam_total_count": len(self.get_all_pams()),
            "pam_note": self.pam_note,
            "seed_positions": self.seed_positions(),
            "tolerant_positions": self.tolerant_positions(),
            "seed_note": self.seed_note,
            "kinetics": self.kinetics,
            "optimal_temp": self.optimal_temperature,
            "temperature": self.temperature,
            "divalent_cation": self.divalent_cation,
            "snv_discrimination": self.snv_discrimination,
            "spacer_length": self.spacer_length,
            "scanner_variant": self.scanner_variant,
            "scoring_trained": self.scoring_trained,
            "references": self.references,
        }
