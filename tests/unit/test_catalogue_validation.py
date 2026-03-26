"""Validate AMR mutation catalogues against expected reference coordinates.

This test catches:
  - Off-by-one errors in codon numbering (especially penA Ambler vs gene-local)
  - Silently dropped rows (parser couldn't parse a mutation notation)
  - Wildtype AA mismatches (catalogue says ref=S but genome annotation says A)
  - Drug mapping failures (unknown abbreviation → Drug.OTHER when it shouldn't be)
  - Gene-presence entries that lack valid gene names
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compass.core.types import Drug, MutationCategory
from compass.targets.catalogue_parser import MutationCatalogueParser

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "mutation_catalogues"

# ---------------------------------------------------------------------------
# Expected mutation counts per catalogue (must match TSV row count minus header
# and comment lines).  If you add/remove rows, update these numbers.
# ---------------------------------------------------------------------------
EXPECTED_COUNTS = {
    "ecoli_esbl_amr.tsv": 14,
    "saureus_mrsa_amr.tsv": 15,
    "ngonorrhoeae_amr.tsv": 16,
}

# ---------------------------------------------------------------------------
# Ground-truth wildtype residues.  Each entry is (gene, position, expected_ref_aa).
# These are manually verified against the reference genome annotation and
# published literature.  If the parser produces a different ref_aa for any
# of these, the catalogue has a coordinate error.
# ---------------------------------------------------------------------------
WILDTYPE_TRUTH = {
    "ecoli_esbl_amr.tsv": [
        ("gyrA", 83, "S"),    # E. coli K-12 gyrA codon 83 = Ser (Yoshida 1990)
        ("gyrA", 87, "D"),    # E. coli K-12 gyrA codon 87 = Asp
        ("parC", 80, "S"),    # E. coli K-12 parC codon 80 = Ser
        ("parC", 84, "E"),    # E. coli K-12 parC codon 84 = Glu
    ],
    "saureus_mrsa_amr.tsv": [
        ("gyrA", 84, "S"),    # S. aureus NCTC 8325 gyrA codon 84 = Ser (Ferrero 1994)
        ("grlA", 80, "S"),    # S. aureus grlA (parC) codon 80 = Ser
        ("rpoB", 481, "H"),   # S. aureus rpoB codon 481 = His (≡ E. coli 526)
        ("rpoB", 464, "S"),   # S. aureus rpoB codon 464 = Ser
        ("fusA", 461, "L"),   # S. aureus fusA codon 461 = Leu (EF-G domain III)
        ("dfrB", 99, "F"),    # S. aureus dfrB codon 99 = Phe
        ("mprF", 295, "S"),   # S. aureus mprF codon 295 = Ser
    ],
    "ngonorrhoeae_amr.tsv": [
        # penA: Ambler PBP2 numbering (Unemo et al. 2019; WHO gonococcal AMR 2023)
        ("penA", 501, "A"),   # N. gonorrhoeae FA 1090 penA Ambler 501 = Ala
        ("penA", 545, "G"),   # Ambler 545 = Gly
        ("penA", 312, "I"),   # Ambler 312 = Ile
        ("penA", 316, "V"),   # Ambler 316 = Val
        ("penA", 483, "T"),   # Ambler 483 = Thr
        # gyrA / parC
        ("gyrA", 91, "S"),    # N. gonorrhoeae FA 1090 gyrA codon 91 = Ser
        ("gyrA", 95, "D"),    # gyrA codon 95 = Asp
        ("parC", 86, "D"),    # parC codon 86 = Asp
        ("parC", 87, "S"),    # parC codon 87 = Ser
        # rRNA (ref is nucleotide, not AA)
        ("23S rRNA", 2611, "C"),  # 23S rRNA position 2611 = C
        ("23S rRNA", 2059, "A"),  # 23S rRNA position 2059 = A
        # Sulfonamide
        ("folP", 228, "R"),   # folP codon 228 = Arg
    ],
}

# ---------------------------------------------------------------------------
# Gene-presence entries: expected gene names per catalogue
# ---------------------------------------------------------------------------
GENE_PRESENCE_EXPECTED = {
    "ecoli_esbl_amr.tsv": [
        "blaCTX-M-15", "blaCTX-M-14", "blaCTX-M-27",
        "blaNDM-1", "blaNDM-5", "blaKPC-2", "blaKPC-3", "blaOXA-48",
        "mcr-1",
    ],
    "saureus_mrsa_amr.tsv": [
        "mecA", "mecC", "blaZ", "vanA", "ermA", "ermC", "cfr",
    ],
    "ngonorrhoeae_amr.tsv": [
        "tetM",
    ],
}

# ---------------------------------------------------------------------------
# Drug resolution: no entry should silently map to Drug.OTHER unless expected
# ---------------------------------------------------------------------------
EXPECTED_OTHER_DRUGS: dict[str, set[str]] = {
    "ecoli_esbl_amr.tsv": set(),
    "saureus_mrsa_amr.tsv": {"fusA"},     # fusidic acid → OTHER is acceptable
    "ngonorrhoeae_amr.tsv": set(),
}


# ===========================================================================
# Tests
# ===========================================================================


class TestCatalogueCompleteness:
    """Verify no rows are silently dropped by the parser."""

    @pytest.mark.parametrize("filename,expected", EXPECTED_COUNTS.items())
    def test_parse_count(self, filename: str, expected: int):
        path = DATA_DIR / filename
        if not path.exists():
            pytest.skip(f"Catalogue file not found: {path}")
        parser = MutationCatalogueParser(path)
        mutations = parser.parse()
        assert len(mutations) == expected, (
            f"{filename}: expected {expected} mutations, got {len(mutations)}. "
            f"Parser may be silently dropping rows."
        )


class TestWildtypeCoordinates:
    """Cross-check that ref_aa in each catalogue entry matches the known
    wildtype residue at that position in the reference genome.

    This is the primary guard against numbering convention errors.
    """

    @pytest.mark.parametrize("filename,checks", WILDTYPE_TRUTH.items())
    def test_ref_aa_matches(self, filename: str, checks: list):
        path = DATA_DIR / filename
        if not path.exists():
            pytest.skip(f"Catalogue file not found: {path}")
        parser = MutationCatalogueParser(path)
        mutations = parser.parse()

        # Build lookup: (gene, position) → ref_aa
        lookup: dict[tuple[str, int], str] = {}
        for m in mutations:
            key = (m.gene, m.position)
            if key not in lookup:
                lookup[key] = m.ref_aa

        for gene, pos, expected_ref in checks:
            key = (gene, pos)
            assert key in lookup, (
                f"{filename}: expected mutation at {gene} position {pos} "
                f"but no entry found. Available {gene} positions: "
                f"{[p for g, p in lookup if g == gene]}"
            )
            actual_ref = lookup[key]
            assert actual_ref == expected_ref, (
                f"{filename}: {gene} position {pos}: catalogue says ref='{actual_ref}' "
                f"but reference genome has '{expected_ref}'. "
                f"Check numbering convention (Ambler vs gene-local vs E. coli equivalent)."
            )


class TestGenePresenceEntries:
    """Verify gene-presence entries are well-formed."""

    @pytest.mark.parametrize("filename,expected_genes", GENE_PRESENCE_EXPECTED.items())
    def test_gene_presence_genes(self, filename: str, expected_genes: list):
        path = DATA_DIR / filename
        if not path.exists():
            pytest.skip(f"Catalogue file not found: {path}")
        parser = MutationCatalogueParser(path)
        mutations = parser.parse()

        # Gene-presence mutations use position=1, ref_aa="N", alt_aa="N"
        gp_genes = [m.gene for m in mutations if m.ref_aa == "N" and m.alt_aa == "N" and m.position == 1]
        for gene in expected_genes:
            assert gene in gp_genes, (
                f"{filename}: expected gene_presence entry for '{gene}' not found. "
                f"Available gene_presence entries: {gp_genes}"
            )


class TestDrugMapping:
    """Verify drug abbreviations resolve correctly (no silent Drug.OTHER)."""

    @pytest.mark.parametrize("filename", EXPECTED_COUNTS.keys())
    def test_no_unexpected_other(self, filename: str):
        path = DATA_DIR / filename
        if not path.exists():
            pytest.skip(f"Catalogue file not found: {path}")
        parser = MutationCatalogueParser(path)
        mutations = parser.parse()

        allowed_other = EXPECTED_OTHER_DRUGS.get(filename, set())
        bad = []
        for m in mutations:
            if m.drug == Drug.OTHER and m.gene not in allowed_other:
                bad.append(f"{m.gene} ({m.notes or 'no notes'})")
        assert not bad, (
            f"{filename}: these mutations unexpectedly mapped to Drug.OTHER: {bad}. "
            f"Add the drug abbreviation to _DRUG_MAP in catalogue_parser.py."
        )


class TestPenACombinatorialConsistency:
    """Verify penA mosaic entries use consistent Ambler numbering."""

    def test_pena_positions_in_expected_range(self):
        """All penA positions should fall in PBP2 transpeptidase domain (300-600 Ambler)."""
        path = DATA_DIR / "ngonorrhoeae_amr.tsv"
        if not path.exists():
            pytest.skip("N. gonorrhoeae catalogue not found")
        parser = MutationCatalogueParser(path)
        mutations = parser.parse()

        pena = [m for m in mutations if m.gene == "penA"]
        assert len(pena) == 6, f"Expected 6 penA mosaic entries, got {len(pena)}"

        for m in pena:
            assert 200 <= m.position <= 700, (
                f"penA position {m.position} out of Ambler PBP2 range (200-700). "
                f"Check numbering convention."
            )

    def test_pena_no_duplicate_positions_same_alt(self):
        """No duplicate (position, alt_aa) pairs in penA."""
        path = DATA_DIR / "ngonorrhoeae_amr.tsv"
        if not path.exists():
            pytest.skip("N. gonorrhoeae catalogue not found")
        parser = MutationCatalogueParser(path)
        mutations = parser.parse()

        pena = [m for m in mutations if m.gene == "penA"]
        seen = set()
        for m in pena:
            key = (m.position, m.alt_aa)
            assert key not in seen, (
                f"Duplicate penA entry: position={m.position}, alt={m.alt_aa}"
            )
            seen.add(key)


class TestCatalogueCategories:
    """Verify mutation categories are correctly assigned."""

    @pytest.mark.parametrize("filename", EXPECTED_COUNTS.keys())
    def test_all_have_category(self, filename: str):
        path = DATA_DIR / filename
        if not path.exists():
            pytest.skip(f"Catalogue file not found: {path}")
        parser = MutationCatalogueParser(path)
        mutations = parser.parse()

        for m in mutations:
            assert m.category is not None, (
                f"{filename}: {m.gene} {m.ref_aa}{m.position}{m.alt_aa} has no category"
            )
