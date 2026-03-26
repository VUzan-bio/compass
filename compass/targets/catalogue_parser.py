"""Parse organism-agnostic AMR mutation catalogues into Mutation objects.

Supports the standard COMPASS catalogue format (TSV/CSV):
  gene | mutation | drug | confidence | category | notes

Handles all mutation types:
  - AA substitution: S83L, D87N, A501V
  - Promoter: A-35del, c.-15C>T, P-42L
  - rRNA: C2611T, A2059G, A1401G
  - Gene presence: gene_presence (for acquired resistance genes)
  - Nucleotide SNP: c.1349C>T

This is the multi-species replacement for WHOCatalogueParser.
WHOCatalogueParser remains for backwards compatibility with existing
WHO TB catalogue files.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from compass.core.types import Drug, Mutation, MutationCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drug abbreviation → Drug enum (covers both full names and COMPASS codes)
# ---------------------------------------------------------------------------
_DRUG_MAP: dict[str, Drug] = {
    # --- Full names (case-insensitive lookup) ---
    "isoniazid": Drug.ISONIAZID,
    "rifampicin": Drug.RIFAMPICIN,
    "ethambutol": Drug.ETHAMBUTOL,
    "pyrazinamide": Drug.PYRAZINAMIDE,
    "levofloxacin": Drug.LEVOFLOXACIN,
    "moxifloxacin": Drug.MOXIFLOXACIN,
    "ciprofloxacin": Drug.CIPROFLOXACIN,
    "amikacin": Drug.AMIKACIN,
    "kanamycin": Drug.KANAMYCIN,
    "streptomycin": Drug.STREPTOMYCIN,
    "gentamicin": Drug.GENTAMICIN,
    "tobramycin": Drug.TOBRAMYCIN,
    "bedaquiline": Drug.BEDAQUILINE,
    "linezolid": Drug.LINEZOLID,
    "clofazimine": Drug.CLOFAZIMINE,
    "delamanid": Drug.DELAMANID,
    "pretomanid": Drug.PRETOMANID,
    "vancomycin": Drug.VANCOMYCIN,
    "teicoplanin": Drug.TEICOPLANIN,
    "methicillin": Drug.METHICILLIN,
    "oxacillin": Drug.OXACILLIN,
    "ampicillin": Drug.AMPICILLIN,
    "penicillin": Drug.PENICILLIN,
    "meropenem": Drug.MEROPENEM,
    "imipenem": Drug.IMIPENEM,
    "ceftriaxone": Drug.CEFTRIAXONE,
    "cefotaxime": Drug.CEFOTAXIME,
    "ceftazidime": Drug.CEFTAZIDIME,
    "colistin": Drug.COLISTIN,
    "azithromycin": Drug.AZITHROMYCIN,
    "clarithromycin": Drug.CLARITHROMYCIN,
    "erythromycin": Drug.ERYTHROMYCIN,
    "tetracycline": Drug.TETRACYCLINE,
    "tigecycline": Drug.TIGECYCLINE,
    "doxycycline": Drug.DOXYCYCLINE,
    "daptomycin": Drug.DAPTOMYCIN,
    "chloramphenicol": Drug.CHLORAMPHENICOL,
    "spectinomycin": Drug.SPECTINOMYCIN,
    "nitrofurantoin": Drug.NITROFURANTOIN,
    "fosfomycin": Drug.FOSFOMYCIN,
    "trimethoprim-sulfamethoxazole": Drug.TRIMETHOPRIM_SULFAMETHOXAZOLE,
    # --- Abbreviation codes (from Drug enum values) ---
    "inh": Drug.ISONIAZID,
    "rif": Drug.RIFAMPICIN,
    "emb": Drug.ETHAMBUTOL,
    "pza": Drug.PYRAZINAMIDE,
    "fq": Drug.FLUOROQUINOLONE,
    "lfx": Drug.LEVOFLOXACIN,
    "mfx": Drug.MOXIFLOXACIN,
    "cip": Drug.CIPROFLOXACIN,
    "amk": Drug.AMIKACIN,
    "kan": Drug.KANAMYCIN,
    "str": Drug.STREPTOMYCIN,
    "gen": Drug.GENTAMICIN,
    "tob": Drug.TOBRAMYCIN,
    "bdq": Drug.BEDAQUILINE,
    "lzd": Drug.LINEZOLID,
    "cfz": Drug.CLOFAZIMINE,
    "dlm": Drug.DELAMANID,
    "pmd": Drug.PRETOMANID,
    "van": Drug.VANCOMYCIN,
    "tec": Drug.TEICOPLANIN,
    "met": Drug.METHICILLIN,
    "oxa": Drug.OXACILLIN,
    "amp": Drug.AMPICILLIN,
    "pen": Drug.PENICILLIN,
    "mem": Drug.MEROPENEM,
    "ipm": Drug.IMIPENEM,
    "car": Drug.CARBAPENEM,
    "cro": Drug.CEFTRIAXONE,
    "ctx": Drug.CEFOTAXIME,
    "caz": Drug.CEFTAZIDIME,
    "cst": Drug.COLISTIN,
    "azm": Drug.AZITHROMYCIN,
    "clr": Drug.CLARITHROMYCIN,
    "ery": Drug.ERYTHROMYCIN,
    "tet": Drug.TETRACYCLINE,
    "tgc": Drug.TIGECYCLINE,
    "dox": Drug.DOXYCYCLINE,
    "dap": Drug.DAPTOMYCIN,
    "chl": Drug.CHLORAMPHENICOL,
    "spt": Drug.SPECTINOMYCIN,
    "nit": Drug.NITROFURANTOIN,
    "fos": Drug.FOSFOMYCIN,
    "sxt": Drug.TRIMETHOPRIM_SULFAMETHOXAZOLE,
    "ag": Drug.AMINOGLYCOSIDE,
    "cap": Drug.CAPREOMYCIN,
    "eth": Drug.ETHIONAMIDE,
    "pas": Drug.PAS,
    "cs": Drug.CYCLOSERINE,
    "mdr": Drug.MULTI_DRUG,
    "other": Drug.OTHER,
}

# ---------------------------------------------------------------------------
# Mutation notation patterns
# ---------------------------------------------------------------------------
_AA_RE = re.compile(r"^([A-Z])(\d+)([A-Z*])$")           # S83L, H481N
_PROMOTER_AA_RE = re.compile(r"^P?([A-Z]?)(-?\d+)([A-Z])$")  # P-42L or A-35del
_RRNA_RE = re.compile(r"^([ATGC])(\d+)([ATGC])$")        # C2611T, A2059G
_NT_CHANGE_RE = re.compile(r"c\.(-?\d+)([ATGC])>([ATGC])")  # c.-15C>T
_PROMOTER_DEL_RE = re.compile(r"^([A-Z]?)(-?\d+)del$")    # A-35del


def _resolve_drug(drug_str: str) -> Drug:
    """Map drug string to Drug enum, case-insensitive."""
    key = drug_str.strip().lower()
    return _DRUG_MAP.get(key, Drug.OTHER)


def _classify_category(cat_str: str) -> Optional[MutationCategory]:
    """Map category string from TSV to MutationCategory enum."""
    cat_map = {
        "aa_substitution": MutationCategory.AA_SUBSTITUTION,
        "nucleotide_snp": MutationCategory.NUCLEOTIDE_SNP,
        "insertion": MutationCategory.INSERTION,
        "deletion": MutationCategory.DELETION,
        "large_deletion": MutationCategory.LARGE_DELETION,
        "mnv": MutationCategory.MNV,
        "promoter": MutationCategory.PROMOTER,
        "rrna": MutationCategory.RRNA,
        "frameshift": MutationCategory.FRAMESHIFT,
        "intergenic": MutationCategory.INTERGENIC,
        "gene_presence": None,  # handled specially
    }
    return cat_map.get(cat_str.strip().lower())


class MutationCatalogueParser:
    """Parse organism-agnostic AMR mutation catalogues.

    Usage:
        parser = MutationCatalogueParser("data/mutation_catalogues/ecoli_esbl_amr.tsv")
        mutations = parser.parse()
        fq_mutations = parser.filter_by_drug(Drug.CIPROFLOXACIN)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._mutations: Optional[list[Mutation]] = None

    def parse(self) -> list[Mutation]:
        """Load and parse the full catalogue."""
        if self._mutations is not None:
            return self._mutations

        df = self._load_dataframe()
        mutations: list[Mutation] = []

        for _, row in df.iterrows():
            mut = self._parse_row(row)
            if mut is not None:
                mutations.append(mut)

        logger.info("Parsed %d mutations from %s", len(mutations), self.path.name)
        self._mutations = mutations
        return mutations

    def filter_by_drug(self, drug: Drug) -> list[Mutation]:
        """Return mutations associated with a specific drug."""
        return [m for m in self.parse() if m.drug == drug]

    def filter_by_gene(self, gene: str) -> list[Mutation]:
        """Return mutations in a specific gene."""
        return [m for m in self.parse() if m.gene == gene]

    def get_panel_mutations(self, panel: list[str]) -> list[Mutation]:
        """Get mutations matching a list of labels."""
        label_set = set(panel)
        return [m for m in self.parse() if m.label in label_set]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_dataframe(self) -> pd.DataFrame:
        suffix = self.path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            return pd.read_csv(self.path, sep=sep, comment="#")
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(self.path)
        raise ValueError(f"Unsupported file format: {suffix}")

    def _parse_row(self, row: pd.Series) -> Optional[Mutation]:
        """Parse a single catalogue row into a Mutation object."""
        gene = str(row.get("gene", row.get("Gene", ""))).strip()
        mutation_str = str(row.get("mutation", row.get("Mutation", ""))).strip()
        drug_str = str(row.get("drug", row.get("Drug", ""))).strip()
        cat_str = str(row.get("category", row.get("Category", ""))).strip()
        notes_str = str(row.get("notes", row.get("Notes", ""))).strip()
        confidence = str(row.get("confidence", row.get("Confidence", ""))).strip()

        if not gene:
            return None

        drug = _resolve_drug(drug_str)

        # --- Gene presence (acquired resistance gene detection) ---
        if mutation_str.lower() == "gene_presence":
            return Mutation(
                gene=gene,
                position=1,  # start of gene
                ref_aa="N",
                alt_aa="N",
                drug=drug,
                category=MutationCategory.LARGE_DELETION,  # reuse for gene-level
                who_confidence=confidence or "assoc w resistance",
                notes=f"Gene presence: {notes_str}" if notes_str else "Gene presence detection",
            )

        # --- Promoter deletion: A-35del ---
        m_del = _PROMOTER_DEL_RE.match(mutation_str)
        if m_del:
            ref_nt = m_del.group(1) or "N"
            pos = int(m_del.group(2))
            return Mutation(
                gene=gene,
                position=pos,
                ref_aa=ref_nt,
                alt_aa="-",
                drug=drug,
                category=MutationCategory.PROMOTER if pos < 0 else MutationCategory.DELETION,
                who_confidence=confidence or "assoc w resistance",
                notes=notes_str or None,
            )

        # --- Standard AA substitution: S83L, H481N ---
        m_aa = _AA_RE.match(mutation_str)
        if m_aa:
            ref_aa, pos, alt_aa = m_aa.group(1), int(m_aa.group(2)), m_aa.group(3)
            cat = _classify_category(cat_str) if cat_str else MutationCategory.AA_SUBSTITUTION
            # Promoter-like AA notation: P-42L
            if pos < 0:
                cat = MutationCategory.PROMOTER
            return Mutation(
                gene=gene,
                position=pos,
                ref_aa=ref_aa,
                alt_aa=alt_aa,
                drug=drug,
                category=cat,
                who_confidence=confidence or "assoc w resistance",
                notes=notes_str or None,
            )

        # --- rRNA mutation: C2611T, A1401G ---
        m_rrna = _RRNA_RE.match(mutation_str)
        if m_rrna and cat_str.lower() == "rrna":
            ref_nt, pos, alt_nt = m_rrna.group(1), int(m_rrna.group(2)), m_rrna.group(3)
            return Mutation(
                gene=gene,
                position=pos,
                ref_aa=ref_nt,
                alt_aa=alt_nt,
                drug=drug,
                category=MutationCategory.RRNA,
                who_confidence=confidence or "assoc w resistance",
                notes=notes_str or None,
            )

        # --- Nucleotide change: c.-15C>T, c.1349C>T ---
        m_nt = _NT_CHANGE_RE.match(mutation_str)
        if m_nt:
            pos = int(m_nt.group(1))
            ref_nt, alt_nt = m_nt.group(2), m_nt.group(3)
            cat = MutationCategory.PROMOTER if pos < 0 else MutationCategory.NUCLEOTIDE_SNP
            return Mutation(
                gene=gene,
                position=pos,
                ref_aa=ref_nt,
                alt_aa=alt_nt,
                nucleotide_change=mutation_str,
                drug=drug,
                category=cat,
                who_confidence=confidence or "assoc w resistance",
                notes=notes_str or None,
            )

        # --- Promoter AA-like: P-42L ---
        m_prom = _PROMOTER_AA_RE.match(mutation_str)
        if m_prom and int(m_prom.group(2)) < 0:
            ref_aa = m_prom.group(1) or "N"
            pos = int(m_prom.group(2))
            alt_aa = m_prom.group(3)
            return Mutation(
                gene=gene,
                position=pos,
                ref_aa=ref_aa,
                alt_aa=alt_aa,
                drug=drug,
                category=MutationCategory.PROMOTER,
                who_confidence=confidence or "assoc w resistance",
                notes=notes_str or None,
            )

        logger.debug("Skipping unparseable mutation: %s %s", gene, mutation_str)
        return None
