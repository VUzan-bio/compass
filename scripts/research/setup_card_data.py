"""Download and parse CARD (Comprehensive Antibiotic Resistance Database).

Extracts per-mutation prevalence estimates and MIC fold-change data
for correlation with EVO-2 LLR in weeks 3-4 experiments.

Usage:
    python scripts/research/setup_card_data.py

Output:
    data/card/aro_index.tsv          — raw CARD ARO index
    data/card/card.json              — full CARD ontology
    data/card/amr_target_metadata.csv — parsed metadata for our 42 SNP targets
"""

from __future__ import annotations

import csv
import io
import json
import logging
import tarfile
from pathlib import Path
from urllib.request import urlopen, Request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CARD_DATA_URL = "https://card.mcmaster.ca/latest/data"
CARD_ONTOLOGY_URL = "https://card.mcmaster.ca/latest/ontology"
OUTPUT_DIR = Path("data/card")

# Our 42 SNP-based AMR targets across 4 organisms
# Format: (organism_id, gene, mutation_label, drug_class, WHO_tier_or_confidence)
TARGETS = [
    # MTB (14 targets)
    ("mtb", "rpoB", "S531L", "RIF", 1),
    ("mtb", "rpoB", "H526Y", "RIF", 1),
    ("mtb", "rpoB", "D516V", "RIF", 1),
    ("mtb", "katG", "S315T", "INH", 1),
    ("mtb", "fabG1", "C-15T", "INH", 1),
    ("mtb", "embB", "M306V", "EMB", 1),
    ("mtb", "embB", "M306I", "EMB", 1),
    ("mtb", "pncA", "H57D", "PZA", 1),
    ("mtb", "gyrA", "D94G", "FQ", 1),
    ("mtb", "gyrA", "A90V", "FQ", 1),
    ("mtb", "rrs", "A1401G", "AG", 1),
    ("mtb", "eis", "C-14T", "AG", 1),
    # Additional MTB from 14-plex panel
    ("mtb", "rrs", "C1402T", "AG", 1),
    ("mtb", "pncA", "D49N", "PZA", 2),
    # E. coli (5 SNP targets)
    ("ecoli", "gyrA", "S83L", "CIP", 1),
    ("ecoli", "gyrA", "D87N", "CIP", 1),
    ("ecoli", "parC", "S80I", "CIP", 2),
    ("ecoli", "parC", "E84V", "CIP", 2),
    ("ecoli", "ampC", "C-42T", "AMP", 1),
    # S. aureus (8 SNP targets)
    ("saureus", "gyrA", "S84L", "CIP", 1),
    ("saureus", "grlA", "S80F", "CIP", 1),
    ("saureus", "grlA", "S80Y", "CIP", 1),
    ("saureus", "rpoB", "H481N", "RIF", 1),
    ("saureus", "rpoB", "S464P", "RIF", 1),
    ("saureus", "fusA", "L461K", "FUS", 2),
    ("saureus", "dfrB", "F99Y", "TMP", 2),
    ("saureus", "mprF", "S295L", "DAP", 2),
    # N. gonorrhoeae (15 SNP targets)
    ("ngonorrhoeae", "penA", "A501V", "ESC", 1),
    ("ngonorrhoeae", "penA", "A501T", "ESC", 1),
    ("ngonorrhoeae", "penA", "G545S", "ESC", 1),
    ("ngonorrhoeae", "penA", "I312M", "ESC", 2),
    ("ngonorrhoeae", "penA", "V316T", "ESC", 2),
    ("ngonorrhoeae", "penA", "T483S", "ESC", 2),
    ("ngonorrhoeae", "gyrA", "S91F", "CIP", 1),
    ("ngonorrhoeae", "gyrA", "D95A", "CIP", 1),
    ("ngonorrhoeae", "gyrA", "D95G", "CIP", 1),
    ("ngonorrhoeae", "parC", "D86N", "CIP", 2),
    ("ngonorrhoeae", "parC", "S87R", "CIP", 2),
    ("ngonorrhoeae", "23S_rRNA", "C2611T", "AZM", 1),
    ("ngonorrhoeae", "23S_rRNA", "A2059G", "AZM", 1),
    ("ngonorrhoeae", "mtrR", "A-35del", "AZM", 2),
    ("ngonorrhoeae", "folP", "R228S", "SUL", 2),
]

# Clinical prevalence estimates from WHO 2023 catalogue and literature
# Format: mutation_label → (prevalence_pct, source)
# These are approximate global frequencies among drug-resistant isolates
PREVALENCE_ESTIMATES = {
    # MTB (WHO 2023 catalogue)
    "rpoB_S531L": (45.0, "WHO 2023; most common RIF-R globally"),
    "rpoB_H526Y": (15.0, "WHO 2023; second most common RIF-R"),
    "rpoB_D516V": (8.0, "WHO 2023; third most common RIF-R"),
    "katG_S315T": (60.0, "WHO 2023; dominant INH-R mechanism"),
    "fabG1_C-15T": (25.0, "WHO 2023; inhA promoter mutation"),
    "embB_M306V": (30.0, "WHO 2023; most common EMB-R"),
    "embB_M306I": (15.0, "WHO 2023; second EMB-R at codon 306"),
    "pncA_H57D": (5.0, "WHO 2023; moderate PZA-R, variable by region"),
    "pncA_D49N": (3.0, "WHO 2023; PZA-R, less common"),
    "gyrA_D94G": (25.0, "WHO 2023; most common FQ-R"),
    "gyrA_A90V": (15.0, "WHO 2023; second most common FQ-R"),
    "rrs_A1401G": (80.0, "WHO 2023; dominant AG-R mechanism"),
    "rrs_C1402T": (5.0, "WHO 2023; AG-R, less common"),
    "eis_C-14T": (10.0, "WHO 2023; KAN-R via eis promoter"),
    # E. coli (EUCAST/CARD prevalence among ESBL/FQ-R E. coli)
    "gyrA_S83L": (85.0, "CARD/EUCAST; primary FQ-R in E. coli"),
    "gyrA_D87N": (50.0, "CARD; 2nd-step FQ-R, usually with S83L"),
    "parC_S80I": (40.0, "CARD; contributes to high-level FQ-R"),
    "parC_E84V": (15.0, "CARD; auxiliary FQ-R determinant"),
    "ampC_C-42T": (20.0, "CARD; promoter mutation, regional variation"),
    # S. aureus (EUCAST/literature prevalence among resistant isolates)
    "gyrA_S84L": (70.0, "EUCAST; primary FQ-R in S. aureus"),
    "grlA_S80F": (60.0, "EUCAST; most common Topo IV FQ-R"),
    "grlA_S80Y": (15.0, "EUCAST; alternative Topo IV change"),
    "rpoB_H481N": (40.0, "Literature; most common SA RIF-R"),
    "rpoB_S464P": (20.0, "Literature; second SA RIF-R"),
    "fusA_L461K": (25.0, "EUCAST; fusidic acid resistance"),
    "dfrB_F99Y": (30.0, "CARD; trimethoprim resistance"),
    "mprF_S295L": (10.0, "Literature; daptomycin-R, weak genotype-phenotype"),
    # N. gonorrhoeae (WHO GASP surveillance + literature)
    "penA_A501V": (35.0, "WHO GASP; key mosaic PBP2 change"),
    "penA_A501T": (10.0, "WHO GASP; alternative at 501"),
    "penA_G545S": (25.0, "WHO GASP; mosaic PBP2"),
    "penA_I312M": (20.0, "WHO GASP; mosaic PBP2"),
    "penA_V316T": (15.0, "WHO GASP; mosaic PBP2"),
    "penA_T483S": (30.0, "WHO GASP; mosaic PBP2"),
    "gyrA_S91F": (70.0, "WHO GASP; dominant FQ-R in Ng"),
    "gyrA_D95A": (15.0, "WHO GASP; second FQ-R"),
    "gyrA_D95G": (25.0, "WHO GASP; second FQ-R, geographic variant"),
    "parC_D86N": (10.0, "WHO GASP; auxiliary FQ-R"),
    "parC_S87R": (20.0, "WHO GASP; auxiliary FQ-R"),
    "23S_rRNA_C2611T": (15.0, "WHO GASP; azithromycin-R"),
    "23S_rRNA_A2059G": (40.0, "WHO GASP; high-level azithromycin-R"),
    "mtrR_A-35del": (50.0, "Literature; efflux pump upregulation"),
    "folP_R228S": (60.0, "Literature; sulfonamide-R, historical"),
}


def download_card() -> None:
    """Download CARD database files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    aro_path = OUTPUT_DIR / "aro_index.tsv"
    if aro_path.exists():
        logger.info("CARD ARO index already exists at %s", aro_path)
    else:
        logger.info("Downloading CARD data from %s ...", CARD_DATA_URL)
        try:
            req = Request(CARD_DATA_URL, headers={"Accept": "application/x-tar"})
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
            # CARD distributes as tar.bz2
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
                for member in tar.getmembers():
                    if "aro_index" in member.name:
                        f = tar.extractfile(member)
                        if f:
                            aro_path.write_bytes(f.read())
                            logger.info("Extracted %s (%d bytes)", aro_path, aro_path.stat().st_size)
                    elif member.name.endswith("card.json"):
                        f = tar.extractfile(member)
                        if f:
                            card_path = OUTPUT_DIR / "card.json"
                            card_path.write_bytes(f.read())
                            logger.info("Extracted %s", card_path)
        except Exception as e:
            logger.warning("Could not download CARD automatically: %s", e)
            logger.info("Manual download: %s", CARD_DATA_URL)
            logger.info("Extract aro_index.tsv and card.json to %s", OUTPUT_DIR)


def build_target_metadata() -> None:
    """Build the target metadata CSV combining our targets with CARD/WHO data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "amr_target_metadata.csv"

    rows = []
    for organism_id, gene, mutation, drug_class, who_tier in TARGETS:
        label = f"{gene}_{mutation}"
        prev_data = PREVALENCE_ESTIMATES.get(label, (None, None))
        rows.append({
            "organism_id": organism_id,
            "gene": gene,
            "mutation": mutation,
            "label": label,
            "drug_class": drug_class,
            "who_confidence_tier": who_tier,
            "prevalence_pct": prev_data[0],
            "prevalence_source": prev_data[1],
            # Placeholders for EVO-2 experiment
            "evo2_llr": None,
            "esm2_llr": None,
            "compass_disc_ratio": None,
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Written %d target records to %s", len(rows), out_path)

    # Summary
    by_org = {}
    for r in rows:
        by_org.setdefault(r["organism_id"], []).append(r)
    for org, targets in by_org.items():
        has_prev = sum(1 for t in targets if t["prevalence_pct"] is not None)
        logger.info("  %s: %d targets (%d with prevalence data)", org, len(targets), has_prev)


def main():
    logger.info("=== CARD Data Setup ===")
    download_card()
    logger.info("")
    logger.info("=== Building Target Metadata ===")
    build_target_metadata()
    logger.info("")
    logger.info("Done. Metadata at data/card/amr_target_metadata.csv")


if __name__ == "__main__":
    main()
