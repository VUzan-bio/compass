"""14-plex MDR-TB diagnostic panel definition.

Mutation selection based on:
  - WHO 2023 Catalogue of mutations in M. tuberculosis
  - Clinical frequency data from TBProfiler (Phelan et al., Genome Med 2019)
  - Drug resistance testing guidelines (WHO 2022)

Panel coverage:
  RIF:  rpoB S531L, H526Y, D516V          (~95% of RIF-R)
  INH:  katG S315T, fabG1 c.-15C>T         (~85% of INH-R)
  EMB:  embB M306V, M306I                  (~60% of EMB-R)
  PZA:  pncA H57D, D49N                    (~30% of PZA-R)
  FQ:   gyrA D94G, A90V                    (~70% of FQ-R)
  AG:   rrs A1401G, C1402T, eis c.-14C>T   (~85% of AG-R)
  MTB:  IS6110 (species ID control — added by pipeline)
"""

from compass.core.types import Drug, Mutation, MutationCategory


def define_mdr_panel() -> list[Mutation]:
    """Define the 14 WHO-critical resistance mutations."""
    return [
        # --- Rifampicin (RIF) ---
        Mutation(
            gene="rpoB", position=531, ref_aa="S", alt_aa="L",
            drug=Drug.RIFAMPICIN, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.42,
            notes="Most common RIF mutation (40-70%). PAM desert — proximity detection.",
        ),
        Mutation(
            gene="rpoB", position=526, ref_aa="H", alt_aa="Y",
            drug=Drug.RIFAMPICIN, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.15,
            notes="Second most common RIF mutation (10-20%).",
        ),
        Mutation(
            gene="rpoB", position=516, ref_aa="D", alt_aa="V",
            drug=Drug.RIFAMPICIN, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.08,
        ),
        # --- Isoniazid (INH) ---
        Mutation(
            gene="katG", position=315, ref_aa="S", alt_aa="T",
            drug=Drug.ISONIAZID, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.65,
            notes="Most common INH mutation (50-90%). katG = Rv1908c (minus strand).",
        ),
        Mutation(
            gene="fabG1", position=-15, ref_aa="C", alt_aa="T",
            nucleotide_change="c.-15C>T",
            drug=Drug.ISONIAZID, who_confidence="assoc w resistance",
            category=MutationCategory.PROMOTER,
            clinical_frequency=0.25,
            notes="fabG1 (mabA/Rv1483) promoter. Upregulates InhA.",
        ),
        # --- Ethambutol (EMB) ---
        Mutation(
            gene="embB", position=306, ref_aa="M", alt_aa="V",
            drug=Drug.ETHAMBUTOL, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.30,
        ),
        Mutation(
            gene="embB", position=306, ref_aa="M", alt_aa="I",
            drug=Drug.ETHAMBUTOL, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.25,
        ),
        # --- Pyrazinamide (PZA) ---
        Mutation(
            gene="pncA", position=57, ref_aa="H", alt_aa="D",
            drug=Drug.PYRAZINAMIDE, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.08,
            notes="pncA = Rv2043c (minus strand).",
        ),
        Mutation(
            gene="pncA", position=49, ref_aa="D", alt_aa="N",
            drug=Drug.PYRAZINAMIDE, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.05,
        ),
        # --- Fluoroquinolones (FQ) ---
        Mutation(
            gene="gyrA", position=94, ref_aa="D", alt_aa="G",
            drug=Drug.FLUOROQUINOLONE, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.35,
            notes="Most common FQ mutation. QRDR position 94.",
        ),
        Mutation(
            gene="gyrA", position=90, ref_aa="A", alt_aa="V",
            drug=Drug.FLUOROQUINOLONE, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.20,
        ),
        # --- Aminoglycosides (AG) ---
        Mutation(
            gene="rrs", position=1401, ref_aa="A", alt_aa="G",
            drug=Drug.AMINOGLYCOSIDE, who_confidence="assoc w resistance",
            category=MutationCategory.RRNA,
            clinical_frequency=0.60,
            notes="rrs A1401G — most common AG mutation (AMK, KAN, CAP). rRNA gene.",
        ),
        Mutation(
            gene="rrs", position=1402, ref_aa="C", alt_aa="T",
            drug=Drug.AMINOGLYCOSIDE, who_confidence="assoc w resistance",
            category=MutationCategory.RRNA,
            clinical_frequency=0.05,
        ),
        Mutation(
            gene="eis", position=-14, ref_aa="C", alt_aa="T",
            nucleotide_change="c.-14C>T",
            drug=Drug.AMINOGLYCOSIDE, who_confidence="assoc w resistance",
            category=MutationCategory.PROMOTER,
            clinical_frequency=0.10,
            notes="eis promoter — KAN/AMK low-level resistance.",
        ),
    ]


def define_mdr_rnasep_panel() -> list[Mutation]:
    """Define the 14-plex MDR-TB panel + RNaseP extraction control.

    RNaseP (RPPH1) is a human housekeeping gene used as an extraction
    and sample adequacy control. Its presence confirms:
      1. DNA extraction succeeded (not inhibited)
      2. Sufficient human material in the specimen (sputum quality)

    This is the standard control recommended by CDC for nucleic acid
    amplification-based TB diagnostics. The pipeline adds IS6110 as
    the MTB species ID control; RNaseP serves as the human sample
    adequacy control — together they gate the diagnostic call.

    The RNaseP target is pre-designed (no mutation, no discrimination)
    and is flagged as a control in the panel output.
    """
    return define_mdr_panel()  # RNaseP control is added by the pipeline runner


# Pre-designed RNaseP control entry for pipeline injection
RNASEP_CONTROL = {
    "gene": "RPPH1",
    "label": "RNaseP_control",
    "drug": "OTHER",
    "detection_strategy": "direct",
    "is_control": True,
    "control_type": "extraction",
    "spacer_seq": "GCGCGAGCGCATGCCTGCAG",
    "pam_seq": "TTTG",
    "organism": "human",
    "notes": "Human extraction/sample adequacy control (CDC standard)",
}


# Drug coverage summary for documentation
DRUG_COVERAGE = {
    "RIF": {"targets": ["rpoB_S531L", "rpoB_H526Y", "rpoB_D516V"], "coverage": "~95%"},
    "INH": {"targets": ["katG_S315T", "fabG1_C-15T"], "coverage": "~85%"},
    "EMB": {"targets": ["embB_M306V", "embB_M306I"], "coverage": "~60%"},
    "PZA": {"targets": ["pncA_H57D", "pncA_D49N"], "coverage": "~30%"},
    "FQ":  {"targets": ["gyrA_D94G", "gyrA_A90V"], "coverage": "~70%"},
    "AG":  {"targets": ["rrs_A1401G", "rrs_C1402T", "eis_C-14T"], "coverage": "~85%"},
}
