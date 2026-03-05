"""Tests for Block 3: Sensitivity-Specificity Optimization Framework.

Comprehensive tests with realistic MDR-TB 14-plex panel mock data.
Each target uses labels matching TARGET_DRUG_CLASS and realistic
score/discrimination distributions based on gene characteristics.

Drug classes tested:
    rifampicin (5 targets): rpoB RRDR, PAM desert, high disc
    isoniazid (3 targets): katG codon 315 + inhA promoter
    ethambutol (2 targets): embB M306V/I, GC-rich region
    pyrazinamide (1 target): pncA H57D, short gene
    fluoroquinolone (2 targets): gyrA D94G/A90V
    aminoglycoside (1 target): rrs A1401G, rRNA
    species_control (1): IS6110
"""

from __future__ import annotations

import pytest

from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    DiscriminationScore,
    HeuristicScore,
    Mutation,
    OffTargetReport,
    PanelMember,
    PAMVariant,
    RPAPrimer,
    RPAPrimerPair,
    ScoredCandidate,
    Strand,
    Target,
)
from compass.optimisation.metrics import (
    DiagnosticMetrics,
    TargetMetrics,
    DrugClassMetrics,
    TARGET_DRUG_CLASS,
    WHO_TPP_SENSITIVITY,
    compute_diagnostic_metrics,
)
from compass.optimisation.profiles import (
    ParameterProfile,
    get_preset,
    list_presets,
    HIGH_SENSITIVITY,
    BALANCED,
    HIGH_SPECIFICITY,
)
from compass.optimisation.sweep import sweep_parameter
from compass.optimisation.pareto import pareto_frontier, generate_profile_grid
from compass.optimisation.top_k import collect_top_k, TargetCandidateSet


# =====================================================================
# Mock data factory: realistic MDR-TB 14-plex panel
# =====================================================================

# Realistic per-target properties based on gene biology
_TARGET_PROPS = {
    # label: (gene, pos, ref, alt, drug, n_cands, best_score, best_disc,
    #         strategy, has_primers, offtargets)
    "rpoB_S450L": ("rpoB", 450, "S", "L", "RIF", 8, 0.52, 6.5, "proximity", True, 1),
    "rpoB_S450W": ("rpoB", 450, "S", "W", "RIF", 6, 0.45, 5.8, "proximity", True, 2),
    "rpoB_H445D": ("rpoB", 445, "H", "D", "RIF", 10, 0.68, 8.2, "direct", True, 0),
    "rpoB_H445Y": ("rpoB", 445, "H", "Y", "RIF", 9, 0.65, 7.9, "direct", True, 0),
    "rpoB_D435V": ("rpoB", 435, "D", "V", "RIF", 7, 0.58, 6.1, "direct", True, 1),
    "katG_S315T": ("katG", 315, "S", "T", "INH", 12, 0.72, 5.3, "direct", True, 0),
    "katG_S315N": ("katG", 315, "S", "N", "INH", 11, 0.69, 4.8, "direct", True, 0),
    "inhA_C-15T": ("inhA", -15, "C", "T", "INH", 8, 0.61, 4.2, "direct", True, 1),
    "embB_M306V": ("embB", 306, "M", "V", "EMB", 5, 0.48, 3.1, "direct", True, 3),
    "embB_M306I": ("embB", 306, "M", "I", "EMB", 4, 0.42, 2.8, "direct", False, 4),
    "pncA_H57D": ("pncA", 57, "H", "D", "PZA", 3, 0.35, 2.2, "direct", False, 2),
    "gyrA_D94G": ("gyrA", 94, "D", "G", "FQ", 9, 0.63, 5.6, "direct", True, 0),
    "gyrA_A90V": ("gyrA", 90, "A", "V", "FQ", 8, 0.59, 4.9, "direct", True, 1),
    "rrs_A1401G": ("rrs", 1401, "A", "G", "AG", 6, 0.55, 7.1, "direct", True, 0),
}

# IS6110 species control
# IS6110 has Mutation.label = "IS6110_A1G", so use that as key.
# Also add this to TARGET_DRUG_CLASS at runtime for tests.
_IS6110_PROPS = {
    "IS6110_A1G": ("IS6110", 1, "A", "G", "CTRL", 15, 0.85, 20.0, "direct", True, 0),
}


def _make_mutation(gene: str, pos: int, ref: str, alt: str) -> Mutation:
    from compass.core.types import Drug
    return Mutation(gene=gene, position=pos, ref_aa=ref, alt_aa=alt, drug=Drug.OTHER)


def _make_target(gene: str, pos: int, ref: str, alt: str) -> Target:
    # Map amino acid refs to valid nucleotide codons for the validator.
    # rRNA/promoter targets use single nucleotide refs (A, C, G, T).
    nt_bases = {"A", "T", "G", "C"}
    if ref in nt_bases and alt in nt_bases:
        ref_codon, alt_codon = ref, alt
    else:
        ref_codon, alt_codon = "TCG", "TTG"
    return Target(
        mutation=_make_mutation(gene, pos, ref, alt),
        genomic_pos=761155 + abs(pos) * 3,
        ref_codon=ref_codon,
        alt_codon=alt_codon,
        flanking_seq="A" * 100,
        flanking_start=761055 + abs(pos) * 3,
    )


def _make_primer_pair() -> RPAPrimerPair:
    return RPAPrimerPair(
        fwd=RPAPrimer(seq="A" * 30, tm=62.0, direction="fwd",
                       amplicon_start=761100, amplicon_end=761300),
        rev=RPAPrimer(seq="T" * 30, tm=63.0, direction="rev",
                       amplicon_start=761100, amplicon_end=761300),
    )


def _make_candidate(
    label: str,
    idx: int,
    efficiency: float,
    disc_ratio: float,
    strategy: str = "direct",
    n_offtargets: int = 0,
) -> ScoredCandidate:
    spacer_list = list("AGCTAGCTAGCTAGCTAGCT")
    spacer_list[idx % 20] = "ACGT"[idx % 4]
    spacer = "".join(spacer_list)

    det_strategy = (
        DetectionStrategy.PROXIMITY if strategy == "proximity"
        else DetectionStrategy.DIRECT
    )

    candidate = CrRNACandidate(
        candidate_id=f"{label}_cand_{idx}",
        target_label=label,
        spacer_seq=spacer,
        pam_seq="TTTG",
        pam_variant=PAMVariant.TTTV,
        strand=Strand.PLUS,
        genomic_start=761155 + idx * 5,
        genomic_end=761155 + idx * 5 + 20,
        gc_content=0.5,
        homopolymer_max=2,
        pam_activity_weight=0.9,
        detection_strategy=det_strategy,
        mutation_position_in_spacer=5 if det_strategy == DetectionStrategy.DIRECT else None,
    )

    heuristic = HeuristicScore(
        seed_position_score=0.5,
        gc_penalty=0.0,
        structure_penalty=0.0,
        homopolymer_penalty=0.0,
        offtarget_penalty=0.0,
        composite=efficiency,
    )

    disc = DiscriminationScore(
        wt_activity=0.3,
        mut_activity=0.3 * disc_ratio,
        model_name="mock",
    )

    offtarget = OffTargetReport(
        candidate_id=f"{label}_cand_{idx}",
        is_clean=n_offtargets <= 3,
    )

    return ScoredCandidate(
        candidate=candidate,
        offtarget=offtarget,
        heuristic=heuristic,
        discrimination=disc,
    )


def _make_mock_panel(
    include_species_control: bool = True,
    all_have_primers: bool = False,
) -> tuple[list[PanelMember], dict[str, list[ScoredCandidate]]]:
    """Generate a realistic mock 14-plex panel.

    Returns (panel_members, candidates_by_target).

    Score/disc distributions:
    - rpoB RRDR: PAM desert for S450L/W (proximity), direct for H445/D435
    - katG: well-characterised, high scores
    - inhA promoter: good PAM, decent disc
    - embB: GC-rich, M306I missing primers
    - pncA: short gene, low score/disc, no primers
    - gyrA: good targets
    - rrs: rRNA, conserved, good disc
    """
    props = dict(_TARGET_PROPS)
    if include_species_control:
        props.update(_IS6110_PROPS)

    members = []
    candidates_by_target = {}

    for label, (gene, pos, ref, alt, _drug, n_cands, best_score, best_disc,
                strategy, has_primers_flag, best_ot) in props.items():

        target = _make_target(gene, pos, ref, alt)
        assert target.label == label, f"Label mismatch: {target.label} != {label}"

        # Generate candidates with varying quality
        candidates = []
        for j in range(n_cands):
            # Best candidate at j=0, decreasing quality
            frac = j / max(n_cands - 1, 1)
            eff = best_score * (1.0 - 0.4 * frac)   # 60-100% of best
            disc = best_disc * (1.0 - 0.3 * frac)    # 70-100% of best
            ot = best_ot + j // 3
            candidates.append(
                _make_candidate(label, j, eff, disc, strategy, ot)
            )

        candidates_by_target[label] = candidates

        primers = _make_primer_pair() if (has_primers_flag or all_have_primers) else None

        members.append(PanelMember(
            target=target,
            selected_candidate=candidates[0],  # best candidate
            primers=primers,
        ))

    return members, candidates_by_target


# =====================================================================
# Test 1: DiagnosticMetrics with realistic mock panel
# =====================================================================

class TestDiagnosticMetrics:

    def test_realistic_panel_metrics(self):
        """14 targets + 1 control with realistic drug-class breakdown."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(
            members, cands,
            efficiency_threshold=0.4,
            discrimination_threshold=3.0,
        )
        # Should have 15 target_metrics (14 resistance + 1 control)
        assert len(metrics.target_metrics) == 15
        # Drug class metrics should exclude species_control
        assert all(d.drug_class != "species_control" for d in metrics.drug_class_metrics)
        # Should have 6 drug classes
        assert len(metrics.drug_class_metrics) == 6
        # Sensitivity between 0.5 and 1.0 (some targets below threshold)
        assert 0.4 <= metrics.sensitivity <= 1.0
        # Specificity should be > 0 (at least some targets assay-ready)
        assert metrics.specificity > 0.0

    def test_sensitivity_excludes_species_control(self):
        """IS6110 should not count toward panel sensitivity."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.3, 2.0)
        # Count resistance targets only
        resistance = [t for t in metrics.target_metrics if t.drug_class != "species_control"]
        assert len(resistance) == 14
        ready_count = sum(1 for t in resistance if t.is_assay_ready)
        expected_sens = ready_count / 14
        assert abs(metrics.sensitivity - expected_sens) < 1e-9

    def test_specificity_proxy_formula(self):
        """Specificity = mean(1 - 1/disc) across assay-ready targets."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.3, 2.0)
        ready = [
            t for t in metrics.target_metrics
            if t.is_assay_ready and t.drug_class != "species_control"
        ]
        if ready:
            import numpy as np
            expected = float(np.mean([1.0 - 1.0 / max(t.best_disc, 1.01) for t in ready]))
            assert abs(metrics.specificity - expected) < 1e-9

    def test_lower_threshold_higher_sensitivity(self):
        """Lowering thresholds should increase or maintain sensitivity."""
        members, cands = _make_mock_panel()
        metrics_low = compute_diagnostic_metrics(members, cands, 0.2, 1.5)
        metrics_high = compute_diagnostic_metrics(members, cands, 0.6, 5.0)
        assert metrics_low.sensitivity >= metrics_high.sensitivity

    def test_drug_class_breakdown(self):
        """Per-drug-class metrics are computed correctly."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.4, 3.0)

        class_names = {d.drug_class for d in metrics.drug_class_metrics}
        expected = {"rifampicin", "isoniazid", "ethambutol", "pyrazinamide",
                    "fluoroquinolone", "aminoglycoside"}
        assert class_names == expected

        # Rifampicin should have 5 targets
        rif = next(d for d in metrics.drug_class_metrics if d.drug_class == "rifampicin")
        assert rif.n_targets == 5

        # Isoniazid should have 3 targets
        inh = next(d for d in metrics.drug_class_metrics if d.drug_class == "isoniazid")
        assert inh.n_targets == 3

    def test_assay_ready_requires_primers(self):
        """is_assay_ready should be False if primers are missing even if covered."""
        members, cands = _make_mock_panel()
        # pncA_H57D: score=0.35, disc=2.2, no primers
        metrics = compute_diagnostic_metrics(members, cands, 0.3, 2.0)
        pnca = next(t for t in metrics.target_metrics if t.label == "pncA_H57D")
        assert pnca.is_covered  # 0.35 >= 0.3 and 2.2 >= 2.0
        assert not pnca.has_primers
        assert not pnca.is_assay_ready

    def test_to_dict_summary_structure(self):
        """summary() returns expected keys and structure."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.4, 3.0)
        s = metrics.summary()
        assert "panel_sensitivity" in s
        assert "panel_specificity" in s
        assert "drug_class_coverage" in s
        assert "who_compliance" in s
        assert "species_control" in s
        assert "cost" in s
        assert "per_drug_class" in s
        assert "per_target" in s
        assert len(s["per_target"]) == 15


# =====================================================================
# Test 2: Presets produce expected ordering
# =====================================================================

class TestParameterProfiles:

    def test_presets_exist(self):
        presets = list_presets()
        assert len(presets) == 3
        names = {p["name"] for p in presets}
        assert names == {"high_sensitivity", "balanced", "high_specificity"}

    def test_get_preset_balanced(self):
        profile = get_preset("balanced")
        assert profile.name == "balanced"
        assert profile.efficiency_threshold == 0.4
        assert profile.discrimination_threshold == 3.0
        assert profile.top_k == 3

    def test_get_preset_invalid(self):
        with pytest.raises(KeyError):
            get_preset("nonexistent")

    def test_threshold_ordering(self):
        """HIGH_SENSITIVITY < BALANCED < HIGH_SPECIFICITY thresholds."""
        assert HIGH_SENSITIVITY.efficiency_threshold < BALANCED.efficiency_threshold
        assert BALANCED.efficiency_threshold < HIGH_SPECIFICITY.efficiency_threshold
        assert HIGH_SENSITIVITY.discrimination_threshold < BALANCED.discrimination_threshold
        assert BALANCED.discrimination_threshold < HIGH_SPECIFICITY.discrimination_threshold

    def test_high_sensitivity_covers_more_targets(self):
        """High sensitivity should detect >= targets than high specificity."""
        members, cands = _make_mock_panel()
        m_sens = compute_diagnostic_metrics(
            members, cands,
            HIGH_SENSITIVITY.efficiency_threshold,
            HIGH_SENSITIVITY.discrimination_threshold,
        )
        m_spec = compute_diagnostic_metrics(
            members, cands,
            HIGH_SPECIFICITY.efficiency_threshold,
            HIGH_SPECIFICITY.discrimination_threshold,
        )
        assert m_sens.sensitivity >= m_spec.sensitivity

    def test_preset_has_clinical_description(self):
        """Each preset should have a non-trivial description."""
        for name in ["high_sensitivity", "balanced", "high_specificity"]:
            profile = get_preset(name)
            assert len(profile.description) > 50


# =====================================================================
# Test 3: Parameter sweep produces monotonic curves
# =====================================================================

class TestSweep:

    def test_sweep_disc_threshold_monotonic(self):
        """Sweeping disc_threshold: sensitivity non-increasing."""
        members, cands = _make_mock_panel()
        result = sweep_parameter(
            "discrimination_threshold",
            [1.0, 2.0, 3.0, 5.0, 8.0, 15.0],
            members, cands,
        )
        sensitivities = [p.metrics.sensitivity for p in result.points]
        # Monotonically non-increasing
        for i in range(len(sensitivities) - 1):
            assert sensitivities[i] >= sensitivities[i + 1], \
                f"Sensitivity increased at disc={result.points[i+1].parameter_value}"

    def test_sweep_eff_threshold_monotonic(self):
        """Sweeping efficiency_threshold: sensitivity non-increasing."""
        members, cands = _make_mock_panel()
        result = sweep_parameter(
            "efficiency_threshold",
            [0.1, 0.3, 0.5, 0.7, 0.9],
            members, cands,
        )
        sensitivities = [p.metrics.sensitivity for p in result.points]
        for i in range(len(sensitivities) - 1):
            assert sensitivities[i] >= sensitivities[i + 1]

    def test_sweep_invalid_param(self):
        members, cands = _make_mock_panel()
        with pytest.raises(ValueError):
            sweep_parameter("invalid_param", [1.0], members, cands)

    def test_sweep_result_serializable(self):
        members, cands = _make_mock_panel()
        result = sweep_parameter(
            "discrimination_threshold", [2.0, 5.0], members, cands,
        )
        d = result.to_dict()
        assert d["parameter_name"] == "discrimination_threshold"
        assert d["n_points"] == 2
        assert len(d["points"]) == 2
        for pt in d["points"]:
            assert "sensitivity" in pt
            assert "specificity" in pt
            assert "cost" in pt

    def test_sweep_extreme_high_threshold_zero_sensitivity(self):
        """Extremely high thresholds should yield 0 sensitivity."""
        members, cands = _make_mock_panel()
        result = sweep_parameter(
            "discrimination_threshold", [100.0], members, cands,
        )
        assert result.points[0].metrics.sensitivity == 0.0


# =====================================================================
# Test 4: Pareto frontier is valid
# =====================================================================

class TestPareto:

    def test_pareto_returns_points(self):
        members, cands = _make_mock_panel()
        frontier = pareto_frontier(
            members, cands,
            disc_values=[1.5, 3.0, 5.0, 10.0],
            score_values=[0.2, 0.4, 0.6],
        )
        assert len(frontier) >= 2

    def test_pareto_non_dominated(self):
        """No point on the frontier should be dominated by another."""
        members, cands = _make_mock_panel()
        frontier = pareto_frontier(
            members, cands,
            disc_values=[1.5, 3.0, 5.0, 10.0],
            score_values=[0.2, 0.4, 0.6],
        )
        for i, p in enumerate(frontier):
            for j, q in enumerate(frontier):
                if i == j:
                    continue
                assert not (
                    q.metrics.sensitivity >= p.metrics.sensitivity
                    and q.metrics.specificity >= p.metrics.specificity
                    and (
                        q.metrics.sensitivity > p.metrics.sensitivity
                        or q.metrics.specificity > p.metrics.specificity
                    )
                ), f"Point {i} dominated by point {j}"

    def test_pareto_sorted_by_sensitivity(self):
        members, cands = _make_mock_panel()
        frontier = pareto_frontier(
            members, cands,
            disc_values=[1.5, 3.0, 5.0],
            score_values=[0.3, 0.5],
        )
        sensitivities = [p.metrics.sensitivity for p in frontier]
        assert sensitivities == sorted(sensitivities, reverse=True)

    def test_generate_profile_grid_size(self):
        """Default grid is 8 disc x 6 score = 48 profiles."""
        grid = generate_profile_grid()
        assert len(grid) == 48
        # Custom grid
        grid2 = generate_profile_grid(disc_values=[2.0, 5.0], score_values=[0.3, 0.6])
        assert len(grid2) == 4

    def test_pareto_serializable(self):
        members, cands = _make_mock_panel()
        frontier = pareto_frontier(
            members, cands,
            disc_values=[2.0, 5.0],
            score_values=[0.3, 0.6],
        )
        for p in frontier:
            d = p.to_dict()
            assert "profile" in d
            assert "sensitivity" in d
            assert "specificity" in d


# =====================================================================
# Test 5: Top-K returns ranked alternatives
# =====================================================================

class TestTopK:

    def test_top_k_returns_one_per_member(self):
        members, cands = _make_mock_panel()
        results = collect_top_k(members, cands, k=5)
        assert len(results) == len(members)

    def test_top_k_alternatives_count(self):
        members, cands = _make_mock_panel()
        results = collect_top_k(members, cands, k=3)
        for r in results:
            # Each target has >= 3 candidates, so at least 2 alternatives
            assert len(r.alternatives) >= 1
            assert len(r.alternatives) <= 3

    def test_top_k_tradeoff_annotations(self):
        members, cands = _make_mock_panel()
        results = collect_top_k(members, cands, k=5)
        valid_tradeoffs = {
            "higher_efficiency", "higher_discrimination",
            "fewer_offtargets", "different_pam", "comparable",
        }
        for r in results:
            for alt in r.alternatives:
                assert len(alt.tradeoffs) >= 1
                for t in alt.tradeoffs:
                    assert t in valid_tradeoffs

    def test_top_k_has_tradeoff_summary(self):
        """Each alternative should have a human-readable tradeoff summary."""
        members, cands = _make_mock_panel()
        results = collect_top_k(members, cands, k=3)
        for r in results:
            for alt in r.alternatives:
                assert isinstance(alt.tradeoff_summary, str)
                assert len(alt.tradeoff_summary) > 0

    def test_top_k_has_drug_class(self):
        members, cands = _make_mock_panel()
        results = collect_top_k(members, cands, k=3)
        for r in results:
            assert r.drug_class in TARGET_DRUG_CLASS.values() or r.drug_class == "unknown"

    def test_top_k_has_selection_reason(self):
        members, cands = _make_mock_panel()
        results = collect_top_k(members, cands, k=3)
        for r in results:
            assert r.selection_reason.startswith("Best composite:")
            assert "score=" in r.selection_reason

    def test_selected_not_in_alternatives(self):
        members, cands = _make_mock_panel()
        results = collect_top_k(members, cands, k=5)
        for r in results:
            sel_spacer = r.selected.candidate.spacer_seq
            for alt in r.alternatives:
                assert alt.candidate.candidate.spacer_seq != sel_spacer

    def test_top_k_serializable(self):
        members, cands = _make_mock_panel()
        results = collect_top_k(members, cands, k=3)
        for r in results:
            d = r.to_dict()
            assert "target_label" in d
            assert "drug_class" in d
            assert "selection_reason" in d
            assert "alternatives" in d
            for alt_d in d["alternatives"]:
                assert "tradeoff_summary" in alt_d
                assert "tradeoffs" in alt_d
                assert "delta_efficiency" in alt_d


# =====================================================================
# Test 6: WHO compliance per drug class
# =====================================================================

class TestWHOCompliance:

    def test_who_compliance_structure(self):
        """who_compliance returns dict with all drug classes."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.4, 3.0)
        compliance = metrics.who_compliance
        assert "rifampicin" in compliance
        assert "isoniazid" in compliance
        assert "ethambutol" in compliance
        assert "pyrazinamide" in compliance
        assert "fluoroquinolone" in compliance
        assert "aminoglycoside" in compliance

    def test_who_compliance_rifampicin_meets_minimal(self):
        """With low thresholds, rifampicin (5 targets, all with primers) should meet minimal."""
        members, cands = _make_mock_panel()
        # Use low thresholds so all rpoB targets pass
        metrics = compute_diagnostic_metrics(members, cands, 0.3, 2.0)
        rif = metrics.who_compliance["rifampicin"]
        # All 5 rpoB targets have high disc and have primers
        assert rif["sensitivity"] == 1.0
        assert rif["meets_minimal"] is True

    def test_who_compliance_fails_with_strict_thresholds(self):
        """With very strict thresholds some drug classes should fail."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.7, 8.0)
        # pncA (score=0.35) should fail
        pza = metrics.who_compliance["pyrazinamide"]
        assert pza["sensitivity"] == 0.0
        assert pza["meets_minimal"] is False

    def test_who_tpp_thresholds_correctness(self):
        """Verify WHO TPP constants match the published values."""
        assert WHO_TPP_SENSITIVITY["rifampicin"]["minimal"] == 0.95
        assert WHO_TPP_SENSITIVITY["isoniazid"]["minimal"] == 0.90
        assert WHO_TPP_SENSITIVITY["fluoroquinolone"]["minimal"] == 0.90
        assert WHO_TPP_SENSITIVITY["ethambutol"]["minimal"] == 0.80
        assert WHO_TPP_SENSITIVITY["pyrazinamide"]["minimal"] == 0.80
        assert WHO_TPP_SENSITIVITY["aminoglycoside"]["minimal"] == 0.80


# =====================================================================
# Test 7: Cost estimation
# =====================================================================

class TestCostEstimation:

    def test_cost_scales_with_assay_ready(self):
        """Cost should scale with number of assay-ready targets."""
        members, cands = _make_mock_panel(all_have_primers=True)
        metrics_low = compute_diagnostic_metrics(members, cands, 0.3, 1.5)
        metrics_high = compute_diagnostic_metrics(members, cands, 0.7, 8.0)
        assert metrics_low.cost["assay_ready_targets"] >= metrics_high.cost["assay_ready_targets"]
        assert metrics_low.cost["oligos_to_order"] >= metrics_high.cost["oligos_to_order"]

    def test_cost_oligo_formula(self):
        """n_oligos = assay_ready_targets * 3 (crRNA + FWD + REV)."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.3, 2.0)
        n_ready = metrics.cost["assay_ready_targets"]
        assert metrics.cost["oligos_to_order"] == n_ready * 3

    def test_cost_per_test(self):
        """cost_per_test_usd = n_ready * 0.30."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.3, 2.0)
        n_ready = metrics.cost["assay_ready_targets"]
        assert metrics.cost["cost_per_test_usd"] == round(n_ready * 0.30, 2)


# =====================================================================
# Test 8: Species control handling
# =====================================================================

class TestSpeciesControl:

    def test_is6110_not_in_sensitivity(self):
        """IS6110 should not affect panel sensitivity calculation."""
        members_with, cands_with = _make_mock_panel(include_species_control=True)
        members_without, cands_without = _make_mock_panel(include_species_control=False)
        m_with = compute_diagnostic_metrics(members_with, cands_with, 0.4, 3.0)
        m_without = compute_diagnostic_metrics(members_without, cands_without, 0.4, 3.0)
        assert abs(m_with.sensitivity - m_without.sensitivity) < 1e-9

    def test_species_control_present(self):
        members, cands = _make_mock_panel(include_species_control=True)
        metrics = compute_diagnostic_metrics(members, cands, 0.3, 2.0)
        ctrl = metrics.species_control
        assert ctrl["present"] is True
        assert ctrl["assay_ready"] is True
        assert ctrl["score"] > 0.0

    def test_species_control_absent(self):
        members, cands = _make_mock_panel(include_species_control=False)
        metrics = compute_diagnostic_metrics(members, cands, 0.3, 2.0)
        ctrl = metrics.species_control
        assert ctrl["present"] is False

    def test_is6110_not_in_drug_class_metrics(self):
        """species_control should not appear in drug_class_metrics."""
        members, cands = _make_mock_panel()
        metrics = compute_diagnostic_metrics(members, cands, 0.4, 3.0)
        for d in metrics.drug_class_metrics:
            assert d.drug_class != "species_control"


# =====================================================================
# Test 9: Edge cases
# =====================================================================

class TestEdgeCases:

    def test_empty_panel(self):
        """No members, no candidates -> 0 sensitivity, 0 specificity."""
        metrics = compute_diagnostic_metrics([], {}, 0.3, 2.0)
        assert metrics.sensitivity == 0.0
        assert metrics.specificity == 0.0
        assert metrics.drug_class_coverage == 0.0

    def test_all_covered(self):
        """All targets assay-ready -> sensitivity = 1.0."""
        members, cands = _make_mock_panel(all_have_primers=True)
        # Very low thresholds so everything passes
        metrics = compute_diagnostic_metrics(members, cands, 0.1, 1.0)
        assert metrics.sensitivity == 1.0

    def test_no_primers(self):
        """All targets covered but none have primers -> sensitivity = 0."""
        members, cands = _make_mock_panel()
        # Remove all primers
        stripped = []
        for m in members:
            stripped.append(PanelMember(
                target=m.target,
                selected_candidate=m.selected_candidate,
                primers=None,
            ))
        metrics = compute_diagnostic_metrics(stripped, cands, 0.1, 1.0)
        # All covered but no primers -> not assay-ready
        assert metrics.sensitivity == 0.0
        for t in metrics.target_metrics:
            assert t.is_covered  # score and disc pass
            assert not t.is_assay_ready  # no primers

    def test_single_target(self):
        """Panel with exactly one resistance target."""
        target = _make_target("rpoB", 450, "S", "L")
        cand = _make_candidate("rpoB_S450L", 0, 0.6, 5.0)
        member = PanelMember(
            target=target,
            selected_candidate=cand,
            primers=_make_primer_pair(),
        )
        metrics = compute_diagnostic_metrics(
            [member],
            {"rpoB_S450L": [cand]},
            0.4, 3.0,
        )
        assert metrics.sensitivity == 1.0
        assert len(metrics.drug_class_metrics) == 1
        assert metrics.drug_class_metrics[0].drug_class == "rifampicin"


# =====================================================================
# Test 10: TARGET_DRUG_CLASS mapping
# =====================================================================

class TestTargetDrugClassMapping:

    def test_all_14_targets_mapped(self):
        """All 14 resistance targets + IS6110 are in the mapping."""
        assert len(TARGET_DRUG_CLASS) == 15  # 14 + IS6110

    def test_drug_class_counts(self):
        """Verify number of targets per drug class."""
        from collections import Counter
        counts = Counter(TARGET_DRUG_CLASS.values())
        assert counts["rifampicin"] == 5
        assert counts["isoniazid"] == 3
        assert counts["ethambutol"] == 2
        assert counts["pyrazinamide"] == 1
        assert counts["fluoroquinolone"] == 2
        assert counts["aminoglycoside"] == 1
        assert counts["species_control"] == 1

    def test_labels_match_mutation_format(self):
        """Labels should match Mutation.label format: gene_refPOSalt."""
        for label in TARGET_DRUG_CLASS:
            if TARGET_DRUG_CLASS[label] == "species_control":
                continue
            # Should have format gene_XnnnY
            parts = label.split("_", 1)
            assert len(parts) == 2, f"Bad label format: {label}"
            assert parts[0] in ("rpoB", "katG", "inhA", "embB", "pncA", "gyrA", "rrs")
