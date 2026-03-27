"""Evo 2 log-likelihood ratio at the mutation site.

Computes a SINGLE SCALAR per target: how "surprised" is Evo 2 by the
resistance mutation? This is passed to Compass-ML as an optional scalar
feature via the n_scalar_features parameter.

NOT a full embedding branch -- just one pre-computed number per target.

The per-position formulation (Fix 10) sums log-likelihood differences
only within a small window around the mutation, avoiding signal dilution
from the 500-nt genomic context.

Requires: Evo 2 7B on GPU (one-time computation for 14 targets).

References:
    Nguyen et al., "Sequence modeling and design from molecular to genome
    scale with Evo." Science 2024. arXiv:2403.19441.
"""

from __future__ import annotations

import torch


def compute_evo2_llr(
    mutant_context: str,
    wildtype_context: str,
    mutation_pos: int,
    window: int = 2,
    model_name: str = "evo2_7b",
) -> float:
    """Per-position Evo 2 log-likelihood ratio at the mutation site.

    Computes:
        LLR = sum_{i in [pos-w, pos+w]} [ log P(mut_i | ctx) - log P(wt_i | ctx) ]

    Only sums over a small window around the mutation. This directly
    measures how "surprising" the SNP is to a genome-scale language model.

    Interpretation:
        LLR < 0: Evo 2 finds the wildtype more likely (expected for most
                  resistance mutations, which are under negative selection).
        |LLR| large: strong evolutionary constraint at this position ->
                     potentially better discrimination power.

    Args:
        mutant_context:  +/-250 bp flanking the mutation (500 nt total).
        wildtype_context: same region with the wildtype allele.
        mutation_pos:     position of the SNP within the context (0-indexed).
        window:           +/- nt around the mutation to sum over.
        model_name:       Evo 2 model identifier.

    Returns:
        Log-likelihood ratio (scalar float).
    """
    try:
        from evo2 import Evo2
    except ImportError:
        raise ImportError(
            "Evo 2 not installed. See: https://github.com/ARC-Institute/evo2"
        )

    model = Evo2(model_name)
    model.model.eval()

    with torch.no_grad():
        mut_ids = torch.tensor(
            model.tokenizer.tokenize(mutant_context), dtype=torch.long,
        ).unsqueeze(0).to("cuda")
        wt_ids = torch.tensor(
            model.tokenizer.tokenize(wildtype_context), dtype=torch.long,
        ).unsqueeze(0).to("cuda")

        mut_logits, _ = model(mut_ids)
        wt_logits, _ = model(wt_ids)

    # Per-position log-probabilities
    mut_logp = torch.nn.functional.log_softmax(mut_logits[0], dim=-1)
    wt_logp = torch.nn.functional.log_softmax(wt_logits[0], dim=-1)

    # Sum LLR over the mutation window only
    start = max(0, mutation_pos - window)
    end = min(len(mutant_context) - 1, mutation_pos + window)

    llr = 0.0
    for i in range(start, end):
        mut_token = mut_ids[0, i + 1]
        wt_token = wt_ids[0, i + 1]
        llr += mut_logp[i, mut_token].item() - wt_logp[i, wt_token].item()

    return llr


def compute_evo2_llr_batch(
    targets: list[dict],
    model_name: str = "evo2_7b",
    window: int = 2,
) -> list[float]:
    """Batch computation for all AMR targets (loads model ONCE).

    Args:
        targets: list of dicts with keys:
            'mutant_context', 'wildtype_context', 'mutation_pos'
        model_name: Evo 2 model identifier.
        window: +/- nt around mutation.

    Returns:
        List of LLR values, one per target.
    """
    try:
        from evo2 import Evo2
    except ImportError:
        raise ImportError(
            "Evo 2 not installed. See: https://github.com/ARC-Institute/evo2"
        )

    # Load model ONCE for all targets (7B params, ~15GB GPU RAM)
    model = Evo2(model_name)
    model.model.eval()

    results = []
    for t in targets:
        llr = _compute_llr_with_model(
            model,
            t["mutant_context"],
            t["wildtype_context"],
            t["mutation_pos"],
            window=window,
        )
        results.append(llr)

    return results


def _compute_llr_with_model(
    model,
    mutant_context: str,
    wildtype_context: str,
    mutation_pos: int,
    window: int = 2,
) -> float:
    """Compute LLR using a pre-loaded EVO-2 model instance."""
    with torch.no_grad():
        mut_ids = torch.tensor(
            model.tokenizer.tokenize(mutant_context), dtype=torch.long,
        ).unsqueeze(0).to("cuda")
        wt_ids = torch.tensor(
            model.tokenizer.tokenize(wildtype_context), dtype=torch.long,
        ).unsqueeze(0).to("cuda")

        mut_logits, _ = model(mut_ids)
        wt_logits, _ = model(wt_ids)

    mut_logp = torch.nn.functional.log_softmax(mut_logits[0], dim=-1)
    wt_logp = torch.nn.functional.log_softmax(wt_logits[0], dim=-1)

    start = max(0, mutation_pos - window)
    end = min(len(mutant_context) - 1, mutation_pos + window)

    llr = 0.0
    for i in range(start, end):
        mut_token = mut_ids[0, i + 1]
        wt_token = wt_ids[0, i + 1]
        llr += mut_logp[i, mut_token].item() - wt_logp[i, wt_token].item()

    return llr
