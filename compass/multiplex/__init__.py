"""Module 7: Multiplex panel optimization and spatial array design."""

from compass.multiplex.optimizer import MultiplexOptimizer
from compass.multiplex.pooling import compute_primer_pools, compute_amplicon_pad_specificity
from compass.multiplex.kinetics import estimate_all_targets, estimate_time_to_result

__all__ = [
    "MultiplexOptimizer",
    "compute_primer_pools",
    "compute_amplicon_pad_specificity",
    "estimate_all_targets",
    "estimate_time_to_result",
]
