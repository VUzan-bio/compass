"""Balanced batch sampler for multi-domain training.

Each batch contains approximately equal numbers of samples from each domain.
Smaller domains are oversampled (with replacement), larger ones are undersampled.
This prevents any single dataset from dominating gradient updates.

Reference:
    Standard practice in domain adaptation. See also:
    Ben-David et al. "A theory of learning from different domains"
    Machine Learning 2010, 79(1):151-175.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class DomainBalancedSampler(Sampler[list[int]]):
    """Yields batches with balanced domain representation.

    Args:
        domain_ids: domain ID for each sample in the dataset.
        batch_size: total batch size (split evenly across domains).
        n_batches_per_epoch: number of batches per epoch.
    """

    def __init__(
        self,
        domain_ids: list[int],
        batch_size: int = 64,
        n_batches_per_epoch: int = 200,
    ):
        self.domain_ids = np.array(domain_ids)
        self.unique_domains = np.unique(self.domain_ids)
        self.n_domains = len(self.unique_domains)
        self.per_domain = batch_size // self.n_domains
        self.n_batches = n_batches_per_epoch

        self.domain_indices = {
            int(d): np.where(self.domain_ids == d)[0]
            for d in self.unique_domains
        }

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for d in self.unique_domains:
                indices = self.domain_indices[int(d)]
                selected = np.random.choice(indices, size=self.per_domain, replace=True)
                batch.extend(selected.tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches
