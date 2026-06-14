"""
mutation.py
───────────
Mutation operators for the Genetic Algorithm.

Key feature: **Adaptive Mutation Rate** — the mutation probability decays
over generations to shift the search from broad exploration early on to
fine-grained exploitation of promising regions later.

Available operators:
  - GaussianMutator    : perturb continuous genes with Gaussian noise
  - UniformMutator     : replace genes with uniformly random values
  - AdaptiveMutator    : wraps any base mutator with a decaying schedule
  - NonUniformMutator  : perturbation shrinks with generation (Michalewicz)
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import numpy as np

from .chromosome import Chromosome, GENE_BOUNDS


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMutator(ABC):
    """Interface for all mutation operators."""

    def __init__(self, mutation_rate: float = 0.05):
        """
        Parameters
        ----------
        mutation_rate : float
            Probability that any given gene is mutated.
        """
        self.mutation_rate = mutation_rate

    @abstractmethod
    def _mutate_gene(self, gene: float, gene_name: str) -> float:
        """Mutate a single gene value."""

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply mutation to a chromosome.

        Returns a mutated deep copy; the original is not modified.
        """
        child = deepcopy(chromosome)
        child.fitness = None   # fitness is invalidated after mutation
        keys = list(GENE_BOUNDS.keys())
        for i, key in enumerate(keys):
            if random.random() < self.mutation_rate:
                child.genes[i] = self._mutate_gene(child.genes[i], key)
        child.repair()
        return child

    def __call__(self, chromosome: Chromosome) -> Chromosome:
        return self.mutate(chromosome)


# ---------------------------------------------------------------------------
# Gaussian Mutator
# ---------------------------------------------------------------------------

class GaussianMutator(BaseMutator):
    """
    Perturbs continuous (float) genes by adding Gaussian noise.
    Integer genes are offset by ±1 or ±2.

    Parameters
    ----------
    mutation_rate : float
        Per-gene mutation probability.
    sigma_fraction : float
        Gaussian std-dev as a fraction of the gene's range width.
        E.g., 0.1 → σ = 10% of the gene's total range.
    """

    def __init__(self, mutation_rate: float = 0.05, sigma_fraction: float = 0.1):
        super().__init__(mutation_rate)
        self.sigma_fraction = sigma_fraction

    def _mutate_gene(self, gene: float, gene_name: str) -> float:
        lo, hi = GENE_BOUNDS[gene_name]
        gene_range = hi - lo
        if isinstance(lo, int):
            # Integer gene: discrete perturbation
            delta = random.choice([-2, -1, 1, 2])
            return int(np.clip(int(gene) + delta, lo, hi))
        else:
            # Float gene: Gaussian perturbation
            sigma = self.sigma_fraction * gene_range
            new_val = float(gene) + random.gauss(0, sigma)
            return round(float(np.clip(new_val, lo, hi)), 4)


# ---------------------------------------------------------------------------
# Uniform Mutator
# ---------------------------------------------------------------------------

class UniformMutator(BaseMutator):
    """
    Replaces selected genes with uniformly random values within bounds.
    Also known as "random resetting" — most disruptive; good early in evolution.
    """

    def _mutate_gene(self, gene: float, gene_name: str) -> float:
        lo, hi = GENE_BOUNDS[gene_name]
        if isinstance(lo, int):
            return random.randint(lo, hi)
        return round(random.uniform(lo, hi), 4)


# ---------------------------------------------------------------------------
# Adaptive Mutator  ← primary recommended operator
# ---------------------------------------------------------------------------

class AdaptiveMutator(BaseMutator):
    """
    Wraps any base mutator with an **exponentially decaying mutation rate**
    schedule.

    The rate starts at `rate_initial` (exploration) and decays to `rate_final`
    (exploitation) following:

        μ(t) = rate_final + (rate_initial - rate_final)
                            * exp(-decay_rate * t / t_max)

    Parameters
    ----------
    rate_initial : float   Initial (high) mutation rate. Default 0.15.
    rate_final   : float   Final  (low)  mutation rate. Default 0.02.
    decay_rate   : float   Controls speed of decay (higher → faster). Default 5.0.
    t_max        : int     Total number of generations planned.
    base_mutator : BaseMutator   Underlying mutation strategy. Defaults to Gaussian.
    """

    def __init__(
        self,
        rate_initial: float = 0.15,
        rate_final: float = 0.02,
        decay_rate: float = 5.0,
        t_max: int = 100,
        base_mutator: Optional[BaseMutator] = None,
    ):
        super().__init__(mutation_rate=rate_initial)
        self.rate_initial = rate_initial
        self.rate_final = rate_final
        self.decay_rate = decay_rate
        self.t_max = t_max
        self.base_mutator = base_mutator or GaussianMutator(mutation_rate=rate_initial)
        self._current_generation = 0

    def step(self, generation: int) -> None:
        """Update the mutation rate for the current generation."""
        self._current_generation = generation
        t = generation
        rate = self.rate_final + (self.rate_initial - self.rate_final) * math.exp(
            -self.decay_rate * t / max(self.t_max, 1)
        )
        self.mutation_rate = rate
        self.base_mutator.mutation_rate = rate

    def _mutate_gene(self, gene: float, gene_name: str) -> float:
        # Delegate to the wrapped mutator's logic
        return self.base_mutator._mutate_gene(gene, gene_name)

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Uses the base mutator's mutate with the current adaptive rate."""
        self.base_mutator.mutation_rate = self.mutation_rate
        return self.base_mutator.mutate(chromosome)

    @property
    def schedule_summary(self) -> str:
        """Human-readable mutation schedule description."""
        return (
            f"AdaptiveMutator: μ₀={self.rate_initial}, μ_f={self.rate_final}, "
            f"decay={self.decay_rate}, T={self.t_max}, "
            f"current gen={self._current_generation}, "
            f"current μ={self.mutation_rate:.4f}"
        )


# ---------------------------------------------------------------------------
# Non-Uniform Mutator (Michalewicz)
# ---------------------------------------------------------------------------

class NonUniformMutator(BaseMutator):
    """
    Michalewicz's Non-Uniform Mutation:
    The perturbation magnitude shrinks with generation `t` so early
    generations explore widely; later generations fine-tune.

        Δ(t, y) = y * (1 - r^((1 - t/T)^b))

    where r ~ Uniform(0,1) and b controls the degree of non-uniformity.

    Parameters
    ----------
    b : float     Non-uniformity degree. Larger b → faster shrinkage. Default 5.0.
    t_max : int   Total planned generations.
    """

    def __init__(self, mutation_rate: float = 0.05, b: float = 5.0, t_max: int = 100):
        super().__init__(mutation_rate)
        self.b = b
        self.t_max = t_max
        self.current_generation = 0

    def _delta(self, y: float, t: int) -> float:
        r = random.random()
        exponent = (1 - t / max(self.t_max, 1)) ** self.b
        return y * (1 - r ** exponent)

    def _mutate_gene(self, gene: float, gene_name: str) -> float:
        lo, hi = GENE_BOUNDS[gene_name]
        if isinstance(lo, int):
            # For integer genes, fall back to small discrete step
            delta = random.choice([-1, 1])
            return int(np.clip(int(gene) + delta, lo, hi))
        # Float genes: non-uniform perturbation
        if random.random() < 0.5:
            new_val = float(gene) + self._delta(hi - float(gene), self.current_generation)
        else:
            new_val = float(gene) - self._delta(float(gene) - lo, self.current_generation)
        return round(float(np.clip(new_val, lo, hi)), 4)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_mutator(name: str, **kwargs) -> BaseMutator:
    registry = {
        "gaussian":    GaussianMutator,
        "uniform":     UniformMutator,
        "adaptive":    AdaptiveMutator,
        "non_uniform": NonUniformMutator,
    }
    if name not in registry:
        raise ValueError(f"Unknown mutator '{name}'. Choose from: {list(registry)}")
    return registry[name](**kwargs)
