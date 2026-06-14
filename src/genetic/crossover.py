"""
crossover.py
────────────
Crossover operators that combine two parent chromosomes to produce offspring.

Available operators:
  - UniformCrossover      : each gene randomly inherited from either parent
  - SinglePointCrossover  : genes split at a random cut point
  - ArithmeticCrossover   : offspring is a weighted average of parents (for floats)
  - BlendCrossover (BLX-α): extends the parental interval by α for continuous genes
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple

import numpy as np

from .chromosome import Chromosome, GENE_BOUNDS


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseCrossover(ABC):
    """Interface for all crossover operators."""

    def __init__(self, prob: float = 0.85):
        """
        Parameters
        ----------
        prob : float
            Probability that crossover is applied. If not applied,
            parents are returned unchanged.
        """
        if not 0.0 <= prob <= 1.0:
            raise ValueError("Crossover probability must be in [0, 1].")
        self.prob = prob

    @abstractmethod
    def cross(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """Produce two offspring from two parents."""

    def __call__(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        if random.random() > self.prob:
            # No crossover — return deep copies of parents
            return deepcopy(parent1), deepcopy(parent2)
        child1, child2 = self.cross(parent1, parent2)
        child1.repair()
        child2.repair()
        return child1, child2


# ---------------------------------------------------------------------------
# Uniform Crossover
# ---------------------------------------------------------------------------

class UniformCrossover(BaseCrossover):
    """
    Each gene is independently drawn from parent1 with probability `gene_swap_prob`
    and from parent2 with probability (1 - gene_swap_prob).

    Default gene_swap_prob=0.5 gives a uniform 50/50 mix.
    """

    def __init__(self, prob: float = 0.85, gene_swap_prob: float = 0.5):
        super().__init__(prob)
        self.gene_swap_prob = gene_swap_prob

    def cross(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        g1, g2 = list(parent1.genes), list(parent2.genes)
        child_genes1, child_genes2 = [], []
        for a, b in zip(g1, g2):
            if random.random() < self.gene_swap_prob:
                child_genes1.append(a)
                child_genes2.append(b)
            else:
                child_genes1.append(b)
                child_genes2.append(a)
        c1 = Chromosome(genes=child_genes1, parents=(id(parent1), id(parent2)))
        c2 = Chromosome(genes=child_genes2, parents=(id(parent1), id(parent2)))
        return c1, c2


# ---------------------------------------------------------------------------
# Single-Point Crossover
# ---------------------------------------------------------------------------

class SinglePointCrossover(BaseCrossover):
    """
    A cut-point is chosen uniformly at random. Child 1 takes genes
    [0..cut] from parent1 and [cut+1..] from parent2; vice versa for child 2.
    """

    def cross(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        n = len(parent1.genes)
        cut = random.randint(1, n - 1)
        g1 = parent1.genes[:cut] + parent2.genes[cut:]
        g2 = parent2.genes[:cut] + parent1.genes[cut:]
        c1 = Chromosome(genes=g1, parents=(id(parent1), id(parent2)))
        c2 = Chromosome(genes=g2, parents=(id(parent1), id(parent2)))
        return c1, c2


# ---------------------------------------------------------------------------
# Arithmetic Crossover
# ---------------------------------------------------------------------------

class ArithmeticCrossover(BaseCrossover):
    """
    Offspring genes are a random convex combination of the parents:
        child1[i] = α * parent1[i] + (1 - α) * parent2[i]
        child2[i] = (1-α) * parent1[i] + α * parent2[i]

    α is drawn per-offspring (whole arithmetic crossover).

    Integer genes are rounded after blending.
    """

    def cross(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        alpha = random.random()
        keys = list(GENE_BOUNDS.keys())
        g1, g2 = [], []
        for i, (a, b) in enumerate(zip(parent1.genes, parent2.genes)):
            v1 = alpha * a + (1 - alpha) * b
            v2 = (1 - alpha) * a + alpha * b
            lo, hi = GENE_BOUNDS[keys[i]]
            if isinstance(lo, int):
                v1, v2 = int(round(v1)), int(round(v2))
            g1.append(v1)
            g2.append(v2)
        c1 = Chromosome(genes=g1, parents=(id(parent1), id(parent2)))
        c2 = Chromosome(genes=g2, parents=(id(parent1), id(parent2)))
        return c1, c2


# ---------------------------------------------------------------------------
# BLX-α Crossover (Blend Crossover)
# ---------------------------------------------------------------------------

class BlendCrossover(BaseCrossover):
    """
    BLX-α crossover extends the search beyond the parental range by α:
        child[i] ~ Uniform(min(p1, p2) - α*δ,  max(p1, p2) + α*δ)
    where δ = |p1[i] - p2[i]|.

    Applied only to continuous (float) genes; integer genes use single-point.

    Parameters
    ----------
    alpha : float
        Extension factor. 0.0 = standard intermediate crossover.
        0.5 is a common default.
    """

    def __init__(self, prob: float = 0.85, alpha: float = 0.5):
        super().__init__(prob)
        self.alpha = alpha

    def cross(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        keys = list(GENE_BOUNDS.keys())
        g1, g2 = [], []
        for i, (a, b) in enumerate(zip(parent1.genes, parent2.genes)):
            lo, hi = GENE_BOUNDS[keys[i]]
            if isinstance(lo, int):
                # Integer genes: single-point style
                if random.random() < 0.5:
                    g1.append(a); g2.append(b)
                else:
                    g1.append(b); g2.append(a)
            else:
                # Float genes: BLX-α
                delta = abs(a - b)
                lower = min(a, b) - self.alpha * delta
                upper = max(a, b) + self.alpha * delta
                v1 = random.uniform(lower, upper)
                v2 = random.uniform(lower, upper)
                g1.append(round(v1, 4))
                g2.append(round(v2, 4))
        c1 = Chromosome(genes=g1, parents=(id(parent1), id(parent2)))
        c2 = Chromosome(genes=g2, parents=(id(parent1), id(parent2)))
        return c1, c2


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_crossover(name: str, **kwargs) -> BaseCrossover:
    registry = {
        "uniform":      UniformCrossover,
        "single_point": SinglePointCrossover,
        "arithmetic":   ArithmeticCrossover,
        "blend":        BlendCrossover,
    }
    if name not in registry:
        raise ValueError(f"Unknown crossover '{name}'. Choose from: {list(registry)}")
    return registry[name](**kwargs)
