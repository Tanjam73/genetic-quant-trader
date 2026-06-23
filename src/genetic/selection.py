"""
selection.py
────────────
Selection operators for the Genetic Algorithm.

Available selectors:
  - TournamentSelector   : k individuals compete; fittest wins (default)
  - RouletteWheelSelector: fitness-proportionate selection
  - RankSelector         : rank-based selection (immune to fitness scaling issues)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .chromosome import Chromosome


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSelector(ABC):
    """Interface for all selection operators."""

    @abstractmethod
    def select(
        self, population: List[Chromosome], n: int
    ) -> List[Chromosome]:
        """
        Select n individuals from population to form the mating pool.

        Parameters
        ----------
        population : list[Chromosome]
            Current population (all individuals must have fitness set).
        n : int
            Number of individuals to select.

        Returns
        -------
        list[Chromosome]  (shallow copies — originals are not modified)
        """


# ---------------------------------------------------------------------------
# Tournament Selection
# ---------------------------------------------------------------------------

class TournamentSelector(BaseSelector):
    """
    Tournament Selection: run `n` tournaments, each comparing `k` randomly
    drawn individuals; the fittest individual in each tournament is selected.

    Parameters
    ----------
    k : int
        Tournament size. Larger k → stronger selection pressure.
        Typical range: 3–7.
    """

    def __init__(self, k: int = 5):
        if k < 2:
            raise ValueError("Tournament size k must be >= 2.")
        self.k = k

    def select(self, population: List[Chromosome], n: int) -> List[Chromosome]:
        evaluated = [c for c in population if c.fitness is not None]
        if not evaluated:
            raise ValueError("No evaluated chromosomes in population.")

        mating_pool: List[Chromosome] = []
        for _ in range(n):
            if self.k <= len(evaluated):
                competitors = random.sample(evaluated, k=self.k)
            else:
                competitors = random.choices(evaluated, k=self.k)
            winner = max(competitors, key=lambda c: c.fitness)
            mating_pool.append(winner)
        return mating_pool


# ---------------------------------------------------------------------------
# Roulette Wheel (Fitness-Proportionate) Selection
# ---------------------------------------------------------------------------

class RouletteWheelSelector(BaseSelector):
    """
    Roulette Wheel Selection: each individual's selection probability is
    proportional to its fitness.

    ⚠ Unstable when fitnesses are negative (e.g., negative Sharpe Ratios).
    A shift is applied automatically: fitness_shifted = fitness - min_fitness + ε.
    """

    def select(self, population: List[Chromosome], n: int) -> List[Chromosome]:
        evaluated = [c for c in population if c.fitness is not None]
        if not evaluated:
            raise ValueError("No evaluated chromosomes in population.")

        fitnesses = np.array([c.fitness for c in evaluated], dtype=float)
        # Shift to ensure non-negative weights
        min_f = fitnesses.min()
        if min_f <= 0:
            fitnesses = fitnesses - min_f + 1e-6
        total = fitnesses.sum()
        probs = fitnesses / total

        indices = np.random.choice(len(evaluated), size=n, replace=True, p=probs)
        return [evaluated[i] for i in indices]


# ---------------------------------------------------------------------------
# Rank-Based Selection
# ---------------------------------------------------------------------------

class RankSelector(BaseSelector):
    """
    Rank Selection: individuals are sorted by fitness and assigned selection
    probabilities based on rank rather than raw fitness value.

    This avoids premature convergence caused by super-fit individuals
    dominating roulette-wheel selection early in evolution.

    Parameters
    ----------
    selection_pressure : float
        Value in [1.0, 2.0]. Higher values increase the chance that
        top-ranked individuals are selected. Default: 1.5.
    """

    def __init__(self, selection_pressure: float = 1.5):
        if not 1.0 <= selection_pressure <= 2.0:
            raise ValueError("selection_pressure must be in [1.0, 2.0].")
        self.sp = selection_pressure

    def select(self, population: List[Chromosome], n: int) -> List[Chromosome]:
        evaluated = [c for c in population if c.fitness is not None]
        if not evaluated:
            raise ValueError("No evaluated chromosomes in population.")

        N = len(evaluated)
        # Sort ascending so rank 1 = worst
        sorted_pop = sorted(evaluated, key=lambda c: c.fitness)
        # Linear rank probabilities
        probs = np.array(
            [
                (2 - self.sp) / N + 2 * (i) * (self.sp - 1) / (N * (N - 1))
                for i in range(1, N + 1)
            ]
        )
        probs /= probs.sum()  # normalize for floating-point safety

        indices = np.random.choice(N, size=n, replace=True, p=probs)
        return [sorted_pop[i] for i in indices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_selector(name: str, **kwargs) -> BaseSelector:
    """
    Factory function to retrieve a selector by name.

    Parameters
    ----------
    name : str
        One of 'tournament', 'roulette', 'rank'.
    **kwargs
        Passed to the selector constructor.

    Returns
    -------
    BaseSelector
    """
    registry = {
        "tournament": TournamentSelector,
        "roulette":   RouletteWheelSelector,
        "rank":       RankSelector,
    }
    if name not in registry:
        raise ValueError(f"Unknown selector '{name}'. Choose from: {list(registry)}")
    return registry[name](**kwargs)
