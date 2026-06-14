"""
population.py
─────────────
Manages the GA population: initialization, generation bookkeeping,
diversity metrics, and elitism.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional
import numpy as np

from .chromosome import Chromosome

logger = logging.getLogger(__name__)


class Population:
    """
    Maintains a fixed-size collection of Chromosome instances and
    provides generation-level statistics.

    Parameters
    ----------
    size : int
        Number of individuals in the population.
    elitism_count : int
        Number of top-fitness chromosomes guaranteed to survive each generation.
    """

    def __init__(self, size: int = 200, elitism_count: int = 5):
        self.size = size
        self.elitism_count = elitism_count
        self.generation: int = 0
        self.individuals: List[Chromosome] = []
        self.history: List[dict] = []          # per-generation stats

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Seed the population with random chromosomes."""
        self.individuals = [
            Chromosome.random(generation=0) for _ in range(self.size)
        ]
        logger.info("Population initialized with %d individuals.", self.size)

    def seed_from(self, chromosomes: List[Chromosome]) -> None:
        """
        Warm-start a population with known-good chromosomes.
        Remaining slots are filled randomly.
        """
        n_seed = min(len(chromosomes), self.size)
        self.individuals = list(chromosomes[:n_seed])
        while len(self.individuals) < self.size:
            self.individuals.append(Chromosome.random(generation=self.generation))
        logger.info("Population seeded with %d provided chromosomes.", n_seed)

    # ------------------------------------------------------------------ #
    #  Generational transition                                             #
    # ------------------------------------------------------------------ #

    def advance(self, offspring: List[Chromosome]) -> None:
        """
        Replace current population with offspring, preserving elite individuals.

        Parameters
        ----------
        offspring : list[Chromosome]
            New individuals produced by crossover/mutation.
        """
        # Sort current population by fitness (descending)
        evaluated = [c for c in self.individuals if c.fitness is not None]
        evaluated.sort(key=lambda c: c.fitness, reverse=True)

        # Elites survive unconditionally
        elite = evaluated[: self.elitism_count]

        # Fill rest from offspring, truncate or pad as needed
        new_gen = elite + offspring
        if len(new_gen) > self.size:
            new_gen = new_gen[: self.size]
        while len(new_gen) < self.size:
            new_gen.append(Chromosome.random(generation=self.generation + 1))

        self.generation += 1
        for c in new_gen:
            c.generation = self.generation
        self.individuals = new_gen

        # Record statistics
        self._record_stats()

    # ------------------------------------------------------------------ #
    #  Queries                                                             #
    # ------------------------------------------------------------------ #

    @property
    def best(self) -> Optional[Chromosome]:
        """Return the chromosome with the highest fitness."""
        evaluated = [c for c in self.individuals if c.fitness is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda c: c.fitness)

    @property
    def median_fitness(self) -> Optional[float]:
        fitnesses = [c.fitness for c in self.individuals if c.fitness is not None]
        if not fitnesses:
            return None
        return float(np.median(fitnesses))

    @property
    def fitness_std(self) -> Optional[float]:
        fitnesses = [c.fitness for c in self.individuals if c.fitness is not None]
        if len(fitnesses) < 2:
            return None
        return float(np.std(fitnesses))

    @property
    def diversity(self) -> float:
        """
        Gene-space diversity measured as the average pairwise Euclidean
        distance between chromosomes (normalized by gene range widths).
        A value near 0 means the population has converged.
        """
        genes = np.array([c.genes for c in self.individuals])
        if len(genes) < 2:
            return 0.0
        # Normalize each gene column to [0,1]
        col_min = genes.min(axis=0)
        col_max = genes.max(axis=0)
        ranges = col_max - col_min
        ranges[ranges == 0] = 1.0   # avoid div-by-zero for converged genes
        normed = (genes - col_min) / ranges
        # Sample pairwise distances (O(n²) is expensive for large populations)
        sample_size = min(50, len(normed))
        idx = random.sample(range(len(normed)), sample_size)
        sample = normed[idx]
        dists = []
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                dists.append(np.linalg.norm(sample[i] - sample[j]))
        return float(np.mean(dists)) if dists else 0.0

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _record_stats(self) -> None:
        best = self.best
        self.history.append(
            {
                "generation": self.generation,
                "best_fitness": best.fitness if best else None,
                "median_fitness": self.median_fitness,
                "fitness_std": self.fitness_std,
                "diversity": self.diversity,
            }
        )
        logger.info(
            "Gen %d | Best=%.4f | Median=%.4f | Diversity=%.4f",
            self.generation,
            best.fitness if best else float("nan"),
            self.median_fitness or float("nan"),
            self.diversity,
        )

    # ------------------------------------------------------------------ #
    #  Dunder                                                              #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __repr__(self) -> str:
        return (
            f"Population(size={self.size}, gen={self.generation}, "
            f"best_fitness={self.best.fitness if self.best else None})"
        )
