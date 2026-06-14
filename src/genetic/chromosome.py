"""
chromosome.py
─────────────
Encodes a mean-reversion trading strategy as a fixed-length chromosome
that the Genetic Algorithm can evolve.

Chromosome layout (6 genes):
  Gene 0  : pair_idx       – integer index into pre-screened cointegrated pairs
  Gene 1  : entry_z        – float  z-score threshold to enter a trade
  Gene 2  : exit_z         – float  z-score threshold to exit a trade
  Gene 3  : holding_min    – integer minimum holding period (days)
  Gene 4  : holding_max    – integer maximum holding period (days)
  Gene 5  : lookback       – integer rolling window for spread statistics (days)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Gene bounds — these define the search space
# ---------------------------------------------------------------------------
GENE_BOUNDS = {
    "pair_idx":    (0, 49),          # index into a pre-screened universe of 50 pairs
    "entry_z":     (1.0, 3.0),       # z-score to open a position
    "exit_z":      (0.0, 1.5),       # z-score to close a position
    "holding_min": (1, 5),           # min days to hold (avoid noise trades)
    "holding_max": (5, 60),          # max days to hold before forced exit
    "lookback":    (20, 252),        # rolling window length in trading days
}


@dataclass
class Chromosome:
    """
    Represents a single candidate mean-reversion trading strategy.

    Attributes
    ----------
    genes : list[float]
        Raw gene values [pair_idx, entry_z, exit_z, holding_min,
        holding_max, lookback].
    fitness : float
        Sharpe Ratio (or other fitness metric) assigned after backtesting.
        None until evaluated.
    generation : int
        Generation in which this chromosome was created.
    parents : tuple[int, int]
        Indices of parent chromosomes (for lineage tracking).
    """

    genes: List[float] = field(default_factory=list)
    fitness: Optional[float] = None
    generation: int = 0
    parents: Tuple[int, int] = (-1, -1)

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, generation: int = 0) -> "Chromosome":
        """Create a chromosome with uniformly random genes within bounds."""
        genes = [
            random.randint(*GENE_BOUNDS["pair_idx"]),
            round(random.uniform(*GENE_BOUNDS["entry_z"]), 3),
            round(random.uniform(*GENE_BOUNDS["exit_z"]), 3),
            random.randint(*GENE_BOUNDS["holding_min"]),
            random.randint(*GENE_BOUNDS["holding_max"]),
            random.randint(*GENE_BOUNDS["lookback"]),
        ]
        # Enforce exit_z < entry_z (logical constraint)
        if genes[2] >= genes[1]:
            genes[2] = round(genes[1] * random.uniform(0.1, 0.8), 3)
        # Enforce holding_min < holding_max
        if genes[3] >= genes[4]:
            genes[3] = max(1, genes[4] - 1)
        return cls(genes=genes, generation=generation)

    # ------------------------------------------------------------------ #
    #  Decoding                                                            #
    # ------------------------------------------------------------------ #

    def decode(self, pair_universe: List[Tuple[str, str]]) -> dict:
        """
        Decode raw gene values into a human-readable strategy specification.

        Parameters
        ----------
        pair_universe : list of (ticker_a, ticker_b)
            Ordered list of cointegrated pairs. Gene 0 indexes into this list.

        Returns
        -------
        dict with keys: pair, entry_z, exit_z, holding_min, holding_max, lookback
        """
        pair_idx = int(self.genes[0]) % len(pair_universe)
        return {
            "pair":        pair_universe[pair_idx],
            "entry_z":     round(float(self.genes[1]), 3),
            "exit_z":      round(float(self.genes[2]), 3),
            "holding_min": int(self.genes[3]),
            "holding_max": int(self.genes[4]),
            "lookback":    int(self.genes[5]),
        }

    # ------------------------------------------------------------------ #
    #  Validity checks                                                     #
    # ------------------------------------------------------------------ #

    def is_valid(self) -> bool:
        """Return True if all genes satisfy logical constraints."""
        _, entry_z, exit_z, hold_min, hold_max, lookback = self.genes
        return (
            exit_z < entry_z
            and hold_min < hold_max
            and int(lookback) >= 5
        )

    def repair(self) -> None:
        """In-place correction of constraint violations after mutation/crossover."""
        # entry_z must be strictly greater than exit_z
        if self.genes[2] >= self.genes[1]:
            self.genes[2] = round(self.genes[1] * 0.5, 3)
        # holding_min must be < holding_max
        if self.genes[3] >= self.genes[4]:
            self.genes[3] = max(1, int(self.genes[4]) - 1)
        # Clamp all genes to their respective bounds
        keys = list(GENE_BOUNDS.keys())
        for i, key in enumerate(keys):
            lo, hi = GENE_BOUNDS[key]
            if isinstance(lo, int):
                self.genes[i] = int(np.clip(self.genes[i], lo, hi))
            else:
                self.genes[i] = float(np.clip(self.genes[i], lo, hi))

    # ------------------------------------------------------------------ #
    #  Serialization                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chromosome":
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Chromosome":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------ #
    #  Dunder                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        g = self.genes
        return (
            f"Chromosome(pair_idx={int(g[0])}, entry_z={g[1]:.2f}, "
            f"exit_z={g[2]:.2f}, hold=[{int(g[3])},{int(g[4])}]d, "
            f"lookback={int(g[5])}d, fitness={self.fitness})"
        )

    def __lt__(self, other: "Chromosome") -> bool:
        """Enable sorting by fitness (descending when reversed)."""
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness
