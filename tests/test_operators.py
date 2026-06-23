"""
tests/test_operators.py
───────────────────────
Unit tests for selection, crossover, and mutation operators.
"""

import math
import pytest
from src.genetic.chromosome import Chromosome
from src.genetic.selection import TournamentSelector, RouletteWheelSelector, RankSelector
from src.genetic.crossover import UniformCrossover, SinglePointCrossover, ArithmeticCrossover
from src.genetic.mutation import GaussianMutator, UniformMutator, AdaptiveMutator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_population(n: int = 20) -> list:
    """Create a population with random chromosomes and assigned fitness."""
    pop = []
    for i in range(n):
        c = Chromosome.random()
        c.fitness = float(i) / n   # fitness 0.0 → ~1.0
        pop.append(c)
    return pop


# ---------------------------------------------------------------------------
# Selection tests
# ---------------------------------------------------------------------------

class TestTournamentSelector:
    def test_returns_correct_count(self):
        pop = make_population(20)
        sel = TournamentSelector(k=3)
        result = sel.select(pop, n=10)
        assert len(result) == 10

    def test_selected_from_population(self):
        pop = make_population(20)
        pop_ids = {id(c) for c in pop}
        sel = TournamentSelector(k=3)
        result = sel.select(pop, n=10)
        for c in result:
            assert id(c) in pop_ids

    def test_raises_on_empty_population(self):
        sel = TournamentSelector()
        with pytest.raises(ValueError):
            sel.select([], n=5)

    def test_higher_k_biased_toward_fit(self):
    """Higher k should select fitter individuals on average."""
    pop = make_population(20)
    best = max(pop, key=lambda c: c.fitness)

    sel_high_k = TournamentSelector(k=15)
    sel_low_k = TournamentSelector(k=2)

    high_k_avg = sum(
        sel_high_k.select(pop, n=1)[0].fitness for _ in range(50)
    ) / 50
    low_k_avg = sum(
        sel_low_k.select(pop, n=1)[0].fitness for _ in range(50)
    ) / 50

    assert high_k_avg > low_k_avg


class TestRouletteWheelSelector:
    def test_returns_correct_count(self):
        pop = make_population(20)
        sel = RouletteWheelSelector()
        result = sel.select(pop, n=15)
        assert len(result) == 15

    def test_works_with_negative_fitness(self):
        pop = make_population(10)
        for c in pop:
            c.fitness -= 2.0    # all negative
        sel = RouletteWheelSelector()
        result = sel.select(pop, n=5)
        assert len(result) == 5


class TestRankSelector:
    def test_returns_correct_count(self):
        pop = make_population(20)
        sel = RankSelector()
        result = sel.select(pop, n=8)
        assert len(result) == 8

    def test_invalid_pressure_raises(self):
        with pytest.raises(ValueError):
            RankSelector(selection_pressure=2.5)


# ---------------------------------------------------------------------------
# Crossover tests
# ---------------------------------------------------------------------------

class TestUniformCrossover:
    def test_offspring_correct_gene_length(self):
        p1 = Chromosome.random()
        p2 = Chromosome.random()
        cx = UniformCrossover()
        c1, c2 = cx(p1, p2)
        assert len(c1.genes) == len(p1.genes)
        assert len(c2.genes) == len(p2.genes)

    def test_offspring_fitness_reset(self):
        p1 = Chromosome.random(); p1.fitness = 1.0
        p2 = Chromosome.random(); p2.fitness = 0.5
        cx = UniformCrossover(prob=1.0)
        c1, c2 = cx(p1, p2)
        assert c1.fitness is None
        assert c2.fitness is None

    def test_no_crossover_when_prob_zero(self):
        p1 = Chromosome.random()
        p2 = Chromosome.random()
        genes1 = list(p1.genes)
        genes2 = list(p2.genes)
        cx = UniformCrossover(prob=0.0)
        c1, c2 = cx(p1, p2)
        assert c1.genes == genes1
        assert c2.genes == genes2

    def test_offspring_genes_come_from_parents(self):
        """Each offspring gene must come from one of the parents."""
        p1 = Chromosome.random()
        p2 = Chromosome.random()
        cx = UniformCrossover(prob=1.0)
        for _ in range(50):
            c1, c2 = cx(p1, p2)
            for i in range(len(c1.genes)):
                assert c1.genes[i] in (p1.genes[i], p2.genes[i])


class TestSinglePointCrossover:
    def test_offspring_valid(self):
        p1 = Chromosome.random()
        p2 = Chromosome.random()
        cx = SinglePointCrossover(prob=1.0)
        c1, c2 = cx(p1, p2)
        assert c1.is_valid()
        assert c2.is_valid()


class TestArithmeticCrossover:
    def test_offspring_within_parent_range(self):
        """Arithmetic crossover offspring should be between parents."""
        p1 = Chromosome(genes=[0, 2.0, 0.5, 2, 20, 60])
        p2 = Chromosome(genes=[1, 1.5, 0.3, 3, 15, 40])
        cx = ArithmeticCrossover(prob=1.0)
        for _ in range(20):
            c1, c2 = cx(p1, p2)
            # Float gene (entry_z): c1 should be between p1 and p2 values
            assert min(p1.genes[1], p2.genes[1]) - 0.01 <= c1.genes[1] <= max(p1.genes[1], p2.genes[1]) + 0.01


# ---------------------------------------------------------------------------
# Mutation tests
# ---------------------------------------------------------------------------

class TestGaussianMutator:
    def test_returns_new_object(self):
        c = Chromosome.random()
        m = GaussianMutator(mutation_rate=1.0)
        c2 = m(c)
        assert c2 is not c

    def test_fitness_invalidated(self):
        c = Chromosome.random(); c.fitness = 1.5
        m = GaussianMutator(mutation_rate=1.0)
        c2 = m(c)
        assert c2.fitness is None

    def test_mutated_genes_within_bounds(self):
        from src.genetic.chromosome import GENE_BOUNDS
        m = GaussianMutator(mutation_rate=1.0)
        for _ in range(100):
            c = Chromosome.random()
            c2 = m(c)
            keys = list(GENE_BOUNDS.keys())
            for i, key in enumerate(keys):
                lo, hi = GENE_BOUNDS[key]
                assert lo <= c2.genes[i] <= hi, (
                    f"Gene {key} = {c2.genes[i]} out of bounds [{lo}, {hi}]"
                )


class TestAdaptiveMutator:
    def test_rate_decreases_with_generation(self):
        m = AdaptiveMutator(
            rate_initial=0.20, rate_final=0.01,
            decay_rate=5.0, t_max=100
        )
        m.step(0)
        rate_early = m.mutation_rate
        m.step(50)
        rate_mid = m.mutation_rate
        m.step(100)
        rate_late = m.mutation_rate
        assert rate_early > rate_mid > rate_late

    def test_rate_starts_near_initial(self):
        m = AdaptiveMutator(rate_initial=0.15, rate_final=0.02, t_max=100)
        m.step(0)
        assert abs(m.mutation_rate - 0.15) < 0.01

    def test_rate_ends_near_final(self):
        m = AdaptiveMutator(rate_initial=0.15, rate_final=0.02, t_max=100)
        m.step(100)
        assert abs(m.mutation_rate - 0.02) < 0.005
