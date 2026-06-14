"""
tests/test_chromosome.py
────────────────────────
Unit tests for Chromosome encoding, decoding, validity, and serialization.
"""

import json
import os
import tempfile
import pytest
from src.genetic.chromosome import Chromosome, GENE_BOUNDS


class TestChromosomeCreation:
    def test_random_creates_correct_length(self):
        c = Chromosome.random()
        assert len(c.genes) == len(GENE_BOUNDS)

    def test_random_genes_within_bounds(self):
        for _ in range(100):
            c = Chromosome.random()
            keys = list(GENE_BOUNDS.keys())
            for i, key in enumerate(keys):
                lo, hi = GENE_BOUNDS[key]
                assert lo <= c.genes[i] <= hi, (
                    f"Gene {key} value {c.genes[i]} out of bounds [{lo}, {hi}]"
                )

    def test_random_respects_exit_lt_entry_constraint(self):
        for _ in range(100):
            c = Chromosome.random()
            assert c.genes[2] < c.genes[1], (
                f"exit_z ({c.genes[2]}) should be < entry_z ({c.genes[1]})"
            )

    def test_random_respects_holding_constraint(self):
        for _ in range(100):
            c = Chromosome.random()
            assert c.genes[3] < c.genes[4], (
                f"holding_min ({c.genes[3]}) should be < holding_max ({c.genes[4]})"
            )

    def test_default_fitness_is_none(self):
        c = Chromosome.random()
        assert c.fitness is None

    def test_generation_set_correctly(self):
        c = Chromosome.random(generation=7)
        assert c.generation == 7


class TestChromosomeDecode:
    PAIR_UNIVERSE = [("AAPL", "MSFT"), ("JPM", "BAC"), ("XOM", "CVX")]

    def test_decode_returns_all_keys(self):
        c = Chromosome.random()
        spec = c.decode(self.PAIR_UNIVERSE)
        expected_keys = {"pair", "entry_z", "exit_z", "holding_min", "holding_max", "lookback"}
        assert set(spec.keys()) == expected_keys

    def test_pair_index_wraps(self):
        # Gene 0 value > len(universe) should wrap via modulo
        c = Chromosome(genes=[100, 2.0, 0.5, 2, 10, 60])
        spec = c.decode(self.PAIR_UNIVERSE)
        assert spec["pair"] in self.PAIR_UNIVERSE

    def test_decode_types(self):
        c = Chromosome.random()
        spec = c.decode(self.PAIR_UNIVERSE)
        assert isinstance(spec["pair"], tuple)
        assert isinstance(spec["entry_z"], float)
        assert isinstance(spec["exit_z"], float)
        assert isinstance(spec["holding_min"], int)
        assert isinstance(spec["holding_max"], int)
        assert isinstance(spec["lookback"], int)


class TestChromosomeValidity:
    def test_valid_chromosome_passes(self):
        c = Chromosome(genes=[0, 2.0, 0.5, 2, 10, 60])
        assert c.is_valid()

    def test_exit_gte_entry_is_invalid(self):
        c = Chromosome(genes=[0, 1.5, 1.5, 2, 10, 60])   # exit_z == entry_z
        assert not c.is_valid()

    def test_holding_min_gte_max_is_invalid(self):
        c = Chromosome(genes=[0, 2.0, 0.5, 10, 10, 60])  # hold_min == hold_max
        assert not c.is_valid()

    def test_repair_fixes_exit_gte_entry(self):
        c = Chromosome(genes=[0, 1.5, 2.0, 2, 10, 60])   # exit > entry
        c.repair()
        assert c.genes[2] < c.genes[1]

    def test_repair_fixes_holding_constraint(self):
        c = Chromosome(genes=[0, 2.0, 0.5, 15, 10, 60])  # min > max
        c.repair()
        assert c.genes[3] < c.genes[4]

    def test_repair_clamps_out_of_range_genes(self):
        # entry_z way above max (3.0)
        c = Chromosome(genes=[0, 100.0, 0.5, 2, 10, 60])
        c.repair()
        lo, hi = GENE_BOUNDS["entry_z"]
        assert lo <= c.genes[1] <= hi


class TestChromosomeSerialization:
    def test_to_dict_roundtrip(self):
        c = Chromosome.random(generation=3)
        c.fitness = 1.23
        d = c.to_dict()
        c2 = Chromosome.from_dict(d)
        assert c2.genes == c.genes
        assert c2.fitness == c.fitness
        assert c2.generation == c.generation

    def test_save_and_load(self):
        c = Chromosome.random()
        c.fitness = 0.85
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            c.save(path)
            c2 = Chromosome.load(path)
            assert c2.genes == c.genes
            assert abs(c2.fitness - c.fitness) < 1e-9
        finally:
            os.unlink(path)


class TestChromosomeSorting:
    def test_sorting_by_fitness(self):
        c1 = Chromosome.random(); c1.fitness = 0.5
        c2 = Chromosome.random(); c2.fitness = 1.5
        c3 = Chromosome.random(); c3.fitness = 1.0
        sorted_chroms = sorted([c1, c2, c3], reverse=True)
        assert sorted_chroms[0].fitness == 1.5
        assert sorted_chroms[-1].fitness == 0.5

    def test_none_fitness_sorts_last(self):
        c1 = Chromosome.random(); c1.fitness = 1.0
        c2 = Chromosome.random(); c2.fitness = None
        sorted_chroms = sorted([c1, c2], reverse=True)
        assert sorted_chroms[-1].fitness is None
