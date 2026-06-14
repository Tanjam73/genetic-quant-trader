"""
run_evolution.py
────────────────
Main entry point for running the Genetic Algorithm evolution.

Usage:
    python scripts/run_evolution.py \
        --config configs/ga_config.yaml \
        --generations 100 \
        --population 200 \
        --output results/run_001

The script:
  1. Loads config and data
  2. Screens pairs for cointegration
  3. Initializes GA population
  4. Runs evolution loop for N generations
  5. Saves the best chromosome + evolution history
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genetic.chromosome import Chromosome
from src.genetic.population import Population
from src.genetic.fitness import evaluate_population
from src.genetic.selection import get_selector
from src.genetic.crossover import get_crossover
from src.genetic.mutation import get_mutator, AdaptiveMutator
from src.data.loader import DataLoader
from src.data.pairs import PairSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evolution")


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

def run_evolution(config: dict, output_dir: Path) -> Chromosome:
    """
    Full GA evolution loop.

    Parameters
    ----------
    config     : merged config dict (GA + backtest settings)
    output_dir : directory to write results

    Returns
    -------
    Best chromosome discovered
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ga_cfg  = config["genetic_algorithm"]
    bt_cfg  = config["backtest"]
    chr_cfg = config["chromosome"]

    n_gen      = ga_cfg["n_generations"]
    pop_size   = ga_cfg["population_size"]
    n_workers  = ga_cfg.get("n_workers", 4)

    # ── Step 1: Load price data ───────────────────────────────────────── #
    logger.info("Loading price data…")
    loader = DataLoader(
        start_date=bt_cfg["start_date"],
        end_date=bt_cfg["end_date"],
    )
    tickers = DataLoader.get_sp500_tickers()
    price_data_raw = loader.load_universe(tickers)

    # ── Step 2: Pair selection ────────────────────────────────────────── #
    logger.info("Screening pairs for cointegration…")
    selector = PairSelector(
        price_data=price_data_raw,
        n_pairs=chr_cfg.get("n_pairs_candidate", 50),
        min_corr=chr_cfg.get("min_correlation", 0.70),
    )
    pair_stats = selector.select()

    if not pair_stats:
        raise RuntimeError("No cointegrated pairs found. Broaden universe or relax filters.")

    pair_universe = [ps.pair for ps in pair_stats]
    logger.info("Pair universe: %d pairs", len(pair_universe))

    # Save pair universe
    with open(output_dir / "pair_universe.json", "w") as f:
        json.dump([list(p) for p in pair_universe], f, indent=2)

    # Build pair price panels
    pair_panels = loader.build_all_pair_panels(pair_universe)

    # Merge everything into config for fitness evaluation
    config["pair_universe"] = pair_universe
    config["pair_panels"]   = pair_panels

    # ── Step 3: Initialize population ────────────────────────────────── #
    logger.info("Initializing population (size=%d)…", pop_size)
    population = Population(size=pop_size, elitism_count=ga_cfg.get("elitism_count", 5))
    population.initialize()

    # ── Step 4: Set up GA operators ──────────────────────────────────── #
    selector_op = get_selector(
        ga_cfg.get("selection", "tournament"),
        k=ga_cfg.get("tournament_size", 5),
    )
    crossover_op = get_crossover(
        ga_cfg.get("crossover", "uniform"),
        prob=ga_cfg.get("crossover_prob", 0.85),
    )
    mutator_op = AdaptiveMutator(
        rate_initial=ga_cfg.get("mutation_rate_initial", 0.15),
        rate_final=ga_cfg.get("mutation_rate_final", 0.02),
        decay_rate=ga_cfg.get("mutation_decay_rate", 5.0),
        t_max=n_gen,
    )

    best_ever: Chromosome = None
    history = []

    # ── Step 5: Evolution loop ────────────────────────────────────────── #
    logger.info("Starting evolution for %d generations…", n_gen)
    start_time = time.time()

    for gen in range(n_gen):
        gen_start = time.time()

        # Update adaptive mutation rate
        mutator_op.step(gen)

        # Evaluate fitness
        individuals = list(population)
        individuals = evaluate_population(
            individuals,
            price_data=pair_panels,
            config=config,
            n_workers=n_workers,
        )

        # Track best
        current_best = population.best
        if current_best and (best_ever is None or current_best.fitness > best_ever.fitness):
            best_ever = Chromosome(
                genes=list(current_best.genes),
                fitness=current_best.fitness,
                generation=gen,
            )
            # Auto-save best
            best_ever.save(output_dir / "best_chromosome.json")

        # Log progress
        gen_time = time.time() - gen_start
        logger.info(
            "Gen %3d/%d | Best=%.4f | Median=%.4f | μ=%.4f | Diversity=%.3f | %.1fs",
            gen + 1, n_gen,
            current_best.fitness if current_best else float("nan"),
            population.median_fitness or float("nan"),
            mutator_op.mutation_rate,
            population.diversity,
            gen_time,
        )

        # Record history
        history.append({
            "generation": gen,
            "best_fitness": current_best.fitness if current_best else None,
            "median_fitness": population.median_fitness,
            "diversity": population.diversity,
            "mutation_rate": mutator_op.mutation_rate,
        })

        # Early stopping: convergence detection
        if gen > 20 and population.diversity < 0.01:
            logger.info("Population converged at gen %d. Stopping early.", gen)
            break

        # ─ Selection → Crossover → Mutation ─────────────────────────── #
        mating_pool = selector_op.select(individuals, n=pop_size)
        offspring = []
        for i in range(0, len(mating_pool) - 1, 2):
            child1, child2 = crossover_op(mating_pool[i], mating_pool[i + 1])
            offspring.append(mutator_op(child1))
            offspring.append(mutator_op(child2))

        # Advance to next generation (with elitism)
        population.advance(offspring)

    # ── Step 6: Save results ─────────────────────────────────────────── #
    elapsed = time.time() - start_time
    logger.info("Evolution complete in %.1f seconds.", elapsed)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Best chromosome: %s", best_ever)
    logger.info(
        "Decoded: %s",
        best_ever.decode(pair_universe) if best_ever else "N/A"
    )

    return best_ever


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Genetic Algorithm evolution to discover mean-reversion strategies."
    )
    parser.add_argument(
        "--config", default="configs/ga_config.yaml",
        help="Path to GA config YAML file."
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Override n_generations from config."
    )
    parser.add_argument(
        "--population", type=int, default=None,
        help="Override population_size from config."
    )
    parser.add_argument(
        "--output", default="results/run_001",
        help="Output directory for results."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes for fitness evaluation."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.generations:
        config["genetic_algorithm"]["n_generations"] = args.generations
    if args.population:
        config["genetic_algorithm"]["population_size"] = args.population
    config["genetic_algorithm"]["n_workers"] = args.workers

    # Load backtest config if separate
    bt_config_path = Path(args.config).parent / "backtest_config.yaml"
    if bt_config_path.exists():
        with open(bt_config_path) as f:
            bt_config = yaml.safe_load(f)
        config.update(bt_config)

    output_dir = Path(args.output)
    best = run_evolution(config=config, output_dir=output_dir)

    if best:
        print("\n" + "=" * 60)
        print("  EVOLUTION COMPLETE")
        print("=" * 60)
        print(f"  Best Sharpe: {best.fitness:.4f}")
        print(f"  Best Chromosome: {best}")
        print(f"  Results saved to: {output_dir}")
        print("=" * 60)
    else:
        print("Evolution failed — no valid chromosomes found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
