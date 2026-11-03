# src/optimization/ga_optimizer.py
"""
Genetic Algorithm for Hyperparameter Optimization.

Classical GA:
- Tournament selection
- Uniform crossover on categorical, BLX-alpha on numeric
- Gaussian mutation
- Elitism
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .search_space import (
    SearchSpace,
    params_to_vector,
    validate_params,
    vector_to_params,
)


@dataclass
class Individual:
    params: Dict[str, Any]
    fitness: float = -1.0
    born_gen: int = 0


class GeneticOptimizer:
    """
    Genetic Algorithm for hyperparameter optimization.
    """

    def __init__(
        self,
        spaces: List[SearchSpace],
        fitness_fn: Callable[[Dict[str, Any]], float],
        population_size: int = 12,
        n_generations: int = 50,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        elitism: int = 2,
        tournament_size: int = 3,
        seed: int = 42,
        verbose: bool = True,
        output_dir: str = "ga_results",
        # ✅ NEW: Early Stopping
        early_stop_patience: int = 8,
        min_improvement: float = 0.002,
        time_limit_hours: float = None,
    ):
        self.spaces = spaces
        self.fitness_fn = fitness_fn
        self.pop_size = population_size
        self.n_gen = n_generations
        self.cross_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.elitism = elitism
        self.tourn_size = tournament_size
        self.verbose = verbose
        self.output_dir = output_dir
        self.early_stop_patience = early_stop_patience
        self.min_improvement = min_improvement
        self.time_limit_hours = time_limit_hours

        random.seed(seed)
        np.random.seed(seed)

        os.makedirs(output_dir, exist_ok=True)

        # State
        self.population: List[Individual] = []
        self.best: Individual = None
        self.history: List[Dict[str, Any]] = []
        self.start_time = None

    # ============================================================
    # ✅ INIT POPULATION
    # ============================================================

    def _random_individual(self) -> Individual:
        params = {s.name: s.sample() for s in self.spaces}
        params = validate_params(params)
        return Individual(params=params)

    def _init_population(self):
        self.population = [self._random_individual() for _ in range(self.pop_size)]

    # ============================================================
    # ✅ SELECTION
    # ============================================================

    def _tournament(self, k: int = None) -> Individual:
        k = k or self.tourn_size
        contenders = random.sample(self.population, k)
        return max(contenders, key=lambda x: x.fitness)

    # ============================================================
    # ✅ CROSSOVER
    # ============================================================

    def _crossover(
        self, p1: Individual, p2: Individual
    ) -> Tuple[Individual, Individual]:
        if random.random() > self.cross_rate:
            return (
                Individual(params=dict(p1.params), born_gen=p1.born_gen),
                Individual(params=dict(p2.params), born_gen=p2.born_gen),
            )

        c1 = {}
        c2 = {}
        for s in self.spaces:
            if random.random() < 0.5:
                c1[s.name] = p1.params[s.name]
                c2[s.name] = p2.params[s.name]
            else:
                c1[s.name] = p2.params[s.name]
                c2[s.name] = p1.params[s.name]

        return (
            Individual(params=validate_params(c1)),
            Individual(params=validate_params(c2)),
        )

    # ============================================================
    # ✅ MUTATION
    # ============================================================

    def _mutate(self, ind: Individual) -> Individual:
        new_params = dict(ind.params)
        for s in self.spaces:
            if random.random() < self.mut_rate:
                new_params[s.name] = s.mutate(new_params[s.name])
        return Individual(params=validate_params(new_params))

    # ============================================================
    # ✅ EVALUATE (with caching to avoid re-eval)
    # ============================================================

    def _evaluate(self, ind: Individual):
        if ind.fitness >= 0:
            return
        try:
            ind.fitness = float(self.fitness_fn(ind.params))
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Eval failed: {e}")
            ind.fitness = 0.0

    # ============================================================
    # ✅ MAIN RUN
    # ============================================================

    def run(self) -> Dict[str, Any]:
        self.start_time = time.time()

        if self.verbose:
            print("\n" + "=" * 80)
            print("🧬 GENETIC ALGORITHM START")
            print("=" * 80)
            print(f"   Population size: {self.pop_size}")
            print(f"   Generations:     {self.n_gen}")
            print(f"   Total trials:    {self.pop_size * self.n_gen}")
            print(f"   Crossover rate:  {self.cross_rate}")
            print(f"   Mutation rate:   {self.mut_rate}")
            print(f"   Elitism:         {self.elitism}")
            print(f"   Spaces:          {len(self.spaces)}")
            print("=" * 80)

        # Initial population
        if self.verbose:
            print("\n📌 Initializing population...")
        self._init_population()

        # Evaluate initial population
        for i, ind in enumerate(self.population):
            if self.verbose:
                print(f"\n🔬 [Gen 0 | Ind {i+1}/{self.pop_size}] Evaluating...")
                print(f"      LR={ind.params['lr']:.2e}, BS={ind.params['batch_size']}, "
                      f"Optim={ind.params['optimizer_name']}")
            self._evaluate(ind)

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best = self.population[0]

        self._log_generation(0)

        # ✅ Early Stop tracking
        no_improve_count = 0
        prev_best = self.best.fitness

        # Generations
        for gen in range(1, self.n_gen + 1):
            # ✅ Time limit check
            if self.time_limit_hours is not None:
                elapsed_h = (time.time() - self.start_time) / 3600
                if elapsed_h >= self.time_limit_hours:
                    print(f"\n⏰ TIME LIMIT REACHED ({elapsed_h:.2f}h)")
                    break

            if self.verbose:
                print(f"\n{'='*80}")
                print(f"🧬 GENERATION {gen}/{self.n_gen}")
                print(f"{'='*80}")
                print(f"   Best so far: {self.best.fitness:.4f}")

                if self.time_limit_hours is not None:
                    elapsed_h = (time.time() - self.start_time) / 3600
                    remaining = self.time_limit_hours - elapsed_h
                    print(f"   Time: {elapsed_h:.2f}h elapsed, {remaining:.2f}h remaining")

            # Elitism
            new_pop = [Individual(params=dict(ind.params),
                                  fitness=ind.fitness,
                                  born_gen=ind.born_gen)
                       for ind in self.population[:self.elitism]]

            # Generate children
            while len(new_pop) < self.pop_size:
                p1 = self._tournament()
                p2 = self._tournament()
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                c1.born_gen = gen
                c2.born_gen = gen
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            # Evaluate new individuals
            for i, ind in enumerate(new_pop[self.elitism:], start=1):
                if self.verbose:
                    print(f"\n🔬 [Gen {gen} | Ind {i}/{self.pop_size - self.elitism}] Evaluating...")
                    print(f"      LR={ind.params['lr']:.2e}, BS={ind.params['batch_size']}, "
                          f"Optim={ind.params['optimizer_name']}")
                self._evaluate(ind)

            # Update population
            new_pop.sort(key=lambda x: x.fitness, reverse=True)
            self.population = new_pop

            # Update best
            if self.population[0].fitness > self.best.fitness:
                self.best = self.population[0]

            self._log_generation(gen)

            # ✅ Early Stop check
            current_best = self.best.fitness
            improvement = current_best - prev_best

            if improvement < self.min_improvement:
                no_improve_count += 1
                if self.verbose:
                    print(f"   ⏸️  No significant improvement ({improvement:.4f} < {self.min_improvement})")
                    print(f"   Early stop counter: {no_improve_count}/{self.early_stop_patience}")
            else:
                no_improve_count = 0

            prev_best = current_best

            if no_improve_count >= self.early_stop_patience:
                print(f"\n🛑 EARLY STOPPING at generation {gen}")
                print(f"   No improvement for {self.early_stop_patience} generations")
                print(f"   Best fitness: {current_best:.4f}")
                break

            # Save checkpoint
            self._save_checkpoint()

        elapsed = time.time() - self.start_time

        if self.verbose:
            print("\n" + "=" * 80)
            print("✅ GA COMPLETE")
            print("=" * 80)
            print(f"   Best fitness:  {self.best.fitness:.4f}")
            print(f"   Total time:    {elapsed/3600:.2f} hours")
            print("=" * 80)

        return {
            "best_params": self.best.params,
            "best_fitness": self.best.fitness,
            "history": self.history,
            "elapsed_hours": elapsed / 3600,
        }

    def _log_generation(self, gen: int):
        fitnesses = [ind.fitness for ind in self.population]
        entry = {
            "generation": gen,
            "best_fitness": float(max(fitnesses)),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "worst_fitness": float(min(fitnesses)),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(entry)

        if self.verbose:
            print(f"\n📊 Gen {gen}: best={entry['best_fitness']:.4f}, "
                  f"mean={entry['mean_fitness']:.4f}, std={entry['std_fitness']:.4f}")

    def _save_checkpoint(self):
        path = os.path.join(self.output_dir, "ga_checkpoint.json")
        data = {
            "best_params": self.best.params,
            "best_fitness": self.best.fitness,
            "history": self.history,
            "elapsed_seconds": time.time() - self.start_time,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4, default=str)