# run_ga.py
"""
Main script to run FAST GA for 4 models.

Usage:
    python run_ga.py                    # All 4 models (default fast mode)
    python run_ga.py --model resnet18   # Only ResNet18
    python run_ga.py --pop 10 --gen 35 --time-limit 3.5  # Custom
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optimization.fitness_evaluator import FitnessEvaluator
from src.optimization.ga_optimizer import GeneticOptimizer
from src.optimization.ga_reporter import GAReporter
from src.optimization.hardware_detector import HardwareDetector
from src.optimization.search_space import get_search_space


MODELS = ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b1"]


def run_ga_for_model(
    model_name: str,
    hardware,
    population_size: int,
    n_generations: int,
    subset_size: int,
    time_limit_hours: float,
    early_stop_patience: int,
    output_root: str = "finalReports/selfSupervisedLearning/GA_Optimization",
):
    """Run FAST GA for a single model."""
    print("\n" + "=" * 80)
    print(f"🧬 FAST GA FOR {model_name.upper()}")
    print("=" * 80)
    print(f"   Population: {population_size}")
    print(f"   Generations: {n_generations}")
    print(f"   Subset: {subset_size}")
    print(f"   Time limit: {time_limit_hours} hours")
    print(f"   Early stop patience: {early_stop_patience}")
    print("=" * 80)

    model_out = os.path.join(output_root, model_name)
    os.makedirs(model_out, exist_ok=True)

    # ✅ Search space
    spaces = get_search_space(
        model_name=model_name,
        hardware_max_batch=hardware.max_batch_size,
        hardware_max_workers=hardware.max_num_workers,
        hardware_mixed_precision=hardware.force_mixed_precision,
    )

    print(f"\n📐 Search space: {len(spaces)} parameters")

    # ✅ Fast Fitness Evaluator
    evaluator = FitnessEvaluator(
        model_name=model_name,
        data_dir="../../finalReports/cropped",
        labels_csv="finalReports/labels.csv",
        subset_size=subset_size,
        verbose=True,
    )

    # ✅ GA
    ga = GeneticOptimizer(
        spaces=spaces,
        fitness_fn=evaluator.evaluate,
        population_size=population_size,
        n_generations=n_generations,
        crossover_rate=0.7,
        mutation_rate=0.2,
        elitism=2,
        tournament_size=3,
        seed=42,
        verbose=True,
        output_dir=model_out,
        early_stop_patience=early_stop_patience,
        min_improvement=0.002,
        time_limit_hours=time_limit_hours,
    )

    # ✅ Run
    result = ga.run()

    # ✅ Save results
    best_params_path = os.path.join(model_out, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "best_fitness": result["best_fitness"],
            "best_params": result["best_params"],
            "history": result["history"],
            "hardware": hardware.to_dict(),
            "elapsed_hours": result["elapsed_hours"],
            "n_generations_run": len(result["history"]) - 1,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=4, default=str)
    print(f"\n✅ Best params saved: {best_params_path}")

    # ✅ PDF report
    reporter = GAReporter(output_dir=model_out)
    report_path = reporter.generate_report(
        model_name=model_name,
        best_params=result["best_params"],
        best_fitness=result["best_fitness"],
        history=result["history"],
        hardware_profile=hardware.to_dict(),
    )
    print(f"✅ PDF report: {report_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["all"] + MODELS)
    parser.add_argument("--pop", type=int, default=10,
                        help="Population size (default 10 for fast mode)")
    parser.add_argument("--gen", type=int, default=35,
                        help="Generations (default 35)")
    parser.add_argument("--subset", type=int, default=100,
                        help="Subset size (default 100 for FAST)")
    parser.add_argument("--time-limit", type=float, default=3.5,
                        help="Time limit in hours per model (default 3.5)")
    parser.add_argument("--early-stop", type=int, default=8,
                        help="Early stop patience (default 8)")
    parser.add_argument(
        "--output",
        type=str,
        default="finalReports/selfSupervisedLearning/GA_Optimization",
    )
    args = parser.parse_args()

    # ✅ Hardware detection
    hw = HardwareDetector(verbose=True)

    # ✅ Which models
    models = MODELS if args.model == "all" else [args.model]

    print(f"\n🎯 Will run FAST GA for: {models}")
    print(f"   Time budget: ~{args.time_limit}h per model")
    print(f"   Total estimate: ~{args.time_limit * len(models)}h")

    all_results = {}
    for m in models:
        result = run_ga_for_model(
            model_name=m,
            hardware=hw.profile,
            population_size=args.pop,
            n_generations=args.gen,
            subset_size=args.subset,
            time_limit_hours=args.time_limit,
            early_stop_patience=args.early_stop,
            output_root=args.output,
        )
        all_results[m] = result

    # ✅ Final summary
    print("\n" + "=" * 80)
    print("🎉 ALL FAST GA RUNS COMPLETE")
    print("=" * 80)
    for m, r in all_results.items():
        gens_run = len(r["history"]) - 1
        print(f"   {m.upper():<18} fitness={r['best_fitness']:.4f}  "
              f"gens={gens_run}  time={r['elapsed_hours']:.2f}h")
    print("=" * 80)
    print(f"\n📁 Results in: {args.output}/")
    print("=" * 80)


if __name__ == "__main__":
    main()