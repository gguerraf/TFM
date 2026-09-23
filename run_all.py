from pathlib import Path
import argparse
import importlib.util
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    "src/pipeline/00_validate_holdings.py",
    "src/pipeline/00_validate_duplicate_holdings.py",
    "src/pipeline/01_load_holdings.py",
    "src/utils/extract_company_universe.py",
    "src/utils/deduplicate_company_universe.py",
    "src/pipeline/02_download_prices.py",
    "src/pipeline/03_check_missing_downloads.py",
    "src/pipeline/04_retry_missing_tickers.py",
    "src/pipeline/05_apply_ticker_mappings.py",
    "src/pipeline/06_download_benchmarks.py",
    "src/pipeline/07_download_energy_benchmark.py",
    "src/pipeline/08_download_gics.py",
    "src/pipeline/09_add_manual_gics_labels.py",
    "src/pipeline/10_download_volume.py",
    "src/pipeline/11_compute_returns.py",
]

MODEL_STEPS = [
    "src/models/20_estimate_baseline_model.py",
    "src/models/21_estimate_improved_model.py",
    "src/models/22_run_robustness_checks.py",
    "src/models/23_run_ml_baselines.py",
    "src/models/24_run_benchmark_robustness.py",
    "src/models/25_run_local_projections.py",
    "src/models/26_run_factor_robustness.py",
    "src/models/27_run_etf_interactions.py",
    "src/models/28_run_quantile_regression.py",
    "src/models/29_compile_etf_results.py",
    "src/models/30_run_specification_comparison.py",
]

ANALYSIS_STEPS = [
    "src/analysis/40_exploratory_analysis.py",
    "src/analysis/41_raw_json_analysis.py",
    "src/analysis/42_integrity_checks.py",
]

DEPENDENCIES = {
    "pipeline": [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("yfinance", "yfinance"),
    ],
    "models": [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("statsmodels", "statsmodels"),
        ("linearmodels", "linearmodels"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
        ("lightgbm", "lightgbm"),
        ("shap", "shap"),
        ("torch", "torch"),
    ],
    "analysis": [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the TFM pipeline scripts in order")
    parser.add_argument(
        "--stage",
        choices=["all", "pipeline", "models", "analysis"],
        default="all",
        help="Scripts to run. Default: all, meaning pipeline and models",
    )
    parser.add_argument(
        "--include-analysis",
        action="store_true",
        help="Also run exploratory analysis scripts after the main workflow",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scripts without running them",
    )
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Skip the import check before running scripts",
    )
    return parser.parse_args()


def selected_steps(args):
    if args.stage == "pipeline":
        steps = PIPELINE_STEPS
    elif args.stage == "models":
        steps = MODEL_STEPS
    elif args.stage == "analysis":
        steps = ANALYSIS_STEPS
    else:
        steps = PIPELINE_STEPS + MODEL_STEPS

    if args.include_analysis and args.stage != "analysis":
        steps = steps + ANALYSIS_STEPS

    return steps


def selected_dependency_groups(args):
    groups = []
    if args.stage in {"all", "pipeline"}:
        groups.append("pipeline")
    if args.stage in {"all", "models"}:
        groups.append("models")
    if args.stage == "analysis" or args.include_analysis:
        groups.append("analysis")
    return groups


def validate_dependencies(args):
    required = []
    seen = set()
    for group in selected_dependency_groups(args):
        for module_name, package_name in DEPENDENCIES[group]:
            if module_name not in seen:
                required.append((module_name, package_name))
                seen.add(module_name)

    missing = [(module_name, package_name) for module_name, package_name in required if importlib.util.find_spec(module_name) is None]
    if not missing:
        return True

    print("Missing Python packages:")
    for _, package_name in missing:
        print(f"  {package_name}")
    print("Install dependencies with:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    return False


def validate_steps(steps):
    missing = [step for step in steps if not (ROOT / step).exists()]
    if missing:
        print("Missing scripts:")
        for step in missing:
            print(f"  {step}")
        return False
    return True


def run_step(step, index, total):
    start = time.perf_counter()
    print(f"[{index:02d}/{total:02d}] {step}", flush=True)
    result = subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"Failed after {elapsed:.1f}s: {step}", flush=True)
        return result.returncode

    print(f"Done in {elapsed:.1f}s", flush=True)
    return 0


def main():
    args = parse_args()
    steps = selected_steps(args)

    if not validate_steps(steps):
        return 1

    if args.dry_run:
        for index, step in enumerate(steps, start=1):
            print(f"[{index:02d}/{len(steps):02d}] {step}")
        return 0

    if not args.skip_dependency_check and not validate_dependencies(args):
        return 1

    start = time.perf_counter()
    for index, step in enumerate(steps, start=1):
        code = run_step(step, index, len(steps))
        if code != 0:
            return code

    elapsed = time.perf_counter() - start
    print(f"All selected scripts completed in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())