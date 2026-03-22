"""Orchestrate the hold dataset experiment: download, combine, train, evaluate, report.

Usage:
    python run_hold_experiment.py
    python run_hold_experiment.py --skip-download --skip-combine
    python run_hold_experiment.py --only gpt_finetune --iteration 1
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "C:/Projects/pythonenv-lerobot/Scripts/python"
DATASET_PREFIX = "irvinh/single-action-shoulder-pan-100-with-hold"
NUM_DATASETS = 15
COMBINED_DATASET = "local/datasets/hold-1500-combined"
BASE_CHECKPOINT_DIR = "local/checkpoints/hold_exp"
BASE_EVAL_DIR = "local/eval/hold_exp"

# Best existing checkpoints for fine-tuning
BEST_GPT_CKPT = "local/checkpoints/gpt_iter6_cosine/best.pth"
BEST_DIFF_CKPT = "local/checkpoints/diff_iter4_wider/best.pth"

EXPERIMENTS = {
    "gpt_finetune": {
        "script": "train_gpt.py",
        "model_type": "gpt",
        "args": [
            "--embed-dim", "384", "--depth", "12", "--num-heads", "12",
            "--lr", "0.0002", "--lr-schedule", "cosine", "--warmup-epochs", "5",
            "--batch-size", "4", "--epochs", "150", "--no-wandb",
        ],
        "fine_tune": BEST_GPT_CKPT,
    },
    "diff_finetune": {
        "script": "train_diffusion.py",
        "model_type": "diffusion",
        "args": [
            "--embed-dim", "512", "--depth", "12", "--num-heads", "16",
            "--lr", "0.0003", "--lr-schedule", "cosine", "--warmup-epochs", "15",
            "--batch-size", "4", "--epochs", "300", "--grad-clip", "1.0",
            "--prediction-type", "sample", "--no-wandb",
        ],
        "fine_tune": BEST_DIFF_CKPT,
    },
    "gpt_scratch": {
        "script": "train_gpt.py",
        "model_type": "gpt",
        "args": [
            "--embed-dim", "384", "--depth", "12", "--num-heads", "12",
            "--lr", "0.0002", "--lr-schedule", "cosine", "--warmup-epochs", "5",
            "--batch-size", "4", "--epochs", "150", "--no-wandb",
        ],
        "fine_tune": None,
    },
    "diff_scratch": {
        "script": "train_diffusion.py",
        "model_type": "diffusion",
        "args": [
            "--embed-dim", "512", "--depth", "12", "--num-heads", "16",
            "--lr", "0.0003", "--lr-schedule", "cosine", "--warmup-epochs", "15",
            "--batch-size", "4", "--epochs", "300", "--grad-clip", "1.0",
            "--prediction-type", "sample", "--no-wandb",
        ],
        "fine_tune": None,
    },
}


def run_cmd(cmd, description, allow_fail=False):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd="c:/Projects/canvas-world-model")
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED ({elapsed:.0f}s): {description}")
        if not allow_fail:
            sys.exit(1)
        return False
    print(f"\n  DONE ({elapsed:.0f}s): {description}")
    return True


def get_dataset_name(i):
    """Get the HuggingFace dataset name for index i (1-15).

    First dataset has no suffix, rest have -2 through -15.
    """
    if i == 1:
        return f"{DATASET_PREFIX}"
    return f"{DATASET_PREFIX}-{i}"


def get_local_dataset_dir(i):
    """Get local dataset directory for index i (1-15)."""
    if i == 1:
        return "local/datasets/single-action-shoulder-pan-100-with-hold"
    return f"local/datasets/single-action-shoulder-pan-100-with-hold-{i}"


def download_datasets():
    """Download and convert all 15 with-hold datasets."""
    for i in range(1, NUM_DATASETS + 1):
        dataset_name = get_dataset_name(i)
        output_dir = get_local_dataset_dir(i)
        if Path(output_dir).exists() and (Path(output_dir) / "dataset_meta.json").exists():
            print(f"  Skipping {dataset_name} (already exists)")
            continue
        run_cmd(
            [PYTHON, "create_dataset.py", "--lerobot-path", dataset_name],
            f"Download dataset {i}/{NUM_DATASETS}: {dataset_name}",
        )


def combine_datasets():
    """Combine all 15 datasets into one."""
    if Path(COMBINED_DATASET).exists() and (Path(COMBINED_DATASET) / "dataset_meta.json").exists():
        print(f"  Combined dataset already exists: {COMBINED_DATASET}")
        return
    inputs = [get_local_dataset_dir(i) for i in range(1, NUM_DATASETS + 1)]
    run_cmd(
        [PYTHON, "combine_datasets.py", "--inputs"] + inputs + ["--output", COMBINED_DATASET],
        "Combine 15 datasets into hold-1500-combined",
    )


def train_experiment(exp_name, exp_config, iteration, extra_args=None):
    """Run a single training experiment."""
    ckpt_dir = f"{BASE_CHECKPOINT_DIR}/iter{iteration}/{exp_name}"
    if Path(ckpt_dir).exists() and (Path(ckpt_dir) / "best.pth").exists():
        print(f"  Skipping {exp_name} iter{iteration} (already trained)")
        return

    cmd = [PYTHON, exp_config["script"], "--dataset", COMBINED_DATASET,
           "--checkpoint-dir", ckpt_dir] + exp_config["args"]

    if exp_config.get("fine_tune"):
        cmd += ["--fine-tune", exp_config["fine_tune"]]

    if extra_args:
        cmd += extra_args

    run_cmd(cmd, f"Train {exp_name} (iteration {iteration})")


def benchmark_experiment(exp_name, exp_config, iteration):
    """Run inference benchmark for an experiment."""
    ckpt_dir = f"{BASE_CHECKPOINT_DIR}/iter{iteration}/{exp_name}"
    output = f"{ckpt_dir}/inference_timing.json"
    if Path(output).exists():
        print(f"  Skipping benchmark {exp_name} iter{iteration} (already done)")
        return

    run_cmd(
        [PYTHON, "benchmark_inference.py",
         "--model-type", exp_config["model_type"],
         "--checkpoint", f"{ckpt_dir}/best.pth",
         "--dataset", COMBINED_DATASET,
         "--output", output],
        f"Benchmark {exp_name} inference (iteration {iteration})",
        allow_fail=True,
    )


def evaluate_experiment(exp_name, exp_config, iteration):
    """Run evaluation for an experiment."""
    ckpt_dir = f"{BASE_CHECKPOINT_DIR}/iter{iteration}/{exp_name}"
    eval_dir = f"{BASE_EVAL_DIR}/iter{iteration}/{exp_name}"
    if Path(eval_dir).exists() and (Path(eval_dir) / "report.json").exists():
        print(f"  Skipping eval {exp_name} iter{iteration} (already done)")
        return

    run_cmd(
        [PYTHON, "evaluate.py",
         "--model-type", exp_config["model_type"],
         "--checkpoint", f"{ckpt_dir}/best.pth",
         "--dataset", COMBINED_DATASET,
         "--output-dir", eval_dir],
        f"Evaluate {exp_name} (iteration {iteration})",
        allow_fail=True,
    )


def generate_report(iteration):
    """Generate comparison report for an iteration."""
    run_cmd(
        [PYTHON, "generate_hold_report.py",
         "--experiment-dir", BASE_CHECKPOINT_DIR,
         "--eval-dir", BASE_EVAL_DIR,
         "--iteration", str(iteration),
         "--output", f"{BASE_EVAL_DIR}/iter{iteration}/comparison_report.md"],
        f"Generate comparison report (iteration {iteration})",
        allow_fail=True,
    )


def parse_main_args():
    p = argparse.ArgumentParser(description="Run hold dataset experiment")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-combine", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--only", type=str, default=None,
                   choices=list(EXPERIMENTS.keys()),
                   help="Run only one experiment")
    p.add_argument("--iteration", type=int, default=0, help="Iteration number")
    p.add_argument("--extra-args", nargs="*", default=None,
                   help="Extra args to pass to training script")
    return p.parse_args()


def main():
    args = parse_main_args()
    iteration = args.iteration

    print(f"\n{'#'*60}")
    print(f"  Hold Dataset Experiment - Iteration {iteration}")
    print(f"{'#'*60}")

    # Phase 1: Download
    if not args.skip_download:
        print("\n--- Phase 1: Download datasets ---")
        download_datasets()

    # Phase 2: Combine
    if not args.skip_combine:
        print("\n--- Phase 2: Combine datasets ---")
        combine_datasets()

    # Select experiments
    experiments = EXPERIMENTS
    if args.only:
        experiments = {args.only: EXPERIMENTS[args.only]}

    # Phase 3: Train
    if not args.skip_train:
        print(f"\n--- Phase 3: Train ({len(experiments)} experiments) ---")
        for name, config in experiments.items():
            train_experiment(name, config, iteration, extra_args=args.extra_args)

    # Phase 4: Benchmark inference
    if not args.skip_eval:
        print(f"\n--- Phase 4: Benchmark inference ---")
        for name, config in experiments.items():
            benchmark_experiment(name, config, iteration)

    # Phase 5: Evaluate
    if not args.skip_eval:
        print(f"\n--- Phase 5: Evaluate ---")
        for name, config in experiments.items():
            evaluate_experiment(name, config, iteration)

    # Phase 6: Report
    if not args.skip_eval:
        print(f"\n--- Phase 6: Generate report ---")
        generate_report(iteration)

    print(f"\n{'#'*60}")
    print(f"  Experiment complete!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
