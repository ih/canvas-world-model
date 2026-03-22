"""Generate comparison report across hold dataset experiments.

Reads timing, inference benchmarks, evaluation reports, and checkpoint configs
from all experiments and iterations to produce a markdown comparison report.

Usage:
    python generate_hold_report.py --experiment-dir local/checkpoints/hold_exp \
        --eval-dir local/eval/hold_exp --iteration 0 \
        --output local/eval/hold_exp/iter0/comparison_report.md
"""

import argparse
import json
from pathlib import Path

import torch

EXP_NAMES = ["gpt_finetune", "gpt_scratch", "diff_finetune", "diff_scratch"]
EXP_LABELS = {
    "gpt_finetune": "GPT Fine-tune",
    "gpt_scratch": "GPT Scratch",
    "diff_finetune": "Diff Fine-tune",
    "diff_scratch": "Diff Scratch",
}


def load_json(path):
    """Load JSON file, return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_checkpoint_config(ckpt_path):
    """Extract training config from checkpoint args dict."""
    p = Path(ckpt_path)
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    return ckpt.get("args", {})


def fmt_time(seconds):
    """Format seconds into human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def fmt_val(val, precision=4):
    """Format a numeric value."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def generate_config_table(config, exp_name, iteration):
    """Generate markdown table for a single experiment's config."""
    if not config:
        return f"*Config not available for {exp_name}*\n"

    lines = [
        f"### {EXP_LABELS.get(exp_name, exp_name)} (Iteration {iteration})\n",
        "| Parameter | Value |",
        "|---|---|",
    ]

    # Key parameters to show
    params = [
        ("embed_dim", "Embed dim"),
        ("depth", "Depth"),
        ("num_heads", "Num heads"),
        ("lr", "Learning rate"),
        ("lr_schedule", "LR schedule"),
        ("warmup_epochs", "Warmup epochs"),
        ("weight_decay", "Weight decay"),
        ("batch_size", "Batch size"),
        ("epochs", "Max epochs"),
        ("grad_clip", "Grad clip"),
        ("prediction_type", "Prediction type"),
        ("beta_schedule", "Beta schedule"),
        ("num_train_timesteps", "Train timesteps"),
        ("fine_tune", "Fine-tune source"),
        ("early_stop_patience", "Early stop patience"),
    ]

    for key, label in params:
        val = config.get(key)
        if val is not None:
            lines.append(f"| {label} | {val} |")

    return "\n".join(lines) + "\n"


def generate_report(experiment_dir, eval_dir, iteration, all_iterations=None):
    """Generate the full comparison report."""
    if all_iterations is None:
        all_iterations = [iteration]

    lines = [
        "# Hold Dataset Experiment: Comparison Report\n",
        f"**Iteration:** {iteration}\n",
    ]

    # Collect data for all experiments
    data = {}
    for exp_name in EXP_NAMES:
        ckpt_dir = Path(experiment_dir) / f"iter{iteration}" / exp_name
        eval_exp_dir = Path(eval_dir) / f"iter{iteration}" / exp_name

        timing = load_json(ckpt_dir / "timing.json")
        inference = load_json(ckpt_dir / "inference_timing.json")
        report = load_json(eval_exp_dir / "report.json")
        config = load_checkpoint_config(ckpt_dir / "best.pth")

        data[exp_name] = {
            "timing": timing,
            "inference": inference,
            "report": report,
            "config": config,
        }

    # --- Configuration Tables ---
    lines.append("## Training Configurations\n")
    for exp_name in EXP_NAMES:
        config = data[exp_name]["config"]
        lines.append(generate_config_table(config, exp_name, iteration))

    # --- Main Comparison Table ---
    lines.append("## Results Comparison\n")
    lines.append("| Metric | " + " | ".join(EXP_LABELS[n] for n in EXP_NAMES) + " |")
    lines.append("|---|" + "|".join(["---"] * len(EXP_NAMES)) + "|")

    # Training time metrics
    def get_timing(exp, key):
        t = data[exp]["timing"]
        return t.get(key) if t else None

    def get_report(exp, key):
        r = data[exp]["report"]
        return r.get(key) if r else None

    def get_inference(exp, key):
        i = data[exp]["inference"]
        return i.get(key) if i else None

    rows = [
        ("Total training time",
         [fmt_time(get_timing(n, "total_training_seconds")) for n in EXP_NAMES]),
        ("Best epoch",
         [fmt_val(get_timing(n, "best_epoch"), 0) for n in EXP_NAMES]),
        ("Time to plateau",
         [fmt_time(get_timing(n, "time_to_plateau_seconds")) for n in EXP_NAMES]),
        ("Best val loss",
         [fmt_val(get_timing(n, "best_val_loss"), 6) for n in EXP_NAMES]),
        ("Val MSE (visual)",
         [fmt_val(get_report(n, "val_mse_visual"), 6) for n in EXP_NAMES]),
        ("Val MSE (motor strip)",
         [fmt_val(get_report(n, "val_mse_motor_strip"), 6) for n in EXP_NAMES]),
        ("SSIM",
         [fmt_val(get_report(n, "ssim"), 4) for n in EXP_NAMES]),
        ("PSNR (dB)",
         [fmt_val(get_report(n, "psnr"), 2) for n in EXP_NAMES]),
        ("Motor direction accuracy",
         [fmt_val(get_report(n, "motor_direction_accuracy") or get_report(n, "motor_dir_accuracy"), 4) for n in EXP_NAMES]),
        ("Action discrimination",
         [fmt_val(get_report(n, "action_discrimination_score") or get_report(n, "action_disc_score"), 4) for n in EXP_NAMES]),
        ("Inference mean (ms)",
         [fmt_val(get_inference(n, "mean_ms"), 1) for n in EXP_NAMES]),
        ("Inference P95 (ms)",
         [fmt_val(get_inference(n, "p95_ms"), 1) for n in EXP_NAMES]),
        ("Inference median (ms)",
         [fmt_val(get_inference(n, "median_ms"), 1) for n in EXP_NAMES]),
    ]

    for label, values in rows:
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    # --- Per-action MSE breakdown ---
    lines.append("\n## Per-Action MSE Breakdown\n")
    action_names = {1: "Move+", 2: "Move-", 3: "Stay/Hold"}
    has_action_data = any(
        data[n]["report"] and any(f"val_mse_action_{a}" in data[n]["report"] for a in action_names)
        for n in EXP_NAMES
    )

    if has_action_data:
        lines.append("| Action | " + " | ".join(EXP_LABELS[n] for n in EXP_NAMES) + " |")
        lines.append("|---|" + "|".join(["---"] * len(EXP_NAMES)) + "|")
        for action_id, action_name in action_names.items():
            values = []
            for exp_name in EXP_NAMES:
                report = data[exp_name]["report"]
                key = f"val_mse_action_{action_id}"
                values.append(fmt_val(report.get(key) if report else None, 6))
            lines.append(f"| {action_name} | " + " | ".join(values) + " |")
    else:
        lines.append("*No per-action MSE data available.*\n")

    # --- Training Convergence Analysis ---
    lines.append("\n## Convergence Analysis\n")
    for exp_name in EXP_NAMES:
        timing = data[exp_name]["timing"]
        if not timing:
            continue

        label = EXP_LABELS[exp_name]
        val_hist = timing.get("val_loss_history", [])
        train_hist = timing.get("train_loss_history", [])
        best_ep = timing.get("best_epoch", 0)
        total = timing.get("num_epochs_run", 0)
        best_val = timing.get("best_val_loss", None)

        lines.append(f"### {label}")
        lines.append(f"- Epochs run: {total}")
        lines.append(f"- Best epoch: {best_ep}")
        if best_val is not None:
            lines.append(f"- Best val loss: {best_val:.6f}")
        if val_hist:
            final_val = val_hist[-1]
            overfitting_ratio = final_val / best_val if best_val and best_val > 0 else 1.0
            lines.append(f"- Final val loss: {final_val:.6f}")
            lines.append(f"- Overfitting ratio (final/best): {overfitting_ratio:.3f}")
            # Check convergence: was val loss still improving in last 10%?
            tail = val_hist[max(0, len(val_hist) - len(val_hist) // 10):]
            if len(tail) >= 2 and tail[-1] < tail[0]:
                lines.append(f"- **Still improving** in final 10% of training")
            elif len(tail) >= 2:
                lines.append(f"- Plateaued or diverging in final 10%")
        if train_hist and val_hist:
            final_gap = train_hist[-1] - val_hist[-1]
            lines.append(f"- Final train-val gap: {final_gap:.6f}")
        lines.append("")

    # --- Fine-tune vs Scratch Comparison ---
    lines.append("## Fine-tune vs Scratch Summary\n")

    for model_type, ft_name, scratch_name in [("GPT", "gpt_finetune", "gpt_scratch"),
                                                ("Diffusion", "diff_finetune", "diff_scratch")]:
        ft_timing = data[ft_name]["timing"]
        sc_timing = data[scratch_name]["timing"]

        if ft_timing and sc_timing:
            ft_best = ft_timing.get("best_val_loss")
            sc_best = sc_timing.get("best_val_loss")
            ft_plateau = ft_timing.get("time_to_plateau_seconds", 0)
            sc_plateau = sc_timing.get("time_to_plateau_seconds", 0)

            lines.append(f"### {model_type}")
            if ft_best and sc_best:
                improvement = (sc_best - ft_best) / sc_best * 100
                lines.append(f"- Fine-tune val loss: {ft_best:.6f} vs Scratch: {sc_best:.6f} ({improvement:+.1f}%)")
            if ft_plateau and sc_plateau and sc_plateau > 0:
                speedup = sc_plateau / ft_plateau
                lines.append(f"- Time to plateau: {fmt_time(ft_plateau)} (FT) vs {fmt_time(sc_plateau)} (scratch), {speedup:.1f}x speedup")
            lines.append("")

    # --- Iteration History (if multiple iterations) ---
    if len(all_iterations) > 1:
        lines.append("## Iteration History\n")
        lines.append("| Iter | Experiment | Val MSE | SSIM | Training Time | Config Changes |")
        lines.append("|---|---|---|---|---|---|")
        for it in all_iterations:
            for exp_name in EXP_NAMES:
                ckpt_dir = Path(experiment_dir) / f"iter{it}" / exp_name
                eval_exp_dir = Path(eval_dir) / f"iter{it}" / exp_name
                t = load_json(ckpt_dir / "timing.json")
                r = load_json(eval_exp_dir / "report.json")
                val_mse = fmt_val(r.get("val_mse_visual") if r else None, 6)
                ssim = fmt_val(r.get("ssim_mean") if r else None, 4)
                train_time = fmt_time(t.get("total_training_seconds") if t else None)
                lines.append(f"| {it} | {EXP_LABELS[exp_name]} | {val_mse} | {ssim} | {train_time} | |")

    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Generate hold experiment comparison report")
    p.add_argument("--experiment-dir", type=str, required=True)
    p.add_argument("--eval-dir", type=str, required=True)
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    # Discover all iterations
    exp_dir = Path(args.experiment_dir)
    all_iterations = sorted([
        int(d.name.replace("iter", ""))
        for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("iter")
    ]) if exp_dir.exists() else [args.iteration]

    report = generate_report(
        args.experiment_dir, args.eval_dir, args.iteration,
        all_iterations=all_iterations,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {args.output}")
    print(f"Iterations covered: {all_iterations}")


if __name__ == "__main__":
    main()
