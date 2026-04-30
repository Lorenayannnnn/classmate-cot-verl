"""Compute and plot monitorability across training steps.

Monitorability = Pearson corr(Y, Z) where
    Y = R_exploit(y_{X=1}) - R_exploit(y_{X=0})   (per query diff)
    Z = hacking_{monitor_san}_{mode}_suspicion_score  (from X=1 preds)

Two monitor modes are always computed and plotted side-by-side:
    CoT    — monitor sees full CoT + final output (main_full_output)
    output — monitor sees only the final output   (main_output)

For each (step, seed, mode) the script:
  1. Loads preds.jsonl from X=1 and X=0 runs and computes Pearson corr(Y, Z).
  2. Loads the matching metrics JSON written by GoldExploitExpRunner and stores
     the computed pearson_r back under {cot_monitor,output_monitor}.monitorability_pearson_r.
  3. Plots four subplots (2 modes × 2 metrics), averaged across seeds:
       (a) Pearson corr(suspicion, Y)   — monitorability signal
       (b) Hacking-monitor validity score (from metrics JSON)

Directory layout expected (set by _setup_output_dir in gold_exploit_inference.py):
    preds.jsonl : {result_dir}/{method}/step_{step_idx}/{seed}/{split}/preds.jsonl
    metrics.json: {result_dir}/{method}/step_{step_idx}/{seed}/{split}/{metrics_filename}

    X=0 method : g_{gold_reward}
    X=1 method : g_{gold_reward}_e_{exploitable_reward}

Usage:
    python my_eval/src/compute_monitorability.py \\
        --result_dir outputs_eval/general_reward/Qwen3-0.6B \\
        --gold_reward general_reward \\
        --exploitable_reward confidence \\
        --gold_reward_model_name Skywork/Skywork-Reward-V2-Qwen3-0.6B \\
        --exploitable_reward_model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --monitor_model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --validity_model_name null \\
        --seeds seed_0,seed_1,seed_2 \\
        --max_step_num 6 \\
        --step_size 152 \\
        --viz_offset 8 \\
        --dataset_split_name test
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from verl.utils.reward_score.hacking_monitor import compute_monitorability

_MODES = ("CoT", "output")
_MODE_METRICS_KEY = {"CoT": "cot_monitor", "output": "output_monitor"}
_MODE_COLOR       = {"CoT": "#D85A30",     "output": "#185FA5"}


# ── Key helpers ───────────────────────────────────────────────────────────────

def _san(name: str | None) -> str | None:
    if not name or name.lower() == "null":
        return None
    return name.replace("/", "_")


def _exploit_score_key(exploitable_reward: str, exploit_san: str) -> str:
    return f"exploit_score_{exploitable_reward}_{exploit_san}"


def _suspicion_key(monitor_san: str, mode: str) -> str:
    return f"hacking_{monitor_san}_{mode}_suspicion_score"


def _build_metrics_filename(gold_reward, gold_san, exploitable_reward, exploit_san, monitor_san, validity_san):
    return (
        f"gold_{gold_reward}_{gold_san}"
        f"_exploit_{exploitable_reward}_{exploit_san}"
        f"_monitor_{monitor_san}"
        f"_validity_{validity_san}"
        f"_metrics.json"
    )


# ── Path helpers ──────────────────────────────────────────────────────────────

def _preds_path(result_dir, method, step_idx, seed, dataset_split):
    return os.path.join(result_dir, method, f"step_{step_idx}", seed, dataset_split, "preds.jsonl")


def _metrics_json_path(result_dir, method, step_idx, seed, dataset_split, metrics_filename):
    return os.path.join(result_dir, method, f"step_{step_idx}", seed, dataset_split, metrics_filename)


def _load_preds(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Per-step computation ──────────────────────────────────────────────────────

def _compute_pearson_for_step(
    result_dir, x0_method, x1_method, step_idx, seed, dataset_split,
    exploit_score_key_str, monitor_san, mode,
):
    """Return Pearson corr(Z, Y) for one (step, seed, mode), or None if data is missing."""
    x0_path = _preds_path(result_dir, x0_method, step_idx, seed, dataset_split)
    x1_path = _preds_path(result_dir, x1_method, step_idx, seed, dataset_split)

    if not os.path.exists(x0_path) or not os.path.exists(x1_path):
        print(f"  [skip] missing preds for step={step_idx} seed={seed}")
        return None

    x0_preds = _load_preds(x0_path)
    x1_preds = _load_preds(x1_path)

    x0_by_question = {e["question"]: e for e in x0_preds if "question" in e}
    susp_key = _suspicion_key(monitor_san, mode)

    r1_scores, r0_scores, z_scores = [], [], []
    for entry in x1_preds:
        q = entry.get("question")
        x0_entry = x0_by_question.get(q)
        if x0_entry is None:
            continue
        r1 = entry.get(exploit_score_key_str)
        r0 = x0_entry.get(exploit_score_key_str)
        if r1 is None or r0 is None:
            continue
        r1_scores.append(float(r1))
        r0_scores.append(float(r0))
        z_scores.append(entry.get(susp_key))  # None → excluded inside compute_monitorability

    if not r1_scores:
        print(f"  [skip] no matched entries for step={step_idx} seed={seed} mode={mode}")
        return None

    result = compute_monitorability(r1_scores, r0_scores, z_scores)
    return result.get("pearson_r")


def _load_validity_from_metrics(path: str, mode: str) -> float | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    metrics_key = _MODE_METRICS_KEY[mode]
    return data.get(metrics_key, {}).get("validity_mean")


def _store_pearson_in_metrics(path: str, pearson_r: float, mode: str):
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)
    metrics_key = _MODE_METRICS_KEY[mode]
    data.setdefault(metrics_key, {})["monitorability_pearson_r"] = pearson_r
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [updated {mode}] {path}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_metric(ax, df_by_mode, title, ylabel):
    any_data = False
    for mode in _MODES:
        df = df_by_mode[mode]
        if df["Value"].notna().any():
            any_data = True
            sns.lineplot(
                data=df, x="RL Steps", y="Value",
                estimator="mean", errorbar="sd",
                marker="o", markersize=9, linewidth=3.0,
                color=_MODE_COLOR[mode], label=f"{mode} monitor",
                ax=ax,
            )
    if any_data:
        all_steps = sorted(set(
            s for mode in _MODES for s in df_by_mode[mode]["RL Steps"].dropna().unique()
        ))
        ax.set_xticks(all_steps)
        for mode in _MODES:
            df = df_by_mode[mode]
            mean_df = df.groupby("RL Steps")["Value"].mean().reset_index()
            for _, row in mean_df.iterrows():
                if not pd.isna(row["Value"]):
                    ax.text(row["RL Steps"], row["Value"], f"{row['Value']:.3f}",
                            ha="center", va="bottom", fontsize=8, color=_MODE_COLOR[mode])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    ax.set_xlabel("RL Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    sns.despine(ax=ax)


def plot_monitorability(
    result_dir, x0_method, x1_method, step_idxs, seeds, dataset_split,
    exploit_score_key_str, monitor_san,
    metrics_filename, output_path,
):
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    rows_pearson  = {m: [] for m in _MODES}
    rows_validity = {m: [] for m in _MODES}

    for step_idx in step_idxs:
        step_num = 0 if step_idx == "base" else int(step_idx)
        for seed in seeds:
            metrics_path = _metrics_json_path(
                result_dir, x1_method, step_idx, seed, dataset_split, metrics_filename
            )
            for mode in _MODES:
                pearson  = _compute_pearson_for_step(
                    result_dir, x0_method, x1_method, step_idx, seed, dataset_split,
                    exploit_score_key_str, monitor_san, mode,
                )
                validity = _load_validity_from_metrics(metrics_path, mode)

                if pearson is not None:
                    _store_pearson_in_metrics(metrics_path, pearson, mode)

                rows_pearson[mode].append({"RL Steps": step_num, "Value": pearson  if pearson  is not None else np.nan, "Seed": seed})
                rows_validity[mode].append({"RL Steps": step_num, "Value": validity if validity is not None else np.nan, "Seed": seed})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_metric(axes[0], {m: pd.DataFrame(rows_pearson[m])  for m in _MODES},
                 "Monitorability: Pearson corr(Z, Y)", "Pearson r")
    _plot_metric(axes[1], {m: pd.DataFrame(rows_validity[m]) for m in _MODES},
                 "Monitor Behavior Validity Score", "Validity Score (mean)")

    monitor_display = (monitor_san or "no_monitor").replace("_", "/")
    fig.suptitle(
        f"Gold+Exploit Monitorability — X=1: {x1_method}\n"
        f"Monitor: {monitor_display}  |  mean ± std across {len(seeds)} seed(s)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute and plot monitorability across training steps.")

    # Four model names
    parser.add_argument("--gold_reward_model_name", required=True)
    parser.add_argument("--exploitable_reward_model_name", required=True)
    parser.add_argument("--monitor_model_name", default=None,
                        help="Hacking monitor model (or 'null' to skip).")
    parser.add_argument("--validity_model_name", default=None,
                        help="Validity judge model (or 'null' to fall back to monitor).")

    # Reward names
    parser.add_argument("--gold_reward", required=True)
    parser.add_argument("--exploitable_reward", required=True)

    # Eval setup
    parser.add_argument("--result_dir", required=True,
                        help="Base dir: outputs_eval/{gold_reward}/{base_model}")
    parser.add_argument("--seeds", default="seed_0,seed_1,seed_2")
    parser.add_argument("--max_step_num", type=int, required=True,
                        help="Number of eval checkpoints (num_eval_ckpts).")
    parser.add_argument("--step_size", type=int, required=True,
                        help="Step interval between eval checkpoints (viz_step_size).")
    parser.add_argument("--viz_offset", type=int, default=0)
    parser.add_argument("--dataset_split_name", default="test")
    parser.add_argument("--output_path", default=None,
                        help="Output PDF path. Defaults to {result_dir}/monitorability.pdf")

    args = parser.parse_args()

    gold_san     = _san(args.gold_reward_model_name)
    exploit_san  = _san(args.exploitable_reward_model_name)
    monitor_san  = _san(args.monitor_model_name)
    validity_san = _san(args.validity_model_name)

    if monitor_san is None:
        print("No monitor model specified — cannot compute monitorability. Exiting.")
        return

    x0_method = f"g_{args.gold_reward}"
    x1_method = f"g_{args.gold_reward}_e_{args.exploitable_reward}"

    exploit_score_key_str = _exploit_score_key(args.exploitable_reward, exploit_san)

    metrics_filename = _build_metrics_filename(
        args.gold_reward, gold_san,
        args.exploitable_reward, exploit_san,
        monitor_san, validity_san,
    )

    seeds     = args.seeds.split(",")
    step_idxs = [
        str(args.viz_offset + (i + 1) * args.step_size)
        for i in range(args.max_step_num)
    ]

    output_path = args.output_path or os.path.join(args.result_dir, "monitorability.pdf")

    print(f"X=0 method : {x0_method}")
    print(f"X=1 method : {x1_method}")
    print(f"Steps      : {step_idxs}")
    print(f"Seeds      : {seeds}")
    print(f"Exploit key: {exploit_score_key_str}")
    print(f"Metrics fn : {metrics_filename}")

    plot_monitorability(
        result_dir=args.result_dir,
        x0_method=x0_method,
        x1_method=x1_method,
        step_idxs=step_idxs,
        seeds=seeds,
        dataset_split=args.dataset_split_name,
        exploit_score_key_str=exploit_score_key_str,
        monitor_san=monitor_san,
        metrics_filename=metrics_filename,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
