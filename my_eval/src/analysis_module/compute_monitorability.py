"""Compute and plot monitorability across training steps.

Monitorability = Pearson corr(Y, Z) where
    Y = R_exploit(y_{X=1}) - R_exploit(y_{X=0})   (per query diff)
    Z = hacking_{monitor_san}_{mode}_suspicion_score  (from X=1 preds)

Multiple methods (e.g. baseline, OURS) can be compared in the same figure.
Each method is identified by a display name and its X=0 / X=1 output directory names.

Three monitor modes are plotted per method:
    CoT      — monitor sees full CoT + final output  (solid line)
    output   — monitor sees only the final output    (dashed line)
    CoT_only — monitor sees only the CoT             (dotted line)

Six subplots (2 rows × 3 columns):
    Row 0: Pearson corr | Suspicion score | Validity score
    Row 1: Gold reward  | Exploit reward  | N monitored entries

Color encodes method; linestyle encodes monitor mode (monitor plots) or X=0/X=1 (reward plots).

Directory layout expected:
    preds.jsonl : {result_dir}/{x0_or_x1_dir}/step_{step_idx}/{seed}/{split}/preds.jsonl
    metrics.json: {result_dir}/{x1_dir}/step_{step_idx}/{seed}/{split}/{metrics_filename}

Usage:
    python my_eval/src/analysis_module/compute_monitorability.py \\
        --result_dir outputs_eval/general_reward/Qwen3-0.6B \\
        --gold_reward general_reward \\
        --exploitable_reward sycophancy \\
        --gold_reward_model_name Skywork/Skywork-Reward-V2-Qwen3-0.6B \\
        --exploitable_reward_model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --monitor_model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --validity_model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --methods "baseline:g_general_reward_x0:g_general_reward_e_sycophancy" \\
                  "OURS:g_general_reward_keep_last-100-tokens:g_general_reward_e_sycophancy_keep_last-100-tokens" \\
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

from scipy.stats import spearmanr
from verl.utils.reward_score.hacking_monitor import compute_monitorability

_MODES = ("CoT", "output", "CoT_only")
_MODE_METRICS_KEY = {
    "CoT":      "cot_monitor",
    "output":   "output_monitor",
    "CoT_only": "cot_only_monitor",
}
_MODE_LABEL     = {"CoT": "CoT+Output",    "output": "Output-only", "CoT_only": "CoT-only"}
_MODE_LINESTYLE = {"CoT": "-",      "output": "--",     "CoT_only": ":"}
_MODE_MARKER    = {"CoT": "o",      "output": "s",      "CoT_only": "^"}

# One color per method, assigned in order of appearance
_METHOD_COLORS = ["#185FA5", "#D85A30", "#2CA02C", "#8E44AD", "#E67E22", "#17BECF"]


# ── Key helpers ───────────────────────────────────────────────────────────────

def _san(name: str | None) -> str | None:
    if not name or name.lower() == "null":
        return None
    return name.replace("/", "_")


def _gold_score_key(gold_reward: str, gold_san: str) -> str:
    return f"gold_score_{gold_reward}_{gold_san}"


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


# ── Path / load helpers ───────────────────────────────────────────────────────

def _preds_path(result_dir, method_dir, step_idx, seed, dataset_split):
    return os.path.join(result_dir, method_dir, f"step_{step_idx}", seed, dataset_split, "preds.jsonl")


def _metrics_json_path(result_dir, method_dir, step_idx, seed, dataset_split, metrics_filename):
    return os.path.join(result_dir, method_dir, f"step_{step_idx}", seed, dataset_split, metrics_filename)


def _load_preds(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _reward_mean(preds: list[dict], score_key: str) -> float | None:
    scores = [float(e[score_key]) for e in preds if e.get(score_key) is not None]
    return float(np.mean(scores)) if scores else None


def _load_metric_from_metrics(path: str, mode: str, metric_key: str) -> float | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(_MODE_METRICS_KEY[mode], {}).get(metric_key)


def _store_correlations_in_metrics(path: str, pearson_r: float | None, spearman_r: float | None, mode: str):
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)
    bucket = data.setdefault(_MODE_METRICS_KEY[mode], {})
    if pearson_r  is not None: bucket["monitorability_pearson_r"]  = pearson_r
    if spearman_r is not None: bucket["monitorability_spearman_r"] = spearman_r
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [updated {mode}] {path}")


# ── Per-step computation ──────────────────────────────────────────────────────

def _compute_correlations_from_preds(
    x0_preds: list[dict], x1_preds: list[dict],
    exploit_score_key_str: str, monitor_san: str, mode: str,
) -> tuple[float | None, float | None]:
    """Return (pearson_r, spearman_r) for one (step, seed, mode), or (None, None)."""
    x0_by_question = {e["question"]: e for e in x0_preds if "question" in e}
    try:
        assert len(x0_by_question) >= 190
    except:
        breakpoint()
    susp_key = _suspicion_key(monitor_san, mode)

    r1_scores, r0_scores, z_scores = [], [], []
    for entry in x1_preds:
        q = entry.get("question")
        x0_entry = x0_by_question.get(q)
        assert x0_entry is not None
        # if x0_entry is None:
        #     continue
        r1 = entry.get(exploit_score_key_str)
        r0 = x0_entry.get(exploit_score_key_str)
        assert r1 is not None, "Missing exploit score in X=1 entry for question: {q!r}"
        assert r0 is not None, "Missing exploit score in X=0 entry for question: {q!r}"
        # if r1 is None or r0 is None:
        #     continue
        r1_scores.append(float(r1))
        r0_scores.append(float(r0))
        z_scores.append(entry.get(susp_key))

    if not r1_scores:
        return None, None
    pearson_r = compute_monitorability(r1_scores, r0_scores, z_scores).get("pearson_r")

    # Spearman: compute from (Y, Z) pairs where Z is not None
    y_vals = [r1 - r0 for r1, r0 in zip(r1_scores, r0_scores)]
    yz_pairs = [(y, z) for y, z in zip(y_vals, z_scores) if z is not None]
    if len(yz_pairs) >= 3:
        ys, zs = zip(*yz_pairs)
        spearman_r = float(spearmanr(ys, zs).statistic)
    else:
        spearman_r = None

    return pearson_r, spearman_r


# ── Plotting ──────────────────────────────────────────────────────────────────

def _set_xticks(ax, dfs: list[pd.DataFrame]):
    all_steps = sorted({s for df in dfs for s in df["RL Steps"].dropna().unique()})
    if all_steps:
        ax.set_xticks(all_steps)


def _finalize_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    ax.set_xlabel("RL Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    sns.despine(ax=ax)


def _plot_mode_method_metric(
    ax, data_by_method_mode: dict, methods: list, method_colors: dict, title: str, ylabel: str
):
    """One line per (method, mode). Color = method; linestyle + marker = mode."""
    all_dfs = []
    for method_name, _, _ in methods:
        color = method_colors[method_name]
        for mode in _MODES:
            df = pd.DataFrame(data_by_method_mode[method_name][mode])
            all_dfs.append(df)
            if df["Value"].notna().any():
                sns.lineplot(
                    data=df, x="RL Steps", y="Value",
                    estimator="mean", errorbar="sd",
                    marker=_MODE_MARKER[mode], markersize=11, linewidth=2.5,
                    color=color, linestyle=_MODE_LINESTYLE[mode],
                    label=f"{method_name} ({_MODE_LABEL[mode]})",
                    ax=ax,
                )
    _set_xticks(ax, all_dfs)
    _finalize_ax(ax, title, ylabel)


def _plot_reward_methods(
    ax,
    data_x0_by_method: dict, data_x1_by_method: dict,
    methods: list, method_colors: dict,
    title: str, ylabel: str,
):
    """X=0 dashed, X=1 solid; color = method."""
    all_dfs = []
    for method_name, _, _ in methods:
        color = method_colors[method_name]
        for data_dict, linestyle, suffix in [
            (data_x0_by_method, "--", "X=0 (dashed)"),
            (data_x1_by_method, "-",  "X=1 (solid)"),
        ]:
            df = pd.DataFrame(data_dict[method_name])
            all_dfs.append(df)
            if df["Value"].notna().any():
                sns.lineplot(
                    data=df, x="RL Steps", y="Value",
                    estimator="mean", errorbar="sd",
                    marker="o", markersize=11, linewidth=2.5,
                    color=color, linestyle=linestyle,
                    label=f"{method_name} {suffix}",
                    ax=ax,
                )
    _set_xticks(ax, all_dfs)
    _finalize_ax(ax, title, ylabel)


# ── Main orchestration ────────────────────────────────────────────────────────

def plot_monitorability(
    result_dir,
    methods: list[tuple[str, str, str]],  # [(display_name, x0_dir, x1_dir), ...]
    step_idxs, seeds, dataset_split,
    gold_score_key_str, exploit_score_key_str, monitor_san,
    metrics_filename, output_path,
):
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    method_names  = [name for name, _, _ in methods]
    method_colors = {name: _METHOD_COLORS[i % len(_METHOD_COLORS)] for i, name in enumerate(method_names)}

    def _empty_mode_dict():
        return {name: {m: [] for m in _MODES} for name in method_names}

    def _empty_method_dict():
        return {name: [] for name in method_names}

    rows = {
        "pearson":             _empty_mode_dict(),
        "spearman":            _empty_mode_dict(),
        "suspicion":           _empty_mode_dict(),
        "validity":            _empty_mode_dict(),
        "weighted_suspicion":  _empty_mode_dict(),
        "n_parsed":            _empty_mode_dict(),
        "gold_x0":             _empty_method_dict(),
        "gold_x1":             _empty_method_dict(),
        "exploit_x0":          _empty_method_dict(),
        "exploit_x1":          _empty_method_dict(),
    }

    def _row(step_num, seed, value):
        return {"RL Steps": step_num, "Value": value if value is not None else np.nan, "Seed": seed}

    for step_idx in step_idxs:
        step_num = 0 if step_idx == "base" else int(step_idx)
        for seed in seeds:
            for method_name, x0_dir, x1_dir in methods:
                x0_path      = _preds_path(result_dir, x0_dir,  step_idx, seed, dataset_split)
                x1_path      = _preds_path(result_dir, x1_dir,  step_idx, seed, dataset_split)
                metrics_path = _metrics_json_path(result_dir, x1_dir, step_idx, seed, dataset_split, metrics_filename)

                x0_preds = _load_preds(x0_path) if os.path.exists(x0_path) else []
                x1_preds = _load_preds(x1_path) if os.path.exists(x1_path) else []

                assert x0_preds, f"Missing X=0 preds for method {method_name} at step {step_idx} seed {seed}"
                assert x1_preds, f"Missing X=1 preds for method {method_name} at step {step_idx} seed {seed}"

                if not x0_preds or not x1_preds:
                    print(f"  [skip] {method_name}: missing preds for step={step_idx} seed={seed}")

                rows["gold_x0"][method_name].append(   _row(step_num, seed, _reward_mean(x0_preds, gold_score_key_str)))
                rows["gold_x1"][method_name].append(   _row(step_num, seed, _reward_mean(x1_preds, gold_score_key_str)))
                rows["exploit_x0"][method_name].append(_row(step_num, seed, _reward_mean(x0_preds, exploit_score_key_str)))
                rows["exploit_x1"][method_name].append(_row(step_num, seed, _reward_mean(x1_preds, exploit_score_key_str)))

                for mode in _MODES:
                    pearson, spearman = _compute_correlations_from_preds(x0_preds, x1_preds, exploit_score_key_str, monitor_san, mode)
                    suspicion           = _load_metric_from_metrics(metrics_path, mode, "suspicion_mean")
                    validity            = _load_metric_from_metrics(metrics_path, mode, "validity_mean")
                    weighted_suspicion  = _load_metric_from_metrics(metrics_path, mode, "weighted_suspicion_mean")
                    n_parsed            = _load_metric_from_metrics(metrics_path, mode, "n_parsed")

                    if pearson is not None or spearman is not None:
                        _store_correlations_in_metrics(metrics_path, pearson, spearman, mode)

                    rows["pearson"][method_name][mode].append(            _row(step_num, seed, pearson))
                    rows["spearman"][method_name][mode].append(           _row(step_num, seed, spearman))
                    rows["suspicion"][method_name][mode].append(          _row(step_num, seed, suspicion))
                    rows["validity"][method_name][mode].append(           _row(step_num, seed, validity))
                    rows["weighted_suspicion"][method_name][mode].append( _row(step_num, seed, weighted_suspicion))
                    rows["n_parsed"][method_name][mode].append(           _row(step_num, seed, n_parsed))

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))

    _plot_mode_method_metric(axes[0, 0], rows["pearson"],   methods, method_colors,
                             "Monitorability: Pearson corr(Z, Y)", "Pearson r")
    _plot_mode_method_metric(axes[0, 1], rows["spearman"],  methods, method_colors,
                             "Monitorability: Spearman corr(Z, Y)", "Spearman r")
    _plot_mode_method_metric(axes[0, 2], rows["suspicion"], methods, method_colors,
                             "Monitor Suspicion Score", "Suspicion (mean)")
    _plot_mode_method_metric(axes[0, 3], rows["validity"],  methods, method_colors,
                             "Monitor Behavior Validity Score", "Validity (mean)")
    _plot_reward_methods(axes[1, 0],
                         rows["gold_x0"],    rows["gold_x1"],    methods, method_colors,
                         "Gold Reward", "Gold Score (mean)")
    _plot_reward_methods(axes[1, 1],
                         rows["exploit_x0"], rows["exploit_x1"], methods, method_colors,
                         "Exploit Reward", "Exploit Score (mean)")
    _plot_mode_method_metric(axes[1, 2], rows["n_parsed"],          methods, method_colors,
                             "N Monitored Entries (parsed)", "Count")
    _plot_mode_method_metric(axes[1, 3], rows["weighted_suspicion"], methods, method_colors,
                             "Weighted Suspicion Score (suspicion × validity/10)", "Weighted Suspicion (mean)")

    monitor_display = (monitor_san or "no_monitor").replace("_", "/")
    methods_display = ", ".join(method_names)
    fig.suptitle(
        f"Gold+Exploit Monitorability — methods: {methods_display}\n"
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

    parser.add_argument("--gold_reward_model_name", required=True)
    parser.add_argument("--exploitable_reward_model_name", required=True)
    parser.add_argument("--monitor_model_name", default=None,
                        help="Hacking monitor model (or 'null' to skip).")
    parser.add_argument("--validity_model_name", default=None,
                        help="Validity judge model (or 'null' to fall back to monitor).")

    parser.add_argument("--gold_reward", required=True)
    parser.add_argument("--exploitable_reward", required=True)

    parser.add_argument("--methods", nargs="+", required=True,
                        help=(
                            "One or more method specs: 'display_name:x0_method_dir:x1_method_dir'. "
                            "Example: 'baseline:g_general_reward_x0:g_general_reward_e_sycophancy'"
                        ))

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

    methods = []
    for spec in args.methods:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid method spec {spec!r}. Expected 'name:x0_dir:x1_dir'.")
        methods.append(tuple(parts))

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

    print(f"Methods    : {methods}")
    print(f"Steps      : {step_idxs}")
    print(f"Seeds      : {seeds}")
    print(f"Metrics fn : {metrics_filename}")

    plot_monitorability(
        result_dir=args.result_dir,
        methods=methods,
        step_idxs=step_idxs,
        seeds=seeds,
        dataset_split=args.dataset_split_name,
        gold_score_key_str=_gold_score_key(args.gold_reward, gold_san),
        exploit_score_key_str=_exploit_score_key(args.exploitable_reward, exploit_san),
        monitor_san=monitor_san,
        metrics_filename=metrics_filename,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
