"""
compute_intersection_metrics.py

Recomputes metrics on the *intersection* of valid entries across all
(method × seed) combinations at each step independently.  At each step,
a shared mask is built from entries valid for every method and seed at
that step, so every line in the figure at a given x-axis point uses the
same set of examples.  N may differ across steps.

Intersection scope:
  - Per-step mask: valid in ALL (method × seed) combos at that step.
  - Base step uses a per-method single preds file (no seed subdirectory).
  - general_reward score is always defined, so it is averaged over all
    valid (non-NaN) entries without intersection.

Saves results as:
  intersection_{monitor_slug}_monitor-{judge_slug}_llm_judge_metrics.json
alongside the regular per-method metrics files (one file per method/seed/step).

Usage:
  python compute_intersection_metrics.py \\
    --result_dir outputs_eval/general_reward/Qwen3-0.6B \\
    --dataset_name general_reward \\
    --methods baseline_all_tokens,baseline_cot_only,OURS_self \\
    --seeds seed_0,seed_1,seed_2 \\
    --max_step_num 6 --step_size 120 --viz_offset 200 \\
    --monitor_model_name gpt-4o-mini \\
    --judge_model_name gpt-4.1-mini
"""

import argparse
import json
import os

import numpy as np

from verl.utils.reward_score.BaseVerifier import get_verifier
from verl.utils.reward_score.GeneralRewardVerifier import BEHAVIOR_TO_MONITOR


def _sanitize(name: str) -> str:
    return name.replace("/", "_")


def _preds_path(result_dir: str, method: str, step_idx, seed, dataset_split_name: str = "test") -> str:
    """Return the path to a method's preds.jsonl for a given (step_idx, seed)."""
    if step_idx == "base":
        return os.path.join(result_dir, method, "step_base", dataset_split_name, "main", "preds.jsonl")
    return os.path.join(result_dir, method, f"step_{step_idx}", seed, dataset_split_name, "main", "preds.jsonl")


def _output_dir(result_dir: str, method: str, step_idx, seed, dataset_split_name: str = "test") -> str:
    """Return the directory where intersection metrics should be saved."""
    if step_idx == "base":
        return os.path.join(result_dir, method, "step_base", dataset_split_name, "main")
    return os.path.join(result_dir, method, f"step_{step_idx}", seed, dataset_split_name, "main")


def compute_intersection_metrics(
    methods_list,
    seeds_list,
    all_step_idx_list,
    result_dir,
    dataset_name,
    monitor_model_name,
    judge_model_name,
    max_new_tokens=None,
    dataset_split_name: str = "test",
):
    monitor_key  = f"{_sanitize(monitor_model_name)}_monitor_score"
    judge_key    = f"{_sanitize(judge_model_name)}_llm_judge_score"
    base_metrics_file = (
        f"{_sanitize(monitor_model_name)}_monitor"
        f"-{_sanitize(judge_model_name)}_llm_judge_metrics.json"
    )
    intersection_file = f"intersection_{base_metrics_file}"

    is_general_reward = dataset_name == "general_reward"
    behavior_keys = (
        BEHAVIOR_TO_MONITOR
        if is_general_reward
        else [dataset_name]
    )

    # ── Phase 1: Load ALL preds (all methods × seeds × steps) ───── #
    # key: (method, seed, step_idx)  — seed=None for base step
    print("\n[intersection] Loading all preds ...")
    all_entries: dict = {}
    for step_idx in all_step_idx_list:
        seeds_for_step = [None] if step_idx == "base" else seeds_list
        for method in methods_list:
            for seed in seeds_for_step:
                fn = _preds_path(result_dir, method, step_idx, seed, dataset_split_name)
                if not os.path.exists(fn):
                    raise ValueError(f"preds not found: {fn}")
                with open(fn) as f:
                    entries = [json.loads(line) for line in f]
                all_entries[(method, seed, step_idx)] = entries
                assert len(entries) == 500, f"  [error_loaded] {fn}  ({len(entries)} entries)"

    n = len(next(iter(all_entries.values())))
    combo_metrics: dict = {key: {} for key in all_entries}

    # ── Phase 2: Per-behavior per-step shared mask ───────────────── #
    # At each step, intersect valid entries across all (method × seed).
    for bk in behavior_keys:
        b_monitor_key = f"{bk}_{monitor_key}"
        b_judge_key   = f"{bk}_{judge_key}"
        verifier = get_verifier(data_source=bk, max_new_tokens=max_new_tokens)
        invalid  = verifier.invalid_score

        m_scores: dict   = {}
        j_scores: dict   = {}
        both_valid: dict = {}
        for key, entries in all_entries.items():
            ms = np.array([e.get(b_monitor_key, invalid) for e in entries])
            js = np.array([e.get(b_judge_key,   invalid) for e in entries])
            m_scores[key]   = ms
            j_scores[key]   = js
            both_valid[key] = (ms != invalid) & (js != invalid)

        for step_idx in all_step_idx_list:
            seeds_for_step = [None] if step_idx == "base" else seeds_list
            # Shared mask: valid in ALL (method × seed) at this step
            shared = np.ones(n, dtype=bool)
            for method in methods_list:
                for seed in seeds_for_step:
                    key = (method, seed, step_idx)
                    n_valid = int(np.sum(both_valid[key]))
                    if n_valid == 0:
                        print(f"  WARNING: {bk} step={step_idx} method={method} seed={seed}: 0/{n} valid — key may be missing from preds.jsonl")
                    shared &= both_valid[key]
            n_shared = int(np.sum(shared))
            print(f"\n  {bk} step={step_idx}: {n_shared}/{n} shared across methods & seeds")

            for method in methods_list:
                for seed in seeds_for_step:
                    key = (method, seed, step_idx)
                    if n_shared == 0:
                        breakpoint()
                        b_metrics = {"total_valid_CoT_entries": int(np.sum(both_valid[key])),
                                     "total_monitored_entries": 0,
                                     "total_entries":           n}
                    elif n_shared < 200:
                        breakpoint()
                    else:
                        b_metrics = verifier.compute_metrics(
                            predictions=m_scores[key][shared].astype(float),
                            ground_truths=j_scores[key][shared].astype(float),
                        )
                        b_metrics["total_valid_CoT_entries"] = int(np.sum(both_valid[key]))
                        b_metrics["total_monitored_entries"] = n_shared
                        b_metrics["total_entries"]           = n
                    combo_metrics[key][bk] = b_metrics

    # ── Phase 3: general_reward score (no intersection — always defined) ── #
    if is_general_reward:
        gr_verifier = get_verifier(data_source="general_reward", max_new_tokens=max_new_tokens)
        gr_invalid  = gr_verifier.invalid_score
        for key, entries in all_entries.items():
            general_reward_arr = np.array(
                [e.get("general_reward_score", gr_invalid) for e in entries], dtype=float
            )
            valid_general = general_reward_arr != gr_invalid
            n_valid = int(np.sum(valid_general))
            combo_metrics[key]["general_reward"] = {
                "mean_score":          float(np.mean(general_reward_arr[valid_general])) if n_valid > 0 else None,
                "total_valid_entries": n_valid,
                "total_entries":       n,
            }

    # ── Phase 4: Save per-(method, seed, step) intersection metrics ─ #
    print("\n[intersection] Saving ...")
    for (method, seed, step_idx), metrics in combo_metrics.items():
        out_dir  = _output_dir(result_dir, method, step_idx, seed, dataset_split_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, intersection_file)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"  [saved]  {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Recompute metrics on the shared valid-entry intersection across methods."
    )
    ap.add_argument("--result_dir",         type=str, required=True,
                    help="Base dir at the {task}/{base_model} level, "
                         "e.g. outputs_eval/general_reward/Qwen3-0.6B")
    ap.add_argument("--dataset_name",       type=str, required=True)
    ap.add_argument("--methods",            type=str, required=True,
                    help="Comma-separated methods to intersect, "
                         "e.g. baseline_all_tokens,OURS_self")
    ap.add_argument("--seeds",              type=str, default="seed_0,seed_1,seed_2")
    ap.add_argument("--max_step_num",       type=int, required=True)
    ap.add_argument("--step_size",          type=int, required=True)
    ap.add_argument("--viz_offset",         type=int, default=0)
    ap.add_argument("--monitor_model_name",   type=str, required=True)
    ap.add_argument("--judge_model_name",     type=str, required=True)
    ap.add_argument("--max_new_tokens",       type=int, default=None)
    ap.add_argument("--dataset_split_name",   type=str, default="test",
                    help="Dataset split to read predictions from (e.g. test, dev).")

    args = ap.parse_args()

    all_step_idx = ["base"] + [
        str(args.viz_offset + (i + 1) * args.step_size)
        for i in range(args.max_step_num)
    ]

    compute_intersection_metrics(
        methods_list       = args.methods.split(","),
        seeds_list         = args.seeds.split(","),
        all_step_idx_list  = all_step_idx,
        result_dir         = args.result_dir,
        dataset_name       = args.dataset_name,
        monitor_model_name = args.monitor_model_name,
        judge_model_name   = args.judge_model_name,
        max_new_tokens     = args.max_new_tokens,
        dataset_split_name = args.dataset_split_name,
    )
