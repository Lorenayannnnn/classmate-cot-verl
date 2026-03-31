import json
import os


def generate_dev_icl_examples(
    monitor_model_name: str = "gpt-4o-mini",
    judge_model_name: str = "gpt-4.1-mini",
    base_dir: str = "outputs_eval/general_reward/Qwen3-0.6B",
    split: str = "dev",
):
    """Generate {behavior}_ICL_examples.json files from dev preds.jsonl files.

    For each preds.jsonl found under base_dir/.../dev/main/, sorts entries per
    behavior by the monitor score (ascending) and picks low/medium/high
    representatives, saving them to dev/{behavior}_ICL_examples.json alongside
    preds.jsonl's parent dev/ directory.
    """
    from verl.utils.reward_score.GeneralRewardVerifier import BEHAVIOR_TO_MONITOR

    monitor_score_suffix = monitor_model_name.replace("/", "_") + "_monitor_score"
    monitor_expl_suffix  = monitor_model_name.replace("/", "_") + "_monitor_explanation"
    judge_score_suffix   = judge_model_name.replace("/", "_") + "_llm_judge_score"
    judge_expl_suffix    = judge_model_name.replace("/", "_") + "_llm_judge_explanation"

    if not os.path.isdir(base_dir):
        print(f"[generate_dev_icl_examples] base_dir not found: {base_dir}")
        return

    for method in sorted(os.listdir(base_dir)):
        method_dir = os.path.join(base_dir, method)
        if not os.path.isdir(method_dir):
            continue

        for step_entry in sorted(os.listdir(method_dir)):
            step_dir = os.path.join(method_dir, step_entry)
            if not os.path.isdir(step_dir):
                continue

            if step_entry == "step_base":
                # step_base/{split}/main/preds.jsonl → ICL saved in step_base/{split}/
                split_dirs = [os.path.join(step_dir, split)]
            elif step_entry.startswith("step_"):
                # step_{N}/{seed}/{split}/main/preds.jsonl → ICL saved in step_{N}/{seed}/{split}/
                split_dirs = [
                    os.path.join(step_dir, seed_entry, split)
                    for seed_entry in sorted(os.listdir(step_dir))
                    if os.path.isdir(os.path.join(step_dir, seed_entry)) and seed_entry.startswith("seed_")
                ]
            else:
                continue

            for split_dir in split_dirs:
                preds_fn = os.path.join(split_dir, "main", "preds.jsonl")
                if not os.path.exists(preds_fn):
                    continue

                with open(preds_fn) as f:
                    entries = [json.loads(line) for line in f]

                for behavior in BEHAVIOR_TO_MONITOR:
                    b_monitor_score = f"{behavior}_{monitor_score_suffix}"
                    b_monitor_expl  = f"{behavior}_{monitor_expl_suffix}"
                    b_judge_score   = f"{behavior}_{judge_score_suffix}"
                    b_judge_expl    = f"{behavior}_{judge_expl_suffix}"

                    valid = [e for e in entries if e.get(b_monitor_score) is not None]
                    if len(valid) < 3:
                        print(f"[skip] {os.path.relpath(preds_fn)} / {behavior}: only {len(valid)} valid entries")
                        continue

                    # Filter to entries where judge score is also available,
                    # so we can measure monitor accuracy (|monitor - judge|).
                    accurate_valid = [e for e in valid if e.get(b_judge_score) is not None]
                    if len(accurate_valid) < 3:
                        print(f"[skip] {os.path.relpath(preds_fn)} / {behavior}: only {len(accurate_valid)} entries with both monitor and judge scores")
                        continue

                    # Selection: divide entries (sorted by monitor score) into low/medium/high
                    # thirds, then pick the entry with smallest |monitor - judge| error within
                    # each third.  This guarantees score-range coverage (via thirds) while
                    # favouring examples where the monitor's prediction was most accurate
                    # (within each range band), making them reliable ICL demonstrations.
                    accurate_valid.sort(key=lambda e: e[b_monitor_score])
                    n = len(accurate_valid)
                    thirds = [
                        accurate_valid[: n // 3],            # low scores
                        accurate_valid[n // 3: 2 * n // 3],  # medium scores
                        accurate_valid[2 * n // 3 :],        # high scores
                    ]
                    chosen = [
                        min(third, key=lambda e: abs(e[b_monitor_score] - e[b_judge_score]))
                        for third in thirds
                    ]

                    icl_examples = [
                        {
                            "user_message":      e["question"],
                            "main_CoT":          e.get("truncated_main_CoT", ""),
                            "main_output":       e.get("main_output", ""),
                            "score":             e[b_monitor_score],
                            "explanation":       e.get(b_monitor_expl, ""),
                            "judge_score":       e.get(b_judge_score),
                            "judge_explanation": e.get(b_judge_expl, ""),
                        }
                        for e in chosen
                    ]

                    out_path = os.path.join(split_dir, f"{behavior}_ICL_examples.json")
                    with open(out_path, "w") as f:
                        json.dump(icl_examples, f, indent=2)
                    print(f"[saved] {os.path.relpath(out_path)}")


if __name__ == "__main__":
    generate_dev_icl_examples()

#python src/analysis_module/generate_dev_icl_examples.py