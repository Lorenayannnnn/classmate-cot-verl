import json
import os


def generate_dev_icl_examples(
    monitor_model_name: str = "gpt-4o-mini",
    judge_model_name: str = "gpt-4.1-mini",
    base_dir: str = "outputs_eval/general_reward/Qwen3-0.6B",
    split: str = "dev",
    behaviors: list | None = None,
    use_dynamic_icl: bool = False,
    no_explanation: bool = False,
):
    """Generate {behavior}_ICL_examples.json files from dev preds.jsonl files.

    For each preds.jsonl found under base_dir/.../dev/main/, sorts entries per
    behavior by the monitor score (ascending) and picks low/medium/high
    representatives, saving them to dev/{behavior}_ICL_examples{_monitor_suffix}.json
    alongside preds.jsonl's parent dev/ directory.

    Args:
        behaviors: List of behavior keys to generate ICL examples for.
            Defaults to BEHAVIOR_TO_MONITOR (general_reward behaviors).
            Pass ["anthropic_sycophancy"] for that dataset.
        use_dynamic_icl: Whether to read monitor scores from the dynamic-ICL variant
            (keys suffixed with _use_dynamic_icl).
        no_explanation: Whether to read monitor scores from the no-explanation variant
            (keys suffixed with _no_expl).
    """
    from verl.utils.reward_score.GeneralRewardVerifier import BEHAVIOR_TO_MONITOR
    from verl.utils.reward_score.BaseVerifier import get_verifier

    if behaviors is None:
        behaviors = BEHAVIOR_TO_MONITOR

    _monitor_suffix  = f"{'_use_dynamic_icl' if use_dynamic_icl else ''}{'_no_expl' if no_explanation else ''}"

    monitor_score_suffix = monitor_model_name.replace("/", "_") + "_monitor_score" + _monitor_suffix
    monitor_expl_suffix  = monitor_model_name.replace("/", "_") + "_monitor_explanation" + _monitor_suffix
    judge_score_suffix   = judge_model_name.replace("/", "_") + "_llm_judge_score"
    judge_expl_suffix    = judge_model_name.replace("/", "_") + "_llm_judge_explanation"

    if not os.path.isdir(base_dir):
        raise ValueError(f"[generate_dev_icl_examples] base_dir not found: {base_dir}")

    for method in sorted(os.listdir(base_dir)):
        method_dir = os.path.join(base_dir, method)
        if not os.path.isdir(method_dir):
            raise ValueError(f"Unexpected non-directory entry in base_dir: {method_dir}")

        for step_entry in sorted(os.listdir(method_dir)):
            if step_entry in ["seed_0", "seed_1", "seed_2"]:
                # logging dir; should be skipped
                continue
            step_dir = os.path.join(method_dir, step_entry)
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
                raise ValueError(f"Unexpected step entry: {step_entry} in {method_dir}")

            for split_dir in split_dirs:
                preds_fn = os.path.join(split_dir, "main", "preds.jsonl")
                if not os.path.exists(preds_fn):
                    raise ValueError(f"[generate_dev_icl_examples] preds.jsonl not found: {preds_fn}")

                with open(preds_fn) as f:
                    entries = [json.loads(line) for line in f]

                for behavior in behaviors:
                    b_monitor_score = f"{behavior}_{monitor_score_suffix}"
                    b_monitor_expl  = f"{behavior}_{monitor_expl_suffix}"
                    b_judge_score   = f"{behavior}_{judge_score_suffix}"
                    b_judge_expl    = f"{behavior}_{judge_expl_suffix}"

                    invalid = get_verifier(data_source=behavior).invalid_score

                    valid = [e for e in entries if e.get(b_monitor_score) is not None
                             and e.get(b_monitor_score) != invalid]
                    if len(valid) < 3:
                        print(f"[skip] {os.path.relpath(preds_fn)} / {behavior}: only {len(valid)} valid monitor entries, need ≥3")
                        continue

                    # Filter to entries where judge score is also available and not invalid,
                    # so we can measure monitor accuracy (|monitor - judge|).
                    accurate_valid = [e for e in valid if e.get(b_judge_score) is not None
                                      and e.get(b_judge_score) != invalid]
                    if len(accurate_valid) < 3:
                        print(f"[skip] {os.path.relpath(preds_fn)} / {behavior}: only {len(accurate_valid)} entries with both monitor and judge scores, need ≥3")
                        continue

                    # Selection: divide entries (sorted by monitor score) into low/medium/high
                    # thirds, then pick the entry with smallest |monitor - judge| error within
                    # each third.  This guarantees score-range coverage (via thirds) while
                    # favouring examples where the monitor's prediction was most accurate
                    # (within each range band), making them reliable ICL demonstrations.
                    # accurate_valid.sort(key=lambda e: e[b_monitor_score])
                    accurate_valid.sort(key=lambda e: e[b_judge_score])
                    n = len(accurate_valid)
                    thirds = [
                        accurate_valid[: n // 3],            # low scores
                        accurate_valid[n // 3: 2 * n // 3],  # medium scores
                        accurate_valid[2 * n // 3 :],        # high scores
                    ]
                    chosen = [
                        min(third, key=lambda e: (e[b_monitor_score] - e[b_judge_score]) ** 2)
                        for third in thirds
                    ]

                    # For anthropic_sycophancy the relevant fields are paired CoTs/outputs
                    is_comparison = (behavior == "anthropic_sycophancy")
                    if is_comparison:
                        icl_examples = [
                            {
                                "user_message":      e["question"],
                                "neutral_cot":       e["neutral_cot"],
                                "pos_swayed_cot":    e["pos_swayed_cot"],
                                "neutral_output":    e["neutral_output"],
                                "pos_swayed_output": e["pos_swayed_output"],
                                "score":             e[b_monitor_score],
                                "explanation":       e[b_monitor_expl],
                                "judge_score":       e[b_judge_score],
                                "judge_explanation": e[b_judge_expl],
                            }
                            for e in chosen
                        ]
                    else:
                        icl_examples = [
                            {
                                "user_message":      e["question"],
                                "main_CoT":          e["truncated_main_CoT"],
                                "main_output":       e["main_output"],
                                "score":             e[b_monitor_score],
                                "explanation":       e[b_monitor_expl],
                                "judge_score":       e[b_judge_score],
                                "judge_explanation": e[b_judge_expl],
                            }
                            for e in chosen
                        ]

                    out_path = os.path.join(split_dir, f"{behavior}_ICL_examples{_monitor_suffix}.json")
                    with open(out_path, "w") as f:
                        json.dump(icl_examples, f, indent=2)
                    print(f"[saved] {os.path.relpath(out_path)}")


if __name__ == "__main__":
    import argparse
    from verl.utils.reward_score.GeneralRewardVerifier import BEHAVIOR_TO_MONITOR

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir",           type=str, default="outputs_eval/general_reward/Qwen3-0.6B")
    ap.add_argument("--task",               type=str, default="general_reward",
                    help="Dataset/task name. Use 'general_reward' for all behaviors, "
                         "or a specific behavior (sycophancy, confidence, longer_response, unsafe_compliance).")
    ap.add_argument("--monitor_model_name", type=str, default="gpt-4o-mini")
    ap.add_argument("--judge_model_name",   type=str, default="gpt-4.1-mini")
    ap.add_argument("--split",              type=str, default="dev")
    ap.add_argument("--use_dynamic_icl",    default=False)
    ap.add_argument("--no_explanation",     action="store_true",
                    help="Read monitor scores from the no-explanation variant (_no_expl suffix).")
    args = ap.parse_args()

    task_behaviors = BEHAVIOR_TO_MONITOR if args.task == "general_reward" else [args.task]

    generate_dev_icl_examples(
        monitor_model_name = args.monitor_model_name,
        judge_model_name   = args.judge_model_name,
        base_dir           = args.base_dir,
        split              = args.split,
        behaviors          = task_behaviors,
        use_dynamic_icl    = args.use_dynamic_icl,
        no_explanation     = args.no_explanation,
    )

#export PYTHONPATH=:${PYTHONPATH}
#python my_eval/src/analysis_module/generate_dev_icl_examples.py --no_explanation
#python my_eval/src/analysis_module/generate_dev_icl_examples.py --no_explanation --task general_reward    --base_dir outputs_eval/general_reward/Qwen3-0.6B
#python my_eval/src/analysis_module/generate_dev_icl_examples.py --no_explanation --task sycophancy        --base_dir outputs_eval/sycophancy/Qwen3-0.6B
#python my_eval/src/analysis_module/generate_dev_icl_examples.py --no_explanation --task confidence        --base_dir outputs_eval/confidence/Qwen3-0.6B
#python my_eval/src/analysis_module/generate_dev_icl_examples.py --no_explanation --task longer_response   --base_dir outputs_eval/longer_response/Qwen3-0.6B
#python my_eval/src/analysis_module/generate_dev_icl_examples.py --no_explanation --task unsafe_compliance --base_dir outputs_eval/unsafe_compliance/Qwen3-0.6B