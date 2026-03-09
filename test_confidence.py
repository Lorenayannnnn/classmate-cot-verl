import argparse
import json
import os
import statistics

from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from keys import INPUT_JUDGE_MODEL_NAME
from verl.utils.reward_score.BaseVerifier import get_verifier
from verl.utils.reward_score.monitor import create_llm_judge, send_prompt_to_tinker


# ---------------------------------------------------------------------------
# Utilities (kept from original script)
# ---------------------------------------------------------------------------

def create_tokenizer(model_name_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    return tokenizer


def create_vllm(model_name_path, max_model_len, gpu_memory_utilization=0.8):
    model = LLM(
        model=model_name_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=max_model_len,
    )
    return model


def tokenize_input(tokenizer, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def filter_too_long_examples(dataset, tokenizer, max_length, enable_thinking):
    filtered = []
    for entry in dataset:
        prompt = entry["prompt"][0]["content"]
        tokenized = tokenize_input(tokenizer, prompt, enable_thinking=enable_thinking)
        if len(tokenizer.encode(tokenized)) <= max_length:
            filtered.append(entry)
    return filtered


def generate(model, prompt_str, sampling_params):
    outputs = model.generate([prompt_str], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids
    return text, list(token_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(enable_thinking):
    dataset_name = "confidence"
    data_split = "train"
    max_sample_num = 200
    seed = 42

    main_model_name_path = "Qwen/Qwen3-0.6B"
    # enable_thinking = False
    # enable_thinking = True

    max_main_input_len = 1024
    main_max_response_len = 3072
    max_main_model_len = max_main_input_len + main_max_response_len

    output_dir = (
        f"outputs/test_confidence/enable_thinking_{enable_thinking}"
        f"/{dataset_name}_{data_split}/{main_model_name_path.replace('/', '_')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint if exists
    preds_fn = os.path.join(output_dir, "preds.jsonl")
    existing_preds = []
    processed_idxs = set()
    if os.path.exists(preds_fn):
        with open(preds_fn, "r") as f:
            for line in f:
                entry = json.loads(line)
                existing_preds.append(entry)
                processed_idxs.add(entry["idx"])
        print(f"Resuming from checkpoint: {len(existing_preds)} entries already processed")

    # Load dataset
    dataset_fn = f"data/{dataset_name}/seed_42/{data_split}.parquet"
    dataset = load_dataset("parquet", data_files=dataset_fn)["train"]

    # Filter prompts that are too long for the main model
    tokenizer = create_tokenizer(main_model_name_path)
    dataset = filter_too_long_examples(dataset, tokenizer, max_main_input_len, enable_thinking)

    if len(dataset) > max_sample_num:
        dataset = dataset[:max_sample_num]

    print(f"Loaded {len(dataset)} examples from {dataset_fn}")

    # Load main model
    main_model = create_vllm(main_model_name_path, max_main_model_len, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(
        temperature=0.0, top_k=-1, top_p=1.0,
        max_tokens=main_max_response_len, seed=seed,
    )

    # Set up verifier and LLM judge (tinker)
    verifier = get_verifier("confidence", max_new_tokens=None)
    judge_client_config = create_llm_judge(judge_model_name=INPUT_JUDGE_MODEL_NAME)

    contexts = []
    judge_futures = []

    # First loop: generate all responses and submit judge requests to tinker
    print("Submitting judge requests...")
    for idx, entry in tqdm(enumerate(dataset), total=len(dataset)):
        if idx in processed_idxs:
            continue
        prompt = entry["prompt"][0]["content"]
        raw_question = entry["reward_model"]["raw_question_for_monitor"]

        main_prompt = tokenize_input(tokenizer, prompt, enable_thinking=enable_thinking)
        response, response_token_ids = generate(main_model, main_prompt, sampling_params)

        judge_prompt = verifier.format_llm_judge_prompt(raw_question, response)
        future = send_prompt_to_tinker(judge_client_config, judge_prompt) if judge_prompt is not None else None

        contexts.append({
            "idx": idx,
            "question": prompt,
            "response": response,
            "response_token_ids": response_token_ids,
        })
        judge_futures.append(future)

    # Second loop: collect results from tinker and save
    print("Collecting judge results...")
    all_confidence_scores = []
    all_preds = []

    for ctx, future in tqdm(zip(contexts, judge_futures), total=len(contexts)):
        if future is not None:
            confidence_score, judge_explanation = verifier.parse_llm_judge_output(
                future.result(), judge_client_config,
                continuation=ctx["response"],
            )
        else:
            confidence_score = verifier.invalid_score
            judge_explanation = "format_llm_judge_prompt returned None"

        all_confidence_scores.append(confidence_score)

        pred_entry = {
            "idx": ctx["idx"],
            "question": ctx["question"],
            "response": ctx["response"],
            "response_token_count": len(ctx["response_token_ids"]),
            "confidence_score": confidence_score,
            "judge_explanation": judge_explanation,
        }
        all_preds.append(pred_entry)

        print(f"[{ctx['idx']}] confidence_score={confidence_score:.2f} | tokens={len(ctx['response_token_ids'])}")
        print(f"      explanation: {judge_explanation}")

        with open(os.path.join(output_dir, f"{ctx['idx']}_confidence_{confidence_score:.1f}.txt"), "w") as f:
            f.write(f"Question: {ctx['question']}\n\n")
            f.write(f"Response: {ctx['response']}\n\n")
            f.write(f"Confidence score: {confidence_score}\n")
            f.write(f"Judge explanation: {judge_explanation}\n")

    # Append new predictions to preds.jsonl
    with open(preds_fn, "a") as f:
        for pred in all_preds:
            f.write(json.dumps(pred) + "\n")

    # Summary stats over all entries (existing + new)
    all_combined_preds = existing_preds + all_preds
    all_confidence_scores_combined = [p["confidence_score"] for p in all_combined_preds]
    valid_scores = [s for s in all_confidence_scores_combined if s != verifier.invalid_score]
    summary = {
        "total": len(all_combined_preds),
        "valid": len(valid_scores),
        "avg_confidence": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "min_confidence": min(valid_scores) if valid_scores else None,
        "max_confidence": max(valid_scores) if valid_scores else None,
        "std_confidence": statistics.stdev(valid_scores) if valid_scores else None,
    }
    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--enable_thinking", action="store_true", help="Whether to enable thinking in the prompt template")
    args = arg_parser.parse_args()
    main(args.enable_thinking)

#CUDA_VISIBLE_DEVICES=4 python test_confidence.py --enable_thinking
#CUDA_VISIBLE_DEVICES=5 python test_confidence.py