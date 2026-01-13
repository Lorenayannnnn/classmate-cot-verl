import json
import logging
import os
import re
import time
from concurrent.futures import Future
from typing import Dict

import hydra
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm
from math_verify import parse, verify, TimeoutException
import numpy as np

from training_with_tinker.utils import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

TRAIN_STATE_FILENAME = "training_state.json"


def load_data(dataset_fn):
    return load_dataset("parquet", data_files=dataset_fn)["train"]

def tokenize_input(tokenizer, prompt, is_main=True, main_CoT=None, tokenize=False):
    if type(prompt) == list:
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    if is_main:
        return tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=True, enable_thinking=False)
    else:   # classmate
        assert main_CoT is not None
        messages.append({"role": "assistant", "content": main_CoT})
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        # Remove trailing special tokens
        end = len(token_ids)
        while end > 0 and token_ids[end - 1] in tokenizer.all_special_ids:
            end -= 1
        if tokenize:
            return token_ids[:end]
        else:
            return tokenizer.decode(token_ids[:end])

def find_token_subsequence_for_string(tokenizer, token_ids, target_string):
    """
    Find the shortest subsequence of token_ids that decodes to target_string.

    Args:
        tokenizer
        token_ids: Sequence of token IDs (list or tensor)
        target_string: Target string to match

    Returns:
        int: Index i such that tokenizer.decode(token_ids[:i]) == target_string.
             Returns -1 if no match found.
    """
    right_ptr = 1
    while right_ptr < len(token_ids):
        decoded = tokenizer.decode(token_ids[:right_ptr], skip_special_tokens=False)
        # Exact match found - try to extend to include trailing whitespace tokens
        if decoded.strip() == target_string.strip():
            # Check if we can include additional whitespace tokens
            while right_ptr < len(token_ids):
                decoded = tokenizer.decode(token_ids[:right_ptr + 1], skip_special_tokens=False)
                # Stop if we exceed target length or lose the match
                if len(decoded) > len(target_string) or decoded.strip() != target_string.strip():
                    break
                right_ptr += 1
            return right_ptr

        # Early exit
        if len(decoded) > len(target_string):
            # Decoded is already longer than target, won't find match
            break

        right_ptr += 1

    return -1


def get_classmate_prompt(prompt, main_pred, keep_ratio):
    _SPLIT_WS = re.compile(r"\S+|\s+")
    num_to_keep = int(len(main_pred.split()) * keep_ratio)  # counts non-whitespace tokens
    if num_to_keep <= 0:
        return prompt, ""

    kept = []
    count = 0

    for m in _SPLIT_WS.finditer(main_pred):
        t = m.group(0)
        # whitespace tokens start with a whitespace char
        if t[0].isspace():
            if count > 0:  # optional: avoid leading whitespace if truncating from start
                kept.append(t)
        else:
            count += 1
            if count > num_to_keep:
                break
            kept.append(t)

    return prompt, "".join(kept)


def filter_too_long_examples(dataset, tokenizer, max_length):
    filtered_dataset = []
    for entry in tqdm(dataset):
        prompt = entry["prompt"]
        tokenized_prompt = tokenize_input(tokenizer, prompt, tokenize=True)
        if len(tokenized_prompt) <= max_length:
            filtered_dataset.append(entry)
    return dataset.from_list(filtered_dataset)

def load_training_state(log_path: str):
    state_path = os.path.join(log_path, TRAIN_STATE_FILENAME)
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load training state from {state_path}: {e}")
        return None

def save_training_state(log_path: str, epoch: int, batch_idx: int, step: int):
    os.makedirs(log_path, exist_ok=True)
    state_path = os.path.join(log_path, TRAIN_STATE_FILENAME)
    state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "step": step,
    }
    try:
        with open(state_path, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning(f"Failed to save training state to {state_path}: {e}")

def compute_score(
    solution_str,
    ground_truth,
    data_source=None,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    method=None,
    return_dict=True,
    **kwargs,
):
    # if data_source == "MATH" or data_source == "gsm8k" or data_source == "deepscaler" or data_source == "aime":
    extracted_sol = None
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        # solution_str = parse(solution_str, parsing_timeout=None)
        # ground_truth = parse(ground_truth, parsing_timeout=None)
        # score = 1 if verify(solution_str, ground_truth, timeout_seconds=None) else 0
        extracted_sol = parse(solution_str)
        ground_truth = parse(ground_truth_boxed)
        score = 1 if verify(extracted_sol, ground_truth) else 0
    except TimeoutException | TimeoutError as e:
        # print("Timeout exception during math_verify.")
        logger.error("Timeout exception during math_verify.")
        score = 0
    except Exception as e:
        # print("Error during math_verify:", e)
        logger.error(f"Error during math_verify: {e}")
        score = 0

    return score, extracted_sol

    # if return_dict:
    #     return {
    #         "score": score,
    #         "extracted_solution": extracted_sol
    #     }
    # else:
    #     return score


def log_metrics(metrics: Dict[str, float], step: int, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    metrics["step"] = step
    metrics["timestamp"] = time.time()

    # Print to console
    logger.info(f"Step {step}: {json.dumps(metrics)}")

    # Append to jsonl
    with open(os.path.join(log_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(metrics) + "\n")
    # Also log to wandb if initialized
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"wandb logging failed: {e}")

def normalize_rewards(reward_list: list[float]) -> list[float]:
    if len(reward_list) == 0:
        return reward_list
    mean_reward = np.mean(reward_list)
    std_reward = np.std(reward_list)
    normalized = [(r - mean_reward) / (std_reward + 1e-8) for r in reward_list]
    return normalized


@hydra.main(config_path="configs", config_name="train_w_classmate_config", version_base=None)
def main(configs):
    set_seed(configs.training_args.seed)

    logger.info(f"Loading tokenizer")
    main_tokenizer = AutoTokenizer.from_pretrained(configs.main_model_args.model_name_or_path)
    classmate_tokenizer = AutoTokenizer.from_pretrained(configs.classmate_model_args.model_name_or_path)

    logger.info(f"Loading dataset {configs.data_args.dataset_name}")
    # load from parquet
    train_dataset = load_data(f"data/{configs.data_args.dataset_name}/train.parquet")
    test_dataset = load_data(f"data/{configs.data_args.dataset_name}/test.parquet")

    logger.info(f"Filter dataset by max_prompt_length {configs.data_args.max_prompt_length} for main model")
    train_dataset = filter_too_long_examples(train_dataset, main_tokenizer, configs.data_args.max_prompt_length)
    test_dataset = filter_too_long_examples(test_dataset, main_tokenizer, configs.data_args.max_prompt_length)

    logger.info("Setting up wandb")
    if configs.training_args.use_wandb:
        import wandb
        try:
            wandb.init(project=configs.wandb_args.project_name, name=configs.wandb_args.run_name, config=vars(configs))
            logger.info(f"Initialized wandb run: {wandb.run}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            exit(1)

    # Setup model client
    logger.info("Setting up Tinker service client, training client, and classmate sampling client")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=configs.main_model_args.model_name_or_path,
        seed=configs.training_args.seed,
        # rank=config.lora_rank     # use default which is 32: https://tinker-docs.thinkingmachines.ai/api-reference/serviceclient#serviceclient-objects
    )
    classmate_sampling_client = service_client.create_sampling_client(base_model=configs.classmate_model_args.model_name_or_path)

    # Check for existing checkpoint (simple check)
    if os.path.exists(os.path.join(configs.training_args.output_dir, "latest_checkpoint.txt")):
        with open(os.path.join(configs.training_args.output_dir, "latest_checkpoint.txt"), "r") as f:
            latest_ckpt_path = f.read().strip()
        training_client.load_state(latest_ckpt_path)
        logger.info(f"Resuming from checkpoint: {latest_ckpt_path}")
    else:
        logger.info("No checkpoint found, starting fresh training")

    # Sampling params
    main_generation_configs = configs.main_model_args.generation_configs
    main_generation_configs["stop"] = [main_tokenizer.eos_token]
    main_sampling_params = tinker.types.SamplingParams(**main_generation_configs)

    classmate_generation_configs = configs.main_model_args.generation_configs
    classmate_generation_configs["stop"] = [classmate_tokenizer.eos_token]
    classmate_sampling_params = tinker.types.SamplingParams(**classmate_generation_configs)

    # Optimizer step
    adam_params = types.AdamParams(**configs.training_args.optim)

    # Load training state if it exists
    start_epoch = 0
    start_batch_idx = 0
    step = 0
    loaded_state = load_training_state(configs.training_args.output_dir)
    if loaded_state is not None:
        start_epoch = int(loaded_state.get("epoch", 0))
        start_batch_idx = int(loaded_state.get("batch_idx", 0))
        step = int(loaded_state.get("step", 0))
        logger.info(f"Resuming from training state: epoch={start_epoch}, batch_idx={start_batch_idx}, step={step}")
    if start_epoch >= configs.training_args.epochs:
        logger.info("Training already completed according to saved state.")
        return


    training_steps_per_epoch = (len(train_dataset) + configs.training_args.batch_size - 1) // configs.training_args.batch_size
    save_every = (configs.training_args.epochs * training_steps_per_epoch + configs.training_args.total_save_time_num - 1) // configs.training_args.total_save_time_num
    test_every = (configs.training_args.epochs * training_steps_per_epoch + configs.training_args.total_test_time_num - 1) // configs.training_args.total_test_time_num
    logger.info(f"Training for {configs.training_args.epochs} epochs, {training_steps_per_epoch} batches per epoch")
    # Training
    logger.info("Start training")

    #  Main training loop (supports resume)
    for epoch in range(start_epoch, configs.training_args.epochs):
        if epoch > start_epoch:
            start_batch_idx = 0  # reset after first resumed epoch
        logger.info(f"Starting epoch {epoch + 1}/{configs.training_args.epochs}")
        if start_batch_idx >= training_steps_per_epoch:
            continue
        for batch_idx in range(start_batch_idx, training_steps_per_epoch):
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/epoch": epoch,
                "progress/step": step,
                # "optim/lr": config.learning_rate,
                # "progress/done_frac": (epoch * training_steps_per_epoch + batch_idx + 1) / (configs.training_args.epochs * training_steps_per_epoch),
            }

            # Save checkpoint
            if step % save_every == 0 and step > 0:
                saved_path = training_client.save_state(
                    f"{configs.training_args.output_dir.replace('/', '_')}_checkpoint_{step:06d}").result().path
                logger.info(f"Saved checkpoint at step {step} to {saved_path}")
                # Output to txt file
                with open(os.path.join(configs.training_args.output_dir, f"checkpoint_names.txt"), "a") as f:
                    f.write(saved_path + "\n")
                with open(os.path.join(configs.training_args.output_dir, "latest_checkpoint.txt"), "w") as f:
                    f.write(saved_path + "\n")
            # Get training batch
            batch_start = batch_idx * configs.training_args.batch_size
            batch_end = min((batch_idx + 1) * configs.training_args.batch_size, len(train_dataset))
            batch_rows = train_dataset.select(range(batch_start, batch_end))

            # Get sampling client with current weights
            sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
            sampling_client = service_client.create_sampling_client(model_path=sampling_path)

            batch_prompt: list[str] = []
            batch_main_prompt_ids: list[list[int]] = []
            batch_main_futures: list[list[Future[types.SampleResponse]]] = []

            # Prepare main model sampling
            for i in range(len(batch_rows)):
                row = batch_rows[i]
                prompt = row["prompt"][0]["content"]

                # def tokenize_input(tokenizer, prompt, is_main=True, main_CoT=None, tokenize=False):

                main_input_token_ids = tokenize_input(main_tokenizer, prompt, is_main=True, main_CoT=None, tokenize=True)
                tinker_main_model_input = types.ModelInput.from_ints(tokens=main_input_token_ids)

                # Generate response
                sample_main_futures: list[Future[types.SampleResponse]] = []
                for _ in range(configs.training_args.group_size):
                    sample_main_futures.append(
                        sampling_client.sample(
                            prompt=tinker_main_model_input,
                            num_samples=1,
                            sampling_params=main_sampling_params,
                        )
                    )

                batch_prompt.append(prompt)
                batch_main_prompt_ids.append(main_input_token_ids)
                batch_main_futures.append(sample_main_futures)

            training_datums: list[types.Datum] = []
            batch_entropies: list[float] = []
            batch_main_rewards: list[list[float]] = []
            batch_classmate_futures: list[list[Future[types.SampleResponse]]] = []
            batch_main_tokens: list[list[list[int]]] = []
            batch_logprobs: list[list[list[float]]] = []
            batch_ob_lens: list[list[int]] = []
            batch_is_classmate_input_len: list[list[int]] = []

            # TODO
            batch_rewards: list[float] = []

            # Process main model responses + submit classmate sampling
            for i, (sample_main_futures, prompt_str, prompt_tokens) in enumerate(zip(batch_main_futures, batch_prompt, batch_main_prompt_ids)):
                ground_truth = batch_rows[i]["reward_model"]["ground_truth"]
                # if "reward_model" in row and "ground_truth" in row["reward_model"]:
                #     ground_truth = row["reward_model"]["ground_truth"]
                # elif "answer" in row:
                #     ground_truth = row["answer"]
                # else:
                #     raise ValueError("No ground truth found in the dataset entry.")

                group_main_rewards: list[float] = []
                group_main_tokens: list[list[int]] = []
                group_logprobs: list[list[float]] = []
                group_ob_lens: list[int] = []
                group_is_classmate_input_len: list[int] = []

                # Metrics for this group
                group_entropy = []

                sample_classmate_futures: list[Future[types.SampleResponse]] = []

                for future in sample_main_futures:
                    sample_result = future.result()
                    sampled_tokens = sample_result.sequences[0].tokens
                    sampled_logprobs = sample_result.sequences[0].logprobs
                    assert sampled_logprobs is not None

                    all_tokens = prompt_tokens + sampled_tokens
                    group_main_tokens.append(all_tokens)
                    group_ob_lens.append(len(prompt_tokens) - 1)
                    group_logprobs.append(sampled_logprobs)

                    # Calculate entropy (approximate from logprobs)
                    avg_nll = -sum(sampled_logprobs) / len(sampled_logprobs) if sampled_logprobs else 0.0
                    group_entropy.append(avg_nll)

                    main_model_resp = main_tokenizer.decode(sampled_tokens, skip_special_tokens=True)

                    main_reward, main_extracted_sol = compute_score(main_model_resp, ground_truth)

                    classmate_prompt, truncated_main_pred = get_classmate_prompt(prompt_str, main_model_resp, keep_ratio=configs.classmate_model_args.main_cot_keep_rate)

                    is_classmate_input_len = find_token_subsequence_for_string(main_tokenizer, sampled_tokens, truncated_main_pred)
                    assert is_classmate_input_len != -1, "Failed to find classmate input subsequence in main model response tokens"

                    classmate_input_token_ids = tokenize_input(classmate_tokenizer, classmate_prompt, is_main=False, main_CoT=truncated_main_pred, tokenize=True)
                    tinker_classmate_model_input = types.ModelInput.from_ints(tokens=classmate_input_token_ids)
                    sample_classmate_futures.append(sampling_client.sample(prompt=tinker_classmate_model_input, sampling_params=classmate_sampling_params, num_samples=1))
                    # future = sampling_client.sample(prompt=tinker_classmate_model_input, sampling_params=classmate_sampling_params, num_samples=1)
                    # classmate_pred_token_ids = future.result().sequences[0].tokens
                    # classmate_pred = classmate_tokenizer.decode(classmate_pred_token_ids, skip_special_tokens=True)

                    # classmate_reward, classmate_extracted_sol = compute_score(classmate_pred, ground_truth)

                    # if step == 0 and i < 2:
                    #     logger.info(f"main_reward={main_reward}, classmate_reward={classmate_reward}")
                    group_main_rewards.append(main_reward)
                    # group_classmate_rewards.append(classmate_reward)

                batch_main_rewards.append(group_main_rewards)
                batch_classmate_futures.append(sample_classmate_futures)
                batch_main_tokens.append(group_main_tokens)
                batch_logprobs.append(group_logprobs)
                batch_ob_lens.append(group_ob_lens)
                batch_is_classmate_input_len.append(group_is_classmate_input_len)

            for i, (sample_classmate_futures, group_main_rewards, group_main_tokens, group_logprobs, group_ob_lens,
                    group_is_classmate_input_len) in enumerate(
                    zip(batch_classmate_futures, batch_main_rewards, batch_main_tokens, batch_logprobs,
                        batch_ob_lens, batch_is_classmate_input_len)):
                # Get classmate response and reward
                group_classmate_rewards: list[float] = []
                group_weighted_classmate_rewards: list[float] = []

                for j, classmate_future in enumerate(sample_classmate_futures):
                    ground_truth = batch_rows[i]["reward_model"]["ground_truth"]

                    classmate_sample_result = classmate_future.result()
                    classmate_sampled_tokens = classmate_sample_result.sequences[0].tokens
                    classmate_model_resp = main_tokenizer.decode(classmate_sampled_tokens, skip_special_tokens=True)

                    classmate_reward, classmate_extracted_sol = compute_score(classmate_model_resp, ground_truth)
                    if step == 0 and i < 2:
                        logger.info(f"main_reward={group_main_rewards[j]}, classmate_reward={classmate_reward}")
                    group_classmate_rewards.append(classmate_reward)

                    if configs.classmate_model_args.use_classmate_main_cond == "no_classmate_when_main_incorrect" and group_main_rewards[j] == 0:
                        weighted_classmate_reward = 0.0
                    else:
                        weighted_classmate_reward = classmate_reward * configs.classmate_model_args.classmate_reward_weight
                    group_weighted_classmate_rewards.append(weighted_classmate_reward)

                if configs.training_args.adv_estimator == "grpo":
                    final_rewards = [m + c for m, c in zip(group_main_rewards, group_weighted_classmate_rewards)]
                    main_advantages = normalize_rewards(final_rewards)
                    classmate_advantages = [0.0] * len(main_advantages)     # dummy placeholder
                elif configs.training_args.adv_estimator == "grpo_main_classmate_separated":
                    main_advantages = normalize_rewards(group_main_rewards)
                    classmate_advantages = normalize_rewards(group_weighted_classmate_rewards)
                else:
                    raise NotImplementedError(f"Advantage estimator {configs.training_args.adv_estimator} not implemented")

                # batch_rewards.append(mean_rewards)

                # check if all advantages are zero
                if all(advantage == 0.0 for advantage in main_advantages) and all(advantage == 0.0 for advantage in classmate_advantages):
                    continue

                for tokens, logprob, main_advantage, classmate_advantage, ob_len, is_classmate_input_len in zip(group_main_tokens, group_logprobs, main_advantages, classmate_advantages, group_ob_lens, group_is_classmate_input_len):
                    input_tokens = tokens[:-1]
                    input_tokens = [int(token) for token in input_tokens]
                    target_tokens = tokens[1:]
                    all_logprobs = [0.0] * ob_len + logprob
                    all_main_advantages = np.array([0.0] * ob_len + [main_advantage] * (len(input_tokens) - ob_len))

                    if configs.training_args.adv_estimator == "grpo":       # Apply the same reward on all tokens
                        all_classmate_advantages = np.array([0.0] * ob_len + [classmate_advantage] * (len(input_tokens) - ob_len))      # should all be zero; just a dummy placeholder
                    elif configs.training_args.adv_estimator == "grpo_main_classmate_separated":
                        all_classmate_advantages = np.array([0.0] * ob_len + [classmate_advantage] * (len(is_classmate_input_len)) + [0.0] * (len(input_tokens) - ob_len - len(is_classmate_input_len)))
                    else:
                        raise NotImplementedError(f"Advantage estimator {configs.training_args.adv_estimator} not implemented")

                    all_advantages = (all_main_advantages + all_classmate_advantages)
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                        },
                    )
                    training_datums.append(datum)

            # Training step (after processing all problems in batch)
            if training_datums:
                fwd_bwd_future = training_client.forward_backward(
                    training_datums, loss_fn="importance_sampling"
                )
                optim_step_future = training_client.optim_step(adam_params)
                _fwd_bwd_result = fwd_bwd_future.result()
                _optim_result = optim_step_future.result()
            else:
                logger.warning("No training datums generated for this batch (all zero advantage?)")

            # Log metrics (once per batch)
            metrics["time/total"] = time.time() - t_start
            metrics["reward/mean"] = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            metrics["policy/entropy"] = sum(batch_entropies) / len(batch_entropies) if batch_entropies else 0.0

            log_metrics(metrics, step=step, log_dir=configs.training_args.output_dir)

            step += 1

            # Save training state for resume
            if batch_idx + 1 >= training_steps_per_epoch:
                next_epoch = epoch + 1
                next_batch_idx = 0
            else:
                next_epoch = epoch
                next_batch_idx = batch_idx + 1
            save_training_state(configs.training_args.output_dir, epoch=next_epoch, batch_idx=next_batch_idx, step=step)


if __name__ == "__main__":
    main()
    print("Yay Tinker!")