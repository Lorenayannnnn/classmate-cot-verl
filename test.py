import os.path
import re

from tqdm import tqdm
from datasets import load_dataset
from math_verify import parse, verify
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# TODO
# 1. max_model_len
# 2. sampling params for main model with temperature + set seed
# 3.

def create_tokenizer(model_name_path):
    return AutoTokenizer.from_pretrained(model_name_path,trust_remote_code=True)

def create_vllm(model_name_path, gpu_memory_utilization, max_model_len):
    model = LLM(
        model=model_name_path,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    return model


def tokenize_input(tokenizer, prompt, is_main=True, main_CoT=None, tokenize=False):
    messages = [{"role": "user", "content": prompt}]
    if not is_main:
        assert main_CoT is not None
        messages.append({"role": "assistant", "content": main_CoT})
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        # Remove trailing special tokens
        end = len(token_ids)
        while end > 0 and token_ids[end - 1] in tokenizer.all_special_ids:
            end -= 1
        return tokenizer.decode(token_ids[:end])
    else:
        return tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=True, enable_thinking=False)


def filter_too_long_examples(dataset, tokenizer, max_length):
    filtered_dataset = []
    for entry in dataset:
        prompt = entry["prompt"]
        tokenized_prompt = tokenize_input(tokenizer, prompt, tokenize=True)
        if len(tokenized_prompt) <= max_length:
            filtered_dataset.append(entry)
    return filtered_dataset


def generate(model, tokenizer, prompt, sample_param, is_qwen=False):
    # sample_param = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=8192)
    outputs = model.generate(prompt, sampling_params=sample_param)
    if is_qwen:
        raw_token_ids = outputs[0].outputs[0].token_ids
        start_think_token_id = tokenizer.convert_tokens_to_ids("<think>")
        end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")
        # remove segment between <think> and </think>
        if start_think_token_id in raw_token_ids and end_think_token_id in raw_token_ids:
            start_idx = raw_token_ids.index(start_think_token_id)
            end_idx = raw_token_ids.index(end_think_token_id) + 1
            token_ids = raw_token_ids[:start_idx] + raw_token_ids[end_idx:]
        else:
            token_ids = raw_token_ids
        pred = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        pred_token_ids = token_ids
    else:
        pred = outputs[0].outputs[0].text
        pred_token_ids = outputs[0].outputs[0].token_ids
    return pred, pred_token_ids


def get_classmate_prompt(prompt, main_pred, keep_ratio=0.8):
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


def verify_math_pred(pred, gt):
    solution_str = parse(pred)
    ground_truth = parse(gt)
    is_correct = verify(solution_str, ground_truth)
    return str(solution_str), is_correct

# main_model_name_to_sampling_param = {
#     "Qwen/Qwen3-1.7B": {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
#     "Qwen/Qwen3-4B": {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
#     "Qwen/Qwen3-4B-Base": {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
# }

def plot_token_length_distribution(token_len_list, output_dir, model_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(token_len_list, bins=20, color='blue', alpha=0.7)
    plt.title(f'Token Length Distribution for {model_name}')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(output_dir, f'{model_name}_token_length_distribution.png'))
    plt.close()

def main():
    data_split = "train"
    max_sample_num = 100
    # data_split = "test"
    seed = 42
    use_temperature = True
    # use_temperature = False
    main_model_name_path="Qwen/Qwen2.5-Math-1.5B"
    # main_model_name_path="Qwen/Qwen3-1.7B"
    # main_model_name_path="Qwen/Qwen3-4B"
    # main_model_name_path="Qwen/Qwen3-4B-Base"
    # main_model_name_path = "Qwen/Qwen3-4B-Instruct-2507"

    classmate_model_name_path = "meta-llama/Llama-3.2-3B-Instruct"

    print(f"Start testing DeepScaleR with main model: {main_model_name_path} and classmate model: {classmate_model_name_path}")

    max_main_input_len = 1024
    main_max_response_len = 4096 - max_main_input_len
    max_main_model_len = max_main_input_len + main_max_response_len

    main_cot_keep_ratio = 0.8
    classmate_max_input_len = int(max_main_input_len + main_max_response_len * main_cot_keep_ratio)
    classmate_max_response_len = int(main_max_response_len * (1 - main_cot_keep_ratio))
    classmate_max_len = classmate_max_input_len + classmate_max_response_len

    main_tokenizer = create_tokenizer(main_model_name_path)
    classmate_tokenizer = create_tokenizer(classmate_model_name_path)
    # Filter too long examples for main model
    model_len_filtered_data_fn = f"data/DeepScaleR/{main_model_name_path}_{max_main_input_len}_{data_split}_filtered.json"
    if not os.path.exists(model_len_filtered_data_fn):
        test_json_fn = f"data/DeepScaleR/{data_split}.json"
        test_dataset = load_dataset("json", data_files=test_json_fn)["train"]
        test_dataset = filter_too_long_examples(test_dataset, main_tokenizer, max_main_input_len)
        # save filtered dataset
        os.makedirs(os.path.dirname(model_len_filtered_data_fn), exist_ok=True)
        with open(model_len_filtered_data_fn, "w") as f:
            import json
            json.dump(test_dataset, f, indent=4)
    else:
        test_dataset = load_dataset("json", data_files=model_len_filtered_data_fn)["train"]

    if len(test_dataset) > max_sample_num:
        if type(test_dataset) == list:
            test_dataset = test_dataset[:max_sample_num]
        else:
            test_dataset = test_dataset.select(range(max_sample_num))

    main_model = create_vllm(main_model_name_path, 0.4, max_main_model_len)
    classmate_model = create_vllm(classmate_model_name_path, 0.4, classmate_max_len)
    if use_temperature:
        main_sampling_params = SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=main_max_response_len, seed=seed)
    else:
        main_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=main_max_response_len, seed=seed)
    classmate_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=classmate_max_response_len, seed=seed)

    output_dir = f"outputs/inference_before_train/deepscaler_{data_split}/main_{main_max_response_len}/{'w_temperature' if use_temperature else 'greedy'}/{main_model_name_path.replace('/', '_')}_with_{classmate_model_name_path.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    main_correct = 0
    classmate_correct = 0
    main_token_len_list = []
    classmate_token_len_list = []
    for idx, entry in tqdm(enumerate(test_dataset)):
        # {
        #     "prompt": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.",
        #     "answer": "204",
        #     "source": "aime",
        #     "id": "0"
        # },
        prompt = entry["prompt"]
        # prompt = entry["prompt"] + " Think step-by-step and provide the final answer in \\boxed{}."
        ground_truth = entry["answer"]

        # Pass to main model
        # main_prompt = tokenize_input(main_tokenizer, prompt + "/no_think", tokenize=False)
        # messages = [{"role": "user", "content": prompt}]
        # main_prompt = main_tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # test = main_model.generate(main_prompt, sampling_params=main_sampling_params)
        # test[0].outputs[0].text
        # classmate_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": truncated_main_pred}]
        # classmate_ids = classmate_tokenizer.apply_chat_template(classmate_messages, return_tensors="pt", tokenize=True, add_generation_prompt=False)
        # classmate_prompt = classmate_tokenizer.apply_chat_template(classmate_messages, return_tensors="pt", tokenize=False, add_generation_prompt=False, add_special_tokens=False)

        main_prompt = tokenize_input(main_tokenizer, prompt, tokenize=False)
        main_pred, main_pred_token_ids = generate(main_model, main_tokenizer, main_prompt, main_sampling_params, is_qwen=True)
        classmate_prompt, truncated_main_pred = get_classmate_prompt(prompt, main_pred, keep_ratio=main_cot_keep_ratio)
        classmate_prompt = tokenize_input(classmate_tokenizer, classmate_prompt, is_main=False, main_CoT=truncated_main_pred, tokenize=False)
        classmate_pred, classmate_pred_token_ids = generate(classmate_model, classmate_tokenizer, classmate_prompt, classmate_sampling_params, is_qwen=False)
        extracted_main, main_is_correct = verify_math_pred(main_pred, ground_truth)
        extracted_classmate, classmate_is_correct = verify_math_pred(classmate_pred, ground_truth)

        main_token_len_list.append(len(main_pred_token_ids))
        classmate_token_len_list.append(len(classmate_pred_token_ids))
        if main_is_correct:
            main_correct += 1
        if classmate_is_correct:
            classmate_correct += 1

        # Print results
        # print("----------------------Start----------------------")
        # print("🐛[Prompt]: ", prompt)
        # print("🐛[Ground Truth]: ", ground_truth)
        # print("----------------------Main----------------------")
        # print("🤖[Main Prompt]: ", main_prompt)
        # print("🤖[Main Pred]: ", main_pred)
        # print("🤖[Main extracted answer]: ", extracted_main)
        # print("🤖[Main Correct]: ", main_is_correct)
        # print("----------------------Classmate------------------------")
        # print("🤖[Classmate Prompt]: ", classmate_prompt)
        # print("🤖[Classmate Pred]: ", classmate_pred)
        # print("🤖[Classmate extracted answer]: ", extracted_classmate)
        # print("🤖[Classmate Correct]: ", classmate_is_correct)
        # print("----------------------End-----------------------")

        with open(os.path.join(output_dir, f"deepscalrr_{idx}.txt"), "a") as f:
            f.write(f"🐛[Prompt]: {prompt}\n")
            f.write(f"🐛[Ground Truth]: {ground_truth}\n")
            f.write(f"----------------------Main----------------------\n")
            f.write(f"🤖[Main Prompt]: {main_prompt}\n")
            f.write(f"🤖[Main Pred]: {main_pred}\n")
            f.write(f"🤖[Main extracted answer]: {extracted_main}\n")
            f.write(f"🤖[Main Correct]: {main_is_correct}\n")
            f.write(f"----------------------Classmate------------------------\n")
            f.write(f"🤖[Classmate Prompt]: {classmate_prompt}\n")
            f.write(f"🤖[Classmate Pred]: {classmate_pred}\n")
            f.write(f"🤖[Classmate extracted answer]: {extracted_classmate}\n")
            f.write(f"🤖[Classmate Correct]: {classmate_is_correct}\n")

    output_dict = {
        "main_acc": f"{main_correct}/{len(test_dataset)}={main_correct/len(test_dataset):.4f}",
        "classmate_acc": f"{classmate_correct}/{len(test_dataset)}={classmate_correct/len(test_dataset):.4f}",
        "main_token_len": f"{main_token_len_list}",
        "max_main_token_len": max(main_token_len_list),
        "min_main_token_len": min(main_token_len_list),
        "avg_main_token_len": sum(main_token_len_list) / len(main_token_len_list),

        "classmate_token_len": f"{classmate_token_len_list}",
        "max_classmate_token_len": max(classmate_token_len_list),
        "min_classmate_token_len": min(classmate_token_len_list),
        "avg_classmate_token_len": sum(classmate_token_len_list) / len(classmate_token_len_list),
    }
    with open(os.path.join(output_dir, f"deepscalrr.txt"), "a") as f:
        f.write(f"=================Final Results=================\n")
        for k, v in output_dict.items():
            f.write(f"{k}: {v}\n")

    # Plot token length distribution
    plot_token_length_distribution(main_token_len_list, output_dir, "Main_Model")
    plot_token_length_distribution(classmate_token_len_list, output_dir, "Classmate_Model")


if __name__ == "__main__":
    main()

#CUDA_VISIBLE_DEVICES=0 python test.py
#CUDA_VISIBLE_DEVICES=1 python test.py
#CUDA_VISIBLE_DEVICES=2 python test.py
#CUDA_VISIBLE_DEVICES=3 python test.py
# 1. think/no-thinking mode / \no_think
# 2. Try to switch to a smaller main model; if works, then start training