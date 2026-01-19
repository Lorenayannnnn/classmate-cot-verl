import os.path
import re

from tqdm import tqdm
from datasets import load_dataset
from math_verify import parse, verify
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from together import Together

from verl.code_utils.testing_util import TimeoutException


def create_tokenizer(model_name_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_path.replace("-Turbo", ""),trust_remote_code=True)
    if "llama" in model_name_path.lower() and not tokenizer.chat_template:
        tokenizer.chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n"""
    elif "OLMo-2-0425-1B" in model_name_path and not tokenizer.chat_template:
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"
    return tokenizer

def create_vllm(model_name_path, max_model_len, gpu_memory_utilization=None, use_together=False):
    if use_together:
        model = Together()
    else:
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
    if type(prompt) == list:
        messages = prompt
    else:
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


def generate(model, model_name_or_path, tokenizer, prompt, sample_param, is_qwen=False):
    # sample_param = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=8192)
    if type(model) == Together:
        # import requests
        #
        # payload = {
        #     "model": model_name_or_path,
        #     "prompt": prompt,
        #     **sample_param
        # }
        # headers = {"Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}","Content-Type": "application/json"}
        # outputs = requests.post("https://api.together.xyz/inference", json=payload, headers=headers).json()
        outputs = model.completions.create(
            model=model_name_or_path,
            prompt=prompt,
            **sample_param
        )
        # outputs.choices[0].text
        # breakpoint()
    else:

        # batch_prompts = [prompt, prompt, prompt, prompt, prompt]
        # batch_outputs = model.generate(batch_prompts, sampling_params=sample_param, use_tqdm=False)
        # breakpoint()
        # texts = [output.outputs[0].text for output in batch_outputs]
        outputs = model.generate(prompt, sampling_params=sample_param, use_tqdm=False)

    if is_qwen:
        raw_token_ids = outputs[0].outputs[0].token_ids
        # start_think_token_id = tokenizer.convert_tokens_to_ids("<think>")
        # end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")
        # # remove segment between <think> and </think>
        # if start_think_token_id in raw_token_ids and end_think_token_id in raw_token_ids:
        #     start_idx = raw_token_ids.index(start_think_token_id)
        #     end_idx = raw_token_ids.index(end_think_token_id) + 1
        #     token_ids = raw_token_ids[:start_idx] + raw_token_ids[end_idx:]
        # else:
        # token_ids = raw_token_ids
        pred = tokenizer.decode(raw_token_ids, skip_special_tokens=True).strip()
        pred_token_ids = raw_token_ids
    else:
        if type(model) == Together:
            pred = outputs.choices[0].text
            pred_token_ids = tokenizer.decode(tokenizer.encode(pred, add_special_tokens=False), skip_special_tokens=False)
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
    extracted_sol = None
    ground_truth_boxed = "\\boxed{" + gt + "}"
    try:
        # solution_str = parse(solution_str, parsing_timeout=None)
        # ground_truth = parse(ground_truth, parsing_timeout=None)
        # score = 1 if verify(solution_str, ground_truth, timeout_seconds=None) else 0
        extracted_sol = parse(pred)
        ground_truth = parse(ground_truth_boxed)
        score = 1 if verify(extracted_sol, ground_truth) else 0
    except TimeoutException | TimeoutError as e:
        # print("Timeout exception during math_verify.")
        # logger.error("Timeout exception during math_verify.")
        score = 0
    except Exception as e:
        # print("Error during math_verify:", e)
        # logger.error(f"Error during math_verify: {e}")
        score = 0
    return str(extracted_sol), True if score == 1 else False

# main_model_name_to_sampling_param = {
#     "Qwen/Qwen3-1.7B": {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
#     "Qwen/Qwen3-4B": {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
#     "Qwen/Qwen3-4B-Base": {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
# }

def plot_token_length_distribution(token_len_list, output_dir, model_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    # subplot token_len_list
    plt.hist(token_len_list["correct"], bins=20, color='green', alpha=0.7, label='Correct')
    plt.hist(token_len_list["wrong"], bins=20, color='red', alpha=0.7, label='Wrong')
    plt.legend()
    plt.title(f'Token Length Distribution for {model_name} (Correct vs Wrong)')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(output_dir, f'AA_{model_name}_token_length_distribution_correct_vs_wrong.png'))
    plt.close()

def main():
    # dataset_name = "GSM_MATH"
    # data_source = "gsm8k"
    # data_source = "MATH"

    # dataset_name = "gsm8k_no_think_prompt"
    # dataset_name = "gsm8k_final_ans_is"

    # dataset_name = "hendrycks_math"
    # dataset_name = "hendrycks_math_think_step_by_step"

    # dataset_name = "gsm8k_minimal_answer_box_prompt"
    # dataset_name = "hendrycks_math_minimal_answer_box_prompt_105_test"

    dataset_name = "DeepScaleR_answer_box"
    data_source = None

    data_split = "train"
    max_sample_num = 100
    # data_split = "test"
    seed = 42
    use_temperature = True
    # use_temperature = False
    # main_model_name_path="Qwen/Qwen2.5-Math-1.5B"
    # main_model_name_path="Qwen/Qwen3-0.6B"
    # main_model_name_path="Qwen/Qwen3-1.7B"
    # main_model_name_path="Qwen/Qwen3-4B"
    # main_model_name_path = "Qwen/Qwen3-4B-Instruct-2507"

    # main_model_name_path="Qwen/Qwen3-0.6B-Base"
    main_model_name_path="Qwen/Qwen3-1.7B-Base"
    # main_model_name_path="Qwen/Qwen3-4B-Base"
    # main_model_name_path="Qwen/Qwen3-8B-Base"

    # main_model_name_path="allenai/OLMo-2-0425-1B"
    # main_model_name_path="allenai/OLMo-2-0425-1B-SFT"
    # main_model_name_path="allenai/OLMo-2-0425-1B-DPO"
    # main_model_name_path="allenai/OLMo-2-0425-1B-RLVR1"

    # main_model_name_path="meta-llama/Llama-3.2-1B"
    # main_model_name_path="meta-llama/Llama-3.2-1B-Instruct"

    classmate_model_name_path = "meta-llama/Llama-3.2-1B-Instruct"
    # use_together = False
    # classmate_model_name_path = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    use_together = False
    main_cot_keep_ratio = 0.5

    print(f"Start testing {dataset_name} with main model: {main_model_name_path} and classmate model: {classmate_model_name_path}")

    max_main_input_len = 1024
    # main_max_response_len = 2048
    # main_max_response_len = 2048
    main_max_response_len = 3072
    # main_max_response_len = 4096
    # main_max_response_len = 8192
    max_main_model_len = max_main_input_len + main_max_response_len

    classmate_max_input_len = int(max_main_input_len + main_max_response_len * main_cot_keep_ratio)
    classmate_max_response_len = int(main_max_response_len * (1 - main_cot_keep_ratio))
    classmate_max_len = classmate_max_input_len + classmate_max_response_len
    main_tokenizer = create_tokenizer(main_model_name_path)

    classmate_tokenizer = create_tokenizer(classmate_model_name_path)
    # Filter too long examples for main model
    model_len_filtered_data_fn = f"data/{dataset_name}/{main_model_name_path}_{max_main_input_len}_{data_split}_filtered.json"
    if not os.path.exists(model_len_filtered_data_fn):
        if dataset_name == "DeepScaleR":
            test_json_fn = f"data/{dataset_name}/{data_split}.json"
            test_dataset = load_dataset("json", data_files=test_json_fn)["train"]
        else:
            # load from parquet
            test_json_fn = f"data/{dataset_name}/{data_split}.parquet"
            test_dataset = load_dataset("parquet", data_files=test_json_fn)["train"]
        test_dataset = filter_too_long_examples(test_dataset, main_tokenizer, max_main_input_len)
        # save filtered dataset
        os.makedirs(os.path.dirname(model_len_filtered_data_fn), exist_ok=True)
        with open(model_len_filtered_data_fn, "w") as f:
            import json
            json.dump(test_dataset, f, indent=4)
    else:
        test_dataset = load_dataset("json", data_files=model_len_filtered_data_fn)["train"]

    if data_source is not None:
        # further filter by data_source
        test_dataset = [entry for entry in test_dataset if entry.get("source", entry.get("data_source", None)) == data_source]

    if len(test_dataset) > max_sample_num:
        if type(test_dataset) == list:
            test_dataset = test_dataset[:max_sample_num]
        else:
            test_dataset = test_dataset.select(range(max_sample_num))

    if use_together:
        main_model = create_vllm(main_model_name_path, max_main_model_len, gpu_memory_utilization=0.9)
        classmate_model = create_vllm(classmate_model_name_path, classmate_max_len, 0.3, use_together=use_together)
        classmate_sampling_params = {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "max_tokens": classmate_max_response_len,
            "seed": seed
        }
    else:
        main_model = create_vllm(main_model_name_path, max_main_model_len, gpu_memory_utilization=0.4)
        classmate_model = create_vllm(classmate_model_name_path, classmate_max_len, gpu_memory_utilization=0.4, use_together=use_together)
        classmate_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=classmate_max_response_len, seed=seed)

    if use_temperature:
        main_sampling_params = SamplingParams(temperature=1, top_k=-1, top_p=1, max_tokens=main_max_response_len, seed=seed)
    else:
        main_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=main_max_response_len, seed=seed)

    output_dir = f"outputs/inference_before_train/{'use_together' if use_together else 'all_vllm'}/{dataset_name}_{data_split}_{data_source}/main_{main_max_response_len}/{'w_temperature' if use_temperature else 'greedy'}/main_cot_keep_ratio_{main_cot_keep_ratio}/{main_model_name_path.replace('/', '_')}_with_{classmate_model_name_path.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    main_correct = 0
    classmate_correct = 0
    main_token_len_list = {
        "correct": [-1],
        "wrong": [-1]
    }
    classmate_token_len_list = {
        "correct": [-1],
        "wrong": [-1]
    }
    for idx, entry in tqdm(enumerate(test_dataset)):
        # {
        #     "prompt": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.",
        #     "answer": "204",
        #     "source": "aime",
        #     "id": "0"
        # },
        prompt = entry["prompt"]
        # prompt = entry["prompt"] + " Think step-by-step and provide the final answer in \\boxed{}."
        if "answer" in entry:
            ground_truth = entry["answer"]
        elif "reward_model" in entry and "ground_truth" in entry["reward_model"]:
            ground_truth = entry["reward_model"]["ground_truth"]
        else:
            raise ValueError("No ground truth found in the dataset entry.")

        # Pass to main model
        # main_prompt = tokenize_input(main_tokenizer, prompt + "/no_think", tokenize=False)
        # messages = [{"role": "user", "content": prompt}]
        # main_prompt = main_tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # test = main_model.generate(main_prompt, sampling_params=main_sampling_params)
        # test[0].outputs[0].text
        # classmate_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": truncated_main_pred}]
        # classmate_ids = classmate_tokenizer.apply_chat_template(classmate_messages, return_tensors="pt", tokenize=True, add_generation_prompt=False)
        # classmate_prompt = classmate_tokenizer.apply_chat_template(classmate_messages, return_tensors="pt", tokenize=False, add_generation_prompt=False, add_special_tokens=False)

        # breakpoint()
        # classmate_messages = prompt
        # classmate_prompt = classmate_tokenizer.apply_chat_template(classmate_messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False)
        # classmate_prompt = classmate_tokenizer.apply_chat_template(prompt, tokenize=False)
        main_prompt = tokenize_input(main_tokenizer, prompt, tokenize=False)
        main_pred, main_pred_token_ids = generate(main_model, main_model_name_path, main_tokenizer, main_prompt, main_sampling_params, is_qwen=True)
        classmate_prompt, truncated_main_pred = get_classmate_prompt(prompt, main_pred, keep_ratio=main_cot_keep_ratio)
        classmate_prompt = tokenize_input(classmate_tokenizer, classmate_prompt, is_main=False, main_CoT=truncated_main_pred, tokenize=False)
        classmate_pred, classmate_pred_token_ids = generate(classmate_model, classmate_model_name_path, classmate_tokenizer, classmate_prompt, classmate_sampling_params, is_qwen=False)
        extracted_main, main_is_correct = verify_math_pred(main_pred, ground_truth)
        extracted_classmate, classmate_is_correct = verify_math_pred(classmate_pred, ground_truth)

        if main_is_correct:
            main_correct += 1
            main_token_len_list["correct"].append(len(main_pred_token_ids))
        else:
            main_token_len_list["wrong"].append(len(main_pred_token_ids))
        if classmate_is_correct:
            classmate_correct += 1
            classmate_token_len_list["correct"].append(len(classmate_pred_token_ids))
        else:
            classmate_token_len_list["wrong"].append(len(classmate_pred_token_ids))

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

        with open(os.path.join(output_dir, f"{idx}_m_{'c' if main_is_correct else 'w'}_cl_{'c' if classmate_is_correct else 'w'}.txt"), "a") as f:
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
        "max_main_token_len": {"correct": max(main_token_len_list["correct"]), "wrong": max(main_token_len_list["wrong"])},
        "min_main_token_len": {"correct": min(main_token_len_list["correct"]), "wrong": min(main_token_len_list["wrong"])},
        "avg_main_token_len": {"correct": sum(main_token_len_list["correct"]) / len(main_token_len_list["correct"]) if len(main_token_len_list["correct"]) > 0 else 0,},

        "classmate_token_len": f"{classmate_token_len_list}",
        "max_classmate_token_len": {"correct": max(classmate_token_len_list["correct"]), "wrong": max(classmate_token_len_list["wrong"])},
        "min_classmate_token_len": {"correct": min(classmate_token_len_list["correct"]), "wrong": min(classmate_token_len_list["wrong"])},
        "avg_classmate_token_len": {"correct": sum(classmate_token_len_list["correct"]) / len(classmate_token_len_list["correct"]) if len(classmate_token_len_list["correct"]) > 0 else 0,},
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

# TOGETHER_API_KEY="6cf968d54220fa0ee7ff5b256b31e7745bc52e252b71798b731deb2b542d9c56"
#CUDA_VISIBLE_DEVICES=2 python test.py
#CUDA_VISIBLE_DEVICES=3 python test.py
# 1. think/no-thinking mode / \no_think
# 2. Try to switch to a smaller main model; if works, then start training