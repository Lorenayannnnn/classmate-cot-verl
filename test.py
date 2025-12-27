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
    return AutoTokenizer.from_pretrained(model_name_path.replace("-Turbo", ""),trust_remote_code=True)

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
        outputs = model.generate(prompt, sampling_params=sample_param)

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
    # dataset_name = "DeepScaleR"
    # data_source = None

    # dataset_name = "GSM_MATH"
    # data_source = "gsm8k"
    # data_source = "MATH"

    dataset_name = "hendrycks_math"
    # dataset_name = "hendrycks_math_think_step_by_step"
    data_source = None

    data_split = "train"
    max_sample_num = 100
    # data_split = "test"
    seed = 42
    use_temperature = True
    # use_temperature = False
    # main_model_name_path="Qwen/Qwen2.5-Math-1.5B"
    # main_model_name_path="Qwen/Qwen3-4B-Base"

    # main_model_name_path="Qwen/Qwen3-0.6B"
    main_model_name_path="Qwen/Qwen3-1.7B"
    # main_model_name_path="Qwen/Qwen3-4B"
    # main_model_name_path = "Qwen/Qwen3-4B-Instruct-2507"

    # classmate_model_name_path = "meta-llama/Llama-3.2-3B-Instruct"
    # use_together = False
    classmate_model_name_path = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    use_together = True

    print(f"Start testing {dataset_name} with main model: {main_model_name_path} and classmate model: {classmate_model_name_path}")

    max_main_input_len = 1024
    # main_max_response_len = 2048
    # main_max_response_len = 2048
    main_max_response_len = 3072
    # main_max_response_len = 4096
    # main_max_response_len = 8192
    max_main_model_len = max_main_input_len + main_max_response_len

    main_cot_keep_ratio = 0.8
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
        classmate_model = create_vllm(classmate_model_name_path, classmate_max_len, use_together=use_together)
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
        main_sampling_params = SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=main_max_response_len, seed=seed)
    else:
        main_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=main_max_response_len, seed=seed)

    output_dir = f"outputs/inference_before_train/{'use_together' if use_together else 'all_vllm'}/{dataset_name}_{data_split}_{data_source}/main_{main_max_response_len}/{'w_temperature' if use_temperature else 'greedy'}/{main_model_name_path.replace('/', '_')}_with_{classmate_model_name_path.replace('/', '_')}"
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


def test_togetherai_api():
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 20 Dec 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

user
Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$.
assistant
<think>

</think><|eot_id|><|start_header_id|>assistant<|end_header_id|>

We are given a triangle $ ABC $ inscribed in a circle $ \omega $, with side lengths $ AB = 5 $, $ BC = 9 $, and $ AC = 10 $. The tangents to $ \omega $ at $ B $ and $ C $ intersect at point $ D $, and the line $ AD $ intersects the circle again at point $ P $. We are to find the length $ AP $, expressed as a reduced fraction $ \frac{m}{n} $, and compute $ m + n $.

---

### Step 1: Use the Power of a Point Theorem

Let’s recall the **Power of a Point Theorem**:

> If a line through a point $ D $ intersects a circle at two points $ X $ and $ Y $, then:
> $$
> DX \cdot DY = \text{Power of } D \text{ w.r.t. } \omega
> $$

In this case, point $ D $ lies outside the circle $ \omega $, and the tangents from $ D $ to $ \omega $ touch the circle at $ B $ and $ C $. So, the power of point $ D $ with respect to $ \omega $ is:
$$
DB^2 = DC^2 = \text{Power of } D
$$

Also, the line $ AD $ intersects the circle again at $ P $, so:
$$
DA \cdot DP = DB^2
$$

So, we can write:
$$
DA \cdot DP = DB^2
$$

Let’s denote:
- $ DA = x $
- $ DP = y $
- Then $ AP = DA + DP = x + y $

But we also have:
$$
DA \cdot DP = DB^2 \Rightarrow x \cdot y = DB^2
$$

So, we can write:
$$
x \cdot y = DB^2 \quad \text{and} \quad x + y = AP
$$

We need to find $ AP = x + y $, given that $ x \cdot y = DB^2 $.

So, if we can find $ DB $, we can find $ x \cdot y $, and then solve for $ x + y $.

---

### Step 2: Use the Law of Cosines to Find $ \angle BAC $

We are given triangle $ ABC $ with sides:
- $ AB = 5 $
- $ BC = 9 $
- $ AC = 10 $

We can use the **Law of Cosines** to find $ \angle BAC $, which is the angle between sides $ AB $ and $ AC $, and is important for computing the angle between the tangents at $ B $ and $ C $, which will help us find $ \angle BDC $, and hence $ \angle BDC = 180^\circ - \angle BAC $.

Let’s compute $ \cos \angle BAC $:

$$
\cos \angle BAC = \frac{AB^2 + AC^2 - BC^2}{2 \cdot AB \cdot AC}
= \frac{5^2 + 10^2 - 9^2}{2 \cdot 5 \cdot 10}
= \frac{25 + 100 - 81}{100}
= \frac{44}{100} = \frac{11}{25}
$$

So, $ \angle BAC = \cos^{-1}\left( \frac{11}{25} \right) $

---

### Step 3: Use the Tangent-Secant Angle Theorem

Let’s recall the **Tangent-Secant Angle Theorem**:

> The angle between a tangent and a chord is equal to the angle in the alternate segment.

So, the angle between tangent $ DB $ and chord $ AB $ is equal to the angle $ \angle ACB $, and the angle between tangent $ DC $ and chord $ AC $ is equal to the angle $ \angle ABC $.

Therefore, the angle $ \angle BDC $ is equal to the angle between the two tangents at $ B $ and $ C $, and is equal to:
$$
\angle BDC = 180^\circ - \angle BAC
$$

So, $ \angle BDC = 180^\circ - \angle BAC $

Now, we can use the **Law of Cosines** in triangle $ BDC $ to find $ DB $, since we know $ BC = 9 $, and we can find $ \angle BDC $.

Let’s compute $ \angle BDC $:

$$
\angle BDC = 180^\circ - \angle BAC = 180^\circ - \cos^{-1}\left( \frac{11}{25} \right)
$$

Let’s compute $ \cos \angle BDC $:

$$
\cos \angle BDC = \cos(180^\circ - \angle BAC) = -\cos \angle BAC = -\frac{11}{25}
$$

Now, in triangle $ BDC $, we have:
- $ BC = 9 $
- $ \angle BDC = \cos^{-1}\left( -\frac{11}{25} \right) $
- $ DB = DC $ (since $ D $ lies on the tangents at $ B $ and $ C $, and the tangents are equal in length from a point outside the circle)

So, triangle $ BDC $ is isosceles with $ DB = DC $, and we can use the Law of Cosines to find $ DB $:

$$
BC^2 = DB^2 + DC^2 - 2 \cdot DB \cdot DC \cdot \cos \angle BDC
$$

Since $ DB = DC $, let’s denote $ DB = DC = x $. Then:

$$
9^2 = x^2 + x^2 - 2x^2 \cdot \left( -\frac{11}{25} \right)
= 2x^2 + \frac{22}{25}x^2
= \left( 2 + \frac{22}{25} \right)x^2
= \frac{72}{25}x^2
$$

So:

$$
81 = \frac{72}{25}x^2 \Rightarrow x^2 = \frac{81 \cdot 25}{72} = \frac{2025}{72}
$$

Therefore:

$$
DB^2 = \frac{2025}{72}
$$

---

### Step 4: Use the Power of a Point to Find $ AP $

We now know that:

$$
DA \cdot DP = DB^2 = \frac{2025}{72}
$$

Let’s denote $ DA = x $, $ DP = y $, so that:

$$
x \cdot y = \frac{2025}{72}
$$

Also, since $ AP = DA + DP = x + y $, we want to find $ x + y $, given that $ x \cdot y = \frac{2025}{72} $

Let’s denote $ x + y = S $, and $ x \cdot y = P = \frac{2025}{72} $

Then, the quadratic"""

    # from openai import OpenAI
    client = Together()
    # response = client.completions.create(model="meta-llama/Llama-3.2-3B-Instruct-Turbo",prompt=prompt,temperature=0, top_k=1, top_p=1.0, max_tokens=409, seed=42, n=1).choices[0].text
    response = client.completions.create(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",prompt=prompt,temperature=0, top_k=1, top_p=1.0, max_tokens=409, seed=42).choices[0].text
    # openai = OpenAI(
    #     api_key="yafq4uYi5X3T2II8d1aFrBhaYkqy16Ur",
    #     base_url="https://api.deepinfra.com/v1/openai",
    # )

    # breakpoint()
    # chat_completion = openai.completions.create(model="meta-llama/Llama-3.2-3B-Instruct", prompt=prompt)
    # chat_completion = openai.chat.completions.create(model="meta-llama/Llama-3.2-3B-Instruct",messages=[{"role": "user", "content": prompt}],)
    #
    # print(chat_completion.choices[0].message.content)
    # print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

if __name__ == "__main__":
    main()
    # test_togetherai_api()

#VLLM_CONFIGURE_LOGGING=0 TOGETHER_API_KEY="f3fec9e45ab2c98b73b83faaf7a329b07069ed7cbc7655614420b47fda16cab1" CUDA_VISIBLE_DEVICES=2 python test.py
#VLLM_CONFIGURE_LOGGING=0 TOGETHER_API_KEY="f3fec9e45ab2c98b73b83faaf7a329b07069ed7cbc7655614420b47fda16cab1" CUDA_VISIBLE_DEVICES=3 python test.py
#VLLM_CONFIGURE_LOGGING=0 TOGETHER_API_KEY="f3fec9e45ab2c98b73b83faaf7a329b07069ed7cbc7655614420b47fda16cab1" CUDA_VISIBLE_DEVICES=4 python test.py
#VLLM_CONFIGURE_LOGGING=0 TOGETHER_API_KEY="f3fec9e45ab2c98b73b83faaf7a329b07069ed7cbc7655614420b47fda16cab1" CUDA_VISIBLE_DEVICES=5 python test.py
#VLLM_CONFIGURE_LOGGING=0 TOGETHER_API_KEY="f3fec9e45ab2c98b73b83faaf7a329b07069ed7cbc7655614420b47fda16cab1" CUDA_VISIBLE_DEVICES=6 python test.py
#CUDA_VISIBLE_DEVICES=0 python test.py
#CUDA_VISIBLE_DEVICES=1 python test.py
#CUDA_VISIBLE_DEVICES=2 python test.py
#CUDA_VISIBLE_DEVICES=3 python test.py
# 1. think/no-thinking mode / \no_think
# 2. Try to switch to a smaller main model; if works, then start training