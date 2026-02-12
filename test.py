import json
import os.path
import re

from tqdm import tqdm
from datasets import load_dataset
from math_verify import parse, verify
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
# from together import Together

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
        raise NotImplementedError()
        # model = Together()
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


def tokenize_input(tokenizer, prompt, is_main=True, main_CoT=None, tokenize=False, enable_thinking=False):
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
        return tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=True, enable_thinking=enable_thinking)


def filter_too_long_examples(dataset, tokenizer, max_length, enable_thinking):
    filtered_dataset = []
    for entry in dataset:
        prompt = entry["prompt"]
        tokenized_prompt = tokenize_input(tokenizer, prompt, tokenize=True, enable_thinking=enable_thinking)
        if len(tokenized_prompt) <= max_length:
            filtered_dataset.append(entry)
    return filtered_dataset


def generate(model, model_name_or_path, tokenizer, prompt, sample_param, is_qwen=False):
    # sample_param = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=8192)
    # if type(model) == Together:
        # import requests
        #
        # payload = {
        #     "model": model_name_or_path,
        #     "prompt": prompt,
        #     **sample_param
        # }
        # headers = {"Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}","Content-Type": "application/json"}
        # outputs = requests.post("https://api.together.xyz/inference", json=payload, headers=headers).json()
        # outputs = model.completions.create(
        #     model=model_name_or_path,
        #     prompt=prompt,
        #     **sample_param
        # )
        # outputs.choices[0].text
        # breakpoint()
    # else:

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
        # if type(model) == Together:
        #     pred = outputs.choices[0].text
        #     pred_token_ids = tokenizer.decode(tokenizer.encode(pred, add_special_tokens=False), skip_special_tokens=False)
        # else:
        pred = outputs[0].outputs[0].text
        pred_token_ids = outputs[0].outputs[0].token_ids

    from vllm import TokensPrompt

    user_q = "Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were\n(A) Adams only.\n(B) Brooks only.\n(C) Case only.\n(D) Adams and Brooks"
    main_cot = """\n\nOkay, let\'s try to figure out this law problem. So, Davis wanted to kill Adams. He went to Adams\'s house. Before he got there, he saw Brooks, who looked like Adams. He thought Brooks was Adams, so he shot at Brooks. The shot missed, but it wounded Case, who was some distance away. Davis didn\'t see Case. Now, the question is about what the prosecutor should indicate as the intended victim(s) under a law that prohibits any attempt to commit murder.\n\nThe options are (A) Adams only, (B) Brooks only, (C) Case only, (D) Adams and Brooks. \n\nFirst, I need to recall the legal principles here. The key points are: Davis intended to kill Adams, thought Brooks was Adams, shot Brooks, but the shot missed and wounded Case. The question is about whether Davis\'s actions count as an attempt to murder, and if so, who is the intended victim.\n\nIn criminal law, an attempt is a specific intent to commit a crime, and the person has taken steps toward the commission of the crime. But the key here is whether the person actually intended to kill someone. But in this case, Davis intended to kill Adams. However, he thought Brooks was Adams. So, he shot Brooks, but he didn\'t intend to kill Brooks. But he was trying to kill Adams. The problem is that the shot missed, but it wounded Case. But Davis didn\'t see Case. \n\nSo, the question is, under the statute that proscribes any attempt to commit murder, the prosecutor should indicate that the intended victim was... \n\nNow, in criminal law, if someone has an intent to kill, even if the shot misses and hits someone else, the attempt is still a separate offense. But the key is whether the person actually intended to kill someone. \n\nHowever, the problem is that Davis intended to kill Adams. But he thought Brooks was Adams. So, he shot Brooks. But he didn\'t intend to kill Brooks. Wait, but the statute says "any attempt to commit murder," so the intent to commit murder is the key. So, even if he shot Brooks, but the intent was to kill Adams, perhaps he\'s not trying to kill Brooks. \n\nWait, but if he intended to kill Adams, but in the process, he shot Brooks. So, is that an attempt to murder Adams? Or is it an attempt to kill Brooks? Because he thought Brooks was Adams. But even if he thought he was shooting Brooks, but he intended to kill Adams, then maybe the attempt is to kill both Adams and Brooks? But that\'s not necessarily the case. \n\nAlternatively, maybe the intent is not to kill Brooks, but the act of shooting Brooks is an attempt to kill Adams. However, the statute is about attempts to commit murder. So, if Davis intended to kill Adams, and he shot Brooks, but didn\'t intend to kill Brooks, then the attempt would be to kill Adams. But since he shot Brooks, which is a different person, but he didn\'t intend to kill Brooks, then the attempt is to kill Adams. But in that case, the prosecutor would have to indicate that the intended victim was Adams. \n\nBut the problem is that the shot wound Case. But Davis didn\'t see Case. So, even though he shot Brooks, he didn\'t intend to kill Case. But he was trying to kill Adams. \n\nSo, the key here is whether the act of shooting Brooks was a mistake, or if the intent is to kill Adams. If the attempted murder is an attempt to kill Adams, then even though he shot Brooks, he\'s still trying to kill Adams, and the attempt is to kill Adams. Therefore, the prosecutor would have to state that the intended victims were Adams. \n\nBut wait, hold on. In some jurisdictions, if a person attempts to kill one person, but the act is directed at another, but the intent was to kill the original, then the attempt is to kill the original. But in this case, he shot someone else, but he intended to kill Adams. So, the attempt is to kill Adams. Therefore, the prosecutor would have to show that the intended victim was Adams, but he shot Brooks. But the question is, under the statute that proscribes any attempt to commit murder, the prosecutor should indicate that the intended victim(s) was/were. \n\nSo, if the intent is to kill Adams, then the intended victim is Adams. So the answer would be (A) Adams only. But wait, but the problem is that he shot Brooks, but the shot missed. So, he didn\'t kill Brooks, but he might have been trying to kill Adams. But perhaps the prosecutor would have to say that he intended to kill Adams, and that the attempt was to kill Adams. However, the law might require that the intent is to kill someone, and if the attempt is made at another person, then the intent is still to kill the original. \n\nBut there\'s another angle here. In the case where the person shoots another person, and the person\'s intent was to kill the original, but in this case, the person\'s intent was to kill Adams, but he shot Brooks, which is a different person. So, in that case, even though he shot Brooks, the intent was to kill Adams. However, the law might require that the prosecutor show that the intended victim was Adams. Because the attempt was made to kill Adams, even though he shot Brooks. \n\nAlternatively, if the person had actually intended to kill Brooks, but the statute says "any attempt to commit murder," then the prosecutor would have to indicate that the intended victims were Brooks. But in this case, the person intended to kill Adams, but thought Brooks was Adams, so he shot Brooks. But the intent was to kill Adams. Therefore, the attempted murder is to kill Adams. So, the answer would be (A) Adams only. \n\nBut wait, there\'s a possibility that the shot at Brooks was an attempt to kill Adams. Even if the shot didn\'t hit Brooks, but he intended to kill Adams. So, the attempt is to kill Adams. Therefore, the prosecutor would have to indicate that the intended victim was Adams. \n\nBut then, what about the case that was wounded? The shot wound Case. But the statute is about attempts to commit murder, not about the actual victims. So, the law would not be concerned with the actual victims, except that if the attempt was made to kill someone, the prosecutor would have to show that the intent was to kill the person. \n\nTherefore, the answer would be (A) Adams only. But wait, but the question is about the intended victim(s) that the prosecutor should indicate. \n\nBut wait, is there an element here where the person actually shot Brooks, but the intent was to kill Adams. So, even though the shot was at Brooks, the intent was to kill Adams. Therefore, the prosecutor would have to say that the intended victims were Adams. \n\nAlternatively, maybe the prosecution has to show that the intent was to kill both Adams and Brooks. But that seems unlikely because he thought Brooks was Adams. So, he thought that Brooks was Adams, and he shot Brooks, but the intent was to kill Adams. Therefore, the intended victims are Adams. \n\nBut in terms of legal principles, an attempt is not necessarily tied to the target. For example, if someone is trying to kill A, but while doing so, they shoot B, but the intent was A. Then, the attempt is a separate act, and the intent is to kill A. So the prosecutor would have to show that the intent was A. \n\nSo, in this case, the prosecutor would indicate that the intended victim was Adams only, so answer (A). But wait, another angle: the statute says "any attempt to commit murder." So, if Davis attempted to commit murder against Adams, but he shot Brooks. But the statute is about any attempt. So, if the act of shooting Brooks was an attempt to commit murder, then the intent is to kill Adams, and the attempt is to kill Adams, regardless of who was shot. \n\nBut then, the prosecutor is required to indicate that the intended victim was Adams, because that\'s the person the intent was directed towards. \n\nBut wait, but what if the shooting at Brooks was an attempt to kill Adams? Even if he shot Brooks, but the intent was Adams, then the attempt was toward Adams. Therefore, the answer would be (A) Adams only. \n\nHowever, in some jurisdictions, if the actual victim is someone else, but the intent was to kill the original, the attempt is still considered to be an attempt to kill the original. Therefore, the prosecutor would have to indicate that the intended victim was Adams. \n\nBut I need to think about any exceptions. For example, in some cases, if the crime is an attempt to kill a specific person, but the act is directed at someone else, but the intent was to kill the original. Then, the attempt is still to kill the original. Therefore, the answer is (A). \n\nAlternatively, maybe the answer is (D) Adams and Brooks. But that would be if the person intended to kill both. But the problem is that he thought Brooks was Adams. So, he was not intending to kill Brooks. But he was trying to kill Adams. So, even if he shot Brooks, but the intent was Adams, he\'s not trying to kill Brooks. Therefore, the answer would be (A). \n\nWait, but the statute says "any attempt to commit murder." So, if Davis intended to kill Adams, and he shot Brooks, that\'s an attempt to commit murder. So, the attempt is to kill Adams, even though the shooting was at Brooks. Therefore, the prosecutor would have to indicate that the intended victim was Adams. So, (A). \n\nBut the answer might be (B) Brooks only. Wait, why? If he thought Brooks was Adams, and he shot Brooks, but he intended to kill Adams, then the attempt is to kill Adams. However, the shooting at Brooks is an attempt, and the intent is Adams. Therefore, the answer is (A). \n\nBut wait, the question is phrased as "the district attorney should indicate that the intended victim(s) was/were". If the prosecution is trying to charge Davis with an attempt to commit murder, then the intent must be to kill someone. So, the prosecution would have to show that the intent was to kill Adams. Therefore, the answer is (A). \n\nBut maybe there\'s another angle. If the shot wound Case, but the prosecution is accusing Davis of an attempt to commit murder. But the statute is about attempts to commit murder, not about the actual injuries. So, the act of shooting Brooks was an attempt, but the intended victim was Adams. Therefore, the answer is (A). \n\nWait, but there\'s another aspect. In some cases, if the person takes an action that is an attempt to kill someone, but it\'s directed at another person, then the attempt is considered to be an attempt to kill the person they intended. But if the person intended to kill someone, but their act is directed at another, then the attempt is to kill the original. \n\nFor example, if you intend to kill A, but shoot B, then the attempt is to kill A. So, the prosecutor would have to indicate that the intended victim was A. Therefore, answer (A). \n\nTherefore, I think the answer is (A) Adams only. But wait, but what if the statute requires that the intended victim is the one who was targeted, even if they were shot by mistake? For instance, if the act is an attempt, the intent is to kill the victim. So, even if the act was directed at Brooks, but the intent was Adams, then the answer is (A). \n\nBut I need to make sure. Another thought: in some cases, if the person shoots someone else, but that someone else was not the intended victim, but the person\'s intent was to kill the original, then the attempt is to kill the original. Therefore, the answer is (A). \n\nTherefore, I think the answer is (A) Adams only. So, the correct choice is (A).\n"""
    user_only_messages = [{"role": "user", "content": user_q}]
    messages = [{"role": "user", "content": user_q}, {"role": "assistant", "content": f"<think>{main_cot}</think>"}]
    user_only_input_tokens = tokenizer.apply_chat_template(user_only_messages, tokenize=True, add_generation_prompt=True)
    input_tokens = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)[:-2]
    input_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)[:-2]
    sample_param.prompt_logprobs = 0
    outputs = model.generate(TokensPrompt(prompt_token_ids=input_tokens), sampling_params=sample_param, use_tqdm=False)
    user_only_outputs = model.generate(TokensPrompt(prompt_token_ids=user_only_input_tokens), sampling_params=sample_param, use_tqdm=False)
    token_prob = [list(tok.values())[0].logprob if tok is not None else float("-inf") for tok in outputs[0].prompt_logprobs]
    user_only_token_prob = [list(tok.values())[0].logprob if tok is not None else float("-inf") for tok in user_only_outputs[0].prompt_logprobs]
    # len(token_prob) - len(user_only_token_prob) == len(input_tokens) - len(user_only_input_tokens)

    #
    breakpoint()
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
        extracted_sol = parse(pred)
        ground_truth = parse(ground_truth_boxed)
        score = 1 if verify(ground_truth, extracted_sol) else 0
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
    # dataset_name = "hendrycks_math_minimal_answer_box_prompt_700_heldout"
    # dataset_name = "hendrycks_math_q_type"

    dataset_name = "mmlu_q_hint_sep"

    reward_type = "vanilla_truncate"
    # reward_type = "help_other_questions"
    # reward_type = "help_other_similar_questions_w_guide_colon"

    # dataset_name = "DeepScaleR_answer_box"
    data_source = None

    data_split = "train"
    max_sample_num = 100
    # data_split = "test"
    seed = 42
    use_temperature = True
    # use_temperature = False

    enable_thinking = True

    # main_model_name_path="Qwen/Qwen2.5-Math-1.5B"
    # main_model_name_path="Qwen/Qwen3-0.6B"
    main_model_name_path="Qwen/Qwen3-1.7B"
    # main_model_name_path="Qwen/Qwen3-4B"
    # main_model_name_path = "Qwen/Qwen3-4B-Instruct-2507"

    # main_model_name_path="Qwen/Qwen3-0.6B-Base"
    # main_model_name_path="Qwen/Qwen3-1.7B-Base"
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
    main_cot_keep_ratio = 0.8

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
        test_dataset = filter_too_long_examples(test_dataset, main_tokenizer, max_main_input_len, enable_thinking)
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

    output_dir = f"outputs/inference_before_train/{'use_together' if use_together else 'all_vllm'}/enable_thinking_{enable_thinking}/{dataset_name}_{data_split}_{data_source}/main_{main_max_response_len}/{'w_temperature' if use_temperature else 'greedy'}/{reward_type if 'help' in reward_type else f'{reward_type}/main_cot_keep_ratio_{main_cot_keep_ratio}'}/{main_model_name_path.replace('/', '_')}_with_{classmate_model_name_path.replace('/', '_')}"
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
        main_prompt = tokenize_input(main_tokenizer, prompt, tokenize=False, enable_thinking=enable_thinking)
        main_pred, main_pred_token_ids = generate(main_model, main_model_name_path, main_tokenizer, main_prompt, main_sampling_params, is_qwen=True)
        wo_main_prompt = tokenize_input(main_tokenizer, prompt, tokenize=False, enable_thinking=False)
        wo_thinking_main_pred, wo_thinking_main_pred_token_ids = generate(main_model, main_model_name_path, main_tokenizer, wo_main_prompt, main_sampling_params, is_qwen=True)

        breakpoint()

        if reward_type == "help_other_questions":
            # find another question with the same type
            cur_type = entry["reward_model"]["type"]
            similar_entry = None
            for other_entry in test_dataset:
                if other_entry == entry:
                    continue
                similar_entry = other_entry  # default to first different question in case no same type found
                other_type = other_entry["reward_model"]["type"]
                if other_type != cur_type:
                    similar_entry = other_entry
                    break
            if similar_entry is not None:
                similar_prompt = similar_entry["prompt"]
                # similar_ground_truth = similar_entry["answer"] if "answer" in similar_entry else similar_entry["reward_model"]["ground_truth"]
                # prompt = f"Question: {similar_prompt}\n"
                # prompt += f"Solution: {main_pred}\n"
                # prompt += f"Based on the above question and solution, please help answer the following similar math question:\n"
                # prompt += f"{prompt}"

            assert similar_entry is not None, "No similar entry found for help_other_similar_questions."

            similar_q = similar_prompt[0]["content"]
            if reward_type == "help_other_similar_questions_w_guide":
                similar_q = "Given the above question and the solution as an example, solve the following question: \n" + similar_q

            messages = [
                {"role": "user", "content": prompt[0]["content"]},
                {"role": "assistant", "content": main_pred},
                {"role": "user", "content": similar_q}
            ]
            classmate_prompt = classmate_tokenizer.apply_chat_template(messages, tokenize=False,
                                                                       add_generation_prompt=True)
            classmate_ground_truth = similar_entry["reward_model"]["ground_truth"]
        elif "help_other_similar_questions" in reward_type:
            # find another question with the same type
            cur_type = entry["reward_model"]["type"]
            similar_entry = None
            for other_entry in test_dataset:
                if other_entry == entry:
                    continue
                similar_entry = other_entry     # default to first different question in case no same type found
                other_type = other_entry["reward_model"]["type"]
                if other_type == cur_type:
                    similar_entry = other_entry
                    break
            if similar_entry is not None:
                similar_prompt = similar_entry["prompt"]
                # similar_ground_truth = similar_entry["answer"] if "answer" in similar_entry else similar_entry["reward_model"]["ground_truth"]
                # prompt = f"Question: {similar_prompt}\n"
                # prompt += f"Solution: {main_pred}\n"
                # prompt += f"Based on the above question and solution, please help answer the following similar math question:\n"
                # prompt += f"{prompt}"

            assert similar_entry is not None, "No similar entry found for help_other_similar_questions."

            similar_q = similar_prompt[0]["content"]
            if reward_type == "help_other_similar_questions_w_guide_colon":
                similar_q = "Given the above question and the solution as an example, solve the following question: \n" + similar_q

            messages = [
                {"role": "user", "content": prompt[0]["content"]},
                {"role": "assistant", "content": main_pred},
                {"role": "user", "content": similar_q}
            ]
            classmate_prompt = classmate_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            classmate_ground_truth = similar_entry["reward_model"]["ground_truth"]
        else:
            classmate_prompt, truncated_main_pred = get_classmate_prompt(prompt, main_pred, keep_ratio=main_cot_keep_ratio)
            classmate_prompt = tokenize_input(classmate_tokenizer, classmate_prompt, is_main=False, main_CoT=truncated_main_pred, tokenize=False)
            classmate_ground_truth = ground_truth

        classmate_pred, classmate_pred_token_ids = generate(classmate_model, classmate_model_name_path, classmate_tokenizer, classmate_prompt, classmate_sampling_params, is_qwen=False)
        extracted_main, main_is_correct = verify_math_pred(main_pred, ground_truth)
        extracted_classmate, classmate_is_correct = verify_math_pred(classmate_pred, classmate_ground_truth)

        breakpoint()

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

    from transformers import AutoTokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{"role": "user", "content": "What is 1+1?"},{"role": "assistant", "content": "<think>\n\n\n\n\nThe answer is 2.\n\n\n\n\n</think>"}]
    input_tokens = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print(input_tokens)


# TOGETHER_API_KEY="6cf968d54220fa0ee7ff5b256b31e7745bc52e252b71798b731deb2b542d9c56"
#CUDA_VISIBLE_DEVICES=2 python test.py
#CUDA_VISIBLE_DEVICES=3 python test.py
# 1. think/no-thinking mode / \no_think
# 2. Try to switch to a smaller main model; if works, then start training