# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any
import logging
import os
import time

import numpy as np
# from together import Together
from transformers import AutoTokenizer
import torch
from tensordict import TensorDict
import requests
from vllm import LLM, SamplingParams, TokensPrompt

import ray
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch, Execute
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ClassmateWorkerConfig:
    model_name_or_path: str
    model_index: int
    # max_new_tokens: int
    max_tokens: int
    classmate_batch_size: int
    classmate_free_cache_engine: bool
    classmate_use_vllm: bool
    enable_thinking: bool

    generation_config: Dict[str, Any]
    """
    max_new_tokens: 128
    do_sample: True
    temperature: 0.7
    top_p: 0.9
    seed: 42
    """

    vllm_config: Dict[str, Any]
    """
    gpu_memory_utilization: 0.9
    tensor_parallel_size: 1
    """

    main_model_max_tokens: int      # For padding for main-classmate importance ratio

    # API server configuration for vLLM remote serving
    # Deprecated. Now using TogetherAI
    # api_host: str = None
    # api_port: int = None


class ClassmateWorker(Worker):
    """Worker for a single classmate model's chain-of-thought continuation sampling.

    Each worker is completely self-contained:
    - Has its own model and tokenizer
    - Handles input preparation (tokenization) using its own tokenizer
    - Performs sampling/generation using its own model
    - Returns processed results
    - Supports vLLM memory management (offload/onload) to free GPU memory when not in use

    This design allows different classmate models to use different tokenizers
    and ensures proper tokenization for each specific model. It also follows the
    same memory management pattern as ActorRolloutRef workers for efficient GPU usage.
    """

    def __init__(self, config: ClassmateWorkerConfig, role: str):
        """Initialize classmate worker with model path and generation config.

        Args:
            config: ClassmateWorkerConfig containing model path, generation config, and vLLM settings
            role: Role name for this worker
        """
        super().__init__()
        self.model_name_or_path = config.model_name_or_path
        self.model_index = config.model_index

        self.batch_size = config.classmate_batch_size
        self.generation_config = config.generation_config

        self.use_vllm = config.classmate_use_vllm
        self.free_cache_engine = config.classmate_free_cache_engine
        self.vllm_config = config.vllm_config

        # Initialize model and tokenizer as None - will be loaded in init_model()
        self.llm = None
        self.special_ids = None

        # API server configuration for remote vLLM serving
        # self.api_base_url = f"http://{config.api_host}:{config.api_port}/v1"
        # self.client = None

        self.classmate_vllm_configs = {
            "model": self.model_name_or_path,
            "gpu_memory_utilization": 0.8,
            "dtype": "auto",
            "max_model_len": config.max_tokens
        }
        self.classmate_vllm = LLM(**self.classmate_vllm_configs, enable_sleep_mode=True)

        self.tokenizer = self.classmate_vllm.get_tokenizer()

        if self.model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
            print("🐛🐛🐛: Applied custom template for DeepSeek-R1-Distill-Qwen-1.5B to include main CoT")
            # change chat template to not omit main CoT
            jinja_fn = "chat_templates/DeepSeek-R1-Distill-Qwen-1.5B.jinja"
            self.tokenizer.chat_template = open(jinja_fn, "r").read()
        self.enable_thinking = config.enable_thinking
        self._offload_model()
        print(f"🚀🚀🚀 Initialized ClassmateWorker {self.model_index} with vLLM engine for model {self.model_name_or_path}")

        self.main_model_max_tokens = config.main_model_max_tokens

    def __del__(self):
        """Cleanup when worker is destroyed."""
        # if self.use_vllm:
        del self.classmate_vllm

    def _generate_via_vllm(self, batch_input_prompt_ids: list, is_eval=False):
        not_none_prompts = [p for p in batch_input_prompt_ids if p is not None]
        is_none_idx = [i for i, p in enumerate(batch_input_prompt_ids) if p is None]
        if is_eval:
            # Greedy
            sampling_params = SamplingParams(
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
                max_tokens=self.generation_config["max_tokens"],
                # prompt_logprobs=0  # Request prompt log probabilities
            )
        else:
            sampling_params = SamplingParams(**self.generation_config)
        outputs = self.classmate_vllm.generate(not_none_prompts, sampling_params=sampling_params, use_tqdm=False)

        completions = []
        completion_token_len = []
        # prompt_logprobs = []
        for i in range(len(batch_input_prompt_ids)):
            if i in is_none_idx:
                completions.append(None)
                completion_token_len.append(0)
                # prompt_logprobs.append(None)
            else:
                output = outputs.pop(0)
                completions.append(output.outputs[0].text)
                completion_token_len.append(len(output.outputs[0].token_ids))

                # token_prob = [list(tok.values())[0].logprob if tok is not None else float("-inf") for tok in output.prompt_logprobs]
                # prompt_logprobs.append(token_prob)

        # return completions, completion_token_len, prompt_logprobs
        return completions, completion_token_len

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the single classmate model for this worker.

        For vLLM mode: Start a separate vLLM API server process
        For HuggingFace mode: Load the model locally
        """
        print(f"Initializing classmate model {self.model_index}: {self.model_name_or_path}")


        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path.replace("-Turbo", ""))       # TogetherAI models have -Turbo suffix
        # Ensure valid padding token and left padding for causal LM
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.special_ids = set(self.tokenizer.all_special_ids)

        try:
            if self.use_vllm:
                # Create OpenAI client
                # from openai import OpenAI
                # self.client = OpenAI(
                #     base_url=self.api_base_url,
                #     api_key="EMPTY",
                # )
                # self.client = Together()
                pass

                # Start vLLM API server process instead of loading model locally
                # self._start_vllm_server()
                # print(f"Successfully started vLLM API server for classmate model {self.model_index}: {self.model_name_or_path}")
            else:
                # Fall back to huggingface
                from transformers import AutoModelForCausalLM
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    device_map=None,
                )
                # Ensure pad token id is set for generation correctness
                try:
                    self.llm.config.pad_token_id = self.tokenizer.pad_token_id
                except Exception:
                    pass
                self.llm.eval()
                print(f"Successfully loaded classmate model {self.model_index}: {self.model_name_or_path}")

        except Exception as e:
            print(f"Failed to load classmate model {self.model_name_or_path}: {e}")
            raise

    def _onload_model(self):
        """Load model to GPU memory.

        For vLLM API mode: No-op since model runs in separate server process
        For HuggingFace mode: Move model to GPU
        """
        if self.use_vllm:
            # No-op for vLLM API mode - model is managed by separate server process
            self.classmate_vllm.wake_up()
            # return
        else:
            if self.llm is None:
                raise RuntimeError(f"Huggingface model for classmate model {self.model_index} is not initialized")
            # Move the whole model to the single visible GPU for this worker
            import torch
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.llm.to(target_device)
            print(f"[Classmate {self.model_index}] Onloaded model to {target_device}; CVD={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
            log_gpu_memory_usage(f"After onloading classmate model {self.model_index}", logger=logger)

    def _offload_model(self):
        """Offload model from GPU memory.

        For vLLM API mode: Destroy vLLM engine to free GPU memory
        For HuggingFace mode: Move model to CPU and clear cache
        """
        if self.use_vllm:
            self.classmate_vllm.sleep(level=1)
        else:
            self.llm.to("cpu")
            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"After offloading classmate model {self.model_index}", logger=logger)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, execute_mode=Execute.ALL, blocking=False)
    def generate_classmate_continuations(self, data: DataProto) -> DataProto:
        """Generate classmate continuations for actor responses using this worker's model and tokenizer.

        Follows the same pattern as ActorRolloutRef workers:
        1. Load model to GPU (onload)
        2. Perform generation
        3. Offload model from GPU
        4. Clean up memory

        Uses vLLM for fast batched inference.
        """
        # data.non_tensor_batch = {
        #     "raw_prompts": np.array(all_prompts),
        #     "main_model_responses": np.array(all_processed_actor_responses),
        #     "keep_rates": np.array(all_keep_rates),
        #     "all_other_prompts": np.array(all_other_prompts) if all_other_prompts is not None else None,
        #     "all_other_prompt_gts": np.array(all_other_prompt_gts) if all_other_prompt_gts is not None else None,
        #     # "prompt_plus_actor_responses": np.array(all_prompt_plus_actor_responses)
        # }

        is_eval = data.meta_info["is_eval"]
        classmate_input_mask = data.non_tensor_batch["classmate_input_mask"]

        classmate_outputs = []
        classmate_output_lens = []
        classmate_prompts = []
        # classmate_prompt_logprobs = []
        # classmate_input_lens = []
        # classmate_input_and_main_cot_lens = []

        # debug_user_only_token_ids = []
        # debug_user_only_prompt_strs = []
        # debug_cls_input_token_ids = []
        # debug_cls_input_prompt_strs = []
        try:
            # ONLOAD: Load model to GPU before generation (no-op for vLLM API mode)
            self._onload_model()
            log_gpu_memory_usage(f"After onloading classmate model {self.model_index} for generation", logger=logger)

            # prompt_plus_actor_responses = data.non_tensor_batch["prompt_plus_actor_responses"]
            # Handle padding: remove globally padded items before minibatching
            batch_prompts = data.non_tensor_batch["raw_prompts"]
            batch_actor_responses = data.non_tensor_batch["main_model_responses"]
            batch_actor_responses_token_ids = data.non_tensor_batch["main_model_response_token_ids"]
            # batch_all_end_with_thinks = data.non_tensor_batch["end_with_thinks"]
            # batch_all_start_with_thinks = data.non_tensor_batch["start_with_thinks"]
            # batch_all_other_prompts = data.non_tensor_batch.get("all_other_prompts", None)

            # Convert numpy array to list if needed
            # if isinstance(prompt_plus_actor_responses, np.ndarray):
            #     prompt_plus_actor_responses = prompt_plus_actor_responses.tolist()
            if isinstance(batch_prompts, np.ndarray):
                batch_prompts = batch_prompts.tolist()
            if isinstance(batch_actor_responses, np.ndarray):
                batch_actor_responses = batch_actor_responses.tolist()
            # Keep batch_actor_responses_token_ids as np.ndarray to avoid re-casting later
            # if isinstance(batch_all_end_with_thinks, np.ndarray):
            #     batch_all_end_with_thinks = batch_all_end_with_thinks.tolist()
            # if isinstance(batch_all_start_with_thinks, np.ndarray):
            #     batch_all_start_with_thinks = batch_all_start_with_thinks.tolist()
            # Keep classmate_input_mask as np.ndarray to avoid re-casting later
            # if isinstance(batch_all_other_prompts, np.ndarray):
            #     batch_all_other_prompts = batch_all_other_prompts.tolist()

            original_length = len(batch_prompts)
            valid_indices = [
                i for i, (p, r) in enumerate(zip(batch_prompts, batch_actor_responses))
                if p is not None and r is not None
            ]
            if len(valid_indices) != original_length:
                batch_prompts = [batch_prompts[i] for i in valid_indices]
                batch_actor_responses = [batch_actor_responses[i] for i in valid_indices]
                batch_actor_responses_token_ids = batch_actor_responses_token_ids[valid_indices]
                # batch_all_end_with_thinks = [batch_all_end_with_thinks[i] for i in valid_indices]
                # batch_all_start_with_thinks = [batch_all_start_with_thinks[i] for i in valid_indices]
                classmate_input_mask = classmate_input_mask[valid_indices]
                # if batch_all_other_prompts is not None:
                #     batch_all_other_prompts = [batch_all_other_prompts[i] for i in valid_indices]

            assert self.generation_config.get("num_return_sequences", 1) == 1 or self.generation_config.get("n", 1) == 1, "Classmate worker currently only supports num_return_sequences=1"

            # Process in batches to manage API requests
            total_samples = len(batch_actor_responses)
            for mini_batch_start in range(0, total_samples, self.batch_size):
                mini_batch_end = min(mini_batch_start + self.batch_size, total_samples)
                mini_batch_prompts = batch_prompts[mini_batch_start:mini_batch_end]
                mini_batch_actor_responses = batch_actor_responses[mini_batch_start:mini_batch_end]
                mini_batch_actor_responses_token_ids = batch_actor_responses_token_ids[mini_batch_start:mini_batch_end]
                # mini_batch_end_with_thinks = batch_all_end_with_thinks[mini_batch_start:mini_batch_end]
                # mini_batch_start_with_thinks = batch_all_start_with_thinks[mini_batch_start:mini_batch_end]
                mini_batch_classmate_input_mask = classmate_input_mask[mini_batch_start:mini_batch_end]

                mini_batch_input_prompts = []
                mini_batch_input_prompt_ids = []

                # Apply classmate's chat template
                for idx, (tmp_prompt, tmp_response, tmp_classmate_input_mask) in enumerate(zip(mini_batch_prompts, mini_batch_actor_responses, mini_batch_classmate_input_mask)):
                    messages = [{"role": "user", "content": tmp_prompt}]
                    user_only_message_token_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

                    # debug_user_only_token_ids.append(user_only_message_token_ids)
                    # debug_user_only_prompt_strs.append(self.tokenizer.decode(user_only_message_token_ids, skip_special_tokens=False))

                    # if batch_other_prompt is not None:      # help other q
                        # messages.append({"role": "user", "content": batch_other_prompt[idx]})
                        # add_generation_prompt = True

                    # token_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
                    # if self.model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" or self.model_name_or_path == "Qwen/Qwen3-0.6B":
                    #     token_ids = token_ids[:-1]       # remove new line at the end
                    #
                    # # Remove trailing special tokens
                    # end = len(token_ids)
                    # while end > 0 and token_ids[end - 1] in self.special_ids:
                    #     end -= 1
                    # token_ids = token_ids[:end]

                    if not tmp_response:
                        # Empty response; return None
                        mini_batch_input_prompts.append(None)
                        mini_batch_input_prompt_ids.append(None)
                        # classmate_input_lens.append(0)
                        # classmate_input_and_main_cot_lens.append(0)

                        # debug_cls_input_token_ids.append(None)
                        # debug_cls_input_prompt_strs.append(None)
                        continue
                    else:
                        if self.enable_thinking:
                            assert "Qwen2.5-Math-1.5B" in self.model_name_or_path or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" in self.model_name_or_path or "Qwen3" in self.model_name_or_path or "Qwen/Qwen3-1.7B" in self.model_name_or_path, f"might need different <think> formatting for classmate {self.model_name_or_path}; double check"
                            tmp_response_token_ids = mini_batch_actor_responses_token_ids[idx][tmp_classmate_input_mask.astype(bool)].tolist()

                            # haha include <think> in CoT?
                            # prefix_token_ids = user_only_message_token_ids
                            # token_ids = prefix_token_ids + tmp_response_token_ids

                            # haha NOT include <think> in CoT?
                            prefix_str = "<think>"
                            suffix_str = "</think>\n\n"
                            prefix_token_ids = user_only_message_token_ids + self.tokenizer.encode(prefix_str)
                            suffix_token_ids = self.tokenizer.encode(suffix_str)
                            token_ids = prefix_token_ids + tmp_response_token_ids + suffix_token_ids

                            # debug_cls_input_token_ids.append(token_ids)
                            # debug_cls_input_prompt_strs.append(self.tokenizer.decode(token_ids, skip_special_tokens=False))

                            # messages.append({"role": "assistant", "content": f"<think>{tmp_response}</think>"})
                            # token_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
                        else:
                            raise NotImplementedError("Currently only support thinking for Qwen3-0.6B and Qwen3-1.7B; need to implement formatting for other models if enable_thinking is True")
                        
                        if len(token_ids) + 1 > self.classmate_vllm_configs["max_model_len"]:       # +1 for generation token (i.e. generate 1 more token will exceed max length)
                            print(f"⚠️ Warning: Input for classmate is too long (have length {len(token_ids)})")
                            mini_batch_input_prompts.append(None)
                            mini_batch_input_prompt_ids.append(None)
                            # classmate_input_lens.append(0)
                            # classmate_input_and_main_cot_lens.append(0)
                            continue
                        else:
                            mini_batch_input_prompts.append(self.tokenizer.decode(token_ids, skip_special_tokens=False))
                            mini_batch_input_prompt_ids.append(TokensPrompt(prompt_token_ids=token_ids))
                            cls_input_len = len(prefix_token_ids)
                            tmp_input_and_main_cot_len = len(prefix_token_ids) + len(tmp_response_token_ids)
                            # classmate_input_lens.append(cls_input_len)
                            # classmate_input_and_main_cot_lens.append(tmp_input_and_main_cot_len)

                    # with open("classmate_worker_debug.txt", "a") as f:
                    #     f.write(f"🐛Original prompt: {tmp_prompt}\n")
                    #     f.write(f"🐛Original actor response: {tmp_response}\n")
                    #     f.write(f"🐛Classmate input prompt: {mini_batch_input_prompts[-1]}\n")
                    #     f.write(f"🐛Classmate input token ids: {token_ids}\n")
                    #     f.write(f"🐛User message only token ids: {user_message_only_token_ids}\n")
                    #     f.write(f"🐛Start with think: {mini_batch_start_with_thinks[idx]}, End with think: {mini_batch_end_with_thinks[idx]}\n")
                    #     f.write("-" * 50 + "\n")
                    # if self.enable_thinking:
                    #     assert self.model_name_or_path == "Qwen/Qwen3-0.6B" or self.model_name_or_path == "Qwen/Qwen3-1.7B", f"might need different <think> formatting for classmate {self.model_name_or_path}; double check"
                    #     if tmp_response and tmp_response[0] != "\n":
                    #         cls_input_len += 1     # for \n
                    #     tmp_input_and_main_cot_len -= 2    # for </think>\n\n
                    #     if tmp_response and tmp_response[-1] != "\n":
                    #         tmp_input_and_main_cot_len -= 1     # for \n


                # Get batch-wise max token len
                # prompt_attn_mask = self.tokenizer(
                #     batch_input_prompts,
                #     padding=True,
                #     return_tensors="pt",
                # )["attention_mask"]
                # batch_max_token_len = prompt_attn_mask.sum(dim=1).max().item()

                # Generate via API for this batch
                # batch_completions = self._generate_via_remote_api(batch_input_prompts, batch_max_token_len)
                # batch_completions = self._generate_via_remote_api(batch_input_prompts)

                # batch_completions, batch_completion_lens, batch_prompt_logprobs = self._generate_via_vllm(
                batch_completions, batch_completion_lens = self._generate_via_vllm(
                    mini_batch_input_prompt_ids, is_eval=is_eval
                )
                classmate_prompts.extend(mini_batch_input_prompts)
                # Extend back to with padding
                classmate_outputs.extend(batch_completions)
                classmate_output_lens.extend(batch_completion_lens)
                # classmate_prompt_logprobs.extend(batch_prompt_logprobs)

                # print(f"🐛🐛🐛 Classmate completions: {classmate_outputs}")
        except Exception as e:
            print(f"❌ Error generating with classmate model {self.model_index}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # OFFLOAD: Offload model from GPU after generation (following ActorRolloutRef pattern)
            self._offload_model()
            log_gpu_memory_usage(f"After offloading classmate model {self.model_index} post-generation", logger=logger)

        # Pad classmate_prompt_logprobs to self.main_model_max_tokens with None for inputs that were too long or None
        # padded_classmate_prompt_logprobs = []
        # padded_classmate_prompt_logprobs_mask = []
        # for idx, (logprobs, cls_input_len, cls_input_and_main_cot_len, cls_input_mask, resp_token_ids) in enumerate(zip(classmate_prompt_logprobs, classmate_input_lens, classmate_input_and_main_cot_lens, classmate_input_mask, batch_actor_responses_token_ids)):
        #     if logprobs is None:        # happens when classmate prompt is None, which happens when e.g. actor response is empty or input is too long
        #         padded_classmate_prompt_logprobs.append([float("-inf")] * self.main_model_max_tokens)
        #         padded_classmate_prompt_logprobs_mask.append([0] * self.main_model_max_tokens)
        #     else:
        #         logprobs = logprobs[cls_input_len:cls_input_and_main_cot_len]
        #         if cls_input_mask is None:
        #             tmp_padded_classmate_prompt_logprobs = logprobs
        #             tmp_padded_classmate_prompt_logprobs_mask = [1] * len(logprobs)
        #         else:
        #             # try:
        #             #     assert len(logprobs) == int(np.sum(cls_input_mask)), f"Length of logprobs after slicing should match sum of cls_input_mask; got {len(logprobs)} vs {int(np.sum(cls_input_mask))}; cls_input_len={cls_input_len}, cls_input_and_main_cot_len={cls_input_and_main_cot_len}, original_logprobs_len={len(logprobs)}, cls_input_mask={cls_input_mask}"
        #             # except Exception as e:
        #                 # with open("debug.txt", "a") as f:
        #                 #     f.write(f"❌ Assertion error when slicing logprobs for classmate prompt: {e}\n")
        #                 #     f.write(f"cls_input_len={cls_input_len}, cls_input_and_main_cot_len={cls_input_and_main_cot_len}, original_logprobs_len={len(logprobs)}, cls_input_mask={cls_input_mask}\n")
        #                 #     f.write(f"resp_token_ids={resp_token_ids}\n")
        #                 #     f.write(f"debug_user_only_token_ids={debug_user_only_token_ids[idx]}\n")
        #                 #     f.write(f"debug_user_only_prompt_strs={debug_user_only_prompt_strs[idx]}\n")
        #                 #     f.write(f"debug_cls_input_token_ids={debug_cls_input_token_ids[idx]}\n")
        #                 #     f.write(f"debug_cls_input_prompt_strs={debug_cls_input_prompt_strs[idx]}\n")
        #                 #     f.write("-" * 50 + "\n")
        #                 # raise
        #
        #             # Pre pad for tokens that got truncated in process_main_cot_helper, and then logprob of main CoT
        #             cls_input_mask = np.array(cls_input_mask)
        #             first_one_candidates = np.flatnonzero(cls_input_mask == 1)
        #             first_one_idx = int(first_one_candidates[0]) if first_one_candidates.size > 0 else self.main_model_max_tokens
        #
        #             tmp_padded_classmate_prompt_logprobs = [float("-inf")] * first_one_idx + logprobs
        #             tmp_padded_classmate_prompt_logprobs_mask = [0] * first_one_idx + [1] * len(logprobs)
        #
        #         # assert len(tmp_padded_classmate_prompt_logprobs) <= self.main_model_max_tokens, f"Length of padded logprobs should not exceed main_model_max_tokens; got {len(tmp_padded_classmate_prompt_logprobs)} vs {self.main_model_max_tokens}; first_one_idx={first_one_idx}, logprobs_len={len(logprobs)}"
        #         # Pad the rest with -inf and 0 to main model max len
        #         pad_len = self.main_model_max_tokens - len(tmp_padded_classmate_prompt_logprobs)
        #         tmp_padded_classmate_prompt_logprobs += [float("-inf")] * pad_len
        #         tmp_padded_classmate_prompt_logprobs_mask += [0] * pad_len
        #
        #         padded_classmate_prompt_logprobs.append(tmp_padded_classmate_prompt_logprobs)
        #         padded_classmate_prompt_logprobs_mask.append(tmp_padded_classmate_prompt_logprobs_mask)
        #         # padded_classmate_prompt_logprobs.append(logprobs + [float("-inf")] * max((self.main_model_max_tokens - len(logprobs), 0)))
        #         # padded_classmate_prompt_logprobs_mask.append([1] * len(logprobs) + [0] * max((self.main_model_max_tokens - len(logprobs), 0)))

        # TODO need to fix this to support num_return_sequences > 1
        classmate_outputs = [[output] for output in classmate_outputs]  # Wrap each output in a list
        classmate_output_lens = [[length] for length in classmate_output_lens]
        # padded_classmate_prompt_logprobs = [[logprobs] for logprobs in padded_classmate_prompt_logprobs]
        # padded_classmate_prompt_logprobs_mask = [[mask] for mask in padded_classmate_prompt_logprobs_mask]

        result_non_tensor_batch = {
            "classmate_prompts": np.array(classmate_prompts),    # Should have shape (bsz,)
            "classmate_outputs": np.array(classmate_outputs),      # Should have shape (bsz, num_return_sequences)
            "classmate_output_lens": np.array(classmate_output_lens),  # Should have shape (bsz, num_return_sequences)
            # "classmate_prompt_logprobs": np.array(padded_classmate_prompt_logprobs),  # Should have shape (bsz, num_return_sequences, input_seq_len)
            # "classmate_prompt_logprobs_mask": np.array(padded_classmate_prompt_logprobs_mask),  # Should have shape (bsz, num_return_sequences, input_seq_len)
        }

        # print(f"🐛 From classmateWorker {self.model_index} classmate_output", result_non_tensor_batch["classmate_outputs"])
        # print(f"Finish sampling from classmate model {self.model_name_or_path} ({self.model_index})")

        return DataProto(batch=None, non_tensor_batch=result_non_tensor_batch)
