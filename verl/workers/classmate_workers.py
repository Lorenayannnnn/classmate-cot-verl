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
from dataclasses import dataclass
from typing import Dict, Any
import logging
import os

import numpy as np
from transformers import AutoTokenizer

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
    generation_config: Dict[str, Any]
    max_prompt_len: int
    model_index: int
    vllm_gpu_memory_utilization: float  # How much GPU memory vLLM can use
    tensor_parallel_size: int # Tensor parallel size for vLLM
    free_cache_engine: bool  # Whether to enable offload/onload of vLLM engine


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
        self.model_path = config.model_name_or_path
        self.generation_config = config.generation_config
        self.model_index = config.model_index
        self.vllm_gpu_memory_utilization = config.vllm_gpu_memory_utilization
        self.vllm_max_model_len = config.max_prompt_len + config.generation_config.max_new_tokens
        self.tensor_parallel_size = config.tensor_parallel_size
        self.free_cache_engine = config.free_cache_engine

        # Initialize model and tokenizer as None - will be loaded in init_model()
        self.model = None
        self.tokenizer = None
        self.llm = None  # vLLM engine

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the single classmate model for this worker.
        
        Follows the same pattern as ActorRolloutRef workers:
        1. Load the model
        2. Immediately offload it to free GPU memory
        3. Model will be loaded on-demand during generation
        """
        print(f"Initializing classmate model {self.model_index}: {self.model_path}")

        try:
            # Load tokenizer (always needed)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Use vLLM for fast inference
            from vllm import LLM
            import torch
            
            # # TODO haha Aggressively clear CUDA cache to free up memory before vLLM init
            # if torch.cuda.is_available():
            #     # Force PyTorch to release all cached memory
            #     torch.cuda.empty_cache()
            #     torch.cuda.synchronize()
            #     # Reset peak memory stats
            #     torch.cuda.reset_peak_memory_stats()
            #     # Try to trigger garbage collection
            #     import gc
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #
            #     # Get actual free memory after cache clear
            #     free_memory_gb = torch.cuda.mem_get_info()[0] / (1024**3)
            #     total_memory_gb = torch.cuda.mem_get_info()[1] / (1024**3)
            #     allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            #     reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            #     print(f"🐛[Classmate {self.model_index}] GPU Memory before init:")
            #     print(f" 🐛 - Free: {free_memory_gb:.2f}GB / Total: {total_memory_gb:.2f}GB")
            #     print(f" 🐛 - PyTorch Allocated: {allocated_gb:.2f}GB")
            #     print(f" 🐛 - PyTorch Reserved: {reserved_gb:.2f}GB")
            #     print(f" 🐛 - gpu_memory_utilization setting: {self.vllm_gpu_memory_utilization}")
            #     print(f" 🐛 - max_model_len setting: {self.vllm_max_model_len}")
                
            # Initialize vLLM engine with additional debugging and fixes
            # Key fix: disable_custom_all_reduce for stability with Ray
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_model_len=self.vllm_max_model_len,
                trust_remote_code=True,
                dtype="auto",
            )
            log_gpu_memory_usage(f"After loading classmate model {self.model_index}", logger=logger)
            print(f"Successfully loaded classmate model {self.model_index} with vLLM: {self.model_path}")
            
            # Immediately offload the model to free GPU memory (following ActorRolloutRef pattern)
            if self.free_cache_engine:
                self._offload_model()
                log_gpu_memory_usage(f"After offloading classmate model {self.model_index}", logger=logger)
                print(f"Offloaded classmate model {self.model_index} to free GPU memory")

        except Exception as e:
            print(f"Failed to load classmate model {self.model_path}: {e}")
            raise

    def _onload_model(self):
        """Load vLLM engine to GPU memory (wake up).
        
        Follows the same pattern as vLLM rollout's resume() method.
        Simply calls wake_up() if available - vLLM handles the internal details.
        """
        if self.llm is None:
            raise RuntimeError(f"vLLM engine for classmate model {self.model_index} is not initialized")
        
        if not self.free_cache_engine:
            return  # No need to wake up if sleep mode is disabled
        
        try:
            # Wake up the vLLM engine (load weights and KV cache back to GPU)
            if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'wake_up'):
                # For LLM with llm_engine attribute
                self.llm.llm_engine.wake_up()
            elif hasattr(self.llm, 'wake_up'):
                # For regular LLM class
                self.llm.wake_up()
            
            log_gpu_memory_usage(f"After onloading classmate model {self.model_index}", logger=logger)
        except Exception as e:
            logger.warning(f"Failed to wake up classmate model {self.model_index}: {e}")

    def _offload_model(self):
        """Offload vLLM engine from GPU memory (sleep).
        
        Follows the same pattern as vLLM rollout's release() method.
        Uses aggressive_empty_cache for cleanup like FSDP workers.
        """
        if self.llm is None:
            return
        
        if not self.free_cache_engine:
            return  # No need to sleep if sleep mode is disabled
        
        try:
            # Put the vLLM engine to sleep (offload weights and KV cache from GPU)
            if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'sleep'):
                # For LLM with llm_engine attribute
                self.llm.llm_engine.sleep()
            elif hasattr(self.llm, 'sleep'):
                # For regular LLM class
                self.llm.sleep()
            
            # Aggressive cache cleanup (following FSDP workers pattern)
            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"After offloading classmate model {self.model_index}", logger=logger)
        except Exception as e:
            logger.warning(f"Failed to sleep classmate model {self.model_index}: {e}")

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
        #     "prompts": all_prompts,
        #     "prompt_plus_actor_responses": all_prompt_plus_actor_responses
        # }
        classmate_outputs = []
        try:
            # ONLOAD: Load model to GPU before generation (following ActorRolloutRef pattern)
            self._onload_model()
            log_gpu_memory_usage(f"After onloading classmate model {self.model_index} for generation", logger=logger)
            
            prompt_plus_actor_responses = data.non_tensor_batch["prompt_plus_actor_responses"]
            
            # Convert numpy array to list if needed
            if isinstance(prompt_plus_actor_responses, np.ndarray):
                prompt_plus_actor_responses = prompt_plus_actor_responses.tolist()
            
            num_return_sequences = self.generation_config.get("num_return_sequences", 1)

            from vllm import SamplingParams
            
            # Convert generation_config to vLLM SamplingParams
            sampling_params = SamplingParams(
                n=num_return_sequences,
                **self.generation_config,
                skip_special_tokens=True,
            )
            
            # Generate with vLLM (batched and efficient)
            outputs = self.llm.generate(prompt_plus_actor_responses, sampling_params)
            
            # Extract outputs (vLLM returns RequestOutput objects)
            for output in outputs:
                sample_outputs = [completion.text for completion in output.outputs]
                classmate_outputs.append(sample_outputs)

        except Exception as e:
            print(f"❌ Error generating with classmate model {self.model_index}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty outputs matching expected shape
            bsz = len(data.non_tensor_batch.get("prompt_plus_actor_responses", []))
            num_return_sequences = self.generation_config.get("num_return_sequences", 1)
            classmate_outputs = [[""] * num_return_sequences for _ in range(bsz)]
        
        finally:
            # OFFLOAD: Offload model from GPU after generation (following ActorRolloutRef pattern)
            self._offload_model()
            log_gpu_memory_usage(f"After offloading classmate model {self.model_index} post-generation", logger=logger)

        result_non_tensor_batch = {
            "classmate_output": np.array(classmate_outputs),      # Should have shape (bsz, num_return_sequences)
        }

        return DataProto(batch=None, non_tensor_batch=result_non_tensor_batch)


def create_classmate_workers(config):
    """Create individual Ray remote workers for each classmate model.

    Args:
        config: Configuration containing classmate model paths and generation config

    Returns:
        List of Ray remote worker references, one per classmate model
    """
    classmate_config = config.get("classmate_cot_reward_configs", {})
    model_paths = classmate_config.get("classmate_model_name_or_path_list", [])

    # Get generation configuration
    generation_config = config.get("classmate_generation_configs", {})

    assert model_paths, "No classmate models configured"

    workers = []
    for i, model_path in enumerate(model_paths):
        # Create a Ray remote worker for each classmate model
        worker_cls = ray.remote(ClassmateWorker)
        worker = worker_cls.remote(
            model_name_or_path=model_path,
            generation_config=generation_config,
            model_index=i
        )
        workers.append(worker)
        print(f"Created classmate worker {i} for model: {model_path}")

    return workers