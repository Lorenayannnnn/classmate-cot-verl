# ClassmateCoT: Collaborative Chain-of-Thought Reinforcement Learning

This system implements a novel approach to reinforcement learning where multiple "classmate" models collaborate to improve the main actor's chain-of-thought reasoning. The actor model generates initial reasoning, which is then continued by classmate models running in parallel using Ray workers. The combined performance guides the training process.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────────────────────────┐    ┌─────────────────┐
│  Actor Model    │───▶│     Classmate Workers (Ray)         │───▶│  Reward Fusion  │
│  (Main Policy)  │    │  ┌─────────┐ ┌─────────┐ ┌─────────┐│    │                 │
│                 │    │  │Model A  │ │Model B  │ │Model C  ││    │ Base + Weighted │
│ Generates CoT   │    │  │Continue │ │Continue │ │Continue ││    │ Classmate Score │
└─────────────────┘    │  │   CoT   │ │   CoT   │ │   CoT   ││    └─────────────────┘
                       │  └─────────┘ └─────────┘ └─────────┘│
                       └─────────────────────────────────────────┘
```

## Key Components

### 1. ClassmateCoTWorker (Ray Remote Actor)
- **Location**: `verl/workers/reward_manager/classmate_cot.py`
- **Purpose**: Ray remote worker for parallel classmate model inference
- **Resources**: 1 GPU + 2 CPUs per worker
- **Functions**:
  - `sample_cot_continuation()`: Continue actor's chain-of-thought
  - `compute_classmate_score()`: Evaluate continuation quality
  - Answer extraction and correctness checking

### 2. ClassmateCoTRewardManager
- **Location**: `verl/workers/reward_manager/classmate_cot.py` 
- **Purpose**: Handles reward computation combining actor and classmate scores
- **Features**:
  - Processes classmate outputs generated during rollout
  - Computes scores using the same function as the actor
  - Combines rewards: `final_reward = base_reward + weight * classmate_score`

### 3. ClassmateCoTRayPPOTrainer
- **Location**: `verl/trainer/ppo/classmate_cot_ray_trainer.py`
- **Purpose**: Extended PPO trainer with classmate worker orchestration
- **Enhancements**:
  - `_init_classmate_workers()`: Creates Ray workers for classmate models
  - `_generate_classmate_continuations()`: Parallel sampling in training loop
  - Integrates seamlessly with existing VERL training pipeline

## Training Flow

1. **Setup Phase**:
   ```python
   # Initialize Ray workers for classmate models
   self._init_classmate_workers()
   ```

2. **Generation Phase** (per training step):
   ```python
   # Actor generates initial reasoning
   actor_output = self.actor_rollout_wg.generate_sequences(batch)
   
   # Classmate models continue reasoning in parallel
   batch_with_classmate = self._generate_classmate_continuations(batch)
   ```

3. **Reward Computation**:
   ```python
   # Evaluate both actor and classmate outputs
   base_reward = compute_score(actor_output, ground_truth)
   classmate_rewards = [compute_score(cm_out, ground_truth) for cm_out in classmate_outputs]
   
   # Combine rewards
   final_reward = base_reward + classmate_weight * max(classmate_rewards)
   ```

4. **Policy Update**:
   - Standard PPO update using combined rewards
   - Actor learns to generate reasoning that enables good classmate continuations

## Configuration

### Basic Setup
```yaml
# verl/trainer/config/classmate_cot_ppo_trainer.yaml
reward_model:
  reward_manager: "classmate_cot"
  reward_kwargs:
    classmate_cot_reward_configs:
      classmate_model_name_or_path_list: 
        - "meta-llama/Llama-3.2-1B-Instruct"
        - "allenai/OLMo-2-0425-1B-SFT"
      classmate_reward_weight: 0.5
    
    classmate_generation_configs:
      max_new_tokens: 256
      do_sample: true
      temperature: 0.7
      top_p: 0.9
      num_return_sequences: 1
```

### Resource Planning
```bash
# Calculate GPU requirements:
# Total GPUs = Actor Model GPUs + Critic GPUs + (1 GPU × Number of Classmate Models)
# Example: 4 (actor) + 2 (critic) + 2 (classmates) = 8 GPUs total

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
```

## Usage Examples

### 1. Command Line Training
```bash
python -m verl.trainer.classmate_cot_main_ppo \
  --config-name="classmate_cot_ppo_trainer" \
  actor_rollout_ref.model.model_path="allenai/OLMo-2-0425-1B-DPO" \
  reward_model.reward_kwargs.classmate_cot_reward_configs.classmate_model_name_or_path_list=["meta-llama/Llama-3.2-1B-Instruct"] \
  reward_model.reward_kwargs.classmate_cot_reward_configs.classmate_reward_weight=0.3
```

### 2. Script-based Training
```bash
# Use the provided example script
./examples/run_classmate_cot_training.sh
```

### 3. Programmatic Usage
```python
from verl.trainer.ppo.classmate_cot_ray_trainer import ClassmateCoTRayPPOTrainer

trainer = ClassmateCoTRayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    classmate_cot_reward_configs={
        "classmate_model_name_or_path_list": ["model1", "model2"],
        "classmate_reward_weight": 0.5
    },
    classmate_generation_configs={
        "max_new_tokens": 256,
        "temperature": 0.7
    }
)

trainer.init_workers()  # This creates classmate workers
trainer.fit()           # This runs training with classmate collaboration
```

## Performance Optimizations

### Parallel Processing
- **Ray Workers**: Each classmate model runs on dedicated GPU worker
- **Async Execution**: All classmate sampling happens simultaneously
- **Batch Processing**: Multiple responses processed together

### Memory Efficiency
- **Model Loading**: `torch.float16` for classmate models with `device_map="auto"`
- **Token Processing**: Efficient attention masking and decoding
- **Resource Management**: Automatic GPU memory management per worker

### Scalability Features
- **Dynamic Workers**: Add/remove classmate models without code changes
- **Load Balancing**: Ray automatically distributes work across workers
- **Error Handling**: Graceful fallback if classmate workers fail

## Expected Learning Behavior

### Training Dynamics
1. **Early Training**: Actor learns basic reasoning patterns
2. **Collaboration Phase**: Actor adapts to generate reasoning that classmate models can extend effectively
3. **Convergence**: Actor produces "collaborative" chain-of-thought that enables strong classmate performance

### Metrics to Monitor
```python
# Available in reward_extra_info
{
  "base_reward": [actor_score],           # Original actor performance
  "classmate_scores": [list_of_scores],   # Individual classmate scores  
  "best_classmate_score": [best_score],   # Top classmate performance
  "avg_classmate_score": [avg_score],     # Average classmate performance
  "final_reward": [combined_score],       # Final training signal
}
```

## File Structure

```
verl/
├── workers/reward_manager/
│   └── classmate_cot.py                    # Ray workers & reward manager
├── trainer/
│   ├── ppo/
│   │   └── classmate_cot_ray_trainer.py   # Extended PPO trainer
│   ├── config/
│   │   └── classmate_cot_ppo_trainer.yaml # Configuration template
│   └── classmate_cot_main_ppo.py          # Main training script
└── examples/
    ├── run_classmate_cot_training.sh      # Usage example
    └── test_classmate_cot.py              # Standalone test
```

## Troubleshooting

### Common Issues

1. **Ray Worker Failures**:
   ```bash
   # Check Ray status
   ray status
   # Verify GPU memory
   nvidia-smi
   ```

2. **Configuration Errors**:
   ```python
   # Verify classmate models are accessible
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("model_path")
   ```

3. **Memory Issues**:
   ```yaml
   # Reduce model size or batch size
   reward_model:
     reward_kwargs:
       classmate_generation_configs:
         max_new_tokens: 128  # Reduce from 256
   ```

### Performance Tuning

1. **Reward Weight Optimization**:
   - Start with `classmate_reward_weight: 0.1`
   - Increase gradually if classmate collaboration improves results
   - Monitor both base and final reward trends

2. **Sampling Parameters**:
   - Higher `temperature` (0.8-1.0) for more diverse continuations
   - Lower `temperature` (0.3-0.5) for more focused completions
   - Adjust `top_p` for creativity vs. coherence balance

3. **Resource Allocation**:
   - More classmate models = better coverage but higher compute cost
   - Balance model diversity vs. available GPU resources
   - Consider using smaller classmate models for efficiency

## Research Applications

This system enables research into:
- **Collaborative Reasoning**: How models can learn to reason together
- **Multi-Agent RL**: Coordination between different model policies  
- **Chain-of-Thought Transfer**: Knowledge sharing through continued reasoning
- **Diverse Sampling**: Using multiple models for robust evaluation

The ClassmateCoT system provides a flexible framework for exploring these collaborative learning paradigms while maintaining compatibility with existing VERL infrastructure.