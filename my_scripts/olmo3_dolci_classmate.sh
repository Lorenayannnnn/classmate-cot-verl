set -x

# TODO 1. code running env
#uvicorn verl.code_utils.api:app --host 0.0.0.0 --port 8001
#kill -9 $(lsof -t -i :8001) 2>/dev/null
#Test connection:
#```
#curl -X GET http://localhost:8001/health
#curl -X POST http://localhost:8001/test_program -H "Content-Type: application/json" -d '{"program": "def add(a, b): return a + b", "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"], "max_execution_time": 1.0}'
#curl -X POST http://localhost:8001/test_program_stdio -H "Content-Type: application/json" -d '{"program": "import sys\nfor line in sys.stdin.read().splitlines():\n    print(int(line.strip()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}, {"input": "100\n", "output": "101\n"}], "max_execution_time": 1.0}'
#```
sandbox_fusion_url=http://localhost:8001

#TODO 2. LLM judge
#CUDA_VISIBLE_DEVICES=0,1 vllm serve "Qwen/Qwen3-32B" --served-model-name "Qwen/Qwen3-32B" --gpu-memory-utilization 0.9 --port 8002 --tensor_parallel_size 2 --max_model_len 8192
#llm_judge_model=hosted_vllm/Qwen/Qwen3-32B
#export HOSTED_VLLM_API_BASE=http://localhost:8002/v1
llm_judge_model=deepinfra/Qwen/Qwen3-32B
export DEEPINFRA_API_KEY=yafq4uYi5X3T2II8d1aFrBhaYkqy16Ur

# TODO 3. Classmate model: Change url and port number in classmate-cot-verl/outputs/host_classmate_models/classmate_model_mapping.json
#CUDA_VISIBLE_DEVICES=0,1 vllm serve "meta-llama/Llama-3.2-1B-Instruct" --served-model-name "meta-llama/Llama-3.2-1B-Instruct" --tensor_parallel_size 2 --gpu-memory-utilization 0.9 --port 8003
#CUDA_VISIBLE_DEVICES=0 vllm serve "meta-llama/Llama-3.2-1B-Instruct" --served-model-name "meta-llama/Llama-3.2-1B-Instruct" --tensor_parallel_size 1 --gpu-memory-utilization 0.9 --port 8003

#TODO change dataset
#data_dir=/data    # run with docker
data_dir=$HOME/data   # run with conda
dataset_name="think-wo_general-Dolci-Think-RL-7B_specified_percentage_general-quality-0_general-quality_ref-0_subset_25500"
train_path=${data_dir}/${dataset_name}/train.parquet
eval_path=${data_dir}/${dataset_name}/eval.parquet
train_size=25138   # After filtering out too long prompts
epoch_num=1    # to make it about 400000 * 8 episodes

train_files="['$train_path']"
eval_files="['$eval_path']"

train_batch_size=64
mini_batch_size_per_gpu=8
gpu_num=8

total_ckpts=10
total_test_times=10

rollout_n=8
train_steps=$(($train_size + train_batch_size - 1 / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}
base_model_name_path=allenai/Olmo-3-7B-Think-DPO
#base_model_name_path=allenai/OLMo-2-1124-7B-DPO
#base_model_name_path=allenai/OLMo-2-0425-1B-DPO

#HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.classmate_cot_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$eval_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${base_model_name_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((mini_batch_size_per_gpu)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='classmate_cot_w_verl' \
    trainer.experiment_name="grpo_${base_model_name_path}_${dataset_name}_with_classmate_reward_llama_${total_episodes}_episodes" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=$((train_steps / total_ckpts)) \
    trainer.test_freq=$((train_steps / total_test_times)) \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=42 \
    reward_model.sandbox_fusion_url=${sandbox_fusion_url} \
    reward_model.llm_judge_model=${llm_judge_model}
#    actor_rollout_ref.model.lora_rank=32 \
#    actor_rollout_ref.model.lora_alpha=32 \
#    actor_rollout_ref.model.target_modules=all-linear \
#    actor_rollout_ref.model.use_shm=True
#    actor_rollout_ref.rollout.free_cache_engine=False


#bash my_scripts/olmo3_dolci_classmate.sh