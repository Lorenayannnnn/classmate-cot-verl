

gpu_idx=0
classmate_model_path="meta-llama/Llama-3.2-1B-Instruct"

CUDA_VISIBLE_DEVICES=${gpu_idx} vllm serve ${classmate_model_path} --served-model-name ${classmate_model_path} \
    --gpu-memory-utilization 0.9 \
    --port 8003 \
#bash my_scripts/host_classmate_model.sh

# TODO: then change port number in outputs/host_classmate_models/classmate_model_mapping.json
