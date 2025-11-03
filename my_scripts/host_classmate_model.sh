
#!/bin/bash

# Trap Ctrl+C and cleanup
trap cleanup SIGINT SIGTERM

cleanup() {
    echo ""
    echo "🛑 Received interrupt signal. Cleaning up..."
    # Kill any background processes started by this script
    jobs -p | xargs -r kill
    exit 0
}

#"meta-llama/Llama-3.2-1B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "allenai/OLMo-2-0425-1B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "allenai/OLMo-2-0425-1B-SFT" "Qwen/Qwen2.5-1.5B-Instruct" "mistralai/Ministral-8B-Instruct-2410"
gpu_idx=3
#outward_hostname="coffee.cs.columbia.edu"
outward_hostname=tea.cs.columbia.edu
list_of_classmate=(
    "meta-llama/Llama-3.2-1B-Instruct"
)

echo "🚀 Starting classmate models with GPU ${gpu_idx}"
echo "💡 Press Ctrl+C to stop all servers"

CUDA_VISIBLE_DEVICES=${gpu_idx} python scripts/host_classmate_model.py \
    ${list_of_classmate[@]} \
    --output-dir outputs/host_classmate_models \
    --outward_hostname ${outward_hostname} \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1
#    --start-port 8000 \
#    --host 127.0.0.1 \
echo "👋 All done!"

#bash my_scripts/host_classmate_model.sh