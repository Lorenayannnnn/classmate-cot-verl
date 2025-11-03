for i in 0 1 2; do
    echo "🚀 Starting HAHA test process $i"
    (
    python scripts/haha_test_vllm_api_simple.py
    ) &
done
#bash my_scripts/haha_start_classmate_vllm_server.sh