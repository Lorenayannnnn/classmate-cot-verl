set -e

#bash my_scripts/scripts_bold_formatting/bold_format_iterate_train.sh

# ── Per-task runner ───────────────────────────────────────────────────────── #
# Args: gpu_idx  method  seed
#   method: "baseline" | "OURS_self"
_run_task() {
  local gpu_idx=$1
  local method=$2
  local seed=$3

  echo "=== Starting bold_formatting ${method} seed_${seed} on GPU ${gpu_idx} ==="

  if [[ "${method}" == "baseline" ]]; then
    seed=${seed} gpu_idx=${gpu_idx} \
      bash my_scripts/scripts_bold_formatting/bold_formatting_baseline.sh
  elif [[ "${method}" == "OURS_self" ]]; then
    seed=${seed} gpu_idx=${gpu_idx} \
      bash my_scripts/scripts_bold_formatting/bold_formatting_classmate_self.sh
  else
    echo "Unknown method: ${method}" && exit 1
  fi

  echo "=== Finished bold_formatting ${method} seed_${seed} on GPU ${gpu_idx} ==="
}

# ── Launch all 6 runs in parallel (one GPU each) ─────────────────────────── #
#  GPU  method     seed
_run_task 0  baseline  0 &
#_run_task 1  baseline  1 &
#_run_task 2  baseline  2 &
#_run_task 3  OURS_self  0 &
#_run_task 4  OURS_self  1 &
#_run_task 5  OURS_self  2 &

wait
echo "All bold_formatting training runs finished."
