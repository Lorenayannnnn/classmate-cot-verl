#!/bin/sh
#
# For now, please specify the following three slurm directives only.
#
# account: your account, i.e. your group
# gpus: from 1 to 8
# time: job run time hh:mm:ss or mm:ss
#
#SBATCH --account=hewittlab
#SBATCH --gpus=4
#SBATCH --time=20:00:00
#SBATCH --job-name=test
# Research goes here.

JOB_NAME="${SLURM_JOB_NAME:-manual}"
JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTDIR="outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-0.6B_anthropic_hh_rlhf_baseline_448000_episodes_seed_42"
OUTFILE="${OUTDIR}/slurm.out"
mkdir -p "$OUTDIR"
exec >"$OUTFILE" 2>&1

export HF_HOME=/scratch/hewittlab/lorenayan/.cache/huggingface
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
bash my_scripts/qwen3_baseline_general_sycophancy.sh

#hostname
#date
#nvidia-smi
#sleep 30

# End of file.

#cd /scratch/hewittlab/lorenayan/classmate-cot-verl/
#sbatch my_scripts/slurm.sh
#srun --account=hewittlab --job-name=test --gres=gpu:4  --pty --time=20:00:00 bash
#srun --account=hewittlab --job-name=test --gres=gpu:4  --pty --time=1:00:00 bash