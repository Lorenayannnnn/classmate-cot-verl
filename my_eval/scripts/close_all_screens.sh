for session in $(screen -ls | grep -o '[0-9]*\.[a-zA-Z0-9_-]*')
do
    screen -S "${session}" -X quit
done


# Replace <job_id_to_keep> with the actual job ID you want to keep
#job_id_to_keep=186361

# Cancel all jobs except the specified job
#squeue -u $USER | awk '{if (NR!=1 && $1 != job_id_to_keep) print $1}' job_id_to_keep=$job_id_to_keep | xargs scancel
