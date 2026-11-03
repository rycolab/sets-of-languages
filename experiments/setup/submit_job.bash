set -euo pipefail

. experiments/include.bash

job_name=setup

result=$( \
  sbatch \
    --job-name="$job_name" \
    --output="$BASE_DIR"/job-outputs/"$job_name".txt \
    --open-mode=append \
    --time=1:00:00 \
    --mem-per-cpu=4G \
    experiments/setup/job.bash \
)
echo "$job_name | $result"
