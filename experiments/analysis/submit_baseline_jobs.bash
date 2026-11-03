set -euo pipefail
. experiments/include.bash

for language in "${LANGUAGES[@]}"; do
  submit_job \
    baseline+"$language" \
    cpu \
    --time=5:00 \
    --mem-per-cpu=1G \
    -- \
    bash recognizers/calibration/run_baseline.bash "$BASE_DIR" "$language"
done
