set -euo pipefail
. experiments/include.bash

for language in "${LANGUAGES[@]}"; do
  if [[ $language = random-context-free-* ]]; then
    device=gpu
  else
    device=cpu
  fi
  submit_job \
    sensitivity-density+"$language" \
    "$device" \
    --time=4:00:00 \
    --mem-per-cpu=4G \
    -- \
    bash recognizers/analysis/estimate_sensitivity_and_density.bash "$BASE_DIR" "$language" "$device"
done
