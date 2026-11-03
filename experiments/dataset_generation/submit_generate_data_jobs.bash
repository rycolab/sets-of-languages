set -euo pipefail
. experiments/include.bash

for language in "${LANGUAGES[@]}"; do
  if [[ $language = random-context-free-* ]]; then
    device=gpu
  else
    device=cpu
  fi
  submit_job \
    generate-data+"$language" \
    "$device" \
    --time=4:00:00 \
    --mem-per-cpu=4G \
    -- \
    bash recognizers/generate_data.bash "$BASE_DIR" "$language" "$device"
done
