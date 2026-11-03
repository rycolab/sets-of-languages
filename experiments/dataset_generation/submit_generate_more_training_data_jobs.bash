set -euo pipefail
. experiments/include.bash

for language in random-{podfa,star-free,regular,context-free}-{1..5}; do
  if [[ $language = random-context-free-* ]]; then
    device=gpu
  else
    device=cpu
  fi
  submit_job \
    generate-more-training-data+"$language" \
    "$device" \
    --time=4:00:00 \
    --mem-per-cpu=4G \
    -- \
    bash recognizers/generate_data_custom_size.bash "$BASE_DIR" "$language" "$device" 100000
done
