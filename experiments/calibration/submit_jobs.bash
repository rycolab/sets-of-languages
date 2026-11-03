set -euo pipefail
. experiments/include.bash

for language_class in "${LANGUAGE_CLASSES[@]}"; do
  if [[ $language_class = context-free ]]; then
    device=gpu
  else
    device=cpu
  fi
  for distribution_no in "${CALIBRATION_DISTRIBUTIONS[@]}"; do
    for language_no in "${CALIBRATION_LANGUAGES[@]}"; do
      submit_job \
        calibrate+$language_class+$distribution_no+$language_no \
        "$device" \
        --time=4:00:00 \
        --mem-per-cpu=4G \
        -- \
        bash recognizers/calibration/generate_data.bash \
          "$BASE_DIR" \
          "$language_class" \
          "$distribution_no" \
          "$language_no" \
          "$device"
    done
  done
done
