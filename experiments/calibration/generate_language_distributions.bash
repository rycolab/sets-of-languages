set -euo pipefail
. experiments/include.bash

bash experiments/run.bash cpu \
  bash recognizers/calibration/generate_random_language_distributions.bash \
    "$BASE_DIR" \
    --language-classes "${LANGUAGE_CLASSES[@]}" \
    --distributions "${CALIBRATION_DISTRIBUTIONS[@]}"
