set -euo pipefail
. experiments/include.bash

bash experiments/run.bash cpu python recognizers/calibration/get_calibrated_distributions.py \
  "$BASE_DIR"/calibration/{podfa,star-free,regular,context-free} \
  --num-distributions "$CALIBRATION_NUM_DISTRIBUTIONS" \
  --num-languages "$CALIBRATION_NUM_LANGUAGES" \
  "$@"
