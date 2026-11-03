set -euo pipefail
. experiments/include.bash

bash experiments/run.bash cpu python recognizers/calibration/get_best_distribution.py \
  "$BASE_DIR"/calibration/"${1?}" \
  --num-languages 30
