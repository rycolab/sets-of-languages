set -euo pipefail
. experiments/include.bash

bash experiments/run.bash cpu python recognizers/analysis/linear_model_best.py "$FIGURES_DIR"/results.csv
