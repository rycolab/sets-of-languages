set -euo pipefail
. experiments/include.bash

bash experiments/run.bash cpu python recognizers/analysis/print_class_summary_tables.py \
  --base-dir "$BASE_DIR" \
  --architectures "${ARCHITECTURES[@]}" \
  --language-classes "${LANGUAGE_CLASSES[@]}" \
  --num-languages "$NUM_LANGUAGES" \
  --num-trials "$NUM_TRIALS"
