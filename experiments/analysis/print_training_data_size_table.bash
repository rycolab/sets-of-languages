set -euo pipefail
. experiments/include.bash

bash experiments/run.bash cpu python recognizers/analysis/print_training_data_size_table.py \
  --base-dir "$BASE_DIR" \
  --architectures "${ARCHITECTURES[@]}" \
  --language-classes "${LANGUAGE_CLASSES[@]}" \
  --num-languages 5 \
  --num-trials "$NUM_TRIALS"
