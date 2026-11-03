set -euo pipefail
. experiments/include.bash

output_file=$FIGURES_DIR/results.csv
mkdir -p "$FIGURES_DIR"
echo "writing $output_file"
bash experiments/run.bash cpu python recognizers/analysis/print_csv.py \
  --base-dir "$BASE_DIR" \
  --architectures "${ARCHITECTURES[@]}" \
  --language-classes "${LANGUAGE_CLASSES[@]}" \
  --num-languages "$NUM_LANGUAGES" \
  --num-trials "$NUM_TRIALS" \
  > "$output_file"
