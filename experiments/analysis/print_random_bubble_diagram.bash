set -euo pipefail
. experiments/include.bash

output_dir=$FIGURES_DIR/random-bubble-diagram
mkdir -p "$output_dir"
bash experiments/run.bash cpu python recognizers/analysis/print_random_bubble_diagram.py \
  --base-dir "$BASE_DIR" \
  --architectures "${ARCHITECTURES[@]}" \
  --num-languages "$NUM_LANGUAGES" \
  --num-trials "$NUM_TRIALS" \
  --num-columns 10 \
  --output-dir "$output_dir"
