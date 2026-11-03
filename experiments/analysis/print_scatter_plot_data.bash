set -euo pipefail
. experiments/include.bash

output_dir=$FIGURES_DIR/scatter-plot-data
mkdir -p "$output_dir"
bash experiments/run.bash cpu python recognizers/analysis/print_scatter_plot_data.py \
  --base-dir "$BASE_DIR" \
  --architectures "${ARCHITECTURES[@]}" \
  --language-classes "${LANGUAGE_CLASSES[@]}" \
  --num-languages "$NUM_LANGUAGES" \
  --num-trials "$NUM_TRIALS" \
  --output-dir "$output_dir"
