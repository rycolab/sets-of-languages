set -euo pipefail
. experiments/include.bash

. src/recognizers/functions.bash

usage() {
  echo "Usage: $0 <language> <architecture>"
}

language=${1-}
architecture=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi

models=()
for trial_no in "${TRIALS[@]}"; do
  models+=("$(get_model_dir "$BASE_DIR" "$language" "$architecture" rec validation-long "$trial_no")")
done
best_model=$(bash experiments/run.bash cpu python recognizers/analysis/print_best_model.py "${models[@]}")
bash experiments/run.bash cpu python recognizers/analysis/sort_examples_by_difficulty.py \
  --model "$best_model" \
  --training-data "$(get_language_dir "$BASE_DIR" "$language")" \
  --dataset test
