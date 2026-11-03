set -euo pipefail
. experiments/include.bash
. src/recognizers/functions.bash

# Insert directories here
model_dirs=( \
)

for model_dir in "${model_dirs[@]}"; do
  [[ $model_dir =~ models/([^/]+)/([^/]+)/rec/validation-long/([^/]+)$ ]]
  language=${BASH_REMATCH[1]}
  architecture=${BASH_REMATCH[2]}
  trial_no=${BASH_REMATCH[3]}
  language_dir=$(get_language_dir "$BASE_DIR" "$language")
  submit_job \
    train+"$language"+"$architecture"+"$trial_no" \
    cpu \
    --time=1:00:00 \
    -- \
    bash recognizers/neural_networks/evaluate.bash \
      "$language_dir" \
      "$model_dir"
done
