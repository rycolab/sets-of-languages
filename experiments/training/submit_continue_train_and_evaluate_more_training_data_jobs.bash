set -euo pipefail
. experiments/include.bash

model_dirs=( \
)

for model_dir in "${model_dirs[@]}"; do
  [[ $model_dir =~ /([^/]+)/([^/]+)/rec/validation-long/([0-9]+)$ ]]
  language=${BASH_REMATCH[1]}
  architecture=${BASH_REMATCH[2]}
  trial_no=${BASH_REMATCH[3]}
  submit_job \
    train-more-data+"$language"+"$architecture"+"$trial_no" \
    cpu \
    --time=72:00:00 \
    --mem-per-cpu=8G \
    -- \
    bash recognizers/neural_networks/continue_train_and_evaluate.bash \
      "$BASE_DIR" \
      "$language" \
      "$architecture" \
      rec \
      validation-long \
      "$trial_no" \
      --num-training-examples 100000 \
      --time-limit 71.5h \
      --no-progress
done
