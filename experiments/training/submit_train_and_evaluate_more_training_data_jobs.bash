set -euo pipefail
. experiments/include.bash

for language in random-{podfa,star-free,regular,context-free}-{1..5}; do
  for architecture in "${ARCHITECTURES[@]}"; do
    for trial_no in "${TRIALS[@]}"; do
      submit_job \
        train-more-data+"$language"+"$architecture"+"$trial_no" \
        cpu \
        --time=72:00:00 \
        --mem-per-cpu=8G \
        -- \
        bash recognizers/neural_networks/train_and_evaluate.bash \
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
  done
done
