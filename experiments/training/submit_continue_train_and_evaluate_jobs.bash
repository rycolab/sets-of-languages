set -euo pipefail
. experiments/include.bash

for language in "${LANGUAGES[@]}"; do
  for architecture in "${ARCHITECTURES[@]}"; do
    for trial_no in "${TRIALS[@]}"; do
      submit_job \
        train+"$language"+"$architecture"+"$trial_no" \
        cpu \
        --time=24:00:00 \
        --mem-per-cpu=8G \
        -- \
        bash recognizers/neural_networks/continue_train_and_evaluate.bash \
          "$BASE_DIR" \
          "$language" \
          "$architecture" \
          rec \
          validation-long \
          "$trial_no" \
          --time-limit 23.5h \
          --no-progress
    done
  done
done
