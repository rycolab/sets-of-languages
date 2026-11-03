set -euo pipefail

cd "$(dirname "$BASH_SOURCE")"/../src

temp_dir=$(mktemp -d)
trap "rm -r -- $temp_dir" EXIT

# Copy over some dummy hyperparameter files for the language distributions.
cp -r ../tests/calibration "$temp_dir"

for language in random-{podfa,star-free,regular,context-free}-1; do
  bash recognizers/generate_data.bash "$temp_dir" "$language" cpu \
    --max-training-length 10 \
    --max-validation-long-length 15 \
    --max-test-length 20
  for architecture in transformer rnn lstm; do
    bash recognizers/neural_networks/train_and_evaluate.bash "$temp_dir" "$language" "$architecture" rec validation-long 1 --max-epochs 1
  done
done
