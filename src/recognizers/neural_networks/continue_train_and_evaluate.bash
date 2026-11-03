set -euo pipefail
. recognizers/functions.bash

usage() {
  echo "Usage: bash $0 <base-directory> <language> <architecture> \\
  <loss-terms> <validation-data> <trial-no> [options...]

Continue training and evalutating a neural network whose training run was
interrupted.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language to run on. Corresponds to the name of a directory
    under <base-directory>/languages/.
  <architecture>
    One of transformer, rnn, lstm.
  <loss-terms>
    Any of the following, joined by \`+\` characters:
    - rec: recognition (binary classification with binary cross-entropy loss)
    - lm: language modeling (cross-entropy loss of next symbol)
    - ns: next set prediction (binary cross-entropy loss of whether every
          symbol at every position is valid in the next position)
    Example: rec+lm for recognition and language modeling.
  <validation-data>
    Which validation set to use. One of: validation-short, validation-long.
  <trial-no>
    A number distinguishing this random restart.

  Options:
  --no-progress
    Don't show progress messages.
"
}

base_dir=${1-}
language=${2-}
architecture=${3-}
loss_terms=${4-}
validation_data=${5-}
trial_no=${6-}
if ! shift 6; then
  usage >&2
  exit 1
fi
alternate_training_set=
extra_train_args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-training-examples)
      shift
      alternate_training_set=training-"$1"
      ;;
    *)
      extra_train_args+=("$1")
      ;;
  esac
  shift
done

language_dir=$(get_language_dir "$base_dir" "$language")
model_dir=$(get_model_dir "$base_dir" "$language" "$architecture" "$loss_terms" "$validation_data" "$trial_no" "$alternate_training_set")
if [[ $alternate_training_set ]]; then
  extra_train_args+=(--training-data-name "$alternate_training_set")
fi
if ! rau is-finished "$model_dir"; then
  python recognizers/neural_networks/train.py \
    --continue \
    --output "$model_dir" \
    --training-data "$language_dir" \
    --validation-data "$validation_data" \
    "${extra_train_args[@]}"
  bash recognizers/neural_networks/evaluate.bash "$language_dir" "$model_dir"
else
  echo "training is already finished for $model_dir"
fi
