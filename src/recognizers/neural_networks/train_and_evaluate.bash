set -euo pipefail
. recognizers/functions.bash

usage() {
  echo "Usage: bash $0 <base-directory> <language> <architecture> \\
  <loss-terms> <validation-data> <trial-no> [options...]

Train and evaluate a neural network on a language.

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
  --num-training-examples <int>
    If given, another training set with the given number of training examples
    will be used instead of the default training set.
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

seed_str="$language $architecture $loss_terms $validation_data $trial_no"
if [[ $alternate_training_set ]]; then
  seed_str+=" $alternate_training_set"
fi
random_arg() {
  local flag=$1
  shift 1
  printf "%s %s" "$flag" "$(random_sample "$seed_str $flag" "$@")"
}

case "$architecture" in
  transformer)
    model_size_args=( \
      $(random_arg --num-layers --int 1 16) \
      $(random_arg --num-heads --int 1 16) \
      --feedforward-size-factor 4 \
    )
    ;;
  rnn)
    model_size_args=( \
      $(random_arg --num-layers --int 1 8) \
    )
    ;;
  lstm)
    model_size_args=( \
      $(random_arg --num-layers --int 1 8) \
    )
    ;;
  *)
    usage >&2
    echo "unknown architecture: $architecture" >&2
    exit 1
    ;;
esac
model_args=($( \
  python recognizers/neural_networks/model_size.py \
    --architecture "$architecture" \
    $(random_arg --parameters --int 100000 1000000) \
    --training-data "$language_dir" \
    "${model_size_args[@]}" \
))

loss_term_flags=()
for loss_term in ${loss_terms//+/ }; do
  case $loss_term in
    rec) ;;
    lm)
      loss_term_flags+=( \
        --use-language-modeling-head \
        $(random_arg --language-modeling-loss-coefficient --log 0.01 10) \
      )
      ;;
    ns)
      loss_term_flags+=( \
        --use-next-symbols-head \
        $(random_arg --next-symbols-loss-coefficient --log 0.01 10) \
      )
      ;;
    *)
      echo "invalid loss term $loss_term" >&2
      exit 1
      ;;
  esac
done

model_dir=$(get_model_dir "$base_dir" "$language" "$architecture" "$loss_terms" "$validation_data" "$trial_no" "$alternate_training_set")
if [[ $alternate_training_set ]]; then
  extra_train_args+=(--training-data-name "$alternate_training_set")
fi
python recognizers/neural_networks/train.py \
  --output "$model_dir" \
  --training-data "$language_dir" \
  --validation-data "$validation_data" \
  --architecture "$architecture" \
  --parameter-seed "$(string_to_seed "$seed_str --parameter-seed")" \
  --random-shuffling-seed "$(string_to_seed "$seed_str --random-shuffling-seed")" \
  "${model_args[@]}" \
  --dropout 0.1 \
  --init-scale 0.1 \
  "${loss_term_flags[@]}" \
  --max-epochs 1000 \
  $(random_arg --max-tokens-per-batch --int 128 4096) \
  --optimizer Adam \
  $(random_arg --initial-learning-rate --log 0.0001 0.01) \
  --gradient-clipping-threshold 5 \
  --early-stopping-patience 10 \
  --learning-rate-patience 5 \
  --learning-rate-decay-factor 0.5 \
  --examples-per-checkpoint 10000 \
  "${extra_train_args[@]}"
bash recognizers/neural_networks/evaluate.bash "$language_dir" "$model_dir"
