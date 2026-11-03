get_model_dir() {
  local base_dir=$1
  local language=$2
  local architecture=$3
  local loss_terms=$4
  local validation_data=$5
  local trial_no=$6
  local alternate_training_set=$7
  local subdir=models
  if [[ $alternate_training_set ]]; then
    subdir+="-$alternate_training_set"
  fi
  printf %s "$base_dir/$subdir/$language/$architecture/$loss_terms/$validation_data/$trial_no"
}

get_language_dir() {
  local base_dir=$1
  local language=$2
  printf %s "$base_dir/languages/$language"
}

string_to_seed() {
  local hash
  hash=$(printf %s "$1" | md5sum | cut -c 1-8)
  printf '%d' "0x$hash"
}

random_sample() {
  local seed_str=$1
  shift 1
  python recognizers/tools/random_sample.py --seed "$(string_to_seed "$seed_str")" "$@"
}

language_sampling_seed() {
  local language_class=$1
  local language_no=$2
  string_to_seed "$language_class-$language_no"
}

STRING_SAMPLING_SEED=123456789

is_saveable_hand_picked_language() {
  local language=$1
  for name in \
    all-strings \
    empty-set \
    repeat-01 \
    even-pairs \
    modular-arithmetic \
    parity \
    cycle-navigation \
    first \
    marked-reversal-cfg
  do
    if [[ $language = $name ]]; then
      return 0
    fi
  done
  if [[ $language =~ ^dyck-[0-9]+-[0-9]+$ ]]; then
    return 0
  fi
  return 1
}
